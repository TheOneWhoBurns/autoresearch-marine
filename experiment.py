"""
Precipitation Nowcasting Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Starting point: Tier 1 gradient boosting with lag features.
The agent evolves this through tiers, improving composite_score (mean macro F1).

Metric: composite_score from prepare.compute_composite_score() — higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from prepare import (
    STATION_IDS, HORIZONS, TIME_BUDGET,
    CACHE_DIR, RESULTS_DIR,
    build_dataset, run_batch_evaluation,
    compute_composite_score, evaluate_predictions,
)

# ---------------------------------------------------------------------------
# TIER: current approach (agent updates this as it progresses)
# ---------------------------------------------------------------------------
TIER = 1  # 1=lag features+GBM, 2=richer features+tuned model, 3=deep learning

# ---------------------------------------------------------------------------
# Hyperparameters (agent tunes these)
# ---------------------------------------------------------------------------

# Feature engineering
LOOKBACK_HOURS = 6          # how many hours of history to use for lag features
USE_ROLLING_STATS = True    # rolling mean/std/min/max of weather variables
USE_CYCLICAL_TIME = True    # sin/cos encoding of hour-of-day, day-of-year
USE_CROSS_STATION = True    # include features from all stations (not just target)
USE_DEWPOINT = True         # compute dewpoint and dewpoint depression
USE_WIND_VECTOR = True      # decompose wind into x/y components
USE_SOIL_TENDENCY = True    # soil moisture change over recent hours
USE_RAIN_HISTORY = True     # cumulative rain in recent windows

# Model
MODEL_TYPE = "gbm"          # "gbm", "rf", "logistic", "mlp"
N_ESTIMATORS = 200
MAX_DEPTH = 5
LEARNING_RATE = 0.1
MIN_SAMPLES_LEAF = 20
SUBSAMPLE = 0.8
CLASS_WEIGHT = "balanced"   # handle class imbalance

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, feature_cols):
    """
    Build feature matrix from the merged weather data.
    Returns a new DataFrame with engineered features only.
    """
    eng = pd.DataFrame(index=df.index)

    # --- Cyclical time features ---
    if USE_CYCLICAL_TIME:
        eng['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        eng['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        eng['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        eng['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        eng['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        eng['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # --- Per-station features ---
    stations_to_use = STATION_IDS if USE_CROSS_STATION else []

    for sid in STATION_IDS:
        prefix = sid

        # Current raw values (always include target station variables)
        for var in ['rain_mm', 'temp_c', 'rh_avg', 'rh_max', 'solar_kw',
                    'net_rad_wm2', 'wind_speed_ms', 'soil_moisture_1',
                    'soil_moisture_2', 'soil_moisture_3', 'leaf_wetness',
                    'cond_1']:
            col = f'{prefix}_{var}'
            if col in df.columns:
                eng[col] = df[col]

        # --- Wind vector decomposition ---
        if USE_WIND_VECTOR:
            wd_col = f'{prefix}_wind_dir'
            ws_col = f'{prefix}_wind_speed_ms'
            if wd_col in df.columns and ws_col in df.columns:
                wd_rad = np.deg2rad(df[wd_col])
                eng[f'{prefix}_wind_x'] = df[ws_col] * np.cos(wd_rad)
                eng[f'{prefix}_wind_y'] = df[ws_col] * np.sin(wd_rad)

        # --- Dewpoint ---
        if USE_DEWPOINT:
            temp_col = f'{prefix}_temp_c'
            rh_col = f'{prefix}_rh_avg'
            if temp_col in df.columns and rh_col in df.columns:
                T = df[temp_col]
                RH = df[rh_col].clip(lower=1)
                alpha = (17.27 * T) / (237.3 + T) + np.log(RH / 100)
                eng[f'{prefix}_dewpoint'] = (237.3 * alpha) / (17.27 - alpha)
                eng[f'{prefix}_dewpoint_depression'] = T - eng[f'{prefix}_dewpoint']

        # --- Rolling statistics ---
        if USE_ROLLING_STATS:
            for window_steps, wlabel in [(4, '1h'), (12, '3h'), (24, '6h')]:
                rain_col = f'{prefix}_rain_mm'
                if rain_col in df.columns and USE_RAIN_HISTORY:
                    eng[f'{prefix}_rain_sum_{wlabel}'] = (
                        df[rain_col].rolling(window_steps, min_periods=1).sum()
                    )
                    eng[f'{prefix}_rain_max_{wlabel}'] = (
                        df[rain_col].rolling(window_steps, min_periods=1).max()
                    )
                    eng[f'{prefix}_rain_count_{wlabel}'] = (
                        (df[rain_col] > 0).rolling(window_steps, min_periods=1).sum()
                    )

                temp_col = f'{prefix}_temp_c'
                if temp_col in df.columns:
                    eng[f'{prefix}_temp_mean_{wlabel}'] = (
                        df[temp_col].rolling(window_steps, min_periods=1).mean()
                    )
                    eng[f'{prefix}_temp_std_{wlabel}'] = (
                        df[temp_col].rolling(window_steps, min_periods=1).std()
                    )

                rh_col = f'{prefix}_rh_avg'
                if rh_col in df.columns:
                    eng[f'{prefix}_rh_mean_{wlabel}'] = (
                        df[rh_col].rolling(window_steps, min_periods=1).mean()
                    )

                ws_col = f'{prefix}_wind_speed_ms'
                if ws_col in df.columns:
                    eng[f'{prefix}_wind_mean_{wlabel}'] = (
                        df[ws_col].rolling(window_steps, min_periods=1).mean()
                    )

        # --- Soil moisture tendency ---
        if USE_SOIL_TENDENCY:
            for var in ['soil_moisture_1', 'soil_moisture_2', 'soil_moisture_3']:
                col = f'{prefix}_{var}'
                if col in df.columns:
                    eng[f'{prefix}_{var}_diff_3h'] = df[col].diff(periods=12)
                    eng[f'{prefix}_{var}_diff_6h'] = df[col].diff(periods=24)

    # --- Rain missing indicators ---
    for sid in STATION_IDS:
        miss_col = f'{sid}_rain_missing'
        if miss_col in df.columns:
            eng[miss_col] = df[miss_col]

    # Fill any NaN from rolling/diff operations
    eng = eng.fillna(0.0)

    return eng


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model():
    """Create the classifier based on MODEL_TYPE."""
    if MODEL_TYPE == "gbm":
        return GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            subsample=SUBSAMPLE,
            random_state=42,
        )
    elif MODEL_TYPE == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight=CLASS_WEIGHT,
            random_state=42,
            n_jobs=-1,
        )
    elif MODEL_TYPE == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=1.0, max_iter=1000, class_weight=CLASS_WEIGHT,
            multi_class='multinomial', random_state=42,
        )
    elif MODEL_TYPE == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

def train_and_predict(df_eng, df_labels, eval_start, label_col):
    """
    Train model on data before eval_start, predict on eval timestamps.
    Returns (model, predictions_df).
    """
    # Split
    train_mask = df_eng.index < eval_start
    eval_mask = df_eng.index >= eval_start

    # Align with labels
    valid_train = train_mask & df_labels[label_col].notna()
    valid_eval = eval_mask & df_labels[label_col].notna()

    # Filter to hourly timestamps for eval
    hourly_eval = valid_eval & (df_eng.index.minute == 0)

    X_train = df_eng.loc[valid_train].values
    y_train = df_labels.loc[valid_train, label_col].values.astype(int)
    X_eval = df_eng.loc[hourly_eval].values
    y_eval = df_labels.loc[hourly_eval, label_col].values.astype(int)

    if len(X_train) == 0 or len(X_eval) == 0:
        return None, np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Handle NaN/inf from scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_eval_scaled = np.nan_to_num(X_eval_scaled, nan=0, posinf=0, neginf=0)

    # Sample training data if too large (for speed)
    max_train = 100000
    if len(X_train_scaled) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train_scaled), max_train, replace=False)
        X_train_scaled = X_train_scaled[idx]
        y_train = y_train[idx]

    # Train
    model = build_model()
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_eval_scaled)
    y_prob = model.predict_proba(X_eval_scaled)
    # Probability of predicted class
    pred_probs = np.array([y_prob[i, p] for i, p in enumerate(y_pred)])

    return model, y_eval, y_pred, pred_probs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Precipitation Nowcasting Autoresearch — Tier {TIER}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "model_type": MODEL_TYPE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "lookback_hours": LOOKBACK_HOURS,
        "use_rolling_stats": USE_ROLLING_STATS,
        "use_cyclical_time": USE_CYCLICAL_TIME,
        "use_cross_station": USE_CROSS_STATION,
        "use_dewpoint": USE_DEWPOINT,
        "use_wind_vector": USE_WIND_VECTOR,
        "use_soil_tendency": USE_SOIL_TENDENCY,
        "use_rain_history": USE_RAIN_HISTORY,
    }
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    # --- Load data ---
    print("\n--- Loading data ---")
    df, label_cols, eval_start, feature_cols = build_dataset()
    t_load = time.time() - t_start
    print(f"Data loaded: {t_load:.1f}s")

    # --- Engineer features ---
    print("\n--- Engineering features ---")
    t1 = time.time()
    df_eng = engineer_features(df, feature_cols)
    eng_cols = list(df_eng.columns)
    print(f"Engineered features: {len(eng_cols)} columns, {time.time()-t1:.1f}s")

    # --- Train and evaluate per station/horizon ---
    print("\n--- Training and evaluating ---")
    all_results = {}
    models = {}

    for sid in STATION_IDS:
        for hname in HORIZONS:
            label_col = f'label_{hname}_{sid}'
            if label_col not in df.columns:
                continue

            key = f'{hname}_{sid}'
            t1 = time.time()

            model, y_true, y_pred, y_prob = train_and_predict(
                df_eng, df[label_cols], eval_start, label_col
            )

            if len(y_true) == 0:
                all_results[key] = evaluate_predictions([], [])
                continue

            result = evaluate_predictions(y_true, y_pred, y_prob)
            all_results[key] = result
            models[key] = model

            elapsed = time.time() - t1
            print(f"  {key}: macro_f1={result['macro_f1']:.3f} "
                  f"micro_f1={result['micro_f1']:.3f} "
                  f"weighted_f1={result['weighted_f1']:.3f} "
                  f"({result['n_samples']} samples, {elapsed:.1f}s)")

            # Time check
            if time.time() - t_start > TIME_BUDGET * 0.9:
                print(f"  WARNING: Approaching time budget, skipping remaining")
                break
        if time.time() - t_start > TIME_BUDGET * 0.9:
            break

    # --- Composite score ---
    composite = compute_composite_score(all_results)

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.json")
    save_data = {"config": config, "results": {}}
    for key, result in all_results.items():
        save_result = {k: v for k, v in result.items() if k != 'report'}
        save_data["results"][key] = save_result
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # --- Feature importance (top model) ---
    if models:
        best_key = max(all_results, key=lambda k: all_results[k].get('macro_f1', 0))
        best_model = models.get(best_key)
        if best_model and hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:15]
            print(f"\n--- Top features ({best_key}) ---")
            for i in top_idx:
                if i < len(eng_cols):
                    print(f"  {eng_cols[i]:40s} {importances[i]:.4f}")

    # --- Final summary (parseable by autoresearch loop) ---
    t_total = time.time() - t_start
    print("\n---")
    print(f"composite_score:  {composite:.6f}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"macro_f1_{key}: {r['macro_f1']:.6f}")

    n_models = len([r for r in all_results.values() if r['n_samples'] > 0])
    avg_weighted = np.mean([r['weighted_f1'] for r in all_results.values() if r['n_samples'] > 0]) if n_models > 0 else 0
    avg_micro = np.mean([r['micro_f1'] for r in all_results.values() if r['n_samples'] > 0]) if n_models > 0 else 0

    print(f"avg_weighted_f1:  {avg_weighted:.6f}")
    print(f"avg_micro_f1:     {avg_micro:.6f}")
    print(f"n_models:         {n_models}")
    print(f"n_features:       {len(eng_cols)}")
    print(f"tier:             {TIER}")
    print(f"model_type:       {MODEL_TYPE}")
    print(f"total_seconds:    {t_total:.1f}")

    # Per-station/horizon detail
    print(f"\n--- Detail ---")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        dist = r.get('class_distribution', {})
        print(f"  {key}: macro={r['macro_f1']:.3f} micro={r['micro_f1']:.3f} "
              f"weighted={r['weighted_f1']:.3f} "
              f"[no_rain={dist.get('no_rain',0)}, light={dist.get('light_rain',0)}, heavy={dist.get('heavy_rain',0)}]")


if __name__ == "__main__":
    main()
