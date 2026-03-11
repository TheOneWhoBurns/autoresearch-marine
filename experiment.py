"""
Precipitation Nowcasting Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Starting point: Tier 1 random forest baseline with lag features.
The agent evolves this through Tiers 1-3, improving the composite_score.

Metric: composite_score = mean weighted F1 across 3h/6h/12h horizons.
Higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
import time
import json
import numpy as np
import pandas as pd

from prepare import (
    HORIZONS, N_CLASSES, CLASS_NAMES,
    PRECIP_THRESHOLD_LIGHT, PRECIP_THRESHOLD_HEAVY,
    TIME_BUDGET, CACHE_DIR, RESULTS_DIR, DEVICE,
    build_dataset, evaluate_predictions, print_detailed_report,
    get_feature_columns,
)

TIER = 1

LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]
USE_TIME_FEATURES = True
USE_LAGS = True
USE_ROLLING = True
USE_WIND_COMPONENTS = True

MODEL = "random_forest"
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_LEAF = 5


def engineer_features(df, feature_cols):
    df = df.copy()

    if USE_TIME_FEATURES and isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        month = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    key_cols = [c for c in ['temp', 'humidity', 'wind_speed', 'solar_kw',
                             'net_radiation', 'soil_moisture_1', 'leaf_wetness_mv',
                             'precip'] if c in df.columns]

    if USE_LAGS:
        for col in key_cols:
            for lag in LAG_HOURS:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

    if USE_ROLLING:
        for col in key_cols:
            for w in ROLLING_WINDOWS:
                df[f'{col}_rmean{w}'] = df[col].rolling(w, min_periods=1).mean()
                df[f'{col}_rstd{w}'] = df[col].rolling(w, min_periods=1).std().fillna(0)

    if USE_WIND_COMPONENTS:
        if 'wind_speed' in df.columns and 'wind_dir' in df.columns:
            wd_rad = np.deg2rad(df['wind_dir'])
            df['wind_u'] = df['wind_speed'] * np.sin(wd_rad)
            df['wind_v'] = df['wind_speed'] * np.cos(wd_rad)

    if 'precip' in df.columns:
        df['precip_sum3'] = df['precip'].rolling(3, min_periods=1).sum()
        df['precip_sum6'] = df['precip'].rolling(6, min_periods=1).sum()
        df['precip_sum12'] = df['precip'].rolling(12, min_periods=1).sum()
        df['precip_sum24'] = df['precip'].rolling(24, min_periods=1).sum()
        df['precip_max3'] = df['precip'].rolling(3, min_periods=1).max()
        df['precip_max6'] = df['precip'].rolling(6, min_periods=1).max()
        df['precip_any_last3'] = (df['precip'].rolling(3, min_periods=1).sum() > 0).astype(int)
        df['precip_any_last6'] = (df['precip'].rolling(6, min_periods=1).sum() > 0).astype(int)
        df['hours_since_rain'] = df['precip'].apply(lambda x: 0 if x > 0.1 else 1).cumsum()

    if 'temp' in df.columns and 'humidity' in df.columns:
        t = df['temp']
        rh = df['humidity'].clip(1, 100)
        df['dewpoint'] = t - ((100 - rh) / 5.0)

    if 'temp' in df.columns:
        df['temp_tendency_3h'] = df['temp'] - df['temp'].shift(3)
        df['temp_tendency_6h'] = df['temp'] - df['temp'].shift(6)

    if 'humidity' in df.columns:
        df['humidity_tendency_3h'] = df['humidity'] - df['humidity'].shift(3)

    if 'solar_kw' in df.columns:
        df['solar_tendency_3h'] = df['solar_kw'] - df['solar_kw'].shift(3)

    return df


def get_all_feature_cols(df):
    exclude = {'precip', '_source_file'}
    exclude.update(c for c in df.columns if c.startswith('target_'))
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def train_and_predict(X_train, y_train, X_val):
    if MODEL == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_val), clf
    else:
        raise ValueError(f"Unknown model: {MODEL}")


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Precipitation Nowcasting — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER, "model": MODEL,
        "lag_hours": LAG_HOURS, "rolling_windows": ROLLING_WINDOWS,
        "use_time_features": USE_TIME_FEATURES,
        "use_lags": USE_LAGS, "use_rolling": USE_ROLLING,
        "use_wind_components": USE_WIND_COMPONENTS,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
    }
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    print("\n--- Loading data ---")
    dataset = build_dataset()
    if dataset is None:
        print("ERROR: No data found. Place CSVs in data/weather_stations/")
        return
    t_load = time.time() - t_start
    print(f"Data loaded: {t_load:.1f}s")

    train_df = dataset['train_df']
    val_df = dataset['val_df']
    raw_feature_cols = dataset['feature_columns']

    print("\n--- Engineering features ---")
    t1 = time.time()
    train_eng = engineer_features(train_df, raw_feature_cols)
    val_eng = engineer_features(val_df, raw_feature_cols)
    feat_cols = get_all_feature_cols(train_eng)
    print(f"Features: {len(feat_cols)} columns, {time.time()-t1:.1f}s")

    max_lag = max(LAG_HOURS) if USE_LAGS else 0
    train_eng = train_eng.iloc[max_lag:]
    val_eng = val_eng.iloc[max_lag:]

    train_eng = train_eng.fillna(0)
    val_eng = val_eng.fillna(0)

    feat_cols = [c for c in feat_cols if c in train_eng.columns and c in val_eng.columns]

    X_train_full = train_eng[feat_cols].values.astype(np.float32)
    X_val_full = val_eng[feat_cols].values.astype(np.float32)

    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_full = np.nan_to_num(X_val_full, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Train samples: {len(X_train_full)}, Val samples: {len(X_val_full)}")

    y_true_dict = {}
    y_pred_dict = {}

    for h in HORIZONS:
        target_col = f'target_{h}h'
        if target_col not in train_eng.columns:
            print(f"WARNING: {target_col} not found, skipping")
            continue

        print(f"\n--- Training {h}h horizon model ---")
        t1 = time.time()

        y_train = train_eng[target_col].values.astype(int)
        y_val = val_eng[target_col].values.astype(int)

        train_mask = y_train >= 0
        val_mask = y_val >= 0

        preds, model = train_and_predict(
            X_train_full[train_mask], y_train[train_mask],
            X_val_full[val_mask],
        )

        y_true_dict[f'{h}h'] = y_val[val_mask]
        y_pred_dict[f'{h}h'] = preds

        print(f"  {h}h model trained: {time.time()-t1:.1f}s")

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            print(f"  Top features:")
            for idx in top_idx:
                print(f"    {feat_cols[idx]}: {importances[idx]:.4f}")

    eval_result = evaluate_predictions(y_true_dict, y_pred_dict, method_name=f"tier{TIER}")
    print_detailed_report(y_true_dict, y_pred_dict)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump({"config": config, "evaluation": eval_result}, f, indent=2)

    t_total = time.time() - t_start
    print("\n---")
    print(f"composite_score:  {eval_result['composite_score']:.6f}")
    for h in HORIZONS:
        key = f'f1_{h}h'
        if key in eval_result:
            print(f"f1_{h}h:            {eval_result[key]:.6f}")
    print(f"n_features:       {len(feat_cols)}")
    print(f"train_samples:    {len(X_train_full)}")
    print(f"val_samples:      {len(X_val_full)}")
    print(f"tier:             {TIER}")
    print(f"model:            {MODEL}")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
