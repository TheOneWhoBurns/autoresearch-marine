"""
Precipitation Nowcasting Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Metric: composite_score = mean weighted F1 across 3h/6h/12h horizons.
Higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from prepare import (
    HORIZONS, N_CLASSES, CLASS_NAMES,
    TIME_BUDGET, CACHE_DIR, RESULTS_DIR, DEVICE,
    build_dataset, evaluate_predictions, print_detailed_report,
    get_feature_columns, load_single_station, resample_to_hourly,
    STATION_DIR,
)

TIER = 1

LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 18
RF_MIN_SAMPLES_LEAF = 3


def build_cross_station_precip():
    from pathlib import Path
    csv_files = sorted(Path(STATION_DIR).glob("*.csv"))

    station_precips = {}
    for fpath in csv_files:
        name = fpath.stem.split('_')[0]
        df = load_single_station(fpath, name)
        hourly = resample_to_hourly(df)
        station_precips[name] = hourly['precip'].fillna(0.0)

    all_timestamps = set()
    for s in station_precips.values():
        all_timestamps.update(s.index)
    idx = pd.DatetimeIndex(sorted(all_timestamps))

    precip_df = pd.DataFrame({f'precip_{n}': s.reindex(idx).fillna(0)
                               for n, s in station_precips.items()})

    cross = pd.DataFrame(index=idx)
    cross['precip_any_station'] = (precip_df > 0.1).any(axis=1).astype(int)
    cross['precip_n_stations'] = (precip_df > 0.1).sum(axis=1)
    cross['precip_mean_stations'] = precip_df.mean(axis=1)
    cross['precip_max_stations'] = precip_df.max(axis=1)
    for w in [3, 6]:
        cross[f'precip_any_station_sum{w}'] = cross['precip_any_station'].rolling(w, min_periods=1).sum()
    return cross


def engineer_features(df, cross_features=None):
    new_cols = {}

    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        new_cols['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        new_cols['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        month = df.index.month
        new_cols['month_sin'] = np.sin(2 * np.pi * month / 12)
        new_cols['month_cos'] = np.cos(2 * np.pi * month / 12)
        new_cols['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        new_cols['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    key_cols = [c for c in ['temp', 'humidity', 'wind_speed', 'solar_kw',
                             'net_radiation', 'soil_moisture_1', 'leaf_wetness_mv',
                             'precip'] if c in df.columns]

    for col in key_cols:
        for lag in LAG_HOURS:
            new_cols[f'{col}_lag{lag}'] = df[col].shift(lag)

    for col in key_cols:
        for w in ROLLING_WINDOWS:
            new_cols[f'{col}_rmean{w}'] = df[col].rolling(w, min_periods=1).mean()
            new_cols[f'{col}_rstd{w}'] = df[col].rolling(w, min_periods=1).std().fillna(0)

    if 'wind_speed' in df.columns and 'wind_dir' in df.columns:
        wd_rad = np.deg2rad(df['wind_dir'])
        new_cols['wind_u'] = df['wind_speed'] * np.sin(wd_rad)
        new_cols['wind_v'] = df['wind_speed'] * np.cos(wd_rad)

    if 'precip' in df.columns:
        new_cols['precip_sum3'] = df['precip'].rolling(3, min_periods=1).sum()
        new_cols['precip_sum6'] = df['precip'].rolling(6, min_periods=1).sum()
        new_cols['precip_sum12'] = df['precip'].rolling(12, min_periods=1).sum()
        new_cols['precip_sum24'] = df['precip'].rolling(24, min_periods=1).sum()
        new_cols['precip_max3'] = df['precip'].rolling(3, min_periods=1).max()
        new_cols['precip_max6'] = df['precip'].rolling(6, min_periods=1).max()
        new_cols['precip_any_last3'] = (df['precip'].rolling(3, min_periods=1).sum() > 0).astype(int)
        new_cols['precip_any_last6'] = (df['precip'].rolling(6, min_periods=1).sum() > 0).astype(int)
        is_rain = (df['precip'] > 0.1).astype(int)
        groups = is_rain.cumsum()
        dry_counter = groups.groupby(groups).cumcount()
        new_cols['hours_since_rain'] = dry_counter * (1 - is_rain)

    if 'soil_moisture_1' in df.columns:
        new_cols['soil_moisture_diff1'] = df['soil_moisture_1'].diff(1)
        new_cols['soil_moisture_diff3'] = df['soil_moisture_1'].diff(3)
    if 'soil_moisture_2' in df.columns:
        new_cols['soil_moisture_2_diff3'] = df['soil_moisture_2'].diff(3)

    if 'leaf_wetness_mv' in df.columns:
        new_cols['leaf_wetness_rmean3'] = df['leaf_wetness_mv'].rolling(3, min_periods=1).mean()
        new_cols['leaf_wetness_diff1'] = df['leaf_wetness_mv'].diff(1)
    for col in ['leaf_wet_min', 'leaf_condensation_min']:
        if col in df.columns:
            new_cols[f'{col}_sum6'] = df[col].rolling(6, min_periods=1).sum()

    if 'temp' in df.columns and 'humidity' in df.columns:
        t = df['temp']
        rh = df['humidity'].clip(1, 100)
        new_cols['dewpoint'] = t - ((100 - rh) / 5.0)

    for col in ['temp', 'humidity']:
        if col in df.columns:
            new_cols[f'{col}_diff3'] = df[col].diff(3)
            new_cols[f'{col}_diff6'] = df[col].diff(6)

    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        new_cols['temp_range'] = df['temp_max'] - df['temp_min']
    if 'humidity_max' in df.columns and 'humidity_min' in df.columns:
        new_cols['humidity_range'] = df['humidity_max'] - df['humidity_min']

    result = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if cross_features is not None:
        cross_aligned = cross_features.reindex(result.index).fillna(0)
        result = pd.concat([result, cross_aligned], axis=1)

    return result


def get_all_feature_cols(df):
    exclude = {'precip', '_source_file'}
    exclude.update(c for c in df.columns if c.startswith('target_'))
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Precipitation Nowcasting — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER, "model": "rf_cross_station",
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
    }
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    print("\n--- Loading data ---")
    dataset = build_dataset()
    if dataset is None:
        print("ERROR: No data found.")
        return
    t_load = time.time() - t_start
    print(f"Data loaded: {t_load:.1f}s")

    print("\n--- Building cross-station features ---")
    t1 = time.time()
    cross_features = build_cross_station_precip()
    print(f"Cross-station features: {cross_features.shape[1]} columns, {time.time()-t1:.1f}s")

    train_df = dataset['train_df']
    val_df = dataset['val_df']

    print("\n--- Engineering features ---")
    t1 = time.time()
    train_eng = engineer_features(train_df, cross_features)
    val_eng = engineer_features(val_df, cross_features)
    feat_cols = get_all_feature_cols(train_eng)
    print(f"Features: {len(feat_cols)} columns, {time.time()-t1:.1f}s")

    max_lag = max(LAG_HOURS)
    train_eng = train_eng.iloc[max_lag:]
    val_eng = val_eng.iloc[max_lag:]
    train_eng = train_eng.fillna(0)
    val_eng = val_eng.fillna(0)

    feat_cols = [c for c in feat_cols if c in train_eng.columns and c in val_eng.columns]

    X_train = train_eng[feat_cols].values.astype(np.float32)
    X_val = val_eng[feat_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    from sklearn.ensemble import RandomForestClassifier

    y_true_dict = {}
    y_pred_dict = {}

    for h in HORIZONS:
        target_col = f'target_{h}h'
        if target_col not in train_eng.columns:
            continue

        print(f"\n--- Training {h}h horizon model ---")
        t1 = time.time()

        y_train = train_eng[target_col].values.astype(int)
        y_val = val_eng[target_col].values.astype(int)

        train_mask = y_train >= 0
        val_mask = y_val >= 0

        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train[train_mask], y_train[train_mask])
        preds = clf.predict(X_val[val_mask])

        y_true_dict[f'{h}h'] = y_val[val_mask]
        y_pred_dict[f'{h}h'] = preds

        print(f"  {h}h trained: {time.time()-t1:.1f}s")

        importances = clf.feature_importances_
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
    print(f"train_samples:    {len(X_train)}")
    print(f"val_samples:      {len(X_val)}")
    print(f"tier:             {TIER}")
    print(f"model:            rf_cross_station")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
