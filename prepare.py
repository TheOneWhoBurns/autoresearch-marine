"""
Fixed data preparation and evaluation for precipitation nowcasting autoresearch.
DO NOT MODIFY — this is the ground truth data loader and evaluation harness.

Data: Weather station observations from San Cristóbal Island, Galápagos.
  4 stations (CER, JUN, MERC, MIRA), 15-min intervals, ~10 years.
Task: Predict precipitation class at 3h, 6h, 12h horizons.
Classes: 0=no_rain (<0.1mm), 1=light_rain (0.1-2.0mm), 2=heavy_rain (>2.0mm)
Metric: Weighted F1-score averaged across horizons.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report


def detect_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"

DEVICE = detect_device()

HORIZONS = [3, 6, 12]
PRECIP_THRESHOLD_LIGHT = 0.1
PRECIP_THRESHOLD_HEAVY = 2.0
CLASS_NAMES = ["no_rain", "light_rain", "heavy_rain"]
N_CLASSES = 3

VAL_FRACTION = 0.2
TIME_BUDGET = 300

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATION_DIR = os.path.join(DATA_DIR, "weather_stations")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

STATION_NAMES = ["CER", "JUN", "MERC", "MIRA"]

COLUMN_RENAME = {
    'TIMESTAMP': 'datetime',
    'Rain_mm_Tot': 'precip',
    'AirTC_Avg': 'temp',
    'AirTC_Max': 'temp_max',
    'AirTC_Min': 'temp_min',
    'RH_Avg': 'humidity',
    'RH_Max': 'humidity_max',
    'RH_Min': 'humidity_min',
    'WS_ms_Avg': 'wind_speed',
    'WS_ms_Max': 'wind_speed_max',
    'WS_ms_Min': 'wind_speed_min',
    'WindDir': 'wind_dir',
    'WindDir_Avg': 'wind_dir_avg',
    'WindDir_Max': 'wind_dir_max',
    'WindDir_Min': 'wind_dir_min',
    'SlrkW_Avg': 'solar_kw',
    'SlrW_Max': 'solar_max',
    'SlrW_Min': 'solar_min',
    'SlrMJ_Tot': 'solar_mj_tot',
    'NR_Wm2_Avg': 'net_radiation',
    'NR_Wm2_Max': 'net_radiation_max',
    'NR_Wm2_Min': 'net_radiation_min',
    'VW': 'soil_moisture_1',
    'VW_Avg': 'soil_moisture_1',
    'VW_2': 'soil_moisture_2',
    'VW_2_Avg': 'soil_moisture_2',
    'VW_3': 'soil_moisture_3',
    'VW_3_Avg': 'soil_moisture_3',
    'LWmV_Avg': 'leaf_wetness_mv',
    'LWMDry_Tot': 'leaf_dry_min',
    'LWMCon_Tot': 'leaf_condensation_min',
    'LWMWet_Tot': 'leaf_wet_min',
    'BattV_Avg': 'battery_v',
    'PTemp_C_Avg': 'panel_temp',
}

SUM_COLS = {'precip', 'solar_mj_tot', 'leaf_dry_min', 'leaf_condensation_min', 'leaf_wet_min'}
MAX_COLS = {'temp_max', 'humidity_max', 'wind_speed_max', 'wind_dir_max', 'solar_max',
            'net_radiation_max'}
MIN_COLS = {'temp_min', 'humidity_min', 'wind_speed_min', 'wind_dir_min', 'solar_min',
            'net_radiation_min'}


def classify_precip(values):
    values = np.asarray(values, dtype=np.float64)
    classes = np.zeros(len(values), dtype=np.int64)
    classes[values >= PRECIP_THRESHOLD_LIGHT] = 1
    classes[values > PRECIP_THRESHOLD_HEAVY] = 2
    return classes


def load_single_station(filepath, station_name=None):
    df = pd.read_csv(filepath, na_values=['NA', 'NAN', 'nan', '', 'NaN'])

    rename_map = {}
    for orig, new in COLUMN_RENAME.items():
        if orig in df.columns:
            rename_map[orig] = new
    df = df.rename(columns=rename_map)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df.sort_values('datetime')
        df = df.set_index('datetime')

    drop_cols = [c for c in ['RECORD', 'battery_v', 'panel_temp'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    df = df.drop(columns=non_numeric, errors='ignore')

    if station_name:
        df['station_id'] = STATION_NAMES.index(station_name) if station_name in STATION_NAMES else 0

    return df


def resample_to_hourly(df):
    agg_dict = {}
    for col in df.columns:
        if col == 'station_id':
            agg_dict[col] = 'first'
        elif col in SUM_COLS:
            agg_dict[col] = 'sum'
        elif col in MAX_COLS:
            agg_dict[col] = 'max'
        elif col in MIN_COLS:
            agg_dict[col] = 'min'
        else:
            agg_dict[col] = 'mean'

    hourly = df.resample('1h').agg(agg_dict)
    return hourly


def load_all_stations(station_dir=None):
    if station_dir is None:
        station_dir = STATION_DIR

    csv_files = sorted(Path(station_dir).glob("*.csv"))
    if not csv_files:
        return None

    all_frames = []
    for fpath in csv_files:
        station_name = fpath.stem.split('_')[0]
        print(f"  Loading: {fpath.name} (station={station_name})")
        df = load_single_station(fpath, station_name)
        print(f"    Raw: {len(df)} rows, {df.index.min()} to {df.index.max()}")

        hourly = resample_to_hourly(df)
        hourly['precip'] = hourly['precip'].fillna(0.0)
        print(f"    Hourly: {len(hourly)} rows")
        all_frames.append(hourly)

    merged = pd.concat(all_frames, ignore_index=False)
    merged = merged.sort_index()
    return merged


def create_horizon_targets(df, horizons=None):
    if horizons is None:
        horizons = HORIZONS

    target_cols = []
    for h in horizons:
        col_name = f'target_{h}h'
        future_precip = df['precip'].shift(-h)
        df[col_name] = classify_precip(future_precip.values)
        df.loc[future_precip.isna(), col_name] = -1
        target_cols.append(col_name)

    return df, target_cols


def temporal_split(df, val_fraction=VAL_FRACTION):
    n = len(df)
    split_idx = int(n * (1 - val_fraction))
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train, val


def get_feature_columns(df):
    exclude = {'precip', '_source_file'}
    exclude.update(c for c in df.columns if c.startswith('target_'))
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
    return feature_cols


def build_dataset(station_dir=None, horizons=None):
    if horizons is None:
        horizons = HORIZONS

    print("Loading station data...")
    df = load_all_stations(station_dir)
    if df is None:
        return None

    print(f"  Merged shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=6)

    print("Creating horizon targets...")
    df, target_cols = create_horizon_targets(df, horizons)

    valid_mask = (df[target_cols] >= 0).all(axis=1)
    df = df[valid_mask]
    print(f"  Valid samples: {len(df)}")

    print("Splitting train/val...")
    train_df, val_df = temporal_split(df)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    for h in horizons:
        col = f'target_{h}h'
        counts = train_df[col].value_counts().sort_index()
        print(f"  Train target_{h}h distribution: {dict(counts)}")

    return {
        'train_df': train_df,
        'val_df': val_df,
        'feature_columns': get_feature_columns(train_df),
        'target_columns': [f'target_{h}h' for h in horizons],
        'horizons': horizons,
    }


def evaluate_predictions(y_true_dict, y_pred_dict, method_name=""):
    results = {'method': method_name}
    f1_scores = []

    for h in HORIZONS:
        key = f'{h}h'
        if key not in y_true_dict or key not in y_pred_dict:
            continue

        y_true = np.asarray(y_true_dict[key])
        y_pred = np.asarray(y_pred_dict[key])

        mask = y_true >= 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_scores.append(f1)

        results[f'f1_{h}h'] = float(f1)
        results[f'n_samples_{h}h'] = int(len(y_true))

        class_counts_true = {CLASS_NAMES[i]: int((y_true == i).sum()) for i in range(N_CLASSES)}
        class_counts_pred = {CLASS_NAMES[i]: int((y_pred == i).sum()) for i in range(N_CLASSES)}
        results[f'true_dist_{h}h'] = class_counts_true
        results[f'pred_dist_{h}h'] = class_counts_pred

    results['composite_score'] = float(np.mean(f1_scores)) if f1_scores else 0.0
    results['n_horizons'] = len(f1_scores)
    return results


def print_detailed_report(y_true_dict, y_pred_dict):
    for h in HORIZONS:
        key = f'{h}h'
        if key not in y_true_dict:
            continue
        y_true = np.asarray(y_true_dict[key])
        y_pred = np.asarray(y_pred_dict[key])
        mask = y_true >= 0
        print(f"\n--- Classification Report: {h}h horizon ---")
        print(classification_report(y_true[mask], y_pred[mask],
                                     target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    print("=" * 60)
    print("Precipitation Nowcasting — Data Preparation")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    os.makedirs(STATION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = build_dataset()
    if result is None:
        print(f"\nNo data files found in {STATION_DIR}")
        sys.exit(1)

    print(f"\nFeature columns ({len(result['feature_columns'])}):")
    for col in result['feature_columns']:
        print(f"  {col}")

    print(f"\nTarget columns: {result['target_columns']}")
    print(f"\nDate range (train): {result['train_df'].index.min()} to {result['train_df'].index.max()}")
    print(f"Date range (val):   {result['val_df'].index.min()} to {result['val_df'].index.max()}")

    precip = result['train_df']['precip']
    print(f"\nPrecip stats (train):")
    print(f"  Zero: {(precip == 0).sum()} ({(precip == 0).mean()*100:.1f}%)")
    print(f"  Light (0.1-2mm): {((precip >= 0.1) & (precip <= 2.0)).sum()}")
    print(f"  Heavy (>2mm): {(precip > 2.0).sum()}")
    print(f"  Max: {precip.max():.1f}mm, Mean: {precip.mean():.3f}mm")

    print("\n" + "=" * 60)
    print("Ready to run experiments: python3 experiment.py")
    print("=" * 60)
