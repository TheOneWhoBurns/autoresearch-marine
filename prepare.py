"""
Fixed data preparation and evaluation for precipitation nowcasting autoresearch.
DO NOT MODIFY — this is the ground truth data loader and evaluation harness.

Adapted from Karpathy's autoresearch pattern for the SALA Hackathon 2026
RainCaster Galapagos precipitation nowcasting challenge.

Data: 4 weather stations on San Cristobal Island, Galapagos (15-min intervals).
  - CER (Cerro Alto, 517m)
  - JUN (El Junco, 548m)
  - MERC (Merceditas, 100m)
  - MIRA (El Mirador, 387m)

Task: 3-class precipitation nowcasting at +1h, +3h, +6h horizons.
  - Class 0: No rain (0 mm)
  - Class 1: Light rain (0 < sum <= threshold)
  - Class 2: Heavy rain (sum > threshold)

Evaluation: Walk-forward validation over the final 365 days.
Primary metric: composite_score based on macro F1 across all stations + horizons.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 180  # experiment time budget in seconds (3 minutes)

STATION_IDS = ['cer', 'jun', 'merc', 'mira']
STATION_FILES = {
    'cer': 'CER_consolid_f15.csv',
    'jun': 'JUN_consolid_f15.csv',
    'merc': 'MERC_consolid_f15.csv',
    'mira': 'MIRA_consolid_f15.csv',
}

STATION_META = {
    'cer':  {'name': 'Cerro Alto',   'lon': -89.53098555, 'lat': -0.887048868, 'alt_m': 517},
    'jun':  {'name': 'El Junco',     'lon': -89.48162446, 'lat': -0.896537076, 'alt_m': 548},
    'merc': {'name': 'Merceditas',   'lon': -89.44202039, 'lat': -0.889712315, 'alt_m': 100},
    'mira': {'name': 'El Mirador',   'lon': -89.53958685, 'lat': -0.886247558, 'alt_m': 387},
}

# Official 3-class precipitation thresholds from raincaster_guidelines.pdf
HORIZONS = {
    '1h': {'steps': 4,  'light_max_mm': 0.254, 'heavy_min_mm': 0.254},
    '3h': {'steps': 12, 'light_max_mm': 0.508, 'heavy_min_mm': 0.508},
    '6h': {'steps': 24, 'light_max_mm': 0.762, 'heavy_min_mm': 0.762},
}

# Column harmonization: maps canonical name -> list of possible CSV column names
COLUMN_MAP = {
    'rain_mm':          ['Rain_mm_Tot'],
    'temp_c':           ['AirTC_Avg'],
    'rh_avg':           ['RH_Avg'],
    'rh_max':           ['RH_Max'],
    'rh_min':           ['RH_Min'],
    'solar_kw':         ['SlrkW_Avg'],
    'solar_mj':         ['SlrMJ_Tot'],
    'net_rad_wm2':      ['NR_Wm2_Avg'],
    'wind_speed_ms':    ['WS_ms_Avg'],
    'wind_dir':         ['WindDir'],
    'soil_moisture_1':  ['VW_Avg', 'VW'],
    'soil_moisture_2':  ['VW_2_Avg', 'VW_2'],
    'soil_moisture_3':  ['VW_3_Avg', 'VW_3'],
    'leaf_wetness':     ['LWmV_Avg'],
    'leaf_wet_minutes': ['LWMWet_Tot'],
    'cond_1':           ['PA_uS_Avg', 'PA_uS'],
    'cond_2':           ['PA_uS_2_Avg', 'PA_uS_2'],
    'cond_3':           ['PA_uS_3_Avg', 'PA_uS_3'],
}

# Walk-forward: final 365 days is the evaluation period
EVAL_DAYS = 365

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATIONS_DIR = os.path.join(DATA_DIR, "weather_stations")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_station(station_id, stations_dir=None):
    """Load a single station CSV, harmonize columns, return DataFrame with DatetimeIndex."""
    if stations_dir is None:
        stations_dir = STATIONS_DIR
    path = os.path.join(stations_dir, STATION_FILES[station_id])
    df = pd.read_csv(path, low_memory=False)

    # Parse timestamp
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m/%d/%Y %H:%M')
    df = df.set_index('TIMESTAMP').sort_index()

    # Multi-candidate column lookup
    rename = {}
    for harmonized, candidates in COLUMN_MAP.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = harmonized
                break

    df = df[list(rename.keys())].rename(columns=rename)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_all_stations(stations_dir=None):
    """Load all 4 stations and return dict of DataFrames."""
    stations = {}
    for sid in STATION_IDS:
        stations[sid] = load_station(sid, stations_dir)
        df = stations[sid]
        missing = [c for c in COLUMN_MAP if c not in df.columns]
        n_rows = df.shape[0]
        date_range = f"{df.index.min().date()} -> {df.index.max().date()}"
        print(f"  {sid:8s}: {date_range} ({n_rows:,} rows, {len(df.columns)} cols)"
              + (f"  missing={missing}" if missing else ""))
    return stations


def merge_stations(stations):
    """Merge all stations into a wide DataFrame with prefixed columns."""
    prefixed = []
    for sid, sdf in stations.items():
        prefixed.append(sdf.add_prefix(f'{sid}_'))
    df = pd.concat(prefixed, axis=1, sort=True)

    # Drop entirely-NaN columns
    all_nan = [c for c in df.columns if df[c].isnull().all()]
    if all_nan:
        print(f"  Dropping {len(all_nan)} entirely-NaN columns")
        df = df.drop(columns=all_nan)

    return df


# ---------------------------------------------------------------------------
# Imputation (fixed strategy)
# ---------------------------------------------------------------------------

def impute(df):
    """Apply robust imputation strategy."""
    for sid in STATION_IDS:
        # Continuous variables: time-interpolate up to 6h (24 steps)
        for var in ['temp_c', 'rh_avg', 'rh_max', 'rh_min', 'solar_kw', 'solar_mj',
                    'net_rad_wm2', 'soil_moisture_1', 'soil_moisture_2',
                    'soil_moisture_3', 'cond_1', 'cond_2', 'cond_3']:
            col = f'{sid}_{var}'
            if col in df.columns:
                df[col] = df[col].interpolate(method='time', limit=24)

        # Wind speed: forward-fill then interpolate
        col = f'{sid}_wind_speed_ms'
        if col in df.columns:
            df[col] = df[col].ffill(limit=4)
            df[col] = df[col].interpolate(method='time', limit=8)

        # Wind direction: forward-fill only (circular)
        col = f'{sid}_wind_dir'
        if col in df.columns:
            df[col] = df[col].ffill(limit=8)

        # Leaf wetness: forward-fill
        for var in ['leaf_wetness', 'leaf_wet_minutes']:
            col = f'{sid}_{var}'
            if col in df.columns:
                df[col] = df[col].ffill(limit=8)

        # Precipitation: zero-fill + missing indicator
        rain_col = f'{sid}_rain_mm'
        if rain_col in df.columns:
            df[f'{sid}_rain_missing'] = df[rain_col].isnull().astype(float)
            df[rain_col] = df[rain_col].fillna(0.0)

    # Global forward/backward fill for remaining short gaps
    df = df.ffill(limit=96).bfill(limit=96)

    # Fill remaining with 0 (long sensor outages)
    still_nan = df.isnull().sum().sum()
    if still_nan > 0:
        n_cols = (df.isnull().sum() > 0).sum()
        print(f"  Filling {still_nan:,} remaining NaN across {n_cols} columns with 0")
    df = df.fillna(0.0)
    df = df.copy()  # defragment
    return df


# ---------------------------------------------------------------------------
# Label creation (official 3-class per guidelines)
# ---------------------------------------------------------------------------

def create_labels(df):
    """
    Create 3-class precipitation labels for all stations and horizons.

    For each station and horizon, the label at time T is based on
    the sum of Rain_mm_Tot in the future window [T+1, T+horizon_steps].

    Classes (per raincaster_guidelines.pdf):
      0: No rain (sum == 0)
      1: Light rain (0 < sum <= threshold)
      2: Heavy rain (sum > threshold)
    """
    label_cols = []

    for sid in STATION_IDS:
        rain_col = f'{sid}_rain_mm'
        if rain_col not in df.columns:
            continue

        for hname, hinfo in HORIZONS.items():
            steps = hinfo['steps']
            threshold = hinfo['heavy_min_mm']

            # Future precipitation sum (shifted forward)
            future_sum = df[rain_col].rolling(steps, min_periods=1).sum().shift(-steps)

            # 3-class labeling
            label_col = f'label_{hname}_{sid}'
            df[label_col] = np.nan
            df.loc[future_sum == 0, label_col] = 0
            df.loc[(future_sum > 0) & (future_sum <= threshold), label_col] = 1
            df.loc[future_sum > threshold, label_col] = 2

            # Also store the raw future sum for reference
            df[f'future_rain_{hname}_{sid}'] = future_sum

            label_cols.append(label_col)

    return df, label_cols


# ---------------------------------------------------------------------------
# Train/eval split
# ---------------------------------------------------------------------------

def get_split_timestamps(df):
    """
    Return timestamps for the walk-forward split.
    Final 365 days = eval period. Everything before = training.
    """
    last_ts = df.index.max()
    eval_start = last_ts - pd.Timedelta(days=EVAL_DAYS)
    return eval_start


def get_eval_timestamps(df, eval_start):
    """
    Get hourly evaluation timestamps within the eval period.
    Walk-forward: one forecast per hour, on the hour.
    """
    eval_mask = df.index >= eval_start
    eval_ts = df.index[eval_mask]
    # Filter to on-the-hour timestamps
    hourly = eval_ts[eval_ts.minute == 0]
    return hourly


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE)
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Evaluate 3-class predictions using the official hackathon metrics.

    Returns dict with macro_f1, micro_f1, weighted_f1, per_class metrics.
    """
    from sklearn.metrics import f1_score, classification_report

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Handle edge cases
    if len(y_true) == 0:
        return {'macro_f1': 0.0, 'micro_f1': 0.0, 'weighted_f1': 0.0,
                'n_samples': 0, 'class_distribution': {}}

    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class breakdown
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                    target_names=['No rain', 'Light rain', 'Heavy rain'],
                                    output_dict=True, zero_division=0)

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'n_samples': len(y_true),
        'class_distribution': {
            'no_rain': int((y_true == 0).sum()),
            'light_rain': int((y_true == 1).sum()),
            'heavy_rain': int((y_true == 2).sum()),
        },
        'report': report,
    }


def compute_composite_score(all_results):
    """
    Compute the primary composite score from per-station, per-horizon results.

    composite_score = mean of macro_f1 across all (station, horizon) pairs.

    This is the single number that drives keep/discard decisions in the
    autoresearch loop. Higher is better.
    """
    macro_f1s = []
    for key, result in all_results.items():
        if result['n_samples'] > 0:
            macro_f1s.append(result['macro_f1'])

    if not macro_f1s:
        return 0.0

    return float(np.mean(macro_f1s))


# ---------------------------------------------------------------------------
# Full evaluation harness
# ---------------------------------------------------------------------------

def run_evaluation(df, predict_fn, label_cols, eval_start):
    """
    Run walk-forward evaluation using the provided predict_fn.

    predict_fn(df_history, station_id, horizon_name) -> (pred_class, pred_prob)
        - df_history: DataFrame with all data up to current timestamp (T0)
        - station_id: one of 'cer', 'jun', 'merc', 'mira'
        - horizon_name: one of '1h', '3h', '6h'
        - Returns: (int class 0/1/2, float probability for predicted class)

    For efficiency, we sample eval timestamps rather than evaluating every hour.
    The autoresearch loop needs to complete within TIME_BUDGET.
    """
    hourly_ts = get_eval_timestamps(df, eval_start)

    # Sample if too many timestamps (keep it under budget)
    max_eval_points = 500  # enough for reliable metrics, fast enough for iteration
    if len(hourly_ts) > max_eval_points:
        rng = np.random.RandomState(42)
        indices = np.sort(rng.choice(len(hourly_ts), max_eval_points, replace=False))
        hourly_ts = hourly_ts[indices]

    print(f"  Evaluating on {len(hourly_ts)} timestamps "
          f"({eval_start.date()} -> {df.index.max().date()})")

    all_results = {}
    all_predictions = {}

    for sid in STATION_IDS:
        for hname in HORIZONS:
            label_col = f'label_{hname}_{sid}'
            if label_col not in df.columns:
                continue

            y_true_list = []
            y_pred_list = []
            y_prob_list = []

            for ts in hourly_ts:
                # Ground truth
                if ts not in df.index:
                    continue
                true_label = df.loc[ts, label_col]
                if pd.isna(true_label):
                    continue

                # Prediction (model only sees data up to ts)
                try:
                    pred_class, pred_prob = predict_fn(df.loc[:ts], sid, hname)
                    pred_class = int(pred_class)
                except Exception as e:
                    pred_class = 0  # default to "no rain" on error
                    pred_prob = 1.0

                y_true_list.append(int(true_label))
                y_pred_list.append(pred_class)
                y_prob_list.append(pred_prob)

            key = f'{hname}_{sid}'
            result = evaluate_predictions(y_true_list, y_pred_list, y_prob_list)
            all_results[key] = result
            all_predictions[key] = {
                'y_true': y_true_list,
                'y_pred': y_pred_list,
                'y_prob': y_prob_list,
            }

    return all_results, all_predictions


# ---------------------------------------------------------------------------
# Efficient batch evaluation (for models that predict all at once)
# ---------------------------------------------------------------------------

def run_batch_evaluation(df, batch_predict_fn, label_cols, eval_start):
    """
    Alternative evaluation for models that can predict in batch.

    batch_predict_fn(df_train, df_eval_timestamps, station_id, horizon_name)
        -> (pred_classes array, pred_probs array)

    This is much faster than the walk-forward loop for models that don't
    need incremental updates (e.g., sklearn classifiers trained on features).
    """
    hourly_ts = get_eval_timestamps(df, eval_start)

    max_eval_points = 2000
    if len(hourly_ts) > max_eval_points:
        rng = np.random.RandomState(42)
        indices = np.sort(rng.choice(len(hourly_ts), max_eval_points, replace=False))
        hourly_ts = hourly_ts[indices]

    print(f"  Batch evaluating on {len(hourly_ts)} timestamps")

    # Training data: everything before eval_start
    df_train = df[df.index < eval_start]

    all_results = {}

    for sid in STATION_IDS:
        for hname in HORIZONS:
            label_col = f'label_{hname}_{sid}'
            if label_col not in df.columns:
                continue

            # Get ground truth for eval timestamps
            eval_labels = df.loc[hourly_ts, label_col].dropna()
            valid_ts = eval_labels.index

            if len(valid_ts) == 0:
                all_results[f'{hname}_{sid}'] = evaluate_predictions([], [])
                continue

            try:
                pred_classes, pred_probs = batch_predict_fn(
                    df_train, valid_ts, sid, hname
                )
                pred_classes = np.asarray(pred_classes, dtype=int)
            except Exception as e:
                print(f"  ERROR predicting {hname}_{sid}: {e}")
                pred_classes = np.zeros(len(valid_ts), dtype=int)
                pred_probs = np.ones(len(valid_ts))

            y_true = eval_labels.values.astype(int)
            result = evaluate_predictions(y_true, pred_classes, pred_probs)
            all_results[f'{hname}_{sid}'] = result

    return all_results


# ---------------------------------------------------------------------------
# Dataset builder (combines all loading steps)
# ---------------------------------------------------------------------------

def build_dataset(stations_dir=None, cache=True):
    """
    Full pipeline: load -> merge -> impute -> label.

    Returns (df, label_cols, eval_start, feature_cols).
    """
    cache_path = os.path.join(CACHE_DIR, "dataset.pkl") if cache else None

    if cache_path and os.path.exists(cache_path):
        print("  Loading cached dataset...")
        df = pd.read_pickle(cache_path)
        label_cols = [c for c in df.columns if c.startswith('label_')]
        feature_cols = [c for c in df.columns
                        if not c.startswith('label_')
                        and not c.startswith('future_rain_')]
        eval_start = get_split_timestamps(df)
        print(f"  Cached: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df, label_cols, eval_start, feature_cols

    print("  Loading stations...")
    stations = load_all_stations(stations_dir)

    print("  Merging stations...")
    df = merge_stations(stations)
    print(f"  Merged: {df.shape[0]:,} rows x {df.shape[1]} cols")

    print("  Imputing missing values...")
    df = impute(df)

    print("  Creating labels...")
    df, label_cols = create_labels(df)

    # Identify feature columns (everything that's not a label or future rain)
    feature_cols = [c for c in df.columns
                    if not c.startswith('label_')
                    and not c.startswith('future_rain_')]

    eval_start = get_split_timestamps(df)
    print(f"  Eval split: train < {eval_start.date()}, eval >= {eval_start.date()}")

    # Cache
    if cache_path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_pickle(cache_path)
        print(f"  Cached to {cache_path}")

    return df, label_cols, eval_start, feature_cols


# ---------------------------------------------------------------------------
# Main (data prep / verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Precipitation Nowcasting Autoresearch — Data Preparation")
    print("=" * 60)

    if not os.path.isdir(STATIONS_DIR):
        print(f"\nNo weather station data found at {STATIONS_DIR}")
        print("Expected CSV files:")
        for sid, fname in STATION_FILES.items():
            print(f"  {STATIONS_DIR}/{fname}")
        print("\nDownload from R2 using the hackathon credentials.")
        sys.exit(1)

    df, label_cols, eval_start, feature_cols = build_dataset()

    print(f"\nDataset summary:")
    print(f"  Total rows: {df.shape[0]:,}")
    print(f"  Total columns: {df.shape[1]}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Label columns: {len(label_cols)}")
    print(f"  Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Eval start: {eval_start.date()}")

    # Class distribution per station/horizon
    print(f"\nClass distribution (eval period):")
    eval_df = df[df.index >= eval_start]
    for sid in STATION_IDS:
        for hname in HORIZONS:
            col = f'label_{hname}_{sid}'
            if col in eval_df.columns:
                counts = eval_df[col].value_counts().sort_index()
                total = counts.sum()
                print(f"  {col}: " + ", ".join(
                    f"class {int(k)}={int(v)} ({v/total:.1%})" for k, v in counts.items()
                ))

    print("\n" + "=" * 60)
    print("Ready to run experiments: python3 experiment.py")
    print("=" * 60)
