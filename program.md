# Precipitation Nowcasting — Autoresearch Program

## Challenge

SALA Hackathon 2026 RainCaster Galapagos: predict precipitation intensity at
+1h, +3h, and +6h horizons for 4 weather stations on San Cristobal Island.

3-class labels per the official guidelines:
- **+1h**: No rain (0mm), Light (0-0.254mm), Heavy (>0.254mm)
- **+3h**: No rain (0mm), Light (0-0.508mm), Heavy (>0.508mm)
- **+6h**: No rain (0mm), Light (0-0.762mm), Heavy (>0.762mm)

Evaluation: walk-forward validation over the final 365 days, macro F1 metric.

## Data

4 stations, 15-min intervals, June 2015 – March 2026 (~375K rows each):
- `CER` (Cerro Alto, 517m) — highland
- `JUN` (El Junco, 548m) — summit freshwater lake
- `MERC` (Merceditas, 100m) — lowland agricultural
- `MIRA` (El Mirador, 387m) — coastal/mid-elevation

Key variables: Rain_mm_Tot, AirTC_Avg, RH_Avg/Max/Min, SlrkW_Avg, NR_Wm2_Avg,
WS_ms_Avg, WindDir, VW (soil moisture x3 horizons), LWmV_Avg, leaf wetness.

Supplementary: NASA LDAS gridded data (daily, 2015-2021) — not yet integrated.

## File Layout

```
prepare.py      — FIXED data loader + evaluation harness (DO NOT MODIFY)
experiment.py   — THE ONLY FILE THE AGENT EDITS
program.md      — this file (research guide)
results.tsv     — experiment tracking log
data/
  weather_stations/   — CSV files from R2 download
  cache/              — cached preprocessed data
  results/            — per-run JSON results
```

## Setup

1. Download data: place CSVs in `data/weather_stations/`
2. Verify: `python3 prepare.py` (shows station summaries + class distribution)
3. First run: `python3 experiment.py > run.log 2>&1`
4. Check: `grep "^composite_score:" run.log`

## How the Autoresearch Loop Works

```
LOOP FOREVER:
1. Check git state
2. Edit experiment.py with next idea
3. git commit -m "experiment: <description>"
4. python3 experiment.py > run.log 2>&1
5. grep "^composite_score:" run.log
6. If empty → crash (debug with: tail -n 50 run.log)
7. Log to results.tsv
8. If composite_score improved → keep commit
9. If worse → git reset --hard HEAD~1
```

TIME_BUDGET = 180 seconds. Kill runs exceeding 5 minutes.

## Primary Metric

`composite_score` = mean of macro_f1 across all (station, horizon) pairs.

This drives keep/discard. The autoresearch loop reads it from stdout.

## Experiment Tracking

Initialize `results.tsv` with a header row only:
```
commit	composite_score	n_models	tier	model_type	status	description
```

After each run, append:
```
a1b2c3d	0.345	12	1	gbm	keep	baseline: GBM with lag features
```

## Hyperparameters in experiment.py

### Feature Engineering
- `LOOKBACK_HOURS` — hours of history for lag features (default 6)
- `USE_ROLLING_STATS` — rolling mean/std/min/max
- `USE_CYCLICAL_TIME` — sin/cos time encoding
- `USE_CROSS_STATION` — multi-station features
- `USE_DEWPOINT` — derived dewpoint + depression
- `USE_WIND_VECTOR` — x/y wind decomposition
- `USE_SOIL_TENDENCY` — soil moisture change rate
- `USE_RAIN_HISTORY` — cumulative rain in recent windows

### Model
- `MODEL_TYPE` — "gbm", "rf", "logistic", "mlp"
- `N_ESTIMATORS` — number of trees (GBM/RF)
- `MAX_DEPTH` — tree depth
- `LEARNING_RATE` — GBM learning rate
- `MIN_SAMPLES_LEAF` — regularization
- `SUBSAMPLE` — GBM stochastic subsampling
- `CLASS_WEIGHT` — "balanced" for imbalance handling

## Research Ideas — Progressive Tiers

### Tier 1: Feature Engineering + Classical ML (current)

Quick wins — try these first:
1. **Tune GBM hyperparameters**: n_estimators=[100,300,500], max_depth=[3,5,7,10], learning_rate=[0.01,0.05,0.1]
2. **Add pressure/tendency features**: compute rate of change in humidity, temperature
3. **Longer rolling windows**: add 12h, 24h rolling stats
4. **Lag features**: explicit lag values (t-1h, t-2h, ..., t-6h) for key variables
5. **Random Forest** instead of GBM: set MODEL_TYPE="rf"
6. **Cross-station gradients**: temp_diff between stations as a feature
7. **Battery voltage as rain proxy**: BattV drops correlate with clouds/rain
8. **Day-vs-night split models**: separate models for daytime vs nighttime
9. **SMOTE or class weights tuning**: experiment with different imbalance strategies
10. **Feature selection**: drop low-importance features to reduce noise

### Tier 2: Advanced Feature Engineering

11. **Station altitude features**: encode elevation differences (weather moves uphill)
12. **Auto-lag selection**: try different lookback windows per station
13. **Spectral features**: FFT of recent temperature/humidity series
14. **Regime detection**: identify dry/wet/transitional periods as meta-features
15. **Spatial interpolation**: use station coordinates for IDW weighting
16. **Interaction features**: humidity * temperature, wind * rain history
17. **Target encoding**: encode historical rain frequency by hour/month
18. **Per-horizon specialized models**: different feature sets per forecast horizon
19. **Stacking/blending**: combine GBM + RF + logistic predictions

### Tier 3: Deep Learning

20. **LSTM/GRU**: sequence model on raw 15-min features (use train_precipitation.py as reference)
21. **Temporal CNN**: 1D convolutions over lookback window
22. **Transformer**: self-attention on multivariate time series
23. **Multi-task learning**: predict all horizons + stations simultaneously
24. **Pretrain on LDAS**: learn atmospheric dynamics from gridded data, fine-tune on stations

### Tier 4: Challenge-Specific Optimizations

25. **Per-station models**: train separate models per station (some are harder than others)
26. **Probability calibration**: Platt scaling or isotonic regression on predicted probabilities
27. **Threshold optimization**: tune class boundaries per station/horizon for optimal F1
28. **Ensemble across horizons**: use 1h prediction as feature for 3h/6h
29. **Walk-forward retraining**: incrementally update model as eval period progresses

## Key Challenges

1. **Severe class imbalance**: it rarely rains. ~85-95% of timestamps are "no rain".
   Always-predict-0 gives high accuracy but 0 macro F1 on minority classes.
2. **Data gaps**: sensors go offline. Imputation strategy matters.
3. **Threshold sensitivity**: the light/heavy boundary is small (0.254mm for 1h).
4. **Temporal leakage**: never use future data. Rolling features must be backward-looking.
5. **Speed**: 180s budget. GBM is fast; RNNs need careful batching.

## Judging Criteria (hackathon)

- Originality & Innovation (30%)
- Technical Execution (30%)
- Impact & Relevance (25%)
- Presentation (15%)

## Rules

- CAN modify: `experiment.py` — hyperparameters, features, models, everything
- CANNOT modify: `prepare.py` (fixed evaluation harness)
- Every run MUST output `composite_score:` to stdout
- External data OK if publicly available and documented
