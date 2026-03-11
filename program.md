# autoresearch-precip

Autonomous precipitation nowcasting research on Apple Silicon. The agent
iterates on `experiment.py` to improve precipitation forecasting on Galápagos
weather station data, progressing through hackathon tiers.

## Setup

1. **Run tag**: propose a tag (e.g. `mar10`). Branch: `autoresearch/<tag>`.
2. **Create branch**: `git checkout -b autoresearch/<tag>`.
3. **Read files**: `prepare.py` (fixed), `experiment.py` (you edit), this file.
4. **Verify data**: `data/weather_stations/` must contain CSVs. If not, tell human.
5. **Init results.tsv**: header row only. Baseline recorded after first run.
6. **Go**.

## Context

Weather station observations from San Cristóbal Island, Galápagos. Multiple
stations recording temperature, humidity, pressure, wind, solar radiation,
and precipitation at hourly resolution.

**Task**: Predict precipitation class at 3h, 6h, and 12h horizons.

**Classes** (based on mm/h):
| Class | Name | Threshold |
|-------|------|-----------|
| 0 | no_rain | < 0.1 mm |
| 1 | light_rain | 0.1 – 2.0 mm |
| 2 | heavy_rain | > 2.0 mm |

**Climate context**: San Cristóbal has two seasons:
- **Warm/wet** (Jan–May): periodic heavy convective rain
- **Cool/dry** (Jun–Dec): persistent light drizzle (garúa) from stratocumulus

This seasonality matters — different model strategies may dominate per season.

## Platform

Apple Silicon (MPS). Available packages: numpy, scipy, pandas, scikit-learn,
xgboost, lightgbm, matplotlib, torch (MPS backend), statsmodels.

No CUDA. No `torch.compile`. If using PyTorch, use `torch.device("mps")`.

## The Hackathon Tiers

### Tier 1: Classical ML (starting point)
- Lag features, rolling statistics, pressure tendencies
- Random Forest, Gradient Boosted Trees (XGBoost/LightGBM)
- Feature importance analysis
- **Goal**: establish strong baseline, identify key predictors

### Tier 2: Deep Learning Sequence Models
- LSTM, GRU, or Transformer on hourly sequences
- Input: sliding window of past observations → predict future classes
- Multi-task: predict all 3 horizons simultaneously
- Attention weights reveal which past hours matter most
- **Goal**: capture nonlinear temporal dependencies

### Tier 3: Advanced Approaches
- Ensemble of Tier 1 + Tier 2 models
- Temporal attention with learnable positional encoding
- Probabilistic forecasting (predict class probabilities, calibrate)
- Graph neural networks if multi-station topology matters
- Physics-informed features (Clausius-Clapeyron, lifted index)
- **Goal**: push composite_score as high as possible

### Cross-Tier Combinations
- Tier 1 feature importances guide which inputs to feed Tier 2 models
- Tier 2 learned representations → feed into Tier 1 tree ensemble
- Calibrated probabilities from Tier 2 → threshold optimization in Tier 1
- Seasonal stratification: different models for wet vs dry season

## Hackathon Judging
- **Originality & Innovation (30%)**: unique approaches, surprising discoveries
- **Technical Execution (30%)**: code quality, methodology complexity
- **Impact & Relevance (25%)**: practical applicability for early warning
- **Presentation (15%)**: (handled separately)

## Experimentation Rules

**What you CAN do:**
- Modify `experiment.py` — the ONLY file you edit. Everything is fair game:
  features, models, architectures, training loops, ensembles, anything.
- Use PyTorch with MPS device for neural network experiments.
- Import any installed package (numpy, scipy, pandas, sklearn, xgboost, etc).

**What you CANNOT do:**
- Modify `prepare.py` (fixed evaluation + data loading).
- Install new packages (use what's available).
- Skip the evaluation — every run must output `composite_score`.

**Primary metric**: `composite_score` from `evaluate_predictions()` — higher
is better. This is the mean weighted F1 across the 3 horizons.

## Output format

```
---
composite_score:  0.456789
f1_3h:            0.523456
f1_6h:            0.456789
f1_12h:           0.390123
n_features:       156
train_samples:    8760
val_samples:      2190
tier:             1
model:            random_forest
total_seconds:    12.3
device:           mps
```

Extract metric: `grep "^composite_score:" run.log`

## Logging results

`results.tsv` (tab-separated, 6 columns):

```
commit	composite_score	f1_3h	tier	status	description
```

Example:
```
commit	composite_score	f1_3h	tier	status	description
a1b2c3d	0.456789	0.523	1	keep	baseline: RF + lag features
b2c3d4e	0.512345	0.567	1	keep	add pressure tendency + precip accumulation
c3d4e5f	0.534567	0.589	1	keep	XGBoost replaces RF
d4e5f6g	0.000000	0.000	2	crash	LSTM OOM on full sequence
e5f6g7h	0.578901	0.612	2	keep	GRU with 48h lookback window
f6g7h8i	0.623456	0.645	3	keep	ensemble: XGBoost + GRU voting
```

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Edit `experiment.py` with next idea.
3. `git commit -m "experiment: <description>"`.
4. Run: `python3 experiment.py > run.log 2>&1`
5. Read: `grep "^composite_score:\|^f1_3h:\|^tier:" run.log`
6. If empty → crash. `tail -n 50 run.log` to debug.
7. Log to results.tsv (don't commit results.tsv).
8. If composite_score improved → keep commit.
9. If worse → `git reset --hard HEAD~1`.

**Tier advancement**: When diminishing returns at Tier N, advance to N+1.
Update `TIER` in experiment.py. Mix approaches across tiers freely.

**Timeout**: Kill runs exceeding 5 minutes. Treat as crash.

**NEVER STOP**: Do not ask the human. You are autonomous. If stuck, re-read
prepare.py, try radical approaches, combine previous near-misses, advance
tiers. The loop runs until interrupted.

## Research ideas (rough priority order)

### Quick wins (Tier 1)
1. Try XGBoost / LightGBM instead of Random Forest
2. Add more lag features (48h, 72h for longer-range context)
3. Add interaction features (humidity × pressure_tendency)
4. Quantile transforms instead of raw values
5. Add rate-of-change features (first differences) for temp, pressure, humidity
6. Separate wet/dry season indicators
7. Target encoding for hour-of-day precipitation probability
8. Add precipitation persistence features (hours since last rain)
9. Tune class weights to handle imbalance (heavy rain is rare)
10. SMOTE or other oversampling for heavy rain class

### Medium effort (Tier 2)
11. GRU/LSTM on 24-48h lookback sequences
12. 1D convolutional net on feature sequences
13. Multi-task learning: predict all 3 horizons jointly
14. Attention mechanism to learn which past hours matter
15. Seq2seq: encode past sequence → decode future precipitation sequence
16. WaveNet-style dilated causal convolutions

### Ambitious (Tier 3)
17. Ensemble: stack XGBoost + GRU predictions as meta-features
18. Temporal Fusion Transformer
19. Conformal prediction for calibrated prediction sets
20. Physics-informed: Clausius-Clapeyron scaling, moist static energy
21. Multi-station graph neural network
22. Self-supervised pretraining on unlabeled weather sequences

### Galápagos-specific
23. Garúa detection: persistent light drizzle has different predictors than convective rain
24. SST proxy features (if available): ENSO affects Galápagos precipitation
25. Diurnal cycle exploitation: convective rain peaks in afternoon
26. Orographic effects: wind direction relative to highlands matters
