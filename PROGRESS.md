# Precipitation Nowcasting — Progress

Branch: `claude/nostalgic-moore`

## Status: Mid-experiment loop, Tier 1

The current `experiment.py` has been committed but **not yet run**. It adds
cross-station features and domain-specific features on top of the RF baseline.
Run it next to see if it beats 0.8695.

## Data

Downloaded from R2 to `data/weather_stations/`:
- 4 stations: CER, JUN, MERC, MIRA
- 15-min intervals, ~10 years (2015-06 to 2026-03)
- ~375K rows per station, ~376K hourly samples after merge
- Class distribution: 87% no_rain, 12% light_rain, 1% heavy_rain

R2 credentials (not secret, hackathon-provided):
```
export R2_ENDPOINT="https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com"
export R2_ACCESS_KEY_ID="93bb95ebfe47d5ef93c45efe3c108ca8"
export R2_SECRET_ACCESS_KEY="cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55"
export R2_BUCKET="sala-2026-hackathon-data"
```

Download command:
```
export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
aws s3 sync s3://$R2_BUCKET/precipitation-nowcasting/weather_stations/ data/weather_stations/ --endpoint-url $R2_ENDPOINT
```

## Environment setup

```
uv venv && source .venv/bin/activate
uv pip install pandas scikit-learn xgboost lightgbm numpy scipy matplotlib torch
```

## Experiment Results

| # | Model | composite_score | f1_3h | f1_6h | f1_12h | Status |
|---|-------|----------------|-------|-------|--------|--------|
| 1 | RF baseline (200 trees, depth=15, balanced) | **0.8695** | 0.879 | 0.869 | 0.860 | BEST |
| 2 | XGBoost (500 trees, manual class weights) | 0.8575 | 0.875 | 0.855 | 0.843 | reverted |
| 3 | XGBoost (300 trees, no class weights) | 0.8499 | 0.861 | 0.847 | 0.841 | reverted |
| 4 | LightGBM (500 trees, balanced) | 0.8624 | 0.874 | 0.863 | 0.850 | reverted |
| 5 | RF + cross-station + threshold tuning | 0.8556 | 0.868 | 0.857 | 0.842 | reverted |
| 6 | RF + cross-station + domain features | ? | ? | ? | ? | **NEXT** |

## Key Findings So Far

1. **RF with `class_weight='balanced'` is hard to beat.** It hits a sweet spot
   for weighted F1 — the balanced weights are moderate enough to not hurt
   majority class accuracy while still helping minority classes.

2. **XGBoost and LightGBM performed worse.** Both with and without class
   weights. XGBoost without weights is too biased toward majority; with manual
   weights it's too aggressive. LightGBM with balanced is decent but slightly
   worse than RF.

3. **Threshold tuning hurts.** Optimizing the no_rain probability threshold on
   a held-out tune set didn't generalize to val. It also costs training data.

4. **Cross-station features are highly informative.** When tested, they ranked
   as the top 4 most important features for 3h prediction. Whether it's
   raining at ANY other station on the island is very predictive.

5. **The bottleneck is the 12h horizon.** At 3h, even minority classes get
   decent F1. At 12h, heavy_rain F1 drops to near zero. Temperature becomes
   the dominant feature at 12h instead of precipitation history.

6. **Class imbalance is the core challenge.** 87/12/1 split means the weighted
   F1 is dominated by no_rain. Improving minority class prediction only helps
   if it doesn't hurt no_rain even slightly.

## What to Try Next

### Immediate (current experiment.py, not yet run)
- RF with cross-station features + soil moisture/leaf wetness tendencies
  + intra-hour variability features (temp_range, humidity_range)
- Same RF params as baseline but with richer features

### High priority ideas
- **Deeper RF trees** (max_depth=None, let it grow) — might overfit but worth trying
- **More trees** (500-1000) — diminishing returns but cheap to test
- **Feature selection**: remove noisy features, keep top 50-100 by importance
- **Per-horizon models with different features**: 3h uses precip features,
  12h uses temperature/seasonal features
- **Cascade classifier**: binary rain/no-rain first, then intensity

### Medium priority (Tier 2)
- GRU/LSTM on 24-48h lookback sequences
- Multi-task learning (predict all horizons jointly)
- Ensemble: average predictions from RF + LightGBM + XGBoost

### Ambitious (Tier 3)
- Temporal Fusion Transformer
- Physics-informed features (Clausius-Clapeyron, moist static energy)
- Stacking ensemble with meta-learner
