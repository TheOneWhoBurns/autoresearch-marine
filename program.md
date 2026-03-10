# autoresearch-marine

Autonomous marine acoustic research on Apple Silicon. The agent iterates on
`experiment.py` to discover ecological patterns in Galapagos hydrophone data,
progressing through hackathon tiers naturally.

## Setup

1. **Run tag**: propose a tag (e.g. `mar10`). Branch: `autoresearch/<tag>`.
2. **Create branch**: `git checkout -b autoresearch/<tag>`.
3. **Read files**: `prepare.py` (fixed), `experiment.py` (you edit), this file.
4. **Verify data**: `data/raw/` must contain WAV files. If not, tell human.
5. **Init results.tsv**: header row only. Baseline recorded after first run.
6. **Go**.

## Context

Underwater recordings from 3 SoundTrap hydrophones in San Cristobal Bay, Galapagos:

| Unit | Sample Rate | Files | Duration/File |
|------|------------|-------|---------------|
| 5783 | 144 kHz | 2 | ~20 min |
| 6478 | 96 kHz | 4 | ~10 min |
| Pilot | 48 kHz | 5 | ~5 min |

Sound types present:
- **Biological**: whale calls, dolphin whistles/clicks, fish drums, snapping shrimp
- **Anthropogenic**: boat engines, sonar
- **Ambient**: waves, rain, currents

Frequency bands:
- **LOW** (50-2000 Hz): ships, whale calls, fish
- **MID** (2-20 kHz): shrimp, dolphins, reef
- **HIGH** (20-24 kHz): echolocation clicks

## Platform

Apple Silicon (MPS). Available packages: numpy, scipy, librosa, scikit-learn,
soundfile, umap-learn, hdbscan, matplotlib, torch (MPS backend).

No CUDA. No `torch.compile`. If using PyTorch, use `torch.device("mps")`.

## The Hackathon Tiers

The experiment naturally progresses through tiers. You don't have to follow
them in order — if a Tier 2 idea will improve the score, jump to it.

### Tier 1: Acoustic Landscape Indices (starting point)
- MFCCs, spectral features, band powers, entropy
- NDSI (Normalized Difference Soundscape Index): bio vs anthro ratio
- Temporal entropy, acoustic complexity index (ACI)
- Clustering with UMAP + HDBSCAN
- **Discovery goal**: identify day/night patterns, boat vs biology separation

### Tier 2: Pretrained Model Embeddings
- PANNs (pretrained audio tagging): 2048-dim embeddings, runs on MPS
- BirdNET / Perch embeddings (if available via pip)
- Use embeddings as features → better clustering
- Cosine similarity between segments for structure discovery
- **Discovery goal**: what sound categories exist? temporal/spatial patterns?

### Tier 3: Custom Classifier with Active Learning
- Use Tier 1+2 findings to bootstrap labels
- Train a small CNN or MLP on mel spectrograms (PyTorch + MPS)
- Active learning: cluster → human-label most uncertain → retrain → repeat
- Fine-tune on marine-specific classes found in exploration
- **Discovery goal**: build a working detector for species in this bay

### Cross-Tier Combinations (the exciting part!)
- Tier 1 indices find "interesting" segments → Tier 2 embeddings classify them
- Tier 2 embeddings reveal clusters → Tier 3 trains on those pseudo-labels
- Temporal patterns from Tier 1 (dawn chorus, boat schedules) contextualize Tier 3
- Band power profiles from Tier 1 validate Tier 2/3 cluster ecological meaning

## Hackathon Judging (for context — guide your research choices)
- **Originality & Innovation (30%)**: unique approaches, surprising discoveries
- **Technical Execution (30%)**: code quality, methodology complexity
- **Impact & Relevance (25%)**: practical applicability for conservation
- **Presentation (15%)**: (we handle this separately)

## Experimentation Rules

**What you CAN do:**
- Modify `experiment.py` — the ONLY file you edit. Everything is fair game:
  architecture, features, models, training loops, anything.
- Use PyTorch with MPS device for neural network experiments.
- Import any installed package (numpy, scipy, librosa, sklearn, torch, etc).

**What you CANNOT do:**
- Modify `prepare.py` (fixed evaluation + data loading).
- Install new packages (use what's available).
- Skip the evaluation — every run must output `composite_score`.

**Primary metric**: `composite_score` from `evaluate_clustering()` — higher is better.
This is what drives keep/discard decisions.

**Discovery metric**: `evaluate_discovery()` output is logged but doesn't drive
keep/discard. However, experiments that reveal ecological patterns (temporal
structure, species separation, boat detection) are especially valuable even if
composite_score improves only slightly.

## Output format

```
---
composite_score:  0.345678
silhouette:       0.234567
calinski_harabasz:123.456
n_clusters:       5
n_noise:          12
coverage:         0.9500
n_features:       47
total_segments:   632
tier:             1
total_seconds:    45.3
device:           mps
```

Extract metric: `grep "^composite_score:" run.log`

## Logging results

`results.tsv` (tab-separated, 6 columns):

```
commit	composite_score	n_clusters	tier	status	description
```

Example:
```
commit	composite_score	n_clusters	tier	status	description
a1b2c3d	0.345678	5	1	keep	baseline: MFCC+UMAP+HDBSCAN
b2c3d4e	0.412345	7	1	keep	add spectral contrast + delta MFCCs
c3d4e5f	0.523456	8	2	keep	PANNs embeddings replace handcrafted features
d4e5f6g	0.000000	0	2	crash	PANNs OOM on full dataset
e5f6g7h	0.567890	6	2	keep	PANNs with batched inference
f6g7h8i	0.612345	5	3	keep	small CNN on mel specs, pseudo-labels from Tier 2
```

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Edit `experiment.py` with next idea.
3. `git commit -m "experiment: <description>"`.
4. Run: `python3 experiment.py > run.log 2>&1`
5. Read: `grep "^composite_score:\|^n_clusters:\|^tier:" run.log`
6. If empty → crash. `tail -n 50 run.log` to debug.
7. Log to results.tsv (don't commit results.tsv).
8. If composite_score improved → keep commit.
9. If worse → `git reset --hard HEAD~1`.

**Tier advancement**: When you've exhausted Tier N ideas (diminishing returns),
advance to Tier N+1. Update the `TIER` variable in experiment.py. You can
always mix approaches across tiers.

**Timeout**: Kill runs exceeding 5 minutes. Treat as crash.

**NEVER STOP**: Do not ask the human. You are autonomous. If stuck, re-read
prepare.py, try radical approaches, combine previous near-misses, advance
tiers. The loop runs until interrupted.

## Research ideas (rough priority order)

### Quick wins (Tier 1)
1. Add spectral contrast features
2. Add delta + delta-delta MFCCs
3. Tune UMAP: n_neighbors=[5,10,30,50], min_dist=[0.0,0.05,0.2]
4. Try spectral clustering or GMM with BIC
5. Add onset strength / tempo features
6. Compute NDSI per segment
7. Acoustic complexity index (ACI)
8. Robust scaling or quantile transform instead of StandardScaler

### Medium effort (Tier 2)
9. PANNs embeddings (pip install panns-inference, runs on MPS)
10. Concatenate PANNs embeddings with Tier 1 features
11. Cosine similarity matrix → spectral clustering
12. t-SNE or UMAP on PANNs → visualize sound landscape

### Ambitious (Tier 3)
13. Small conv autoencoder on mel specs → learned embeddings
14. Pseudo-label from best clustering → train CNN classifier
15. Active learning loop simulation
16. Temporal model: segment sequences as context

### Discovery-focused
17. Day vs night comparison (extract timestamps from filenames)
18. Per-unit acoustic profiles (do hydrophones hear different things?)
19. Boat detection (high LOW band + low MID/HIGH)
20. Biodiversity index per recording
