# Autoresearch: Marine Acoustic Clustering — Agent ML Experiment Log

## Goal
Automatically cluster marine acoustic recordings from 3 hydrophone units (24/48/96kHz) into meaningful acoustic categories. The recordings are primarily for whale monitoring. An AI agent (Claude Code) iteratively modifies `experiment.py`, uploads to a g4dn GPU instance, runs experiments, analyzes results, and optimizes.

## Infrastructure
- **Compute**: AWS g4dn.xlarge (Tesla T4, 15GB VRAM, 16GB RAM) — shared with precip experiment
- **Host**: ubuntu@3.236.252.38
- **Data**: 123 recordings → 4451 segments (10s each), 3 hydrophone units
- **Code**: Single file `experiment.py` — the ONLY file the agent modifies
- **Caches**: tier1_features.npy, mels.npy, noaa_scores.npy in /opt/autoresearch/data/cache/
- **Metric**: `composite_score` from `prepare.evaluate_clustering()` (higher = better)

## Pipeline Architecture
```
Tier 1: Acoustic indices (MFCC, spectral, band power, ZCR, RMS) → 113 features
    ↓
UMAP (20 dims, n_neighbors=15, min_dist=0.0) → Dimensionality reduction
    ↓
HDBSCAN (min_cluster=10, min_samples=5) → Pseudo-labels (63 clusters)
    ↓
NOAA whale labeling: segments with max whale score > 0.8 → single "whale" class
    ↓
Tier 3a: CNN on mel spectrograms (128×938) using enhanced pseudo-labels → 64-dim features
    ↓
Re-cluster CNN features → evaluate all combinations
    ↓
Tier 3b: Contrastive (SimCLR) for unsupervised features
    ↓
NOAA combos, kitchen sink, per-unit normalization, HDBSCAN sweep
    ↓
Best composite_score wins
```

## Key Discoveries

### NOAA Whale Detection as Domain-Expert Labeler
- NOAA humpback whale detector (TensorFlow Hub) scores each segment for whale probability
- 345 segments (7.8%) have high-confidence whale calls (max score > 0.8)
- These whale segments were **scattered across 43 of 63 HDBSCAN clusters** — the unsupervised clustering split whale calls by superficial acoustic differences
- Merging whale segments into one class before CNN training gives CNN real biological signal
- Result: NOAA-labeled CNN (0.963) significantly beats plain CNN (0.912)

### TF/PyTorch GPU Memory Conflict
- TensorFlow (used by NOAA model and implicitly by UMAP) grabs GPU memory
- PyTorch CNN/contrastive then crashes with CUDA OOM
- Fix: `tf.config.set_visible_devices([], 'GPU')` at script startup — TF runs on CPU only
- NOAA features cached to .npy so TF model rarely needs to load

### Unit-Driven Clustering
- 3 hydrophone units with different sample rates create unit-specific acoustic profiles
- unit_diversity=1.2 across all runs — per-unit normalization hasn't fully fixed this
- CNN features help bridge units better than raw acoustic indices

### Contrastive Learning Underperforms
- SimCLR consistently worst: 0.59 solo, degrades combos
- Likely because mel augmentations (time/freq masking) are too aggressive for marine acoustics
- Worth trying: gentler augmentations, or supervised contrastive with NOAA whale labels

## Run History (Full Dataset — 4451 segments)

| Run | Best Score | Key Change | Notes |
|-----|-----------|------------|-------|
| 12a | 0.894 | First baseline | Combined CNN+T1 won |
| 12c | 0.965 | — | Crashed CUDA OOM (TF/PyTorch conflict) |
| 12d | — | Per-unit norm, HDBSCAN sweep | Crashed (TF during UMAP) |
| 12e | 0.964 | TF forced to CPU | Clean run, CNN-only wins |
| 12f | 0.963 | NOAA-as-labeler | NOAA-labeled CNN beats plain CNN (0.963 vs 0.912), 99.5% coverage |
| 12g | pending | Cosine UMAP, faster epochs | In progress |

## What the Agent Learned (Operational)
1. **Never use `pkill` on shared machines** — killed precip's experiment once
2. **Cache everything** — feature extraction takes 20 min, crashes in later phases waste it
3. **TF must be CPU-only** — any TF GPU usage blocks PyTorch
4. **Reduce epochs early** — CNN converges by 300, contrastive by 200, no need for 500/300
5. **Compare before removing** — always train a baseline for A/B testing before dropping approaches
6. **Background timers with `sleep N && ssh`** work; agent-based timers often fail to sleep

## Optimization Ideas Not Yet Tried
- Lower whale threshold (0.5-0.7) — captures 2-3x more whale segments
- Supervised contrastive with NOAA whale labels (whale pairs should be close)
- Deeper CNN / attention mechanism
- OPTICS instead of HDBSCAN
- Per-file normalization (not just per-unit)
- Whale sub-classes from NOAA score distribution (song vs social calls)
- Cross-unit transfer learning
