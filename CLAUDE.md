# Marine Acoustic Autoresearch

## Infrastructure
- g4dn.2xlarge: ubuntu@3.227.0.49 (Tesla T4, 15GB VRAM, 32GB RAM, 8 vCPUs) — dedicated marine instance
- Second g4dn (3.236.17.112) is for BRUV — not ours.
- NEVER use `pkill` or broad kill patterns on shared machines — only `kill <specific_PID>` for our process.

## Experiment
- Single file: `experiment.py` (local) → `/opt/autoresearch/experiment.py` (g4dn)
- Logs: `/opt/autoresearch/run*.log`
- Caches: `/opt/autoresearch/data/cache/` (tier1_features.npy, mels.npy, noaa_scores.npy)
- Full dataset: 4451 segments, 123 recordings, 3 hydrophone units
- TensorFlow must be forced to CPU-only (`tf.config.set_visible_devices([], 'GPU')`) to avoid GPU memory conflict with PyTorch

## Autoresearch Loop
- Keep improving composite_score continuously — don't stop after one run.
- Don't ask for confirmation — just iterate.
