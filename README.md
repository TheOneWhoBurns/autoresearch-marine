# Autoresearch Marine

Autonomous marine acoustic research agent for Apple Silicon. Iteratively analyzes Galapagos hydrophone data to discover ecological patterns in underwater soundscapes.

## What it does

An AI agent that autonomously edits and runs `experiment.py` to analyze underwater recordings from 3 SoundTrap hydrophones deployed in San Cristobal Bay, Galapagos. It classifies biological signals (whale calls, dolphin clicks, snapping shrimp), anthropogenic noise (boat engines, sonar), and ambient sound.

## Sound types detected

| Band | Frequency | Sources |
|------|-----------|---------|
| LOW | 50–2000 Hz | Ships, whale calls, fish |
| MID | 2–20 kHz | Shrimp, dolphins, reef |
| HIGH | 20–24 kHz | Echolocation clicks |

## Stack

- Python + NumPy + SciPy + librosa
- scikit-learn + UMAP + HDBSCAN for clustering
- PyTorch (MPS backend) for Apple Silicon acceleration
- matplotlib for visualization
