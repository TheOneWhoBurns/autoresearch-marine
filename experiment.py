"""
Marine Acoustic Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Cross-tier approach: Tier 1 acoustic indices for clustering +
Tier 2 PANNs predictions for ecological interpretation +
temporal analysis for discovery.

Metric: composite_score from prepare.evaluate_clustering() — higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs
import re
import time
import json
import numpy as np

# Prevent TensorFlow from grabbing GPU memory — reserve GPU for PyTorch
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # TF uses CPU only
except Exception:
    pass

import librosa
from prepare import (
    TARGET_SR, SEGMENT_SECONDS, SEGMENT_SAMPLES,
    N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX,
    TIME_BUDGET, CACHE_DIR, RESULTS_DIR, DEVICE,
    BAND_LOW, BAND_MID, BAND_HIGH,
    find_wav_files, load_audio, segment_audio, highpass_filter,
    compute_melspec, compute_band_power, compute_rms,
    evaluate_clustering, evaluate_discovery,
    build_dataset,
)

# ---------------------------------------------------------------------------
# TIER: current approach (agent updates this as it progresses)
# ---------------------------------------------------------------------------
TIER = 3  # 1=acoustic indices, 2=pretrained embeddings, 3=custom classifier

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Feature extraction (Tier 1)
N_MFCC = 20
USE_DELTA_MFCC = True
USE_BAND_POWER = True
USE_SPECTRAL = True
USE_SPECTRAL_CONTRAST = True
USE_ZCR = True
USE_RMS = True

# PANNs (Tier 2 — used for interpretation, not clustering)
USE_PANNS_LABELS = True

# NOAA humpback whale detector (Tier 2c — whale detection scores as features)
USE_NOAA_WHALE = True  # Run 12b: re-enabled with per-file resampling optimization

# Dimensionality reduction
REDUCER = "umap"
N_COMPONENTS = 20
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"  # 12g: cosine beat euclidean for CNN features

# Clustering
CLUSTERER = "hdbscan"
HDBSCAN_MIN_CLUSTER = 10
HDBSCAN_MIN_SAMPLES = 5
KMEANS_K = 8


# ---------------------------------------------------------------------------
# Feature extraction (Tier 1: acoustic indices)
# ---------------------------------------------------------------------------

def extract_features(segments):
    """Extract feature vectors from audio segments. Returns (N, D) array."""
    import librosa

    features_list = []
    for i, seg in enumerate(segments):
        feats = []

        mfcc = librosa.feature.mfcc(y=seg, sr=TARGET_SR, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc, axis=1))
        if USE_DELTA_MFCC:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            feats.extend(np.mean(delta, axis=1))
            feats.extend(np.mean(delta2, axis=1))

        if USE_BAND_POWER:
            bp = compute_band_power(seg)
            feats.extend([bp["low"], bp["mid"], bp["high"]])

        if USE_SPECTRAL:
            centroid = librosa.feature.spectral_centroid(y=seg, sr=TARGET_SR,
                                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
            bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=TARGET_SR,
                                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
            rolloff = librosa.feature.spectral_rolloff(y=seg, sr=TARGET_SR,
                                                        n_fft=N_FFT, hop_length=HOP_LENGTH)
            feats.extend([np.mean(centroid), np.std(centroid),
                          np.mean(bandwidth), np.std(bandwidth),
                          np.mean(rolloff), np.std(rolloff)])

        if USE_SPECTRAL_CONTRAST:
            contrast = librosa.feature.spectral_contrast(y=seg, sr=TARGET_SR,
                                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
            feats.extend(np.mean(contrast, axis=1))
            feats.extend(np.std(contrast, axis=1))

        if USE_ZCR:
            zcr = librosa.feature.zero_crossing_rate(seg, frame_length=N_FFT,
                                                      hop_length=HOP_LENGTH)
            feats.extend([np.mean(zcr), np.std(zcr)])

        if USE_RMS:
            feats.append(compute_rms(seg))

        # NDSI
        bp_vals = compute_band_power(seg) if not USE_BAND_POWER else bp
        bio = 10**(bp_vals["mid"]/10)
        anthro = 10**(bp_vals["low"]/10)
        ndsi = (bio - anthro) / (bio + anthro + 1e-12)
        feats.append(ndsi)

        # Mel band stats
        mel = compute_melspec(seg)
        feats.extend(np.mean(mel, axis=1)[::4])
        feats.extend(np.std(mel, axis=1)[::4])

        features_list.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  Features: {i+1}/{len(segments)}")

    return np.array(features_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# PANNs AudioSet predictions (Tier 2 — for cluster interpretation)
# ---------------------------------------------------------------------------

# Marine-relevant AudioSet classes
MARINE_CLASSES = {
    "Water": "ambient", "Ocean": "ambient", "Rain": "ambient",
    "Stream": "ambient", "Waves, surf": "ambient",
    "Boat, Water vehicle": "anthropogenic", "Ship": "anthropogenic",
    "Engine": "anthropogenic", "Motor vehicle (road)": "anthropogenic",
    "Vehicle": "anthropogenic", "Motorboat, speedboat": "anthropogenic",
    "Mechanical fan": "anthropogenic",
    "Whale vocalization": "biological", "Animal": "biological",
    "Bird": "biological", "Insect": "biological",
    "Click": "biological", "Squeak": "biological",
    "Chirp, tweet": "biological", "Splash, splashing": "ambient",
    "Rumble": "ambiguous", "Hum": "ambiguous",
    "White noise": "ambient", "Noise": "ambient",
    "Silence": "ambient", "Static": "ambient",
}


def get_panns_labels(segments, n_per_cluster=3):
    """Get PANNs AudioSet predictions for representative segments."""
    try:
        import torch
        from panns_inference import AudioTagging
        from panns_inference.config import labels as audioset_labels
        import librosa
    except ImportError:
        print("  PANNs not available, skipping AudioSet labels")
        return None

    panns_sr = 32000
    print("  Loading PANNs model...")
    at = AudioTagging(checkpoint_path=None, device='cpu')

    results = {}
    for i, seg in enumerate(segments):
        resampled = librosa.resample(seg, orig_sr=TARGET_SR, target_sr=panns_sr)
        audio_tensor = resampled[np.newaxis, :]
        import torch as th
        with th.no_grad():
            clip_probs, emb = at.inference(th.from_numpy(audio_tensor))

        top_indices = np.argsort(clip_probs[0])[-10:][::-1]
        top_labels = [(audioset_labels[idx], float(clip_probs[0][idx])) for idx in top_indices]
        results[i] = top_labels

    return results


def classify_segment_panns(panns_preds):
    """Classify a segment based on PANNs top predictions into ecological categories."""
    if panns_preds is None:
        return "unknown"

    bio_score = 0.0
    anthro_score = 0.0
    ambient_score = 0.0

    for label, prob in panns_preds:
        category = MARINE_CLASSES.get(label, None)
        if category == "biological":
            bio_score += prob
        elif category == "anthropogenic":
            anthro_score += prob
        elif category == "ambient":
            ambient_score += prob

    if anthro_score > 0.1:
        return "anthropogenic"
    elif bio_score > 0.05:
        return "biological"
    elif ambient_score > 0.2:
        return "ambient"
    return "unknown"


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

def parse_timestamp(filename):
    """Extract timestamp from SoundTrap filename like '6478.230723191251.wav'."""
    match = re.search(r'\.(\d{12})\.', filename)
    if match:
        ts = match.group(1)
        # Format: YYMMDDHHMMSS
        year = 2000 + int(ts[0:2])
        month = int(ts[2:4])
        day = int(ts[4:6])
        hour = int(ts[6:8])
        minute = int(ts[8:10])
        second = int(ts[10:12])
        return {"year": year, "month": month, "day": day,
                "hour": hour, "minute": minute, "second": second}
    return None


def analyze_temporal(labels, metadata, segments):
    """Analyze temporal patterns across clusters."""
    print(f"\n{'='*60}")
    print("Temporal Analysis")
    print(f"{'='*60}")

    # Parse timestamps for all segments
    timestamps = []
    for m in metadata:
        ts = parse_timestamp(m["file"])
        if ts:
            # Adjust for segment offset within file
            total_seconds = ts["hour"] * 3600 + ts["minute"] * 60 + ts["second"]
            total_seconds += m["offset_s"]
            ts["total_seconds"] = total_seconds
            ts["hour_decimal"] = total_seconds / 3600
        timestamps.append(ts)

    valid_ts = [t for t in timestamps if t is not None]
    if not valid_ts:
        print("  No timestamps found in filenames")
        return {}

    print(f"  Parsed {len(valid_ts)}/{len(timestamps)} timestamps")

    # Cluster temporal distribution
    temporal_info = {}
    for c in sorted(set(labels[labels >= 0])):
        cmask = labels == c
        cluster_hours = [timestamps[i]["hour"] for i in range(len(timestamps))
                        if cmask[i] and timestamps[i] is not None]
        if cluster_hours:
            unique_hours = sorted(set(cluster_hours))
            temporal_info[f"cluster_{c}"] = {
                "hours": unique_hours,
                "n_segments": int(cmask.sum()),
                "hour_range": f"{min(unique_hours):02d}:00-{max(unique_hours):02d}:59",
            }
            print(f"  Cluster {c}: hours={unique_hours}, "
                  f"n={cmask.sum()}")

    # Day vs night analysis (rough: day=6-18, night=18-6)
    day_mask = np.array([timestamps[i] is not None and 6 <= timestamps[i]["hour"] < 18
                         for i in range(len(timestamps))])
    night_mask = np.array([timestamps[i] is not None and (timestamps[i]["hour"] >= 18 or timestamps[i]["hour"] < 6)
                          for i in range(len(timestamps))])

    if day_mask.any() and night_mask.any():
        print(f"\n  Day segments (6-18h): {day_mask.sum()}")
        print(f"  Night segments (18-6h): {night_mask.sum()}")

        # Per-cluster day/night breakdown
        for c in sorted(set(labels[labels >= 0])):
            cmask = labels == c
            day_count = (cmask & day_mask).sum()
            night_count = (cmask & night_mask).sum()
            total = day_count + night_count
            if total > 0:
                day_pct = day_count / total * 100
                print(f"  Cluster {c}: {day_pct:.0f}% day, {100-day_pct:.0f}% night ({total} segs)")

    return temporal_info


# ---------------------------------------------------------------------------
# Tier 2b: BirdNET embeddings (marine-aware bioacoustic features)
# ---------------------------------------------------------------------------

def extract_birdnet_embeddings(segments):
    """Extract BirdNET v2.4 embeddings (1024-dim) for each segment.
    BirdNET expects 3s chunks at 48kHz, so we split 10s segments and average."""
    try:
        from birdnet.models.v2m4 import AudioModelV2M4TFLite
    except ImportError:
        print("  BirdNET not available, skipping")
        return None

    print("  Loading BirdNET model...")
    model = AudioModelV2M4TFLite()
    interp = model._audio_interpreter
    chunk_samples = int(model.chunk_size_s * model.sample_rate)  # 3s * 48kHz = 144000
    embedding_idx = 545  # GLOBAL_AVG_POOL layer

    embeddings = []
    for i, seg in enumerate(segments):
        # Segment is at TARGET_SR (48kHz), BirdNET expects 48kHz — no resample needed
        # Split 10s segment into 3s chunks with overlap
        chunk_embeds = []
        for start in range(0, len(seg) - chunk_samples + 1, chunk_samples):
            chunk = seg[start:start + chunk_samples].astype(np.float32)
            interp.resize_tensor_input(0, [1, chunk_samples])
            interp.allocate_tensors()
            interp.set_tensor(0, chunk.reshape(1, -1))
            interp.invoke()
            emb = interp.get_tensor(embedding_idx)
            chunk_embeds.append(emb[0])

        # Handle remaining audio if any
        if not chunk_embeds:
            # Segment shorter than 3s — pad
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[:len(seg)] = seg
            interp.resize_tensor_input(0, [1, chunk_samples])
            interp.allocate_tensors()
            interp.set_tensor(0, padded.reshape(1, -1))
            interp.invoke()
            emb = interp.get_tensor(embedding_idx)
            chunk_embeds.append(emb[0])

        # Average chunk embeddings for segment-level representation
        embeddings.append(np.mean(chunk_embeds, axis=0))

        if (i + 1) % 100 == 0:
            print(f"  BirdNET embeddings: {i+1}/{len(segments)}")

    result = np.array(embeddings, dtype=np.float32)
    print(f"  BirdNET embeddings: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Tier 3: CNN classifier with pseudo-labels from HDBSCAN
# ---------------------------------------------------------------------------

def train_cnn_classifier(all_mels_array, pseudo_labels):
    """Train a small CNN on mel spectrograms using HDBSCAN pseudo-labels.
    Returns learned features (penultimate layer) for re-clustering.
    Accepts precomputed normalized mel spectrograms (N, n_mels, time_dim)."""
    import torch
    import torch.nn as nn

    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"  CNN device: {device}")

    # Only use labeled segments (exclude noise label -1)
    mask = pseudo_labels >= 0
    labeled_indices = np.where(mask)[0]
    n_classes = len(set(pseudo_labels[mask]))
    print(f"  Training CNN: {len(labeled_indices)} labeled segments, {n_classes} classes")

    X_all = all_mels_array
    X = X_all[labeled_indices]
    y = pseudo_labels[mask]

    # Simple CNN
    class MarineCNN(nn.Module):
        def __init__(self, n_mels, time_dim, n_classes, feat_dim=64):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
                nn.Dropout2d(0.1),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                nn.Dropout2d(0.1),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.Dropout2d(0.15),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                nn.Dropout2d(0.2),
            )
            dummy = torch.zeros(1, 1, n_mels, time_dim)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel()
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, feat_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(feat_dim, n_classes)

        def forward(self, x):
            h = self.conv(x)
            feats = self.features(h)
            logits = self.classifier(feats)
            return logits, feats

    time_dim = X_all.shape[2]
    model = MarineCNN(128, time_dim, n_classes, feat_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_tensor = torch.from_numpy(X[:, np.newaxis, :, :])
    y_tensor = torch.from_numpy(y.astype(np.int64))

    batch_size = 32
    n = len(X_tensor)
    n_epochs = 300

    warmup_epochs = 20
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0
        correct = 0
        total = 0
        for i in range(0, n, batch_size):
            batch_x = X_tensor[perm[i:i+batch_size]].to(device)
            batch_y = y_tensor[perm[i:i+batch_size]].to(device)
            batch_x = spec_augment(batch_x)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += len(batch_y)
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            acc = correct / total * 100
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={total_loss:.4f}, acc={acc:.1f}%")

    # Extract features for ALL segments (reuse pre-computed mels)
    model.eval()
    X_all_tensor = torch.from_numpy(X_all[:, np.newaxis, :, :])

    features = []
    with torch.no_grad():
        for i in range(0, len(X_all_tensor), batch_size):
            batch = X_all_tensor[i:i+batch_size].to(device)
            _, feats = model(batch)
            features.append(feats.cpu().numpy())

    learned_features = np.vstack(features)
    print(f"  CNN learned features: {learned_features.shape}")
    return learned_features


# ---------------------------------------------------------------------------
# Tier 3b: Contrastive learning (SimCLR-style) — no labels needed
# ---------------------------------------------------------------------------

def spec_augment(batch, freq_mask_param=20, time_mask_param=30):
    """SpecAugment-style data augmentation."""
    import torch
    b, c, f, t = batch.shape
    augmented = batch.clone()
    for idx in range(b):
        f_start = torch.randint(0, max(1, f - freq_mask_param), (1,)).item()
        f_width = torch.randint(0, freq_mask_param + 1, (1,)).item()
        augmented[idx, :, f_start:f_start+f_width, :] = 0
        t_start = torch.randint(0, max(1, t - time_mask_param), (1,)).item()
        t_width = torch.randint(0, time_mask_param + 1, (1,)).item()
        augmented[idx, :, :, t_start:t_start+t_width] = 0
    return augmented


def train_contrastive(all_mels_array, temperature=0.1, feat_dim=64, n_epochs=300):
    """Train a CNN encoder with NT-Xent contrastive loss (SimCLR).
    No labels required — learns by comparing augmented views of same segment.
    Returns learned features for all segments.
    Accepts precomputed normalized mel spectrograms (N, n_mels, time_dim)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"  Contrastive device: {device}")

    X_all = all_mels_array

    class Encoder(nn.Module):
        def __init__(self, n_mels, time_dim, feat_dim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            )
            dummy = torch.zeros(1, 1, n_mels, time_dim)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel()
            # Feature head
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, feat_dim),
                nn.ReLU(),
            )
            # Projection head (used during training, discarded for clustering)
            self.projector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
            )

        def forward(self, x, return_projection=False):
            h = self.conv(x)
            feats = self.features(h)
            if return_projection:
                proj = self.projector(feats)
                return feats, F.normalize(proj, dim=1)
            return feats

    time_dim = X_all.shape[2]
    model = Encoder(128, time_dim, feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)

    X_tensor = torch.from_numpy(X_all[:, np.newaxis, :, :])
    n = len(X_tensor)
    batch_size = 64  # larger batches = more negatives = better contrastive signal

    warmup_epochs = 15
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Training contrastive: {n} segments, {n_epochs} epochs, temp={temperature}")
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0
        n_batches = 0
        for i in range(0, n, batch_size):
            batch_x = X_tensor[perm[i:i+batch_size]].to(device)
            if len(batch_x) < 4:  # need enough for contrastive pairs
                continue

            # Two augmented views of same batch
            view1 = spec_augment(batch_x, freq_mask_param=25, time_mask_param=35)
            view2 = spec_augment(batch_x, freq_mask_param=25, time_mask_param=35)

            _, z1 = model(view1, return_projection=True)
            _, z2 = model(view2, return_projection=True)

            # NT-Xent loss
            bsz = z1.shape[0]
            z = torch.cat([z1, z2], dim=0)  # (2B, D)
            sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

            # Mask out self-similarity
            mask = torch.eye(2 * bsz, dtype=torch.bool, device=device)
            sim.masked_fill_(mask, -1e9)

            # Positive pairs: (i, i+B) and (i+B, i)
            labels = torch.cat([
                torch.arange(bsz, 2 * bsz, device=device),
                torch.arange(0, bsz, device=device),
            ])
            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1}/{n_epochs}: contrastive_loss={avg_loss:.4f}")

    # Extract features (use feature head, not projection head)
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            feats = model(batch, return_projection=False)
            features.append(feats.cpu().numpy())

    learned_features = np.vstack(features)
    print(f"  Contrastive features: {learned_features.shape}")
    return learned_features


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(features):
    """Reduce feature dimensions. Returns (N, D') array."""
    from sklearn.preprocessing import StandardScaler
    features_scaled = StandardScaler().fit_transform(features)

    if REDUCER == "none":
        return features_scaled
    elif REDUCER == "umap":
        import umap
        return umap.UMAP(
            n_components=N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=42,
        ).fit_transform(features_scaled)
    elif REDUCER == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=N_COMPONENTS, random_state=42).fit_transform(features_scaled)
    else:
        raise ValueError(f"Unknown reducer: {REDUCER}")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster(features_reduced):
    """Cluster features. Returns labels array."""
    if CLUSTERER == "hdbscan":
        import hdbscan
        return hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER,
            min_samples=HDBSCAN_MIN_SAMPLES,
        ).fit_predict(features_reduced)
    elif CLUSTERER == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10).fit_predict(features_reduced)
    else:
        raise ValueError(f"Unknown clusterer: {CLUSTERER}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_clusters(labels, metadata, features, segments=None, panns_labels=None, noaa_scores=None):
    """Print cluster summary with ecological interpretation."""
    unique_labels = sorted(set(labels))
    print(f"\n{'='*60}")
    print(f"Cluster Analysis: {len([l for l in unique_labels if l >= 0])} clusters")
    print(f"{'='*60}")

    for label in unique_labels:
        mask = labels == label
        count = mask.sum()
        cluster_meta = [metadata[i] for i in range(len(metadata)) if mask[i]]
        units = [m["unit"] for m in cluster_meta]
        unit_counts = {u: units.count(u) for u in set(units)}
        files = set(m["file"] for m in cluster_meta)

        name = "NOISE (unclustered)" if label == -1 else f"Cluster {label}"
        print(f"\n  {name}: {count} segments")
        print(f"    Units: {unit_counts}")
        print(f"    Files: {len(files)} unique")

        # Band power profile
        if segments is not None:
            sample_indices = [i for i in range(len(labels)) if mask[i]][:10]
            if sample_indices:
                powers = [compute_band_power(segments[i]) for i in sample_indices]
                avg_low = np.mean([p["low"] for p in powers])
                avg_mid = np.mean([p["mid"] for p in powers])
                avg_high = np.mean([p["high"] for p in powers])
                print(f"    Band profile: LOW={avg_low:.1f}dB  MID={avg_mid:.1f}dB  HIGH={avg_high:.1f}dB")

                # NDSI for cluster
                ndsi_vals = []
                for idx in sample_indices:
                    bp = compute_band_power(segments[idx])
                    bio_lin = 10**(bp["mid"]/10)
                    anthro_lin = 10**(bp["low"]/10)
                    ndsi_vals.append((bio_lin - anthro_lin) / (bio_lin + anthro_lin + 1e-12))
                avg_ndsi = np.mean(ndsi_vals)
                print(f"    NDSI: {avg_ndsi:.3f} ({'biophony-dominated' if avg_ndsi > 0 else 'anthrophony-dominated'})")

                # RMS energy
                rms_vals = [compute_rms(segments[idx]) for idx in sample_indices]
                print(f"    RMS energy: {np.mean(rms_vals):.6f} ({'loud' if np.mean(rms_vals) > 0.01 else 'quiet'})")

                # Improved ecological interpretation
                mid_low_ratio = avg_mid - avg_low
                high_mid_ratio = avg_high - avg_mid
                rms_mean = np.mean(rms_vals)

                # Classify based on energy, band ratios, and NDSI
                if rms_mean < 0.001:
                    eco_label = "near-silence / deep ambient"
                elif rms_mean < 0.005 and avg_low < -85:
                    eco_label = "quiet ambient (very low energy)"
                elif avg_low > -30 and mid_low_ratio < -15:
                    eco_label = "boat/ship engine noise (loud)"
                elif mid_low_ratio < -20 and rms_mean > 0.01:
                    eco_label = "vessel noise (strong low-freq dominance)"
                elif mid_low_ratio < -10 and avg_mid < -110 and rms_mean > 0.005:
                    eco_label = "vessel/engine (low-freq only, MID silent)"
                elif avg_ndsi > 0.3 and rms_mean > 0.005:
                    eco_label = "strong biological activity (reef/shrimp)"
                elif avg_ndsi > -0.2 and avg_mid > -85:
                    eco_label = "biological sounds (shrimp/reef/fish)"
                elif avg_high > -80 and high_mid_ratio > 5:
                    eco_label = "echolocation clicks (high-freq)"
                elif avg_mid > -100 and avg_mid < -80 and rms_mean > 0.005:
                    eco_label = "moderate biological (mid-freq activity)"
                elif rms_mean > 0.01 and avg_low > -80 and avg_mid < -100:
                    eco_label = "low-freq acoustic event"
                elif avg_low < -100 and avg_mid < -110:
                    eco_label = "quiet ambient (low energy)"
                else:
                    eco_label = "mixed soundscape"
                print(f"    -> Ecological: {eco_label}")

        # PANNs labels for representative samples
        if panns_labels is not None:
            cluster_panns_indices = [idx for idx in panns_labels.keys()
                                     if mask[idx]]
            if cluster_panns_indices:
                categories = []
                all_top_labels = []
                for idx in cluster_panns_indices:
                    cat = classify_segment_panns(panns_labels[idx])
                    categories.append(cat)
                    top3 = panns_labels[idx][:3]
                    all_top_labels.extend([l for l, _ in top3])
                from collections import Counter
                cat_counts = Counter(categories)
                label_counts = Counter(all_top_labels).most_common(5)
                print(f"    PANNs: {dict(cat_counts)}")
                print(f"    PANNs top: {[l for l, c in label_counts]}")

        # NOAA whale scores per cluster
        if noaa_scores is not None:
            cluster_noaa = noaa_scores[mask]  # (n_in_cluster, 3): mean, max, std
            whale_mean = cluster_noaa[:, 0].mean()
            whale_max = cluster_noaa[:, 1].max()
            whale_high = (cluster_noaa[:, 1] >= 0.8).sum()
            whale_med = ((cluster_noaa[:, 1] >= 0.5) & (cluster_noaa[:, 1] < 0.8)).sum()
            whale_pct = whale_high / count * 100
            if whale_high > 0:
                whale_label = f"WHALE CLUSTER ({whale_pct:.0f}% high-confidence)"
            elif whale_med > 0:
                whale_label = f"possible whale ({whale_med} medium-confidence)"
            else:
                whale_label = "non-whale"
            print(f"    NOAA whale: mean={whale_mean:.3f}, max={whale_max:.3f}, "
                  f"high={whale_high}/{count}, -> {whale_label}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 60)
    print(f"Marine Acoustic Autoresearch — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "feature_mode": "tier1_clustering + panns_interpretation",
        "n_mfcc": N_MFCC, "use_delta_mfcc": USE_DELTA_MFCC,
        "use_band_power": USE_BAND_POWER,
        "use_spectral": USE_SPECTRAL, "use_spectral_contrast": USE_SPECTRAL_CONTRAST,
        "use_zcr": USE_ZCR, "use_rms": USE_RMS,
        "use_panns_labels": USE_PANNS_LABELS,
        "reducer": REDUCER, "n_components": N_COMPONENTS,
        "umap_n_neighbors": UMAP_N_NEIGHBORS, "umap_min_dist": UMAP_MIN_DIST,
        "clusterer": CLUSTERER,
        "hdbscan_min_cluster": HDBSCAN_MIN_CLUSTER,
        "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
    }
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    # ---------------------------------------------------------------------------
    # Feature caching: skip extraction if cache exists with correct segment count
    # ---------------------------------------------------------------------------
    import gc
    cache_features_path = os.path.join(CACHE_DIR, "tier1_features.npy")
    cache_mels_path = os.path.join(CACHE_DIR, "mels.npy")
    cache_meta_path = os.path.join(CACHE_DIR, "metadata.json")

    recordings = find_wav_files()
    print(f"Found {len(recordings)} recordings")

    # Try loading from cache
    cache_hit = False
    if os.path.exists(cache_features_path) and os.path.exists(cache_mels_path) and os.path.exists(cache_meta_path):
        print("\n--- Loading cached features ---")
        try:
            features = np.load(cache_features_path)
            all_mels_array = np.load(cache_mels_path)
            with open(cache_meta_path, "r") as f:
                metadata = json.load(f)
            n_segs = len(metadata)
            print(f"Cache loaded: {n_segs} segments, features={features.shape}, mels={all_mels_array.shape}")
            cache_hit = True
        except Exception as e:
            print(f"Cache load failed: {e}, re-extracting...")
            cache_hit = False

    if not cache_hit:
        # Load data in streaming fashion to minimize peak memory
        print("\n--- Loading data & extracting features (streaming) ---")
        print(f"Loading {len(recordings)} recordings...")

        all_features = []
        all_mels_list = []
        metadata = []

        for rec in recordings:
            print(f"  {rec['unit']}/{rec['filename']} ({rec['duration_s']:.0f}s @ {rec['sample_rate']}Hz)")
            audio, sr = load_audio(rec["path"])
            audio = highpass_filter(audio, sr)
            file_segments = segment_audio(audio, sr)
            del audio

            for i, seg in enumerate(file_segments):
                # Extract tier1 features
                feat_vec = []
                mel = compute_melspec(seg)
                all_mels_list.append(mel)
                mfcc = librosa.feature.mfcc(y=seg, sr=48000, n_mfcc=N_MFCC)
                feat_vec.extend(mfcc.mean(axis=1))
                feat_vec.extend(mfcc.std(axis=1))
                if USE_DELTA_MFCC:
                    d1 = librosa.feature.delta(mfcc)
                    feat_vec.extend(d1.mean(axis=1))
                    feat_vec.extend(d1.std(axis=1))
                if USE_BAND_POWER:
                    bp = compute_band_power(seg)
                    feat_vec.extend([bp["low"], bp["mid"], bp["high"]])
                if USE_SPECTRAL:
                    sc = librosa.feature.spectral_centroid(y=seg, sr=48000)[0]
                    sb = librosa.feature.spectral_bandwidth(y=seg, sr=48000)[0]
                    sr_feat = librosa.feature.spectral_rolloff(y=seg, sr=48000)[0]
                    sf_feat = librosa.feature.spectral_flatness(y=seg)[0]
                    for x in [sc, sb, sr_feat, sf_feat]:
                        feat_vec.extend([x.mean(), x.std()])
                if USE_SPECTRAL_CONTRAST:
                    scon = librosa.feature.spectral_contrast(y=seg, sr=48000)
                    feat_vec.extend(scon.mean(axis=1))
                    feat_vec.extend(scon.std(axis=1))
                if USE_ZCR:
                    z = librosa.feature.zero_crossing_rate(seg)[0]
                    feat_vec.extend([z.mean(), z.std()])
                if USE_RMS:
                    r = librosa.feature.rms(y=seg)[0]
                    feat_vec.extend([r.mean(), r.std()])
                mel_stats = mel
                feat_vec.extend([mel_stats.mean(), mel_stats.std(),
                                mel_stats.max(), np.percentile(mel_stats, 90)])
                all_features.append(feat_vec)

                metadata.append({
                    "file": rec["filename"], "unit": rec["unit"],
                    "segment_idx": i, "offset_s": i * 10,
                    "path": rec["path"],
                })
            del file_segments
            gc.collect()

        n_segs = len(all_features)
        features = np.array(all_features, dtype=np.float32)
        del all_features
        gc.collect()

        # Build mel array
        time_dim = min(m.shape[1] for m in all_mels_list)
        all_mels_array = np.array([m[:, :time_dim] for m in all_mels_list], dtype=np.float32)
        all_mels_array = (all_mels_array - all_mels_array.min()) / (all_mels_array.max() - all_mels_array.min() + 1e-8)
        del all_mels_list
        gc.collect()

        # Save cache
        print("\n--- Saving feature cache ---")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_features_path, features)
        np.save(cache_mels_path, all_mels_array)
        with open(cache_meta_path, "w") as f:
            json.dump(metadata, f)
        print(f"Cache saved: features={features.shape}, mels={all_mels_array.shape}")

    t_load = time.time() - t_start
    print(f"Total segments: {len(metadata)}")
    print(f"Features: {features.shape}, loaded in {t_load:.1f}s")
    print(f"Mels: {all_mels_array.shape}")
    n_segs = len(metadata)

    segments = None  # Raw audio never kept — PANNs/temporal will be skipped

    # ---------------------------------------------------------------------------
    # NOAA humpback whale detection (per-file resampling for speed)
    # ---------------------------------------------------------------------------
    noaa_features = None
    if USE_NOAA_WHALE:
        cache_noaa_path = os.path.join(CACHE_DIR, "noaa_scores.npy")
        if os.path.exists(cache_noaa_path):
            print("\n--- Loading cached NOAA scores ---")
            noaa_features = np.load(cache_noaa_path)
            print(f"NOAA cache loaded: {noaa_features.shape}")
        else:
            print("\n--- NOAA Humpback Whale Detection (per-file resampling) ---")
            t1 = time.time()
            try:
                import tensorflow_hub as hub
                import tensorflow as tf
                print("  Loading NOAA model...")
                noaa_model = hub.load("https://tfhub.dev/google/humpback_whale/1")
                noaa_score_fn = noaa_model.signatures["score"]
                noaa_sr = 10000
                all_noaa_scores = []
                for rec in recordings:
                    # Resample entire file ONCE to 10kHz
                    audio, sr = load_audio(rec["path"])
                    audio = highpass_filter(audio, sr)
                    # Segment at native SR first (matches tier1 segmentation)
                    file_segments = segment_audio(audio, sr)
                    del audio
                    for seg in file_segments:
                        # Resample each segment to 10kHz for NOAA
                        seg_10k = librosa.resample(seg, orig_sr=TARGET_SR, target_sr=noaa_sr)
                        waveform = tf.constant(seg_10k.astype(np.float32).reshape(1, -1, 1))
                        context_step = tf.constant(noaa_sr, dtype=tf.int64)
                        result = noaa_score_fn(waveform=waveform, context_step_samples=context_step)
                        scores = result["scores"].numpy().flatten()
                        all_noaa_scores.append([float(scores.mean()), float(scores.max()), float(scores.std())])
                    del file_segments
                    gc.collect()
                    if len(all_noaa_scores) % 500 == 0:
                        print(f"  NOAA: {len(all_noaa_scores)}/{n_segs} segments")

                noaa_features = np.array(all_noaa_scores, dtype=np.float32)
                np.save(cache_noaa_path, noaa_features)
                print(f"  NOAA scores: {noaa_features.shape}, "
                      f"mean={noaa_features[:, 0].mean():.4f}, max={noaa_features[:, 1].max():.4f}, "
                      f"{time.time()-t1:.1f}s")
            except Exception as e:
                print(f"  NOAA failed: {e}")
                noaa_features = None
            finally:
                try:
                    del noaa_model, noaa_score_fn
                except Exception:
                    pass
                gc.collect()
                print("  TF GPU memory freed")

    features = features  # use all 113 features (MI selection didn't help in 12j)

    # Reduce dimensions
    print("\n--- Reducing dimensions ---")
    t1 = time.time()
    features_reduced = reduce_dimensions(features)
    print(f"Reduced: {features_reduced.shape}, {time.time()-t1:.1f}s")

    # Cluster
    print("\n--- Clustering ---")
    t1 = time.time()
    labels = cluster(features_reduced)
    print(f"Clustering: {time.time()-t1:.1f}s")

    # Evaluate (primary metric)
    eval_result = evaluate_clustering(labels, features_reduced, method_name=f"tier{TIER}")

    # Discovery insights
    discovery = evaluate_discovery(labels, metadata, features_reduced, segments)

    # Tier 3a: CNN classifier with NOAA-enhanced pseudo-labels
    labels_hdbscan = labels.copy()
    cnn_labels = labels.copy()
    if noaa_features is not None:
        # Sweep whale thresholds quickly (just counting, no training)
        for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
            n = (noaa_features[:, 1] >= thr).sum()
            print(f"  NOAA threshold={thr}: {n} whale segments ({n/len(labels)*100:.1f}%)")
        # Use 0.7 — captures more whale segments (512 vs 345 at 0.8)
        whale_threshold = 0.7
        whale_mask = noaa_features[:, 1] >= whale_threshold
        n_whale = whale_mask.sum()
        if n_whale > 0:
            whale_class = cnn_labels.max() + 1 if cnn_labels.max() >= 0 else 0
            whale_clusters = set(cnn_labels[whale_mask])
            cnn_labels[whale_mask] = whale_class
            print(f"  Using threshold={whale_threshold}: {n_whale} segments "
                  f"→ class {whale_class} (from {len(whale_clusters)} clusters)")

    print("\n--- Tier 3a: CNN Classifier ---")
    t1 = time.time()
    cnn_features = train_cnn_classifier(all_mels_array, cnn_labels)
    from sklearn.preprocessing import StandardScaler
    cnn_norm = StandardScaler().fit_transform(cnn_features)

    features_reduced_t3 = reduce_dimensions(cnn_norm)
    labels_t3 = cluster(features_reduced_t3)
    eval_t3 = evaluate_clustering(labels_t3, features_reduced_t3, method_name="tier3_cnn")

    tier1_norm = StandardScaler().fit_transform(features)
    combined = np.hstack([tier1_norm, cnn_norm])
    features_reduced_t3c = reduce_dimensions(combined)
    labels_t3c = cluster(features_reduced_t3c)
    eval_t3c = evaluate_clustering(labels_t3c, features_reduced_t3c, method_name="tier3_combined")
    print(f"  CNN-only: {eval_t3['composite_score']:.6f}, "
          f"Combined: {eval_t3c['composite_score']:.6f}")

    if eval_t3c['composite_score'] > eval_t3['composite_score']:
        eval_t3 = eval_t3c
        labels_t3 = labels_t3c
        features_reduced_t3 = features_reduced_t3c
    print(f"  Tier 3a: {eval_t3['composite_score']:.6f} "
          f"(vs Tier 1: {eval_result['composite_score']:.6f}), {time.time()-t1:.1f}s")

    if eval_t3['composite_score'] > eval_result['composite_score']:
        print("  ** Tier 3a CNN IMPROVED! **")
        labels = labels_t3
        features_reduced = features_reduced_t3
        eval_result = eval_t3

    # UMAP n_components sweep on CNN features
    print("\n--- UMAP n_components sweep (CNN features) ---")
    t1c = time.time()
    import umap
    for nc in [15, 25, 30]:
        if nc == N_COMPONENTS:
            continue  # already tested
        reducer_nc = umap.UMAP(n_components=nc, n_neighbors=UMAP_N_NEIGHBORS,
                               min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=42)
        fr_nc = reducer_nc.fit_transform(cnn_norm)
        lb_nc = cluster(fr_nc)
        ev_nc = evaluate_clustering(lb_nc, fr_nc, method_name=f"cnn_nc{nc}")
        print(f"  n_components={nc}: {ev_nc['composite_score']:.6f}")
        if ev_nc['composite_score'] > eval_result['composite_score']:
            print(f"  ** n_components={nc} IMPROVED: {ev_nc['composite_score']:.6f} **")
            labels = lb_nc
            features_reduced = fr_nc
            eval_result = ev_nc
    print(f"  Component sweep time: {time.time()-t1c:.1f}s")

    # Tier 3b: Contrastive learning (skipped — consistently underperforms)
    print("\n--- Tier 3b: Contrastive Learning (skipped, consistently <0.6) ---")
    # No contrastive features this run

    # NOAA combos (if available)
    if noaa_features is not None:
        print("\n--- NOAA Combos ---")
        t1 = time.time()
        noaa_candidates = [
            (np.hstack([tier1_norm, noaa_features]), "noaa+tier1"),
            (np.hstack([cnn_norm, noaa_features]), "noaa+cnn"),
            (np.hstack([tier1_norm, cnn_norm, noaa_features]), "noaa+tier1+cnn"),
        ]
        for combo_feats, name in noaa_candidates:
            fr = reduce_dimensions(combo_feats)
            lb = cluster(fr)
            ev = evaluate_clustering(lb, fr, method_name=name)
            print(f"  {name}: {ev['composite_score']:.6f}")
            if ev['composite_score'] > eval_result['composite_score']:
                print(f"  ** {name} IMPROVED: {ev['composite_score']:.6f} **")
                labels = lb
                features_reduced = fr
                eval_result = ev
        print(f"  NOAA combos time: {time.time()-t1:.1f}s")

    # Kitchen sink: combine all available feature types
    print("\n--- Kitchen Sink (all features) ---")
    t1 = time.time()
    sink_parts = [tier1_norm, cnn_norm]
    sink_names = ["T1", "CNN"]
    if noaa_features is not None:
        sink_parts.append(noaa_features)
        sink_names.append("NOAA")
    combined_sink = np.hstack(sink_parts)
    features_reduced_sink = reduce_dimensions(combined_sink)
    labels_sink = cluster(features_reduced_sink)
    eval_sink = evaluate_clustering(labels_sink, features_reduced_sink,
                                    method_name="kitchen_sink_" + "+".join(sink_names))
    print(f"  {'+'.join(sink_names)}: {eval_sink['composite_score']:.6f} "
          f"({combined_sink.shape[1]} dims)")
    print(f"  Kitchen sink time: {time.time()-t1:.1f}s")
    if eval_sink['composite_score'] > eval_result['composite_score']:
        print(f"  ** Kitchen sink IMPROVED: {eval_sink['composite_score']:.6f} **")
        labels = labels_sink
        features_reduced = features_reduced_sink
        eval_result = eval_sink

    # ---------------------------------------------------------------------------
    # Optimization: Per-unit feature normalization
    # Clusters split by hydrophone unit due to different sample rates (24/48/96kHz)
    # Normalizing features per-unit removes unit-driven artifacts
    # ---------------------------------------------------------------------------
    print("\n--- Per-Unit Feature Normalization ---")
    t1 = time.time()
    units = [m["unit"] for m in metadata]
    unique_units = sorted(set(units))
    print(f"  Units: {unique_units}")

    # Normalize tier1 features per-unit
    tier1_unit_norm = features.copy()
    for unit in unique_units:
        mask = np.array([u == unit for u in units])
        if mask.sum() > 1:
            unit_mean = tier1_unit_norm[mask].mean(axis=0)
            unit_std = tier1_unit_norm[mask].std(axis=0) + 1e-8
            tier1_unit_norm[mask] = (tier1_unit_norm[mask] - unit_mean) / unit_std
    tier1_un = StandardScaler().fit_transform(tier1_unit_norm)

    # Try per-unit norm with best feature combos
    un_candidates = [
        (tier1_un, "unitnorm_T1"),
        (np.hstack([tier1_un, cnn_norm]), "unitnorm_T1+CNN"),
    ]
    if noaa_features is not None:
        un_candidates.append((np.hstack([tier1_un, cnn_norm, noaa_features]), "unitnorm_T1+CNN+NOAA"))
    for combo_feats, name in un_candidates:
        fr = reduce_dimensions(combo_feats)
        lb = cluster(fr)
        ev = evaluate_clustering(lb, fr, method_name=name)
        print(f"  {name}: {ev['composite_score']:.6f}")
        if ev['composite_score'] > eval_result['composite_score']:
            print(f"  ** {name} IMPROVED: {ev['composite_score']:.6f} **")
            labels = lb
            features_reduced = fr
            eval_result = ev
    print(f"  Per-unit norm time: {time.time()-t1:.1f}s")

    # ---------------------------------------------------------------------------
    # Optimization: HDBSCAN parameter sweep on best features so far
    # ---------------------------------------------------------------------------
    print("\n--- HDBSCAN Parameter Sweep ---")
    t1 = time.time()
    best_sweep_eval = eval_result
    best_sweep_labels = labels
    best_sweep_name = "current_best"
    import hdbscan as hdbscan_lib
    for min_cluster in [8, 10, 12, 15, 20]:
        for min_samples in [3, 5, 7, 10]:
            lb = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster, min_samples=min_samples
            ).fit_predict(features_reduced)
            ev = evaluate_clustering(lb, features_reduced,
                                     method_name=f"sweep_mc{min_cluster}_ms{min_samples}")
            n_cl = len(set(lb[lb >= 0]))
            noise_pct = (lb == -1).sum() / len(lb) * 100
            if ev['composite_score'] > best_sweep_eval['composite_score']:
                best_sweep_eval = ev
                best_sweep_labels = lb
                best_sweep_name = f"mc{min_cluster}_ms{min_samples}"
                print(f"  ** mc={min_cluster} ms={min_samples}: {ev['composite_score']:.6f} "
                      f"({n_cl} clusters, {noise_pct:.1f}% noise) **")
    if best_sweep_name != "current_best":
        print(f"  Sweep winner: {best_sweep_name} = {best_sweep_eval['composite_score']:.6f}")
        labels = best_sweep_labels
        eval_result = best_sweep_eval
    else:
        print(f"  No improvement from sweep (best remains {eval_result['composite_score']:.6f})")
    print(f"  Sweep time: {time.time()-t1:.1f}s")

    # ---------------------------------------------------------------------------
    # Optimization: Noise reassignment via KNN
    # ---------------------------------------------------------------------------
    n_noise = (labels == -1).sum()
    if n_noise > 0:
        print(f"\n--- Noise Reassignment ({n_noise} noise points) ---")
        t1 = time.time()
        from sklearn.neighbors import KNeighborsClassifier
        noise_mask = labels == -1
        labeled_mask = labels >= 0
        if labeled_mask.sum() > 0:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(features_reduced[labeled_mask], labels[labeled_mask])
            noise_preds = knn.predict(features_reduced[noise_mask])
            labels_reassigned = labels.copy()
            labels_reassigned[noise_mask] = noise_preds
            ev_reassigned = evaluate_clustering(labels_reassigned, features_reduced,
                                                method_name="noise_reassigned")
            print(f"  After reassignment: {ev_reassigned['composite_score']:.6f} "
                  f"(was {eval_result['composite_score']:.6f}), "
                  f"coverage: {ev_reassigned['coverage']:.4f}")
            if ev_reassigned['composite_score'] > eval_result['composite_score']:
                print(f"  ** Noise reassignment IMPROVED! **")
                labels = labels_reassigned
                eval_result = ev_reassigned
            print(f"  Reassignment time: {time.time()-t1:.1f}s")

    # Update discovery with potentially new labels
    discovery = evaluate_discovery(labels, metadata, features_reduced, segments)

    # PANNs labels for cluster representatives (Tier 2 interpretation)
    panns_labels = None
    if USE_PANNS_LABELS:
        print("\n--- PANNs AudioSet Labels (Tier 2) ---")
        t1 = time.time()
        # Get representative segments from each cluster (closest to centroid)
        representative_indices = []
        for c in sorted(set(labels[labels >= 0])):
            cmask = labels == c
            cluster_indices = np.where(cmask)[0]
            cluster_features = features_reduced[cmask]
            centroid = cluster_features.mean(axis=0)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest = cluster_indices[np.argsort(distances)[:3]]
            representative_indices.extend(closest.tolist())

        # Also add a few noise samples if any
        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0:
            representative_indices.extend(noise_indices[:3].tolist())

        if segments is not None:
            rep_segments = [segments[i] for i in representative_indices]
            raw_panns = get_panns_labels(rep_segments)
            if raw_panns is not None:
                panns_labels = {representative_indices[k]: v for k, v in raw_panns.items()}
                print(f"  PANNs labels for {len(representative_indices)} representatives, {time.time()-t1:.1f}s")
        else:
            print("  PANNs skipped (raw audio freed for memory)")

    # Analysis with improved interpretation
    analyze_clusters(labels, metadata, features, segments, panns_labels, noaa_scores=noaa_features)

    # Temporal analysis
    if segments is not None:
        temporal = analyze_temporal(labels, metadata, segments)
    else:
        temporal = None
        print("  Temporal analysis skipped (raw audio freed for memory)")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.json")
    result_data = {
        "config": config, "evaluation": eval_result,
        "discovery": discovery, "temporal": temporal,
    }
    if panns_labels:
        # Serialize PANNs labels (convert int keys to strings for JSON)
        result_data["panns_labels"] = {str(k): v for k, v in panns_labels.items()}
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    # =========================================================================
    # PHASE 2: Cross-Unit Marine Sound Classifier
    # Use clustering + NOAA to create labeled dataset, train classifier,
    # validate across hydrophone units, export for inference on new data.
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Cross-Unit Marine Sound Classifier")
    print("="*60)
    t_phase2 = time.time()

    # Step 1: Create labeled dataset from clustering + NOAA + acoustics
    print("\n--- Creating labeled dataset ---")
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, f1_score

    # Compute per-segment band power for ecological classification
    # features columns: MFCC(40) + delta(40) + band(3) + spectral(8) + contrast(14) + zcr(2) + rms(2) + mel(4) = 113
    # band power is at indices 40+40 = 80,81,82 (low, mid, high)
    band_low_idx, band_mid_idx, band_high_idx = 80, 81, 82
    rms_mean_idx = 108  # rms mean is near the end

    eco_labels = []
    units = [m["unit"] for m in metadata]
    for i in range(n_segs):
        noaa_max = noaa_features[i, 1] if noaa_features is not None else 0
        noaa_mean = noaa_features[i, 0] if noaa_features is not None else 0
        low_db = features[i, band_low_idx]
        mid_db = features[i, band_mid_idx]
        high_db = features[i, band_high_idx]
        rms_val = features[i, rms_mean_idx]
        mid_low = mid_db - low_db

        # Whale detection (NOAA-based)
        if noaa_max >= 0.7:
            if noaa_mean >= 0.5:
                eco_labels.append("whale_strong")  # sustained whale vocalization
            else:
                eco_labels.append("whale_brief")   # brief whale call in segment
        elif noaa_max >= 0.4:
            eco_labels.append("possible_whale")
        # Anthropogenic (low-freq dominated, loud)
        elif mid_low < -15 and rms_val > 0.005:
            eco_labels.append("vessel_noise")
        # Biological (mid-freq activity)
        elif mid_db > -85 and mid_low > -5:
            eco_labels.append("biophony")  # reef, fish, shrimp
        # Quiet ambient
        elif rms_val < 0.001:
            eco_labels.append("silence")
        # General ambient
        else:
            eco_labels.append("ambient")

    eco_labels = np.array(eco_labels)
    from collections import Counter
    label_counts = Counter(eco_labels)
    print(f"  Label distribution: {dict(label_counts)}")

    # Create numeric labels
    label_names = sorted(set(eco_labels))
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    y_all = np.array([label_to_idx[l] for l in eco_labels])
    n_eco_classes = len(label_names)
    print(f"  Classes: {label_names} ({n_eco_classes} classes)")

    # Step 2: Leave-one-unit-out cross-validation
    print("\n--- Leave-One-Unit-Out Cross-Validation ---")
    unique_units = sorted(set(units))
    unit_arr = np.array(units)

    all_fold_accs = []
    all_fold_f1s = []

    for held_out_unit in unique_units:
        test_mask = unit_arr == held_out_unit
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train = all_mels_array[train_mask]
        y_train = y_all[train_mask]
        X_test = all_mels_array[test_mask]
        y_test = y_all[test_mask]

        # Check if test set has all unknown — skip
        if len(set(y_test)) < 2 and label_names[y_test[0]] == "unknown":
            print(f"  Unit {held_out_unit}: skipped (all unknown)")
            continue

        # Train CNN classifier
        time_dim = X_train.shape[2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class EcoClassifier(nn.Module):
            def __init__(self, n_mels, time_dim, n_classes):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.fc = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, n_classes),
                )

            def forward(self, x):
                x = x.unsqueeze(1)  # (B, 1, n_mels, time)
                h = self.conv(x)
                h = h.view(h.size(0), -1)
                return self.fc(h)

        model = EcoClassifier(128, time_dim, n_eco_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Class-weighted loss for imbalanced classes
        class_counts = np.bincount(y_train, minlength=n_eco_classes).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * n_eco_classes
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

        # Train with cosine annealing
        n_train_epochs = 150
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_train_epochs)

        model.train()
        for epoch in range(n_train_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate on held-out unit
        model.eval()
        X_te = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_te), 64):
                batch = X_te[i:i+64].to(device)
                pred = model(batch).argmax(dim=1).cpu().numpy()
                preds.extend(pred)
        preds = np.array(preds)

        acc = (preds == y_test).mean()
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        all_fold_accs.append(acc)
        all_fold_f1s.append(f1)
        print(f"  Unit {held_out_unit}: acc={acc:.4f}, f1={f1:.4f} "
              f"(train={train_mask.sum()}, test={test_mask.sum()})")

        # Per-class report for this fold
        report = classification_report(y_test, preds, target_names=label_names,
                                        zero_division=0, output_dict=True)
        for cls_name in label_names:
            if cls_name in report:
                r = report[cls_name]
                print(f"    {cls_name}: prec={r['precision']:.3f} rec={r['recall']:.3f} "
                      f"f1={r['f1-score']:.3f} n={r['support']}")

    if all_fold_accs:
        mean_acc = np.mean(all_fold_accs)
        mean_f1 = np.mean(all_fold_f1s)
        print(f"\n  Cross-unit mean: acc={mean_acc:.4f}, f1={mean_f1:.4f}")

    # Step 3: Train final model on all data and export
    print("\n--- Training final model (all units) ---")
    final_model = EcoClassifier(128, all_mels_array.shape[2], n_eco_classes).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Class-weighted loss
    all_class_counts = np.bincount(y_all, minlength=n_eco_classes).astype(np.float32)
    all_class_weights = 1.0 / (all_class_counts + 1)
    all_class_weights = all_class_weights / all_class_weights.sum() * n_eco_classes
    final_criterion = nn.CrossEntropyLoss(weight=torch.tensor(all_class_weights).to(device))

    X_all_t = torch.tensor(all_mels_array, dtype=torch.float32)
    y_all_t = torch.tensor(y_all, dtype=torch.long)
    all_dataset = torch.utils.data.TensorDataset(X_all_t, y_all_t)
    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=64, shuffle=True)
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=200)

    final_model.train()
    for epoch in range(200):
        total_loss = 0
        for xb, yb in all_loader:
            xb, yb = xb.to(device), yb.to(device)
            final_optimizer.zero_grad()
            loss = final_criterion(final_model(xb), yb)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item()
        final_scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/200: loss={total_loss:.4f}")

    # Export model
    model_path = os.path.join(RESULTS_DIR, "marine_classifier.pt")
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "label_names": label_names,
        "label_to_idx": label_to_idx,
        "n_mels": 128,
        "time_dim": all_mels_array.shape[2],
        "n_classes": n_eco_classes,
        "cross_unit_acc": mean_acc if all_fold_accs else None,
        "cross_unit_f1": mean_f1 if all_fold_f1s else None,
    }, model_path)
    print(f"  Model exported: {model_path}")
    print(f"  Phase 2 time: {time.time()-t_phase2:.1f}s")

    # Final summary
    t_total = time.time() - t_start
    print("\n---")
    print(f"composite_score:  {eval_result['composite_score']:.6f}")
    print(f"silhouette:       {eval_result['silhouette']:.6f}")
    print(f"calinski_harabasz:{eval_result['calinski_harabasz']:.6f}")
    print(f"n_clusters:       {eval_result['n_clusters']}")
    print(f"n_noise:          {eval_result['n_noise']}")
    print(f"coverage:         {eval_result['coverage']:.4f}")
    print(f"n_features:       {features.shape[1]}")
    print(f"total_segments:   {n_segs}")
    print(f"tier:             {TIER}")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")

    # Discovery summary
    print(f"\n--- Discovery ---")
    print(f"temporal_spread:  {discovery['mean_temporal_spread']:.1f}")
    print(f"unit_diversity:   {discovery['mean_unit_diversity']:.1f}")
    if discovery["band_profiles"]:
        print("band_profiles:")
        for k, v in discovery["band_profiles"].items():
            print(f"  {k}: LOW={v['low_db']:.1f} MID={v['mid_db']:.1f} HIGH={v['high_db']:.1f} ({v['n_segments']} segs)")


if __name__ == "__main__":
    main()
