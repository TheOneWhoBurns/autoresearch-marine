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
import re
import time
import json
import numpy as np

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

# Dimensionality reduction
REDUCER = "umap"
N_COMPONENTS = 20
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "euclidean"

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
# Tier 3: CNN classifier with pseudo-labels from HDBSCAN
# ---------------------------------------------------------------------------

def train_cnn_classifier(segments, pseudo_labels):
    """Train a small CNN on mel spectrograms using HDBSCAN pseudo-labels.
    Returns learned features (penultimate layer) for re-clustering."""
    import torch
    import torch.nn as nn

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  CNN device: {device}")

    # Only use labeled segments (exclude noise label -1)
    mask = pseudo_labels >= 0
    labeled_indices = np.where(mask)[0]
    n_classes = len(set(pseudo_labels[mask]))
    print(f"  Training CNN: {len(labeled_indices)} labeled segments, {n_classes} classes")

    # Build mel spectrograms
    mels = []
    for idx in labeled_indices:
        mel = compute_melspec(segments[idx])  # (128, T)
        mels.append(mel)
    time_dim = min(m.shape[1] for m in mels)
    X = np.array([m[:, :time_dim] for m in mels], dtype=np.float32)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    y = pseudo_labels[mask]

    # Simple CNN
    class MarineCNN(nn.Module):
        def __init__(self, n_mels, time_dim, n_classes, feat_dim=64):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            )
            # Compute conv output size
            dummy = torch.zeros(1, 1, n_mels, time_dim)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.numel()
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, feat_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(feat_dim, n_classes)

        def forward(self, x):
            h = self.conv(x)
            feats = self.features(h)
            logits = self.classifier(feats)
            return logits, feats

    model = MarineCNN(128, time_dim, n_classes, feat_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.from_numpy(X[:, np.newaxis, :, :])
    y_tensor = torch.from_numpy(y.astype(np.int64))

    batch_size = 64
    n = len(X_tensor)
    n_epochs = 30

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0
        correct = 0
        total = 0
        for i in range(0, n, batch_size):
            batch_x = X_tensor[perm[i:i+batch_size]].to(device)
            batch_y = y_tensor[perm[i:i+batch_size]].to(device)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += len(batch_y)
        if (epoch + 1) % 10 == 0:
            acc = correct / total * 100
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={total_loss:.4f}, acc={acc:.1f}%")

    # Extract features for ALL segments (including noise)
    model.eval()
    all_mels = []
    for seg in segments:
        mel = compute_melspec(seg)
        all_mels.append(mel[:, :time_dim])
    X_all = np.array(all_mels, dtype=np.float32)
    X_all = (X_all - X_all.min()) / (X_all.max() - X_all.min() + 1e-8)
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

def analyze_clusters(labels, metadata, features, segments=None, panns_labels=None):
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

    # Load data
    print("\n--- Loading data ---")
    segments, metadata = build_dataset()
    t_load = time.time() - t_start
    print(f"Data loaded: {t_load:.1f}s")

    # Extract features (Tier 1 for clustering)
    print("\n--- Extracting features ---")
    t1 = time.time()
    features = extract_features(segments)
    print(f"Features: {features.shape}, {time.time()-t1:.1f}s")

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

    # Tier 3: CNN classifier with pseudo-labels
    print("\n--- Tier 3: CNN Classifier ---")
    t1 = time.time()
    cnn_features = train_cnn_classifier(segments, labels)
    # Re-cluster using CNN features (combined with Tier 1)
    from sklearn.preprocessing import StandardScaler
    tier1_norm = StandardScaler().fit_transform(features)
    cnn_norm = StandardScaler().fit_transform(cnn_features)
    combined = np.hstack([tier1_norm, cnn_norm])
    print(f"  Combined features: {combined.shape}")

    # Reduce and re-cluster
    features_reduced_t3 = reduce_dimensions(combined)
    labels_t3 = cluster(features_reduced_t3)
    eval_t3 = evaluate_clustering(labels_t3, features_reduced_t3, method_name="tier3_cnn")
    print(f"  Tier 3 composite: {eval_t3['composite_score']:.6f} "
          f"(vs Tier 1: {eval_result['composite_score']:.6f})")
    print(f"  Tier 3 time: {time.time()-t1:.1f}s")

    # Use Tier 3 results if better
    if eval_t3['composite_score'] > eval_result['composite_score']:
        print("  ** Tier 3 CNN features IMPROVED clustering! Using Tier 3 results. **")
        labels = labels_t3
        features_reduced = features_reduced_t3
        eval_result = eval_t3
        features = combined
    else:
        print("  Tier 3 CNN did not improve clustering. Keeping Tier 1 results.")

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

        rep_segments = [segments[i] for i in representative_indices]
        raw_panns = get_panns_labels(rep_segments)
        if raw_panns is not None:
            panns_labels = {representative_indices[k]: v for k, v in raw_panns.items()}
            print(f"  PANNs labels for {len(representative_indices)} representatives, {time.time()-t1:.1f}s")

    # Analysis with improved interpretation
    analyze_clusters(labels, metadata, features, segments, panns_labels)

    # Temporal analysis
    temporal = analyze_temporal(labels, metadata, segments)

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
    print(f"total_segments:   {len(segments)}")
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
