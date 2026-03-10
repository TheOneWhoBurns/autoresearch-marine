"""
Marine Acoustic Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Starting point: Tier 1 acoustic indices baseline.
The agent evolves this through Tiers 1-3, choosing the best approach.

Metric: composite_score from prepare.evaluate_clustering() — higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
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
TIER = 1  # 1=acoustic indices, 2=pretrained embeddings, 3=custom classifier

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Feature extraction
N_MFCC = 20
USE_DELTA_MFCC = True       # delta + delta-delta MFCCs
USE_BAND_POWER = True
USE_SPECTRAL = True
USE_SPECTRAL_CONTRAST = True # spectral contrast (7 bands)
USE_ZCR = True
USE_RMS = True

# Dimensionality reduction
REDUCER = "umap"        # "umap", "pca", or "none"
N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "euclidean"

# Clustering
CLUSTERER = "hdbscan"   # "hdbscan", "kmeans", "spectral", "gmm"
HDBSCAN_MIN_CLUSTER = 5
HDBSCAN_MIN_SAMPLES = 3
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

        # MFCCs (mean + std) + deltas
        mfcc = librosa.feature.mfcc(y=seg, sr=TARGET_SR, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc, axis=1))
        if USE_DELTA_MFCC:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            feats.extend(np.mean(delta, axis=1))
            feats.extend(np.mean(delta2, axis=1))

        # Ecological band powers
        if USE_BAND_POWER:
            bp = compute_band_power(seg)
            feats.extend([bp["low"], bp["mid"], bp["high"]])

        # Spectral shape
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

        # Spectral contrast
        if USE_SPECTRAL_CONTRAST:
            contrast = librosa.feature.spectral_contrast(y=seg, sr=TARGET_SR,
                                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
            feats.extend(np.mean(contrast, axis=1))
            feats.extend(np.std(contrast, axis=1))

        # Zero crossing rate
        if USE_ZCR:
            zcr = librosa.feature.zero_crossing_rate(seg, frame_length=N_FFT,
                                                      hop_length=HOP_LENGTH)
            feats.extend([np.mean(zcr), np.std(zcr)])

        # RMS energy
        if USE_RMS:
            feats.append(compute_rms(seg))

        features_list.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  Features: {i+1}/{len(segments)}")

    return np.array(features_list, dtype=np.float32)


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

def analyze_clusters(labels, metadata, features, segments=None):
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

        # Band power profile for ecological interpretation
        if segments is not None:
            sample_indices = [i for i in range(len(labels)) if mask[i]][:5]
            if sample_indices:
                powers = [compute_band_power(segments[i]) for i in sample_indices]
                avg_low = np.mean([p["low"] for p in powers])
                avg_mid = np.mean([p["mid"] for p in powers])
                avg_high = np.mean([p["high"] for p in powers])
                print(f"    Band profile: LOW={avg_low:.1f}dB  MID={avg_mid:.1f}dB  HIGH={avg_high:.1f}dB")

                # Ecological guess
                if avg_low > avg_mid and avg_low > avg_high:
                    print(f"    -> Likely: boat noise or large whale vocalizations")
                elif avg_mid > avg_low and avg_mid > avg_high:
                    print(f"    -> Likely: snapping shrimp, dolphin whistles, or reef activity")
                elif avg_high > avg_mid:
                    print(f"    -> Likely: echolocation clicks or ultrasonic biological sounds")


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
        "n_mfcc": N_MFCC, "use_delta_mfcc": USE_DELTA_MFCC,
        "use_band_power": USE_BAND_POWER,
        "use_spectral": USE_SPECTRAL, "use_spectral_contrast": USE_SPECTRAL_CONTRAST,
        "use_zcr": USE_ZCR, "use_rms": USE_RMS,
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

    # Extract features
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

    # Analysis
    analyze_clusters(labels, metadata, features, segments)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump({"config": config, "evaluation": eval_result, "discovery": discovery}, f, indent=2)

    # Final summary (parseable — the autoresearch loop reads these)
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
