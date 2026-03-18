"""
explore_sample.py — Prueba rápida de Tier 1 + Tier 2 sobre un sample pequeño.

Tier 1 (scikit-maad): índices acústicos por archivo — entropía, potencia por banda, NDSI
Tier 2 (PANNs):       embeddings de 2048 dims → similitud entre archivos

Uso:
    python -m src.explore_sample
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import librosa
import maad
from scipy.signal import welch
from panns_inference import AudioTagging

# ---------------------------------------------------------------------------
# Sample: 4 archivos de 6478 en distintas horas del 2023-07-24
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent
HACKATHON_DATA = DATA_DIR / "src/marine-acoustic-monitoring/hackathon_data/marine-acoustic"

SAMPLE_FILES = [
    (HACKATHON_DATA / "6478/6478.230724044251.wav",  "04:42 (madrugada)"),
    (HACKATHON_DATA / "6478/6478.230724091251.wav",  "09:12 (mañana)"),
    (HACKATHON_DATA / "6478/6478.230724141251.wav",  "14:12 (tarde)"),
    (HACKATHON_DATA / "6478/6478.230724234251.wav",  "23:42 (noche)"),
]

LOAD_DURATION_S = 60   # primeros 60s de cada archivo (rápido)
SR_TARGET       = 96_000  # sample rate nativo de 6478

# Bandas adaptadas para audio submarino (no las defaults de maad que son para bosques)
BAND_LOW  = (50,   2_000)   # barcos, peces
BAND_MID  = (2_000, 20_000)  # camarones, delfines
BAND_HIGH = (20_000, 48_000) # ecolocalización (hasta Nyquist de 96kHz)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clip(path: Path, duration_s: float = LOAD_DURATION_S) -> tuple:
    """Carga los primeros duration_s segundos como float32 mono."""
    audio, sr = librosa.load(str(path), sr=None, mono=True, duration=duration_s)
    return audio, sr


def band_power_db(audio: np.ndarray, sr: int, fmin: int, fmax: int) -> float:
    """Potencia media en dB dentro de una banda de frecuencias."""
    freqs, psd = welch(audio, fs=sr, nperseg=min(4096, len(audio)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    power = np.mean(psd[mask]) if mask.any() else 0.0
    return 10 * np.log10(power + 1e-12)


def compute_ndsi(audio: np.ndarray, sr: int) -> float:
    """
    NDSI adaptado para audio submarino:
        biophony  = potencia en BAND_MID
        anthrophony = potencia en BAND_LOW
    NDSI = (bio - anthro) / (bio + anthro)  ∈ [-1, 1]
    """
    bio   = band_power_db(audio, sr, *BAND_MID)
    antro = band_power_db(audio, sr, *BAND_LOW)
    # Convertir de dB a lineal para el ratio
    bio_lin   = 10 ** (bio / 10)
    antro_lin = 10 ** (antro / 10)
    return (bio_lin - antro_lin) / (bio_lin + antro_lin + 1e-12)


def compute_entropy(audio: np.ndarray, sr: int) -> float:
    """Entropía temporal de Wiener via maad."""
    try:
        Ht = maad.features.temporal_entropy(audio)
        return float(Ht)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Tier 2: PANNs embeddings
# ---------------------------------------------------------------------------

def load_panns() -> AudioTagging:
    print("Cargando modelo PANNs... ", end="", flush=True)
    at = AudioTagging(checkpoint_path=None, device="cpu")
    print("listo.")
    return at


def get_embedding(at: AudioTagging, audio: np.ndarray, sr: int) -> np.ndarray:
    """Devuelve embedding de 2048 dims. Resamplea a 32kHz si hace falta."""
    if sr != 32_000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=32_000)
    _, embedding = at.inference(audio[None, :])
    return embedding[0]  # (2048,)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("TIER 1 — Índices acústicos (primeros 60s por archivo)")
    print("=" * 60)

    results = []
    for path, label in SAMPLE_FILES:
        if not path.exists():
            print(f"  [SKIP] {path.name} no encontrado")
            continue
        print(f"\n{label}  ({path.name})")
        audio, sr = load_clip(path)

        low  = band_power_db(audio, sr, *BAND_LOW)
        mid  = band_power_db(audio, sr, *BAND_MID)
        high = band_power_db(audio, sr, *BAND_HIGH)
        ndsi = compute_ndsi(audio, sr)
        ht   = compute_entropy(audio, sr)

        print(f"  Banda LOW  (barcos/peces)   : {low:7.1f} dB")
        print(f"  Banda MID  (delfines)       : {mid:7.1f} dB")
        print(f"  Banda HIGH (ecolocalización): {high:7.1f} dB")
        print(f"  NDSI (bio vs antro)         : {ndsi:+.3f}  {'↑ más biológico' if ndsi > 0 else '↓ más antropogénico'}")
        print(f"  Entropía temporal           : {ht:.4f}")

        results.append({"label": label, "path": path, "audio": audio, "sr": sr,
                         "low": low, "mid": mid, "high": high, "ndsi": ndsi, "ht": ht})

    # Resumen comparativo
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("RESUMEN COMPARATIVO")
        print("=" * 60)
        print(f"{'Hora':<22} {'LOW':>8} {'MID':>8} {'HIGH':>8} {'NDSI':>8} {'Ht':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['label']:<22} {r['low']:>8.1f} {r['mid']:>8.1f} {r['high']:>8.1f} {r['ndsi']:>+8.3f} {r['ht']:>8.4f}")

    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TIER 2 — PANNs embeddings + similitud entre archivos")
    print("=" * 60)

    at = load_panns()
    embeddings = []
    for r in results:
        emb = get_embedding(at, r["audio"], r["sr"])
        embeddings.append((r["label"], emb))
        print(f"  {r['label']:<22}  embedding shape={emb.shape}  norm={np.linalg.norm(emb):.2f}")

    if len(embeddings) >= 2:
        print("\nSimilitud coseno entre archivos (1.0 = idénticos, 0.0 = sin relación):")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i][1], embeddings[j][1])
                print(f"  {embeddings[i][0]}  ↔  {embeddings[j][0]}:  {sim:.4f}")


if __name__ == "__main__":
    main()
