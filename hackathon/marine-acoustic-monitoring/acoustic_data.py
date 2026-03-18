"""
acoustic_data.py — Load, explore, and visualize marine acoustic monitoring data.

Single-file helper for hackathon participants. Handles:
- SoundTrap file naming conventions (unit_id.YYMMDDHHMMSS.wav, YYMMDD_seq.wav)
- XML metadata parsing (deployment info, timestamps, sample rates)
- Audio loading with chunked reading for large files
- Spectrogram computation and visualization helpers

Deps: numpy, soundfile, matplotlib, scipy (all in standard scientific Python stack)
"""

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

# Optional imports — fail gracefully so the module loads even without all deps
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from scipy.signal import butter, sosfilt, spectrogram as sp_spectrogram
except ImportError:
    butter = sosfilt = sp_spectrogram = None


# ============================================================================
# Dataset inventory
# ============================================================================

UNITS = {
    "5783": {
        "sample_rate": 144_000,
        "pattern": r"^5783\.(\d{12})\.wav$",
        "description": "SoundTrap unit 5783, 144 kHz, 20-min recordings",
    },
    "6478": {
        "sample_rate": 96_000,
        "pattern": r"^6478\.(\d{12})\.wav$",
        "description": "SoundTrap unit 6478, 96 kHz, 10-min recordings",
    },
    "pilot": {
        "sample_rate": 48_000,
        "pattern": r"^(\d{6})_(\d+)\.wav$",
        "description": "Pilot deployment, 48 kHz, ~5-min recordings",
    },
}

# Map directory names to unit keys
_DIR_TO_UNIT = {
    "5783": "5783",
    "6478": "6478",
    "Music_Soundtrap_Pilot": "pilot",
}


# ============================================================================
# Timestamp parsing
# ============================================================================

def parse_soundtrap_timestamp(filename, unit=None):
    """Extract UTC datetime from a SoundTrap WAV filename.

    Args:
        filename: Just the filename (not full path), e.g. '6478.230723151251.wav'
        unit: Optional unit key ('5783', '6478', 'pilot'). Auto-detected if None.

    Returns:
        datetime object, or None if parsing fails.
    """
    name = Path(filename).name

    # === Unit 5783 / 6478: {unit_id}.{YYMMDDHHMMSS}.{ext} ===
    m = re.match(r"^\d{4}\.(\d{12})\.wav$", name)
    if m:
        ts_str = m.group(1)
        try:
            return datetime.strptime(ts_str, "%y%m%d%H%M%S")
        except ValueError:
            return None

    # === Pilot: {YYMMDD}_{sequence}.wav ===
    m = re.match(r"^(\d{6})_(\d+)\.wav$", name)
    if m:
        date_str = m.group(1)
        try:
            return datetime.strptime(date_str, "%y%m%d")
        except ValueError:
            return None

    return None


# ============================================================================
# Dataset discovery
# ============================================================================

def find_data_dir(base_path=None):
    """Locate the acoustic monitoring data directory.

    Searches common locations if base_path is not provided.
    Returns the path containing 5783/, 6478/, Music_Soundtrap_Pilot/.
    """
    if base_path and Path(base_path).exists():
        return Path(base_path)

    # Common locations to check
    candidates = [
        Path("./data"),
        Path("./acoustic_data"),
        Path("./raw_data/acoustic_monitoring"),
        Path("../raw_data/acoustic_monitoring"),
    ]
    for c in candidates:
        if c.exists() and (c / "6478").exists():
            return c

    raise FileNotFoundError(
        "Could not find acoustic data directory. "
        "Pass base_path= pointing to the folder containing 5783/, 6478/, Music_Soundtrap_Pilot/"
    )


def list_recordings(data_dir, unit=None):
    """List all WAV recordings with parsed metadata.

    Args:
        data_dir: Path to the data root (containing 5783/, 6478/, Music_Soundtrap_Pilot/)
        unit: Optional unit key to filter ('5783', '6478', 'pilot'). None = all units.

    Returns:
        List of dicts with keys: path, filename, unit, timestamp, sample_rate
    """
    data_dir = Path(data_dir)
    recordings = []

    dirs = _DIR_TO_UNIT.items()
    if unit:
        dirs = [(k, v) for k, v in dirs if v == unit]

    for dirname, unit_key in dirs:
        unit_dir = data_dir / dirname
        if not unit_dir.exists():
            continue

        unit_info = UNITS[unit_key]
        # Filter out macOS resource fork files (._*)
        wavs = sorted(p for p in unit_dir.glob("*.wav") if not p.name.startswith("._"))

        for wav_path in wavs:
            ts = parse_soundtrap_timestamp(wav_path.name)
            recordings.append({
                "path": wav_path,
                "filename": wav_path.name,
                "unit": unit_key,
                "timestamp": ts,
                "sample_rate": unit_info["sample_rate"],
            })

    return recordings


def inventory(data_dir):
    """Print a summary of available recordings per unit."""
    data_dir = Path(data_dir)
    recs = list_recordings(data_dir)

    print(f"{'Unit':<8} {'WAVs':>6} {'Sample Rate':>12} {'Time Range'}")
    print("-" * 60)

    for unit_key in ["5783", "6478", "pilot"]:
        unit_recs = [r for r in recs if r["unit"] == unit_key]
        if not unit_recs:
            continue
        sr = unit_recs[0]["sample_rate"]
        timestamps = [r["timestamp"] for r in unit_recs if r["timestamp"]]
        if timestamps:
            t_min = min(timestamps).strftime("%Y-%m-%d")
            t_max = max(timestamps).strftime("%Y-%m-%d")
            time_range = f"{t_min} → {t_max}"
        else:
            time_range = "unknown"
        print(f"{unit_key:<8} {len(unit_recs):>6} {sr:>10} Hz  {time_range}")

    print(f"\nTotal: {len(recs)} WAV files")
    return recs


# ============================================================================
# XML metadata
# ============================================================================

def parse_xml_metadata(xml_path):
    """Parse a SoundTrap .log.xml file for deployment metadata.

    Returns dict with available fields: start_time, stop_time, sample_rate,
    temperature_c, battery_v, gain_db, hardware_id.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta = {}

    for pe in root.findall(".//PROC_EVENT"):
        for child in pe:
            attrs = child.attrib
            if "SamplingStartTimeUTC" in attrs:
                try:
                    meta["start_time"] = datetime.strptime(
                        attrs["SamplingStartTimeUTC"], "%Y-%m-%dT%H:%M:%S"
                    )
                except ValueError:
                    pass
            if "SamplingStopTimeUTC" in attrs:
                try:
                    meta["stop_time"] = datetime.strptime(
                        attrs["SamplingStopTimeUTC"], "%Y-%m-%dT%H:%M:%S"
                    )
                except ValueError:
                    pass
            if "SampleRate" in attrs:
                meta["sample_rate"] = int(attrs["SampleRate"])
            if "Temperature" in attrs:
                meta["temperature_c"] = float(attrs["Temperature"])
            if "BatteryState" in attrs:
                meta["battery_v"] = float(attrs["BatteryState"])
            if "Gain" in attrs:
                meta["gain_db"] = float(attrs["Gain"])

    # Hardware ID from top-level
    hw = root.find(".//HARDWARE")
    if hw is not None and "SerialNumber" in hw.attrib:
        meta["hardware_id"] = hw.attrib["SerialNumber"]

    return meta


# ============================================================================
# Audio loading
# ============================================================================

def load_audio(path, duration_s=None, offset_s=0.0, target_sr=None):
    """Load a WAV file (or a segment of it) as a float32 numpy array.

    Args:
        path: Path to WAV file.
        duration_s: Max seconds to load. None = full file.
        offset_s: Start offset in seconds.
        target_sr: If provided, resample to this rate (requires librosa).

    Returns:
        (audio, sample_rate) — audio is float32 in [-1, 1].
    """
    path = str(path)

    if sf is not None:
        info = sf.info(path)
        sr = info.samplerate
        start_frame = int(offset_s * sr)
        n_frames = int(duration_s * sr) if duration_s is not None else -1
        audio, sr = sf.read(path, start=start_frame, frames=n_frames, dtype="float32")
    else:
        # Fallback: stdlib wave module (no offset support, loads full file)
        import wave
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_total = wf.getnframes()
            start_frame = int(offset_s * sr)
            if duration_s is not None:
                n_frames = min(int(duration_s * sr), n_total - start_frame)
            else:
                n_frames = n_total - start_frame
            wf.setpos(start_frame)
            raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

    if target_sr and target_sr != sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            raise ImportError(
                f"librosa is required for resampling ({sr} → {target_sr} Hz): "
                "pip install librosa"
            )

    return audio, sr


def highpass_filter(audio, sr, cutoff_hz=50, order=4):
    """Apply a highpass filter to remove DC offset and low-frequency self-noise.

    Args:
        audio: 1D float32 array.
        sr: Sample rate.
        cutoff_hz: Cutoff frequency (default 50 Hz).
        order: Filter order.

    Returns:
        Filtered audio array.
    """
    if butter is None:
        raise ImportError("scipy is required: pip install scipy")
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ============================================================================
# Spectrogram computation
# ============================================================================

def compute_spectrogram(audio, sr, n_fft=2048, hop_length=512, f_min=0, f_max=None):
    """Compute a power spectrogram in dB.

    Args:
        audio: 1D float32 array.
        sr: Sample rate.
        n_fft: FFT window size.
        hop_length: Hop size between frames.
        f_min: Minimum frequency to include (Hz).
        f_max: Maximum frequency to include (Hz). None = Nyquist.

    Returns:
        (S_db, freqs, times) — S_db is power in dB (ref max), freqs and times are axes.
    """
    if sp_spectrogram is None:
        raise ImportError("scipy is required: pip install scipy")

    freqs, times, Sxx = sp_spectrogram(
        audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
        scaling="spectrum",
    )

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # Frequency band selection
    f_max = f_max or sr / 2
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]

    return Sxx_db, freqs, times


def plot_spectrogram(audio, sr, title=None, duration_s=None, f_max=None,
                     n_fft=2048, hop_length=512, figsize=(14, 4),
                     cmap="magma", vmin=None, vmax=None, ax=None):
    """Plot a spectrogram from raw audio.

    Args:
        audio: 1D float32 array.
        sr: Sample rate.
        title: Plot title.
        duration_s: Trim audio to this many seconds before plotting.
        f_max: Max frequency to display (Hz). None = Nyquist.
        n_fft, hop_length: FFT parameters.
        figsize: Figure size (only used if ax is None).
        cmap: Colormap.
        vmin, vmax: dB range clipping.
        ax: Matplotlib axes to plot into. Creates new figure if None.

    Returns:
        (fig, ax) if ax was None, else ax.
    """
    if plt is None:
        raise ImportError("matplotlib is required: pip install matplotlib")

    if duration_s is not None:
        audio = audio[: int(duration_s * sr)]

    S_db, freqs, times = compute_spectrogram(
        audio, sr, n_fft=n_fft, hop_length=hop_length, f_max=f_max
    )

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.pcolormesh(times, freqs / 1000, S_db,
                       shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel("Frequency (kHz)")
    ax.set_xlabel("Time (s)")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power (dB)")

    if created_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_spectrogram_bands(audio, sr, title_prefix="", figsize=(14, 10)):
    """Plot spectrograms at three ecologically relevant frequency bands.

    Bands:
    - LOW (50–2000 Hz): Fish vocalizations, boat noise
    - MID (2–20 kHz): Snapping shrimp, dolphin whistles
    - HIGH (20+ kHz): Echolocation clicks, high-frequency transients

    Args:
        audio: 1D float32 array.
        sr: Sample rate.
        title_prefix: Prefix for subplot titles.
        figsize: Figure size.

    Returns:
        (fig, axes)
    """
    if plt is None:
        raise ImportError("matplotlib is required: pip install matplotlib")

    nyquist = sr / 2
    bands = [
        ("LOW (50–2000 Hz): fish, boats", 50, 2000, 1024),
        ("MID (2–20 kHz): shrimp, dolphin whistles", 2000, 20000, 2048),
    ]
    if nyquist > 24000:
        bands.append(
            (f"HIGH (20–{nyquist/1000:.0f} kHz): echolocation", 20000, nyquist, 4096)
        )

    fig, axes = plt.subplots(len(bands), 1, figsize=figsize)
    if len(bands) == 1:
        axes = [axes]

    for ax, (label, f_min, f_max, nfft) in zip(axes, bands):
        f_max = min(f_max, nyquist)
        plot_spectrogram(
            audio, sr, title=f"{title_prefix}{label}",
            f_max=f_max, n_fft=nfft, ax=ax,
        )
        # Override y-axis to show only this band
        ax.set_ylim(f_min / 1000, f_max / 1000)

    fig.tight_layout()
    return fig, axes


# ============================================================================
# Quick-listen helper
# ============================================================================

def listen(path, duration_s=10, offset_s=0):
    """Create an IPython audio widget for in-notebook listening.

    Automatically resamples to 22050 Hz for browser playback.
    Only works inside Jupyter/Colab.
    """
    try:
        from IPython.display import Audio, display
    except ImportError:
        print("IPython not available — use an external audio player.")
        return

    audio, sr = load_audio(path, duration_s=duration_s, offset_s=offset_s)

    # Resample to 22050 Hz for browser playback
    if sr > 22050:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            sr = 22050
        except ImportError:
            # Crude decimation fallback
            factor = sr // 22050
            audio = audio[::factor]
            sr = sr // factor

    display(Audio(audio, rate=sr))
