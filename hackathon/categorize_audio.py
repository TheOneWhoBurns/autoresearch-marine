"""
Audio silence categorization filter.

Reads WAV files in streaming chunks, classifies them as containing audio or being silent,
and copies/moves them to labeled subfolders (with_sound/ or silent/).

Designed to be used as a filter within an orchestrator pipeline:

    from categorize_audio import SilenceFilter

    f = SilenceFilter()
    results = f.run()
"""

import logging
import shutil
import wave
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 50.0       # int16 units; ~-56 dBFS
DEFAULT_CHUNK_FRAMES = 144000  # 1 second at 144kHz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"


# ---------------------------------------------------------------------------
# Low-level helpers (module-level, reusable by other filters)
# ---------------------------------------------------------------------------

def get_wav_info(path: Path) -> dict:
    """Read WAV header metadata."""
    with wave.open(str(path), "rb") as wf:
        return {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "framerate": wf.getframerate(),
            "n_frames": wf.getnframes(),
            "duration_s": wf.getnframes() / wf.getframerate(),
        }


def iter_chunk_rms(
    path: Path, chunk_frames: int = DEFAULT_CHUNK_FRAMES
) -> Generator[float, None, None]:
    """Generator yielding RMS per chunk for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        if sample_width not in (2, 3):
            raise ValueError(
                f"Only 16-bit or 24-bit PCM WAV is supported, got {sample_width * 8}-bit: {path}"
            )
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            if sample_width == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            else:
                # 24-bit little-endian: [b0, b1, b2] → int32 con extensión de signo
                raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                sign_byte = np.where(raw_bytes[:, 2] >= 128, np.uint8(0xFF), np.uint8(0x00)).reshape(-1, 1)
                int32_bytes = np.concatenate([raw_bytes, sign_byte], axis=1)
                samples = int32_bytes.view(np.int32).astype(np.float32)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            yield float(np.sqrt(np.mean(samples ** 2)))


def detect_sound(
    path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> Tuple[bool, float]:
    """
    Early-exit detection: returns (is_sound, max_rms).
    Stops as soon as a chunk exceeds the threshold.
    """
    max_rms = 0.0
    for chunk_idx, rms in enumerate(iter_chunk_rms(path, chunk_frames)):
        logger.debug("Chunk %d: RMS=%.2f (threshold=%.2f)", chunk_idx, rms, threshold)
        if rms > max_rms:
            max_rms = rms
        if rms > threshold:
            logger.debug("Sound detected at chunk %d (RMS=%.2f)", chunk_idx, rms)
            return True, max_rms
    return False, max_rms


# ---------------------------------------------------------------------------
# Filter class
# ---------------------------------------------------------------------------

class SilenceFilter:
    """
    Filter that classifies WAV files from a source folder into
    with_sound/ and silent/ subfolders inside the output directory.

    Example:
        filter = SilenceFilter()
        results = filter.run()

        # Custom paths:
        filter = SilenceFilter(
            source_dir=Path("data/raw_data"),
            output_dir=Path("data"),
            threshold=80.0,
            move=True,
        )
        results = filter.run()
    """

    def __init__(
        self,
        source_dir: Path = RAW_DATA_DIR,
        output_dir: Path = DATA_DIR,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        move: bool = False,
        recursive: bool = False,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.chunk_frames = chunk_frames
        self.move = move
        self.recursive = recursive

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[dict]:
        """
        Process all WAV files in source_dir.
        Returns a list of result dicts: {file, label, max_rms, dest}.
        """
        wav_files = self._collect_files()
        if not wav_files:
            logger.warning("No .wav files found in: %s", self.source_dir)
            return []

        logger.info("SilenceFilter: processing %d file(s)", len(wav_files))
        results = []
        for wav_path in wav_files:
            try:
                result = self._process_file(wav_path)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to process %s: %s", wav_path, exc)

        self._log_summary(results)
        return results

    def classify(self, wav_path: Path) -> dict:
        """Classify a single WAV file without copying/moving it."""
        is_sound, max_rms = detect_sound(wav_path, self.threshold, self.chunk_frames)
        return {
            "file": wav_path,
            "label": "with_sound" if is_sound else "silent",
            "max_rms": max_rms,
            "dest": None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_files(self) -> List[Path]:
        pattern = "**/*.wav" if self.recursive else "*.wav"
        all_files = list(self.source_dir.glob(pattern))
        skip_dirs = {self.source_dir / "with_sound", self.source_dir / "silent"}
        return [f for f in all_files if not any(f.is_relative_to(d) for d in skip_dirs)]

    def _process_file(self, wav_path: Path) -> dict:
        is_sound, max_rms = detect_sound(wav_path, self.threshold, self.chunk_frames)
        label = "with_sound" if is_sound else "silent"
        logger.info("%s -> %s (max_rms=%.2f)", wav_path.name, label, max_rms)

        dest_dir = self.output_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / wav_path.name

        if dest_path.exists():
            logger.warning("Already exists, skipping: %s", dest_path)
        elif self.move:
            shutil.move(str(wav_path), str(dest_path))
        else:
            shutil.copy2(str(wav_path), str(dest_path))

        return {"file": wav_path, "label": label, "max_rms": max_rms, "dest": dest_path}

    def _log_summary(self, results: List[dict]) -> None:
        with_sound = sum(1 for r in results if r["label"] == "with_sound")
        silent = len(results) - with_sound
        logger.info("Done — with_sound: %d | silent: %d", with_sound, silent)