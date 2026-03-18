"""
AudioClipper — recorta WAVs a segmentos con actividad real.

Lee un WAV completo en chunks, detecta segmentos activos (RMS > threshold),
aplica padding/merge/filtrado y escribe clips precisos en output_dir.

Uso:
    from src.clip_audio import AudioClipper

    clips = AudioClipper().run()
"""

import logging
import shutil
import wave
from pathlib import Path
from typing import List, Tuple

from src.categorize_audio import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_THRESHOLD,
    DATA_DIR,
    RAW_DATA_DIR,
    iter_chunk_rms,
)

logger = logging.getLogger(__name__)


class AudioClipper:
    """
    Recorta WAVs a clips de segmentos con actividad real.

    Example:
        AudioClipper().run()

        # Opciones personalizadas:
        AudioClipper(
            source_dir=Path("data/raw_data"),
            output_dir=Path("data/with_sound"),
            padding_s=1.5,
            merge_gap_s=3.0,
            min_segment_s=1.0,
        ).run()
    """

    def __init__(
        self,
        source_dir: Path = RAW_DATA_DIR,
        output_dir: Path = DATA_DIR / "with_sound",
        threshold: float = DEFAULT_THRESHOLD,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        padding_s: float = 1.0,
        merge_gap_s: float = 2.0,
        min_segment_s: float = 2.0,
        clean_output: bool = True,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.chunk_frames = chunk_frames
        self.padding_s = padding_s
        self.merge_gap_s = merge_gap_s
        self.min_segment_s = min_segment_s
        self.clean_output = clean_output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[dict]:
        """
        Procesa todos los WAVs en source_dir.
        Devuelve lista de dicts: {source, clips, segments_written}.
        """
        if self.clean_output and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(self.source_dir.glob("*.wav"))
        if not wav_files:
            logger.warning("No .wav files found in: %s", self.source_dir)
            return []

        logger.info("AudioClipper: processing %d file(s)", len(wav_files))
        results = []
        for wav_path in wav_files:
            try:
                result = self.clip_file(wav_path)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to process %s: %s", wav_path, exc)

        total_clips = sum(r["segments_written"] for r in results)
        logger.info("Done — %d clip(s) written to %s", total_clips, self.output_dir)
        return results

    def clip_file(self, wav_path: Path) -> dict:
        """
        Procesa un único archivo WAV y escribe sus clips.
        Devuelve {source, clips, segments_written}.
        """
        wav_path = Path(wav_path)
        rms_list = self._scan_rms(wav_path)

        with wave.open(str(wav_path), "rb") as wf:
            total_frames = wf.getnframes()
            framerate = wf.getframerate()

        segments = self._find_segments(rms_list, total_frames, framerate)

        if not segments:
            logger.warning("No active segments found in: %s", wav_path.name)
            return {"source": wav_path, "clips": [], "segments_written": 0}

        written_clips = []
        with wave.open(str(wav_path), "rb") as src:
            for idx, (start_frame, n_frames) in enumerate(segments, 1):
                clip_name = f"{wav_path.stem}_seg{idx:03d}.wav"
                dst_path = self.output_dir / clip_name
                self._write_clip(src, start_frame, n_frames, dst_path)
                duration_s = n_frames / framerate
                written_clips.append({"path": dst_path, "duration_s": round(duration_s, 2)})
                logger.info(
                    "  %s: frame %d + %d frames (%.2fs)",
                    clip_name, start_frame, n_frames, duration_s,
                )

        logger.info(
            "%s -> %d clip(s)", wav_path.name, len(written_clips)
        )
        return {
            "source": wav_path,
            "clips": written_clips,
            "segments_written": len(written_clips),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan_rms(self, path: Path) -> List[float]:
        """Devuelve lista completa de RMS por chunk (sin early-exit)."""
        return list(iter_chunk_rms(path, self.chunk_frames))

    def _find_segments(
        self, rms_list: List[float], total_frames: int, framerate: int
    ) -> List[Tuple[int, int]]:
        """
        A partir de la lista de RMS, devuelve lista de (start_frame, n_frames)
        para cada segmento activo tras padding, merge y filtrado.
        """
        total_chunks = len(rms_list)
        if total_chunks == 0:
            return []

        pad_chunks = max(1, round(self.padding_s * framerate / self.chunk_frames))
        merge_gap_chunks = max(0, round(self.merge_gap_s * framerate / self.chunk_frames))
        min_seg_chunks = max(1, round(self.min_segment_s * framerate / self.chunk_frames))

        # 1. Índices activos
        active = [i for i, rms in enumerate(rms_list) if rms > self.threshold]
        if not active:
            return []

        # 2. Agrupar en runs consecutivos
        runs: List[Tuple[int, int]] = []
        run_start = active[0]
        run_end = active[0]
        for idx in active[1:]:
            if idx == run_end + 1:
                run_end = idx
            else:
                runs.append((run_start, run_end))
                run_start = idx
                run_end = idx
        runs.append((run_start, run_end))

        # 3. Padding + clampear
        padded: List[Tuple[int, int]] = []
        for s, e in runs:
            ps = max(0, s - pad_chunks)
            pe = min(total_chunks - 1, e + pad_chunks)
            padded.append((ps, pe))

        # 4. Merge si solapan o gap <= merge_gap_chunks
        merged: List[Tuple[int, int]] = [padded[0]]
        for s, e in padded[1:]:
            ms, me = merged[-1]
            if s <= me + merge_gap_chunks + 1:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))

        # 5. Filtrar segmentos cortos
        filtered = [(s, e) for s, e in merged if (e - s + 1) >= min_seg_chunks]

        # 6. Convertir a frames, clampear al total real
        segments: List[Tuple[int, int]] = []
        for s, e in filtered:
            start_frame = s * self.chunk_frames
            end_frame = min((e + 1) * self.chunk_frames, total_frames)
            n_frames = end_frame - start_frame
            if n_frames > 0:
                segments.append((start_frame, n_frames))

        return segments

    def _write_clip(
        self, src: wave.Wave_read, start_frame: int, n_frames: int, dst: Path
    ) -> None:
        """Lee frames del WAV fuente y escribe un clip con los mismos parámetros."""
        src.setpos(start_frame)
        raw = src.readframes(n_frames)
        with wave.open(str(dst), "wb") as out:
            out.setnchannels(src.getnchannels())
            out.setsampwidth(src.getsampwidth())
            out.setframerate(src.getframerate())
            out.writeframes(raw)
