"""
Tester: reproduce archivos WAV de with_sound para verificación manual.

Uso directo:
    from tester import AudioTester
    AudioTester().run()

    # Con opciones:
    AudioTester(one_by_one=True, limit=5).run()
    AudioTester(wav_dir=Path("data/with_sound")).run()
"""

import subprocess
import wave
from pathlib import Path

WITH_SOUND_DIR = Path(__file__).resolve().parent.parent / "data" / "with_sound"


class AudioTester:
    """
    Reproduce archivos WAV de una carpeta para verificación manual.

    Example:
        AudioTester().run()
        AudioTester(one_by_one=True, limit=3).run()
    """

    def __init__(
        self,
        wav_dir: Path = WITH_SOUND_DIR,
        one_by_one: bool = False,
        limit: int = 0,
    ) -> None:
        self.wav_dir = Path(wav_dir)
        self.one_by_one = one_by_one
        self.limit = limit

    def run(self) -> None:
        wav_files = sorted(self.wav_dir.glob("*.wav"))
        if self.limit > 0:
            wav_files = wav_files[: self.limit]

        if not wav_files:
            print(f"No se encontraron archivos .wav en: {self.wav_dir}")
            return

        print(f"Encontrados {len(wav_files)} archivo(s) en: {self.wav_dir}\n")

        for i, wav_path in enumerate(wav_files, 1):
            duration = self._get_duration(wav_path)
            print(f"[{i}/{len(wav_files)}] {wav_path.name}  ({duration:.2f}s)")

            if self.one_by_one:
                try:
                    input("  Presiona Enter para reproducir (Ctrl+C para salir)... ")
                except KeyboardInterrupt:
                    print("\nSaliendo.")
                    break

            self._play(wav_path)

        print("\nFin.")

    def _get_duration(self, path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0

    def _play(self, path: Path) -> None:
        result = subprocess.run(["aplay", str(path)], capture_output=True)
        if result.returncode != 0:
            print(f"  [ERROR] {result.stderr.decode().strip()}")
