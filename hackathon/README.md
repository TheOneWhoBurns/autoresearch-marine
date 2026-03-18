# Marine Acoustic Monitoring — Pipeline de Procesamiento

Pipeline local para filtrar, recortar y verificar archivos WAV de hidrofonos del Golfo de San Cristóbal, Galápagos.

## Contexto

Los hidrofonos SoundTrap graban audio submarino continuo en archivos WAV de 5–20 minutos. La mayoría del tiempo es silencio o ruido de fondo. Este pipeline extrae automáticamente sólo los segmentos con actividad acústica real.

## Estructura del proyecto

```
hackathon-proposal/
├── data/
│   ├── raw_data/          # WAVs originales de los hidrofonos
│   └── with_sound/        # Clips recortados con actividad (generado)
├── src/
│   ├── categorize_audio.py   # Detección de silencio por RMS (helpers de bajo nivel)
│   ├── clip_audio.py         # AudioClipper — recorta WAVs a segmentos activos
│   ├── audio_tester.py       # AudioTester — reproducción manual de clips
│   ├── tester.py             # Script principal: clipper + tester
│   └── marine-acoustic-monitoring/
│       ├── acoustic_data.py      # Helper de carga, filtrado y espectrogramas
│       ├── acoustic_explorer.ipynb
│       └── README.md             # Guía del hackathon (dataset, ideas de proyecto)
├── r2_download.py         # Descarga del dataset desde Cloudflare R2
├── manifest.json          # Manifesto de archivos disponibles
└── data_download.ipynb    # Notebook de descarga interactiva
```

## Dataset

Tres unidades de hidrofono en la Bahía de San Cristóbal:

| Unidad | Sample Rate | Archivos WAV | Duración/archivo | Total audio |
|--------|------------|-------------|-----------------|-------------|
| 5783   | 144 kHz    | 16          | 20 min          | ~5.3 h      |
| 6478   | 96 kHz     | 189         | 10 min          | ~31.5 h     |
| Pilot  | 48 kHz     | 721         | ~5 min          | ~60 h       |

Formato de archivos:
- **5783/6478:** `{unit_id}.{AAMMDDHHMMSS}.wav` — e.g. `6478.230723151251.wav`
- **Pilot:** `{AAMMDD}_{secuencia}.wav` — e.g. `190806_3754.wav`

Formatos soportados: **16-bit PCM** y **24-bit PCM**.

## Pipeline principal

### Correr el pipeline completo

```bash
python -m src.tester
```

Esto:
1. Limpia `data/with_sound/`
2. Escanea `data/raw_data/` en busca de WAVs
3. Recorta cada archivo a sus segmentos activos → `*_seg001.wav`, `*_seg002.wav`, etc.
4. Abre reproducción interactiva de los clips (Enter para reproducir, Ctrl+C para salir)

### Usar los módulos individualmente

```python
from src.clip_audio import AudioClipper

# Con parámetros por defecto
results = AudioClipper().run()

# Personalizado
results = AudioClipper(
    threshold=80.0,      # RMS mínimo para considerar "activo" (int16, 0–32767)
    padding_s=1.5,       # Segundos de padding alrededor de cada segmento
    merge_gap_s=3.0,     # Silencio máximo para fusionar dos segmentos
    min_segment_s=1.0,   # Duración mínima de un clip válido
    clean_output=False,  # No borrar clips previos
).run()

# Resultado
# [{'source': Path('...wav'), 'clips': [{'path': Path('...'), 'duration_s': 12.0}, ...], 'segments_written': 6}]
```

```python
from src.audio_tester import AudioTester

AudioTester(one_by_one=True, limit=5).run()  # reproducir primeros 5 clips
```

## Cómo funciona el recorte (`AudioClipper`)

El archivo se escanea en **chunks de 1 segundo**. Para cada chunk se calcula el RMS:

```
Chunks:   [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]
RMS:       8   12    9  210  185  170   11    9   180  160
                        ^-- activos --^              ^-- activos --^
```

Pasos del algoritmo:

1. **Activos** — chunks donde `RMS > threshold`
2. **Runs** — agrupa índices consecutivos en rangos `(inicio, fin)`
3. **Padding** — expande cada run ±`padding_s` segundos en ambos extremos
4. **Merge** — fusiona runs cuyo gap (tras padding) es ≤ `merge_gap_s`
5. **Filtrado** — descarta segmentos más cortos que `min_segment_s`
6. **Escritura** — lee frames directamente del WAV fuente con `wave`, sin cargar todo en memoria

### Parámetros de corte

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `threshold` | `50.0` | RMS mínimo (~-56 dBFS) para considerar un chunk activo |
| `padding_s` | `1.0` | Margen de silencio a incluir antes y después de cada zona activa |
| `merge_gap_s` | `2.0` | Silencio ≤ 2s entre segmentos → fusionar en uno solo |
| `min_segment_s` | `2.0` | Clips más cortos que esto se descartan |
| `clean_output` | `True` | Borra `with_sound/` al inicio de cada `run()` |

## Módulo de exploración (notebooks)

`src/marine-acoustic-monitoring/acoustic_data.py` provee helpers para notebooks:

```python
import acoustic_data as hd

recs = hd.inventory("./data")              # resumen del dataset
audio, sr = hd.load_audio(recs[0]["path"], duration_s=30.0)
audio = hd.highpass_filter(audio, sr, cutoff_hz=50)
fig, ax = hd.plot_spectrogram(audio, sr)
fig, axes = hd.plot_spectrogram_bands(audio, sr)
hd.listen(recs[0]["path"], duration_s=10)  # sólo en Jupyter/Colab
```

## Dependencias

```
numpy
scipy (para filtros y espectrogramas)
matplotlib (para visualización)
soundfile (carga de audio, opcional — fallback a stdlib wave)
librosa (resampleo, opcional)
```

Reproducción en terminal: requiere `aplay` (Linux/ALSA).

## Ideas de proyecto (hackathon)

Ver [`src/marine-acoustic-monitoring/README.md`](src/marine-acoustic-monitoring/README.md) para:
- **Tier 1:** Índices acústicos del paisaje sonoro (sin etiquetas)
- **Tier 2:** Modelos preentrenados (Perch, BirdNET, humpback detector de Google/NOAA)
- **Tier 3:** Clasificador con active learning
- **Bonus:** WhAM — modelo generativo de audio de ballenas (Project CETI)
