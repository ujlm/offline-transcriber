# Backend (Python sidecar)

The Tauri Rust process spawns this directory's `main.py` per transcription job. Communication is one-way: CLI args in, JSON Lines on stdout.

## Run standalone (useful for debugging)

```bash
source ../.venv/bin/activate
python main.py \
  --input /path/to/audio.wav \
  --output-dir /tmp \
  --models-dir ../models \
  --language en
```

You should see JSON Lines events stream to stdout, ending with a `done` event and the produced file paths.

## Files

| File | Responsibility |
|---|---|
| `main.py` | CLI entry. Parses args, sets up signal handlers, runs the pipeline. |
| `events.py` | JSON-line event emitter. The single source of truth for the protocol. |
| `audio_io.py` | Decode any input file to 16 kHz mono float32. |
| `vad.py` | Silero VAD chunking. |
| `diarize.py` | Pyannote 3.1 speaker diarization (offline). |
| `asr.py` | Cohere Transcribe via `transformers`. English only in Stage 1. |
| `fusion.py` | Map VAD chunks to speaker labels via max-overlap. |
| `export.py` | Write `.txt` and `.docx`. |
| `pipeline.py` | Orchestrates the six stages and emits progress events. |
