# Software Requirements Specification (SRS)

**Project:** Offline Audio Transcription & Diarization App
**Core Stack:** Tauri 2, Vanilla TypeScript, Python sidecar, Cohere Transcribe, Pyannote 3.1, Silero VAD
**Document version:** 0.2 (replaces v0.1 PDF)
**Last updated:** 2026-04-29
**Current stage:** **Stage 1 — Minimum Viable Pipeline**

---

## 1. System Overview

A privacy-first, fully offline desktop application (Windows/macOS) that lets a user select a local audio file via a graphical interface, transcribes and diarizes it entirely on-device using state-of-the-art open-source models, and writes the result to a local file.

### Privacy & Compliance

- All processing runs on the host machine. No audio, transcripts or telemetry leave the device.
- No external API is contacted at runtime; offline guards (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`) are set in the sidecar process.
- All model weights are downloaded once during install/setup and stored locally under `models/`.
- No account, no login, no key required to use the app.

---

## 2. Delivery Plan (Staged)

The product is built incrementally. Each stage is shippable and validated before the next is started.

### Stage 1 — Minimum Viable Pipeline *(current stage)*

**Goal:** prove end-to-end on a single machine, single language, single output format. No packaging/signing yet.

In scope:

- Tauri 2 desktop shell (Vanilla TS, single screen).
- Python sidecar invoked via the Python interpreter (no PyInstaller yet).
- Pipeline: audio decode → 16 kHz mono → Silero VAD → Cohere Transcribe (`transformers`, English only) → Pyannote 3.1 → segment-level speaker fusion.
- Output: `.txt` and `.docx`.
- Stdout JSON-line event protocol from sidecar to Rust to frontend.
- Cancel button (graceful SIGTERM).
- Runs on the developer's primary OS only.
- Models loaded from local `models/` directory only; offline env vars enforced.

Out of scope (deferred to later stages):

- Cross-platform packaging, code signing, notarization.
- ONNX INT8/INT4 acceleration.
- Forced word-level alignment.
- Languages other than English.
- Subtitle outputs (`.srt`/`.vtt`).
- Speaker label editing UI.
- Multiple files / batch mode.
- Auto-update.

### Stage 2 — Quality & UX polish

- Language selector (14 supported by Cohere Transcribe).
- Pyannote `min_speakers` / `max_speakers` controls.
- ETA in progress UI; richer status messages.
- Editable speaker labels (find/replace) before export.
- `.srt`, `.vtt`, and `.json` outputs.
- Better error model + user-facing error messages.

### Stage 3 — ONNX acceleration & smaller bundle

- Swap `transformers` Cohere model for `onnxruntime` INT8/INT4 community port.
- Remove redundant ML runtimes (drop torch where possible).
- Benchmark accuracy and throughput on a small held-out set.
- Optional: word-level timestamps via `ctc-forced-aligner`.

### Stage 4 — Cross-platform packaging

- PyInstaller-bundled sidecar with proper Tauri target-triple suffixes:
  - `backend-sidecar-aarch64-apple-darwin`
  - `backend-sidecar-x86_64-apple-darwin`
  - `backend-sidecar-x86_64-pc-windows-msvc`
- GitHub Actions matrix (macOS + Windows runners).
- macOS code signing + notarization; Windows code signing.
- First-run model download UX (alternative to bundling 3+ GB of weights).
- Auto-update channel.

### Stage 5 — Optional upgrades

- Live (microphone) transcription mode.
- GPU acceleration where available (CUDA / Metal / DirectML).
- Built-in audio preview and waveform UI.
- Project files (re-open and re-edit prior transcripts).

---

## 3. Stage 1 Architecture

```
┌────────────────────────────────────────────┐
│ Tauri Window (Vanilla TS)                  │
│  • file picker • status • cancel • open    │
└──────────────────┬─────────────────────────┘
                   │ invoke("process_audio", {input, output})
                   │ listen("transcription-event")
┌──────────────────▼─────────────────────────┐
│ Rust (Tauri command + sidecar manager)     │
│  spawns python interpreter, pipes stdout,  │
│  emits Tauri events per JSON line          │
└──────────────────┬─────────────────────────┘
                   │ stdin args + stdout JSON lines
┌──────────────────▼─────────────────────────┐
│ Python sidecar (pipeline.py)               │
│  1. audio_io     — decode → 16 kHz mono    │
│  2. vad          — Silero VAD chunks       │
│  3. diarize      — Pyannote 3.1 segments   │
│  4. asr          — Cohere Transcribe (en)  │
│  5. fusion       — VAD ↔ speaker mapping   │
│  6. export       — .txt and .docx          │
└────────────────────────────────────────────┘
```

### 3.1 Stdout event protocol (JSON Lines)

Every line from the sidecar's stdout is a single JSON object. The Rust process forwards each line to the frontend as a `transcription-event` Tauri event.

```jsonc
{ "event": "progress", "stage": "decode",     "percent": 5  }
{ "event": "progress", "stage": "vad",        "percent": 15 }
{ "event": "progress", "stage": "diarize",    "percent": 40 }
{ "event": "progress", "stage": "transcribe", "percent": 80 }
{ "event": "progress", "stage": "export",     "percent": 95 }
{ "event": "log",      "level": "info", "message": "Loaded Cohere Transcribe (cpu)" }
{ "event": "done",     "txt_path": "...", "docx_path": "..." }
{ "event": "error",    "code": "DECODE_FAILED", "message": "..." }
```

### 3.2 Cancellation

The Rust side kills the sidecar with SIGTERM (Windows: `TerminateProcess`). The Python sidecar installs a SIGTERM handler that emits `{"event":"cancelled"}` and exits with code 130.

### 3.3 Compatibility shims (technical debt to retire)

`backend/main.py` installs two startup shims to keep pyannote.audio 3.x usable on the modern stack. Remove them once pyannote ships a fix (or we move to pyannote 4.x):

- **`use_auth_token` → `token`** for `huggingface_hub.hf_hub_download` (and friends). pyannote 3.x still passes the deprecated kwarg; `huggingface_hub` 1.0 removed it.
- **`torch.load(weights_only=False)`** override. PyTorch 2.6 flipped the default, and pyannote checkpoints contain metadata pickles (`TorchVersion`, omegaconf containers) outside torch's default safe-globals allowlist. Safe in our context because we only load locally-bundled, gate-accepted weights.

Additionally, `backend/diarize.py` rewrites the relative paths in `models/pyannote-3.1/config.yaml` to absolute paths in a temp file before calling `Pipeline.from_pretrained`, because pyannote uses `os.path.isfile()` against the literal config value and would otherwise interpret bare names as HF repo IDs.

---

## 4. Required Packages & Models

### 4.1 Frontend (Tauri 2 + Vite + Vanilla TS)

| Dependency | Purpose |
|---|---|
| `@tauri-apps/api` | Frontend ↔ Rust IPC and event bus |
| `@tauri-apps/plugin-dialog` | Native file picker |
| `@tauri-apps/plugin-shell` | (Rust side) spawn the Python sidecar process |
| `vite`, `typescript` | Build tooling |

### 4.2 Rust (Tauri 2)

| Dependency | Purpose |
|---|---|
| `tauri` 2.x | Desktop shell |
| `tauri-plugin-dialog` 2.x | Open-file dialog |
| `tauri-plugin-shell` 2.x | Reserved for future sidecar ergonomics |
| `tauri-plugin-fs` 2.x | Open output folder |
| `tauri-plugin-opener` 2.x | Reveal file in OS file manager |
| `serde`, `serde_json` | JSON event parsing |

### 4.3 Python sidecar (Stage 1)

```text
# `lightning` is currently quarantined on PyPI; install from Lightning AI's
# GitHub release wheel until the quarantine is lifted.
lightning @ https://github.com/Lightning-AI/pytorch-lightning/releases/download/2.6.1/lightning-2.6.1-py3-none-any.whl

transformers>=5.4.0
torch>=2.4,<2.9          # <2.9 because torchaudio 2.9 dropped AudioMetaData
torchaudio>=2.4,<2.9     # which pyannote.audio 3.x still imports
huggingface_hub>=0.25
soundfile
librosa
sentencepiece
protobuf
silero-vad>=5
pyannote.audio>=3.3,<4   # 3.x library can load the speaker-diarization-3.1 model
python-docx
numpy
pyyaml
```

> **About the version numbers:** "3.1" in `pyannote/speaker-diarization-3.1` refers to the Hugging Face *model* version, not the *library* version. The library is on 3.3.x/3.4.x and is the version we depend on. We deliberately stay under 4.x to avoid pulling in `pyannoteai-sdk` (cloud SDK) and `torchcodec` (mandatory ffmpeg).

### 4.4 Models (downloaded once into `models/`)

| Model | HF repo | License | Approx. size | Where it goes |
|---|---|---|---:|---|
| Cohere Transcribe (FP32) | `CohereLabs/cohere-transcribe-03-2026` | Apache 2.0 | ~4 GB | `models/cohere-transcribe-03-2026/` |
| Pyannote segmentation 3.0 | `pyannote/segmentation-3.0` | MIT (gated) | ~6 MB | `models/pyannote-3.1/segmentation-3.0/` |
| Pyannote diarization 3.1 | `pyannote/speaker-diarization-3.1` | MIT (gated) | ~5 MB | (config copied into `models/pyannote-3.1/config.yaml`) |
| Pyannote embeddings | `pyannote/wespeaker-voxceleb-resnet34-LM` | CC-BY 4.0 (gated) | ~26 MB | `models/pyannote-3.1/wespeaker-voxceleb-resnet34-LM/` |
| Silero VAD | bundled in `silero-vad` pip package | MIT | ~2 MB | (no manual step) |

> **Note on Cohere model size:** Stage 1 uses the FP32 weights via `transformers`. Stage 3 will swap to the community ONNX INT8/INT4 port (`cstr/cohere-transcribe-onnx-int8` or `-int4`) which is roughly 2.9 GB / 1.5 GB respectively.

---

## 5. Directory Layout (Stage 1)

```
cohere-transcription/
├── SRS.md                         # this document
├── README.md                      # quickstart and manual setup
├── package.json                   # frontend deps
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/                           # frontend (Vanilla TS)
│   ├── main.ts
│   └── styles.css
├── src-tauri/                     # Tauri Rust app
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── build.rs
│   ├── capabilities/default.json
│   └── src/
│       ├── main.rs
│       └── lib.rs
├── backend/                       # Python sidecar
│   ├── requirements.txt
│   ├── README.md
│   ├── main.py                    # CLI entry (stdin args, stdout JSON events)
│   ├── pipeline.py                # orchestrates the 6 stages
│   ├── audio_io.py                # decode + resample
│   ├── vad.py                     # Silero VAD
│   ├── diarize.py                 # Pyannote 3.1 (offline)
│   ├── asr.py                     # Cohere Transcribe (transformers)
│   ├── fusion.py                  # speaker x VAD-segment fusion
│   ├── export.py                  # .txt and .docx writers
│   └── events.py                  # JSON-line event helpers
├── models/                        # local model weights
│   ├── README.md                  # how to download
│   ├── cohere-transcribe-03-2026/
│   └── pyannote-3.1/
│       ├── config.yaml            # local Pyannote pipeline config
│       ├── segmentation-3.0/
│       └── wespeaker-voxceleb-resnet34-LM/
└── scripts/
    └── download_models.py         # one-shot helper
```

---

## 6. Stage 1 GUI

A single window with:

1. **Select audio file** — opens native file dialog (`.wav`, `.flac`, `.ogg`; `.mp3`/`.m4a` if ffmpeg present).
2. **Selected file** — text label.
3. **Output folder** — defaults to the input file's directory; user may change.
4. **Start transcription** — disabled until a file is selected.
5. **Cancel** — visible while running.
6. **Progress** — stage label + percent bar.
7. **Open output folder** — appears after completion.

No language picker in Stage 1 (English hardcoded).

---

## 7. Stage 1 Acceptance Criteria

The build is considered complete when, on the developer's primary OS:

1. `npm run tauri dev` launches the window.
2. Selecting a 1–10 minute English `.wav` and clicking **Start** produces:
   - A `.txt` file with one paragraph per speaker turn.
   - A `.docx` file with bold speaker labels and turn paragraphs.
3. The progress bar advances through all six stages.
4. **Cancel** stops the process within 2 seconds and removes any partial output.
5. The pipeline runs **with the network turned off** (verifiable: disable Wi-Fi, run again, succeed).
6. Errors (missing model, bad audio, etc.) surface as a dismissable error toast in the UI, not a crash.

---

## 8. Risks & Open Questions

- **CPU runtime for Pyannote 3.1** is slow (~0.1–0.3× realtime per minute of audio on M-series chips). Stage 1 simply documents this; Stage 3 may explore ONNX exports.
- **`transformers` ASR memory pressure**: a 2B FP32 model is ~8 GB in memory. Machines with <16 GB RAM may struggle. Stage 1 will load with `torch_dtype=torch.float32` for correctness; Stage 3 switches to INT8.
- **MP3/M4A support** depends on ffmpeg being installed. Stage 1 documents WAV/FLAC/OGG as guaranteed; MP3 is best-effort.
- **First-run model size** (~4 GB Cohere alone) may be too large to bundle. Stage 4 will ship a first-run downloader instead.

---

## 9. Glossary

- **VAD** — Voice Activity Detection.
- **Diarization** — segmenting audio by *who* spoke *when*.
- **Sidecar** — an auxiliary executable (here: the Python pipeline) that the Tauri Rust process invokes.
- **JSON Lines** — newline-delimited JSON, one object per line.
