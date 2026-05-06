# Offline Transcriber — Stage 1 (MVP)

A privacy-first desktop app that transcribes and diarizes audio entirely on your machine using Cohere Transcribe and Pyannote 3.1.

> See `SRS.md` for the full multi-stage plan. **You are reading the Stage 1 quickstart.**

---

## What Stage 1 does

- Single-screen Tauri window: pick a `.wav` (or `.flac`/`.ogg`), pick an output folder, click **Start**.
- Pipeline (English only): decode → Silero VAD → Pyannote 3.1 diarization → Cohere Transcribe → fusion → `.txt` + `.docx`.
- Live progress reported from the Python sidecar over a JSON Lines stdout protocol, forwarded to the UI as Tauri events.
- Cancel button.
- 100 % offline at runtime (offline env vars enforced; verifiable by running with Wi-Fi off).

---

## Prerequisites

Install these once on your machine.

| Tool | Min version | macOS | Windows |
|---|---|---|---|
| Node.js | 20+ | `brew install node` | https://nodejs.org |
| Rust | 1.78+ (stable) | `curl https://sh.rustup.rs -sSf \| sh` | https://rustup.rs |
| Python | 3.11 (recommended) — **not 3.13+** | `brew install python@3.11` | https://www.python.org |
| Tauri CLI | 2.x | `cargo install tauri-cli --version "^2"` | same |
| (optional) ffmpeg | for `.mp3`/`.m4a` | `brew install ffmpeg` | https://www.gyan.dev/ffmpeg/builds |
| Hugging Face account | — | https://huggingface.co/join (free) | same |

> Python 3.11 is the safest choice today — `torch` and `pyannote.audio` 3.1 occasionally lag on the latest 3.13 wheels.

---

## Manual setup (do these once, in order)

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Create the Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows PowerShell

pip install --upgrade pip
pip install -r backend/requirements.txt
```

The first `pip install` will pull torch CPU wheels and is ~1.5 GB. Be patient.

### 3. Accept the Hugging Face model gates (in your browser)

You must visit each of these pages while logged into Hugging Face and click **Agree and access**:

1. https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/speaker-diarization-3.1
4. https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

### 4. Create a Hugging Face access token

Go to https://huggingface.co/settings/tokens, create a **Read** token, and export it in your shell:

```bash
export HF_TOKEN="hf_xxx_your_token_here"
```

### 5. Download the models into `models/`

```bash
python scripts/download_models.py
```

This script (using your `HF_TOKEN`) downloads:

- `CohereLabs/cohere-transcribe-03-2026` → `models/cohere-transcribe-03-2026/`
- `pyannote/segmentation-3.0` → `models/pyannote-3.1/segmentation-3.0/`
- `pyannote/wespeaker-voxceleb-resnet34-LM` → `models/pyannote-3.1/wespeaker-voxceleb-resnet34-LM/`

Total disk: ~4 GB. Coffee break recommended.

The Pyannote pipeline `config.yaml` is already checked into the repo at `models/pyannote-3.1/config.yaml` and references the two local models above with relative paths.

> Silero VAD ships inside the `silero-vad` PyPI package — no manual download.

### 6. Add Tauri icons (one-time)

Tauri requires icons to build. The simplest path:

```bash
cd src-tauri
mkdir -p icons
# Drop any 1024x1024 PNG into icons/source.png, then:
cargo tauri icon icons/source.png
cd ..
```

(For dev only, you can also delete the `bundle.icon` array in `src-tauri/tauri.conf.json` to bypass this — but you'll need them before building a real installer.)

### 7. Run it

In one terminal, activate the Python venv:

```bash
source .venv/bin/activate
```

In the same terminal:

```bash
npm run tauri dev
```

This launches the Tauri dev window. The Rust backend will spawn `python backend/main.py` as a subprocess for each transcription job, picking up `.venv/bin/python` automatically because the venv is activated.

> Set `OFFLINE_TRANSCRIBER_PYTHON` to override the Python interpreter path the Rust process uses. Default: `python3` on PATH.

---

## Acceptance test

1. Disable Wi-Fi.
2. Pick a 1–5 min English WAV.
3. Click **Start**.
4. Observe stage transitions: decode → vad → diarize → transcribe → export.
5. Open the `.docx` — speaker turns should be labeled `SPEAKER_00`, `SPEAKER_01`, etc., with one paragraph per turn.

If all of that works, Stage 1 is done.

---

## Layout

See `SRS.md` § 5 for the full directory layout. The two interfaces you care about most:

- `backend/main.py` — sidecar CLI: `python backend/main.py --input <wav> --output-dir <dir> --models-dir <dir>`. Emits JSON events on stdout.
- `src-tauri/src/lib.rs` — defines the `process_audio` Tauri command that spawns the sidecar and streams events to the frontend.

---

## Common issues

- **`Could not download 'pyannote/...' pipeline`** — you skipped step 3 or 4. Re-accept the gates and re-run `download_models.py`.
- **`OSError: ... .so` on macOS** when loading torch — you're on macOS arm64 with an x86_64 venv. Recreate the venv with the matching arch Python.
- **Out of memory loading Cohere** — Stage 1 runs the FP32 2B model (~8 GB RAM). Close other apps, or wait for Stage 3 (INT8 ONNX, ~3 GB).
- **Sidecar exits silently** — run `python backend/main.py --input test.wav --output-dir /tmp` directly to see Python tracebacks. The Tauri UI only shows clean error events.
- **`No matching distribution found for lightning`** — PyPI has currently quarantined the `lightning` package. `backend/requirements.txt` works around this by installing the wheel directly from Lightning AI's GitHub release. If you bumped pip or got the file from elsewhere and see this, restore the `lightning @ https://...` line and re-install.
- **`module 'torchaudio' has no attribute 'AudioMetaData'`** — torchaudio 2.9 removed this API but pyannote.audio 3.x still uses it. The requirements pin `torch<2.9`/`torchaudio<2.9`; if you upgraded those manually, downgrade with `pip install "torch<2.9" "torchaudio<2.9"`.
- **`hf_hub_download() got an unexpected keyword argument 'use_auth_token'`** — pyannote 3.x calls a kwarg that huggingface_hub 1.0+ removed. `backend/main.py` installs a shim at startup that rewrites `use_auth_token=` → `token=`. If you see this it usually means you ran `python pipeline.py` directly without going through `main.py`; use `main.py` as the entry point.
- **`UnpicklingError: Weights only load failed`** with `TorchVersion` — PyTorch 2.6 made `torch.load(weights_only=True)` the default and pyannote checkpoints contain metadata pickles outside torch's allowlist. `backend/main.py` patches `torch.load` to default to `weights_only=False` for our trusted bundled weights. Same fix path: enter via `main.py`.
