# Local model weights

Everything in this folder must be downloaded **once** before the app can run. The recommended path is to use the helper script:

```bash
export HF_TOKEN="hf_xxx_your_token_here"
python scripts/download_models.py
```

Below is what that script downloads, in case you'd rather do it manually.

## Required HuggingFace gate accepts

Before downloading, log in at https://huggingface.co and click **Agree and access** on each of these pages:

1. https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 — Apache 2.0
2. https://huggingface.co/pyannote/segmentation-3.0 — MIT (gated)
3. https://huggingface.co/pyannote/speaker-diarization-3.1 — MIT (gated)
4. https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM — CC-BY 4.0 (gated)

## Layout (after download)

```
models/
├── README.md                                         (this file)
├── cohere-transcribe-03-2026/                        ~4 GB — full HF repo snapshot
│   ├── config.json
│   ├── model.safetensors                            (or model.safetensors.index.json + shards)
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── ... (any other files in the HF repo)
└── pyannote-3.1/
    ├── config.yaml                                  (committed in repo)
    ├── segmentation-3.0/                            ~6 MB
    │   ├── pytorch_model.bin
    │   └── config.yaml
    └── wespeaker-voxceleb-resnet34-LM/              ~26 MB
        ├── pytorch_model.bin
        └── config.yaml
```

The `pyannote-3.1/config.yaml` (committed in this repo) references the two child directories with **paths relative to itself**, so the whole `models/` directory is portable.

## Manual download (alternative to the script)

```bash
# 1. Install the HF CLI
pip install -U "huggingface_hub[cli]"

# 2. Log in (interactive)
huggingface-cli login

# 3. Cohere Transcribe
huggingface-cli download CohereLabs/cohere-transcribe-03-2026 \
  --local-dir models/cohere-transcribe-03-2026

# 4. Pyannote segmentation 3.0
huggingface-cli download pyannote/segmentation-3.0 \
  --local-dir models/pyannote-3.1/segmentation-3.0

# 5. Pyannote embeddings (wespeaker)
huggingface-cli download pyannote/wespeaker-voxceleb-resnet34-LM \
  --local-dir models/pyannote-3.1/wespeaker-voxceleb-resnet34-LM
```

After downloading, verify that `models/pyannote-3.1/segmentation-3.0/pytorch_model.bin` exists — that's the file referenced by `models/pyannote-3.1/config.yaml`.

## Silero VAD

No manual step. The `silero-vad` PyPI package bundles its own model weights (~2 MB).
