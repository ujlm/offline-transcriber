"""One-shot helper to download every model required for Stage 1.

Requires the ``HF_TOKEN`` environment variable to be set with a token that has
accepted the gates listed in ``models/README.md``.

Usage:
    export HF_TOKEN="hf_xxx_your_token_here"
    python scripts/download_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

DOWNLOADS = [
    (
        "CohereLabs/cohere-transcribe-03-2026",
        MODELS_DIR / "cohere-transcribe-03-2026",
    ),
    (
        "pyannote/segmentation-3.0",
        MODELS_DIR / "pyannote-3.1" / "segmentation-3.0",
    ),
    (
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        MODELS_DIR / "pyannote-3.1" / "wespeaker-voxceleb-resnet34-LM",
    ),
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "error: HF_TOKEN environment variable is not set.\n"
            "       Create a token at https://huggingface.co/settings/tokens\n"
            "       and: export HF_TOKEN=hf_xxx_your_token_here",
            file=sys.stderr,
        )
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "error: huggingface_hub is not installed. "
            "Activate your venv and run: pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for repo_id, local_dir in DOWNLOADS:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==> Downloading {repo_id}")
        print(f"    -> {local_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
            )
        except Exception as e:
            print(f"\nFailed to download {repo_id}: {e}", file=sys.stderr)
            print(
                "Check that you have accepted the gate for this model on huggingface.co\n"
                "(see models/README.md for the list).",
                file=sys.stderr,
            )
            return 2

    config_path = MODELS_DIR / "pyannote-3.1" / "config.yaml"
    if not config_path.is_file():
        print(
            f"\nwarning: {config_path} is missing. "
            "It should have been committed in the repo.",
            file=sys.stderr,
        )

    print("\nAll models downloaded successfully.")
    print(f"Total disk usage under {MODELS_DIR}:")
    _print_dir_size(MODELS_DIR)
    return 0


def _print_dir_size(path: Path) -> None:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    gb = total / (1024 ** 3)
    print(f"  {gb:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
