"""ASR backend protocol and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from vad import Segment


class AsrBackend(Protocol):
    device: str | None
    dtype: object | None
    encoder_attn_mask_fix: bool
    encoder_attn_mask_fix_layers: int
    encoder_fp32: bool
    encoder_eager_attempts: int
    encoder_eager_recovered: int
    encoder_fallback_attempts: int
    encoder_fallback_recovered: int
    mps_retry_attempts: int
    mps_retry_recovered: int
    cpu_retry_attempts: int
    cpu_retry_recovered: int
    cpu_fallback_loaded: bool

    def _ensure_loaded(self) -> None: ...

    def transcribe_segments(
        self, audio: np.ndarray, segments: Iterable[Segment], *, language: str | None = None
    ) -> list: ...

    def reset_retry_counters(self) -> None: ...


def make_backend(
    name: str,
    *,
    model_dir: Path,
    language: str = "en",
    device: str | None = None,
) -> AsrBackend:
    key = name.strip().lower()
    if key == "transformers":
        from asr import TransformersCohereAsr

        return TransformersCohereAsr(model_dir, language=language, device=device)
    raise ValueError(f"Unsupported ASR backend '{name}'.")
