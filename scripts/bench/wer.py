"""WER/speed metric helpers for benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_NORMALIZER = None
_JIWER_WER: Callable[[str, str], float] | None = None


@dataclass(frozen=True)
class ProfileMetrics:
    total_duration_s: float
    transcribe_duration_s: float
    audio_seconds_processed: float
    rtfx: float
    diarize_duration_s: float
    decode_duration_s: float


def normalize_text(text: str) -> str:
    global _NORMALIZER
    if _NORMALIZER is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
        except Exception as exc:
            raise RuntimeError(
                "whisper-normalizer is required for benchmark WER. "
                "Install with: pip install -r backend/requirements-bench.txt"
            ) from exc
        _NORMALIZER = EnglishTextNormalizer()
    return _NORMALIZER(text)


def compute_wer(reference: str, hypothesis: str) -> float:
    global _JIWER_WER
    if _JIWER_WER is None:
        try:
            from jiwer import wer as jiwer_wer
        except Exception as exc:
            raise RuntimeError(
                "jiwer is required for benchmark WER. "
                "Install with: pip install -r backend/requirements-bench.txt"
            ) from exc
        _JIWER_WER = jiwer_wer
    ref_norm = normalize_text(reference or "").strip()
    hyp_norm = normalize_text(hypothesis or "").strip()
    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0
    assert _JIWER_WER is not None
    return float(_JIWER_WER(ref_norm, hyp_norm))


def load_profile_metrics(path: Path) -> ProfileMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages") or []
    by_name = {str(s.get("stage")): s for s in stages if isinstance(s, dict)}
    transcribe = by_name.get("transcribe", {})
    diarize = by_name.get("diarize", {})
    decode = by_name.get("decode", {})
    transcribe_duration_s = float(transcribe.get("duration_s") or 0.0)
    audio_seconds_processed = float(
        (transcribe.get("details") or {}).get("audio_seconds_processed") or 0.0
    )
    rtfx = (
        (audio_seconds_processed / transcribe_duration_s)
        if transcribe_duration_s > 0
        else 0.0
    )
    return ProfileMetrics(
        total_duration_s=float(payload.get("total_duration_s") or 0.0),
        transcribe_duration_s=transcribe_duration_s,
        audio_seconds_processed=audio_seconds_processed,
        rtfx=rtfx,
        diarize_duration_s=float(diarize.get("duration_s") or 0.0),
        decode_duration_s=float(decode.get("duration_s") or 0.0),
    )


def peak_torch_memory_bytes(device: str) -> int | None:
    """Best-effort torch memory peak for the selected device."""
    try:
        import torch
    except Exception:
        return None

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        return int(torch.cuda.max_memory_allocated(idx))

    if device == "mps" and getattr(torch.backends, "mps", None) is not None:
        try:
            return int(torch.mps.current_allocated_memory())
        except Exception:
            return None

    return None


def reset_torch_peak_memory(device: str) -> None:
    try:
        import torch
    except Exception:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        torch.cuda.reset_peak_memory_stats(idx)
