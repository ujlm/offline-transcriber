"""Pyannote 3.1 speaker diarization, loaded fully offline from local weights.

The pipeline is configured via ``models/pyannote-3.1/config.yaml`` which points
at the segmentation and embedding model files using paths relative to that
config file. At call time we materialise an absolute-path copy in a temp file
because pyannote.audio resolves checkpoint paths via ``os.path.isfile``, which
only succeeds when the path is absolute (or relative to whichever cwd the
sidecar happens to be running from).

Hardware acceleration: the pipeline is moved to MPS on Apple Silicon (or CUDA
when present) and falls back to CPU otherwise.
``PYTORCH_ENABLE_MPS_FALLBACK=1`` (set in ``main.py``) lets unsupported ops
fall back to CPU silently so partial MPS support never breaks a run.

Caching: the pipeline is wrapped in :class:`DiarizePipeline` so the warm
sidecar can hold one instance and reuse it across jobs. The first
inference compiles MPS kernels (~21 s on a 4-min clip on an M-series Mac);
subsequent inferences on the same process reuse those kernels and run
~9 s faster. ``Pipeline.from_pretrained`` itself is cheap (~0.8 s); the
real win from caching is keeping MPS kernels warm across jobs.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_io import SAMPLE_RATE


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DiarizationResult:
    segments: list[SpeakerSegment]
    device: str
    min_speakers: int | None
    max_speakers: int | None


def _resolve_device() -> str:
    """Pick the fastest available torch device for diarization."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DiarizePipeline:
    """Lazy, cacheable wrapper around a pyannote ``Pipeline`` instance.

    Construct once per ``models_dir``; reuse across jobs in the warm
    sidecar. ``_ensure_loaded()`` is idempotent — calling it on an
    already-loaded instance is a no-op, mirroring :class:`CohereAsr`.

    Why a class and not a closure: we want a stable identity to cache
    against in ``main.py`` (so swapping ``models_dir`` invalidates the
    cache cleanly), and we want ``_ensure_loaded()`` to be callable
    from a background thread separately from the actual inference.
    """

    def __init__(self, models_dir: Path, *, device: str | None = None) -> None:
        self.models_dir = Path(models_dir)
        self._requested_device = device
        self._pipeline: Any | None = None
        self._device: str | None = None

    @property
    def device(self) -> str | None:
        """Device the pipeline is loaded on, or ``None`` if not loaded."""
        return self._device

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        import torch
        import yaml
        from pyannote.audio import Pipeline

        config_dir = (self.models_dir / "pyannote-3.1").resolve()
        config_path = config_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Pyannote config not found at {config_path}. "
                "Run `python scripts/download_models.py` first."
            )

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        params = config.get("pipeline", {}).get("params", {}) or {}
        for key in ("segmentation", "embedding"):
            params[key] = _resolve_weight_path(config_dir, key, params.get(key))

        # pyannote resolves checkpoint paths via ``os.path.isfile``; rewrite
        # the config to absolute paths and load from a temp file so the
        # resolution succeeds regardless of cwd.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", dir=str(config_dir), delete=False, encoding="utf-8"
        ) as tmp:
            yaml.safe_dump(config, tmp)
            tmp_path = Path(tmp.name)
        try:
            pipeline = Pipeline.from_pretrained(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        chosen_device = self._requested_device or _resolve_device()
        try:
            pipeline.to(torch.device(chosen_device))
        except Exception:
            # If moving to the chosen device fails for any reason (e.g.
            # MPS unsupported op surfaced eagerly), fall back to CPU
            # rather than killing the whole run.
            chosen_device = "cpu"
            pipeline.to(torch.device("cpu"))

        self._pipeline = pipeline
        self._device = chosen_device

    def diarize(
        self,
        audio: np.ndarray,
        *,
        min_speakers: int | None = 1,
        max_speakers: int | None = 4,
    ) -> DiarizationResult:
        import torch

        self._ensure_loaded()
        assert self._pipeline is not None  # for type-checkers
        assert self._device is not None

        waveform = torch.from_numpy(audio).unsqueeze(0)
        diarize_kwargs: dict = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = int(max_speakers)

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            **diarize_kwargs,
        )

        out: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            out.append(
                SpeakerSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
            )
        out.sort(key=lambda s: s.start)
        return DiarizationResult(
            segments=out,
            device=self._device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )


def diarize(
    audio: np.ndarray,
    models_dir: Path,
    *,
    min_speakers: int | None = 1,
    max_speakers: int | None = 4,
    device: str | None = None,
) -> DiarizationResult:
    """One-shot convenience wrapper.

    Constructs a fresh :class:`DiarizePipeline`, runs once, returns. The
    pipeline is discarded on return — for repeated calls (warm sidecar)
    construct a :class:`DiarizePipeline` directly and reuse it.
    """
    pipeline = DiarizePipeline(models_dir, device=device)
    return pipeline.diarize(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )


def _resolve_weight_path(config_dir: Path, key: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"pyannote config has no '{key}' entry")

    candidate = (config_dir / value).resolve()
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        weights = candidate / "pytorch_model.bin"
        if weights.is_file():
            return str(weights)
    raise FileNotFoundError(
        f"Pyannote '{key}' weights not found. "
        f"Resolved '{value}' to {candidate}; expected a file (or a directory containing pytorch_model.bin). "
        "Re-run `python scripts/download_models.py`."
    )
