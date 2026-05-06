"""Dataset preparation for benchmark/regression runs.

This module downloads a tiny public benchmark set once, then reuses the
cached WAV + reference text files for offline reruns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16_000
BENCH_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
DATA_ROOT = BENCH_ROOT / "data"


@dataclass(frozen=True)
class BenchItem:
    item_id: str
    suite: str
    source: str
    audio_path: Path
    reference_path: Path
    duration_s: float


def prepare_dataset(suite: str) -> list[BenchItem]:
    """Return cached dataset items for ``suite`` (download if needed)."""
    suite = suite.strip().lower()
    if suite not in {"smoke", "full"}:
        raise ValueError(f"Unsupported suite '{suite}'. Expected smoke|full.")

    manifest_path = DATA_ROOT / f"{suite}.manifest.json"
    if manifest_path.is_file():
        return _read_manifest(manifest_path)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    items = _build_suite(suite)
    _write_manifest(manifest_path, items)
    return items


def _build_suite(suite: str) -> list[BenchItem]:
    from datasets import load_dataset

    if suite == "smoke":
        earnings22_limit = 2
        earnings21_limit = 0
    else:
        earnings22_limit = 4
        earnings21_limit = 1

    built: list[BenchItem] = []
    earnings22_cfg, earnings22_stream = _load_with_config_fallback(
        load_dataset,
        "hf-audio/open-asr-leaderboard",
        ("earnings22-open", "earnings22"),
    )
    built.extend(
        _materialize_stream(
            stream=earnings22_stream,
            suite=suite,
            source=earnings22_cfg,
            limit=earnings22_limit,
        )
    )
    if earnings21_limit > 0:
        earnings21_cfg, earnings21_stream = _load_with_config_fallback(
            load_dataset,
            "hf-audio/asr-leaderboard-longform",
            ("earnings21-lb", "earnings21"),
        )
        built.extend(
            _materialize_stream(
                stream=earnings21_stream,
                suite=suite,
                source=earnings21_cfg,
                limit=earnings21_limit,
                max_duration_s=600.0,
            )
        )
    return built


def _load_with_config_fallback(load_dataset, repo: str, configs: tuple[str, ...]):
    """Load dataset stream trying known config aliases in order."""
    from datasets import Audio

    last_error: Exception | None = None
    for cfg in configs:
        try:
            stream = load_dataset(repo, cfg, split="test", streaming=True)
            # Keep audio as raw bytes/path so we do not depend on torchcodec.
            stream = stream.cast_column("audio", Audio(decode=False))
            return cfg, stream
        except ValueError as exc:
            # e.g. "BuilderConfig '<name>' not found"
            if "BuilderConfig" in str(exc):
                last_error = exc
                continue
            raise
    if last_error is not None:
        raise last_error
    raise ValueError(f"No valid configs found for {repo}: {configs}")


def _materialize_stream(
    *,
    stream: Iterable[dict],
    suite: str,
    source: str,
    limit: int,
    max_duration_s: float | None = None,
) -> list[BenchItem]:
    import numpy as np
    import soundfile as sf

    out: list[BenchItem] = []
    for idx, sample in enumerate(stream):
        if len(out) >= limit:
            break
        audio, sr = _extract_audio(sample)
        text = _extract_text(sample)
        if max_duration_s is not None:
            max_samples = int(max_duration_s * sr)
            audio = audio[:max_samples]
        if sr != SAMPLE_RATE:
            audio = _resample(audio, src_sr=sr, dst_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        if audio.ndim != 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32, copy=False)
        duration_s = float(audio.shape[0]) / float(sr)
        item_id = f"{source}-{idx:03d}"
        audio_path = DATA_ROOT / f"{item_id}.wav"
        reference_path = DATA_ROOT / f"{item_id}.txt"
        sf.write(str(audio_path), audio, sr)
        reference_path.write_text(text.strip() + "\n", encoding="utf-8")
        out.append(
            BenchItem(
                item_id=item_id,
                suite=suite,
                source=source,
                audio_path=audio_path,
                reference_path=reference_path,
                duration_s=round(duration_s, 4),
            )
        )
    return out


def _extract_audio(sample: dict) -> tuple[np.ndarray, int]:
    import io

    import numpy as np
    import soundfile as sf

    audio = sample.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("dataset sample does not expose an 'audio' dict")
    array = audio.get("array")
    sr = audio.get("sampling_rate")
    if array is not None and sr is not None:
        return np.asarray(array, dtype=np.float32), int(sr)

    # decode=False format from datasets.Audio gives {bytes, path}
    raw = audio.get("bytes")
    path = audio.get("path")
    if raw is not None:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return np.asarray(wav, dtype=np.float32), int(sr)
    if isinstance(path, str) and path:
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            return np.asarray(wav, dtype=np.float32), int(sr)
        except Exception:
            # fallback for odd containers handled better by librosa/audioread
            import librosa

            wav, sr = librosa.load(path, sr=None, mono=False)
            return np.asarray(wav, dtype=np.float32), int(sr)

    raise ValueError("dataset audio is missing array/sampling_rate and bytes/path")


def _extract_text(sample: dict) -> str:
    for key in ("text", "transcript", "sentence", "normalized_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("dataset sample missing transcript text field")


def _resample(audio: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    try:
        import librosa
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError(
            "librosa is required to resample benchmark audio"
        ) from exc
    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr)


def _read_manifest(path: Path) -> list[BenchItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[BenchItem] = []
    for row in payload.get("items", []):
        out.append(
            BenchItem(
                item_id=row["item_id"],
                suite=row["suite"],
                source=row["source"],
                audio_path=Path(row["audio_path"]),
                reference_path=Path(row["reference_path"]),
                duration_s=float(row["duration_s"]),
            )
        )
    return out


def _write_manifest(path: Path, items: list[BenchItem]) -> None:
    payload = {
        "items": [
            {
                **asdict(item),
                "audio_path": str(item.audio_path),
                "reference_path": str(item.reference_path),
            }
            for item in items
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
