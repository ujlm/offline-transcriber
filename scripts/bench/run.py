"""Run headless transcription benchmarks and emit machine-readable results."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.datasets import BenchItem, prepare_dataset  # noqa: E402
from scripts.bench.wer import (  # noqa: E402
    compute_wer,
    load_profile_metrics,
    peak_torch_memory_bytes,
    reset_torch_peak_memory,
)


@dataclass(frozen=True)
class BenchCaseResult:
    item_id: str
    source: str
    audio_path: str
    profile_path: str
    txt_path: str
    duration_s: float
    total_duration_s: float
    transcribe_duration_s: float
    diarize_duration_s: float
    decode_duration_s: float
    rtfx: float
    wer: float
    peak_python_mb: float
    peak_torch_mb: float | None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run offline ASR benchmark suite.")
    p.add_argument("--suite", default="smoke", choices=("smoke", "full"))
    p.add_argument(
        "--backend",
        default="transformers",
        choices=("transformers",),
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--language", default="en")
    p.add_argument("--diarization-enabled", action="store_true")
    p.add_argument(
        "--models-dir",
        default=str(ROOT / "models"),
        help="Directory containing local model snapshots.",
    )
    p.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults to benchmarks/results/<ts>.json",
    )
    return p.parse_args()


def _resolve_device(requested: str) -> tuple[str, bool]:
    if requested == "auto":
        try:
            import torch
        except Exception:
            return "cpu", True
        if torch.cuda.is_available():
            return "cuda", True
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", True
        return "cpu", True

    if requested == "cpu":
        return "cpu", True

    try:
        import torch
    except Exception:
        return requested, False

    if requested == "cuda":
        return "cuda", bool(torch.cuda.is_available())
    if requested == "mps":
        return "mps", bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        )
    return requested, False


def _make_backend(name: str, *, models_dir: Path, language: str, device: str):
    from asr_backends import make_backend  # type: ignore

    cohere_dir = models_dir / "cohere-transcribe-03-2026"
    return make_backend(
        name,
        model_dir=cohere_dir,
        language=language,
        device=None if device == "auto" else device,
    )


def _read_reference(item: BenchItem) -> str:
    return item.reference_path.read_text(encoding="utf-8").strip()


def _read_hypothesis(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8").strip()


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{ts}.json"


def main() -> int:
    args = _parse_args()
    import main as sidecar_main  # type: ignore
    from pipeline import PipelineConfig, run  # type: ignore

    sidecar_main._enforce_offline()
    sidecar_main._install_hf_hub_compat_shim()
    sidecar_main._install_torch_load_compat_shim()

    resolved_device, device_available = _resolve_device(args.device)
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "backend": args.backend,
        "language": args.language,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "device_available": device_available,
        "diarization_enabled": bool(args.diarization_enabled),
        "models_dir": str(Path(args.models_dir).resolve()),
        "items": [],
        "aggregate": {},
    }

    if not device_available:
        payload["warning"] = f"Requested device '{args.device}' is not available."
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Device unavailable, wrote {output_path}")
        return 0

    items = prepare_dataset(args.suite)
    models_dir = Path(args.models_dir)
    run_dir = output_path.parent / output_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    asr_backend = _make_backend(
        args.backend, models_dir=models_dir, language=args.language, device=resolved_device
    )

    results: list[BenchCaseResult] = []
    for item in items:
        output_dir = run_dir / item.item_id
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = PipelineConfig(
            input_path=item.audio_path,
            output_dir=output_dir,
            models_dir=models_dir,
            language=args.language,
            diarization_enabled=bool(args.diarization_enabled),
        )
        reset_torch_peak_memory(resolved_device)
        tracemalloc.start()
        t0 = time.monotonic()
        result = run(cfg, asr=asr_backend, diarize_pipeline=None)
        wall_s = time.monotonic() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        profile_metrics = load_profile_metrics(result.profile_path)
        reference = _read_reference(item)
        hypothesis = _read_hypothesis(result.txt_path)
        torch_peak = peak_torch_memory_bytes(resolved_device)
        case = BenchCaseResult(
            item_id=item.item_id,
            source=item.source,
            audio_path=str(item.audio_path),
            profile_path=str(result.profile_path),
            txt_path=str(result.txt_path),
            duration_s=item.duration_s,
            total_duration_s=profile_metrics.total_duration_s or wall_s,
            transcribe_duration_s=profile_metrics.transcribe_duration_s,
            diarize_duration_s=profile_metrics.diarize_duration_s,
            decode_duration_s=profile_metrics.decode_duration_s,
            rtfx=profile_metrics.rtfx,
            wer=compute_wer(reference, hypothesis),
            peak_python_mb=round(float(peak_bytes) / (1024 * 1024), 4),
            peak_torch_mb=(
                round(float(torch_peak) / (1024 * 1024), 4)
                if torch_peak is not None
                else None
            ),
        )
        results.append(case)

    payload["items"] = [asdict(r) for r in results]
    payload["aggregate"] = _aggregate(results)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {output_path}")
    print(_summary_table(results))
    return 0


def _aggregate(items: list[BenchCaseResult]) -> dict:
    if not items:
        return {}
    n = float(len(items))
    avg_wer = sum(x.wer for x in items) / n
    avg_rtfx = sum(x.rtfx for x in items) / n
    total_audio_s = sum(x.duration_s for x in items)
    total_transcribe_s = sum(x.transcribe_duration_s for x in items)
    throughput_rtfx = (total_audio_s / total_transcribe_s) if total_transcribe_s > 0 else 0.0
    return {
        "cases": int(n),
        "avg_wer": round(avg_wer, 6),
        "avg_rtfx": round(avg_rtfx, 6),
        "throughput_rtfx": round(throughput_rtfx, 6),
        "avg_peak_python_mb": round(sum(x.peak_python_mb for x in items) / n, 6),
    }


def _summary_table(items: list[BenchCaseResult]) -> str:
    lines = [
        "",
        "| case | WER | RTFx | transcribe_s |",
        "|---|---:|---:|---:|",
    ]
    for item in items:
        lines.append(
            f"| {item.item_id} | {item.wer:.4f} | {item.rtfx:.3f} | {item.transcribe_duration_s:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
