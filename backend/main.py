"""Sidecar CLI entry.

Spawned by the Tauri Rust process. Communicates via JSON Lines on stdout.

Two modes:

* **One-shot CLI** (default, used for ad-hoc runs and integration tests): the
  process is invoked with ``--input/--output-dir/--models-dir/--language``,
  runs the pipeline once, emits ``done`` and exits.

* **Warm sidecar server** (``--server``, used by the Tauri app): the process
  stays alive and reads one JSON job per line from stdin. The Cohere ASR
  model is loaded on the first job and reused across subsequent jobs, so
  files 2..N skip the ~22 s model load cost. EOF on stdin or a
  ``{"command": "shutdown"}`` line terminates the loop cleanly. Any pipeline
  error is reported via an ``error`` event but does *not* take down the
  server — the next job still runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any


_CANCELLED = False

# Hard-coded fallback if reading config.json fails for any reason. Kept in
# sync with ``models/cohere-transcribe-03-2026/config.json::supported_languages``.
_FALLBACK_SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "el",
    "ar",
    "ja",
    "zh",
    "vi",
    "ko",
)


def _supported_languages(models_dir: Path) -> tuple[str, ...]:
    """Read the ASR model's supported languages from its config.json.

    Falls back to a baked-in list if the file is missing or malformed.
    The model itself is multilingual; this is purely a sanity check so we
    can reject typos like ``"english"`` or ``"EN"`` before paying the
    model-load cost.
    """
    cfg_path = models_dir / "cohere-transcribe-03-2026" / "config.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        langs = cfg.get("supported_languages")
        if isinstance(langs, list) and all(isinstance(x, str) for x in langs):
            return tuple(langs)
    except (OSError, json.JSONDecodeError):
        pass
    return _FALLBACK_SUPPORTED_LANGUAGES


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _CANCELLED
        _CANCELLED = True
        try:
            import events as _events

            _events.cancelled()
        finally:
            sys.exit(130)

    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, handler)


def _enforce_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Pyannote 3.1 on Apple Silicon MPS occasionally hits ops that aren't
    # implemented on the MPS backend (e.g. some convolution variants). Enabling
    # automatic CPU fallback keeps the pipeline working while still running the
    # bulk on the GPU. Must be set before torch is imported.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _install_hf_hub_compat_shim() -> None:
    """Bridge `use_auth_token=` (pyannote 3.x) to `token=` (huggingface_hub >= 1.0).

    huggingface_hub 1.0 dropped the `use_auth_token` kwarg entirely; pyannote
    3.x still passes it. Without this shim, any HF Hub call from inside pyannote
    raises `TypeError: hf_hub_download() got an unexpected keyword argument
    'use_auth_token'`. We patch a handful of HF Hub entry points to translate
    the old kwarg to the new one. Removing this is safe once pyannote 3.x is
    updated upstream or we move to pyannote 4.x.
    """
    try:
        import huggingface_hub  # noqa: WPS433
    except ImportError:
        return

    def _translate_kwargs(kwargs: dict) -> dict:
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            kwargs.setdefault("token", tok)
        return kwargs

    for fn_name in (
        "hf_hub_download",
        "snapshot_download",
        "model_info",
        "list_repo_files",
    ):
        original = getattr(huggingface_hub, fn_name, None)
        if original is None:
            continue

        def _make_wrapper(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **_translate_kwargs(kwargs))

            wrapper.__wrapped__ = fn
            return wrapper

        setattr(huggingface_hub, fn_name, _make_wrapper(original))


def _install_torch_load_compat_shim() -> None:
    """Allow pyannote 3.x ``pytorch_model.bin`` to load under PyTorch >= 2.6.

    PyTorch 2.6 flipped ``torch.load(weights_only=True)`` on by default for
    safety. Pyannote's checkpoints include small metadata pickles
    (``torch.torch_version.TorchVersion``, omegaconf containers, etc.) that
    aren't on torch's default allowlist, so loading aborts with an
    ``UnpicklingError``.

    We trust our own bundled weights, so we monkey-patch ``torch.load`` to
    default to ``weights_only=False`` when the caller didn't ask explicitly.
    """
    try:
        import torch
    except ImportError:
        return

    original_load = torch.load

    def patched_load(*args, **kwargs):
        # Force weights_only=False even when callers (e.g. lightning_fabric)
        # set it explicitly. We trust our bundled, gated weights.
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    patched_load.__wrapped__ = original_load
    torch.load = patched_load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline transcription sidecar")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Stay alive after first job; read jobs as JSON lines on stdin.",
    )
    parser.add_argument("--input", help="Path to input audio file (one-shot mode)")
    parser.add_argument(
        "--output-dir", help="Directory to write outputs into (one-shot mode)"
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Directory containing model weights",
    )
    parser.add_argument(
        "--language",
        default="en",
        help=(
            "ISO 639-1 language code. Supported: "
            + ", ".join(_FALLBACK_SUPPORTED_LANGUAGES)
            + " (default: en)"
        ),
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip speaker diarization and export a single-speaker transcript.",
    )
    args = parser.parse_args()
    if not args.server and not args.input:
        parser.error("--input is required unless --server is given")
    if not args.server and not args.output_dir:
        parser.error("--output-dir is required unless --server is given")
    return args


def _build_config(
    job: dict[str, Any], default_models_dir: Path, default_language: str
) -> "PipelineConfig":  # type: ignore[name-defined]  # forward import
    from pipeline import PipelineConfig

    try:
        input_path = Path(job["input"])
    except KeyError as e:
        raise ValueError(f"missing required job field: {e!s}") from e
    output_dir = Path(job.get("output_dir") or input_path.parent)
    models_dir = Path(job.get("models_dir") or default_models_dir)
    language = job.get("language") or default_language
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return PipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        models_dir=models_dir,
        language=language,
        diarization_enabled=bool(job.get("diarization_enabled", True)),
    )


def _run_one_shot(args: argparse.Namespace) -> int:
    import events
    from pipeline import PipelineConfig, run

    models_dir = Path(args.models_dir)
    supported = _supported_languages(models_dir)
    if args.language not in supported:
        events.error(
            "UNSUPPORTED_LANGUAGE",
            f"Language '{args.language}' is not supported. "
            f"Supported: {', '.join(supported)}.",
        )
        return 2

    input_path = Path(args.input)
    if not input_path.is_file():
        events.error("INPUT_NOT_FOUND", f"Input file not found: {input_path}")
        return 2

    cfg = PipelineConfig(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        models_dir=models_dir,
        language=args.language,
        diarization_enabled=not args.no_diarization,
    )

    try:
        result = run(cfg)
    except FileNotFoundError as e:
        events.error("MODEL_NOT_FOUND", str(e))
        return 3
    except Exception as e:
        events.error("PIPELINE_FAILED", f"{e.__class__.__name__}: {e}")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        return 1

    events.done(
        str(result.txt_path),
        str(result.docx_path),
        str(result.profile_path),
        transcript=result.transcript,
        diarization_enabled=result.diarization_enabled,
    )
    return 0


def _run_server(args: argparse.Namespace) -> int:
    """Read JSON jobs from stdin until EOF; reuse the loaded ASR across jobs.

    Job line format::

        {"input": "/path/to/audio.wav", "output_dir": "/some/dir",
         "language": "en", "diarization_enabled": true}

    ``output_dir``, ``language`` and ``diarization_enabled`` are optional;
    ``models_dir`` defaults to the ``--models-dir`` passed at startup but can
    be overridden per job.

    A line of ``{"command": "shutdown"}`` exits the loop cleanly. EOF on
    stdin (the host process closes its write end) does the same.
    """
    import events
    from asr_backends import make_backend
    from diarize import DiarizePipeline
    from pipeline import run

    default_models_dir = Path(args.models_dir)
    default_language = args.language
    supported = _supported_languages(default_models_dir)

    # The ASR model is multilingual: language is selected per call via the
    # decoder prompt, so we cache one instance per model directory and
    # reuse it across language switches without reloading weights.
    # Pyannote's pipeline is similarly stateless across calls — caching
    # it here also keeps its MPS kernels warm, which is the bulk of the
    # second-run-onwards speedup on the diarize stage.
    cached_asr = None
    cached_asr_dir: Path | None = None
    cached_diarize: DiarizePipeline | None = None
    cached_diarize_dir: Path | None = None

    events.log(
        "Sidecar entered server mode; awaiting jobs on stdin "
        f"(supported languages: {', '.join(supported)})"
    )

    while True:
        line = sys.stdin.readline()
        if not line:
            events.log("Sidecar stdin closed; shutting down server loop")
            return 0
        line = line.strip()
        if not line:
            continue

        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            events.error("BAD_JSON", f"could not parse job line: {e}")
            continue

        if isinstance(job, dict) and job.get("command") == "shutdown":
            events.log("Sidecar received shutdown command")
            return 0

        try:
            cfg = _build_config(job, default_models_dir, default_language)
        except (KeyError, ValueError) as e:
            events.error("BAD_JOB", str(e))
            continue
        except FileNotFoundError as e:
            events.error("INPUT_NOT_FOUND", str(e))
            continue

        if cfg.language not in supported:
            events.error(
                "UNSUPPORTED_LANGUAGE",
                f"Language '{cfg.language}' is not supported. "
                f"Supported: {', '.join(supported)}.",
            )
            continue

        cohere_dir = cfg.models_dir / "cohere-transcribe-03-2026"
        if cached_asr is not None and cached_asr_dir == cohere_dir:
            asr = cached_asr
        else:
            if cached_asr is not None:
                events.log(
                    "Discarding cached ASR (model directory changed)",
                    level="warning",
                )
            asr = make_backend(
                "transformers",
                model_dir=cohere_dir,
                language=cfg.language,
            )
            cached_asr = asr
            cached_asr_dir = cohere_dir

        if not cfg.diarization_enabled:
            diarize_pipeline = None
        elif cached_diarize is not None and cached_diarize_dir == cfg.models_dir:
            diarize_pipeline = cached_diarize
        else:
            if cached_diarize is not None:
                events.log(
                    "Discarding cached pyannote pipeline (models directory changed)",
                    level="warning",
                )
            diarize_pipeline = DiarizePipeline(cfg.models_dir)
            cached_diarize = diarize_pipeline
            cached_diarize_dir = cfg.models_dir

        try:
            result = run(cfg, asr=asr, diarize_pipeline=diarize_pipeline)
        except FileNotFoundError as e:
            events.error("MODEL_NOT_FOUND", str(e))
            continue
        except Exception as e:
            events.error("PIPELINE_FAILED", f"{e.__class__.__name__}: {e}")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            continue

        events.done(
            str(result.txt_path),
            str(result.docx_path),
            str(result.profile_path),
            transcript=result.transcript,
            diarization_enabled=result.diarization_enabled,
        )


def main() -> int:
    _enforce_offline()
    _install_hf_hub_compat_shim()
    _install_torch_load_compat_shim()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    _install_signal_handlers()
    args = _parse_args()

    if args.server:
        return _run_server(args)
    return _run_one_shot(args)


if __name__ == "__main__":
    sys.exit(main())
