"""JSON Lines event protocol for stdout communication with the Rust host.

Every public function here writes exactly one JSON object followed by a newline
to stdout and flushes. The Rust process forwards each line to the frontend as a
`transcription-event` Tauri event.

A pipeline can also attach a `RunProfile` sink via `set_profile`; while
attached, every `log(...)` call is mirrored into the profile so we can later
write a JSON profile of the run.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol


class _ProfileSink(Protocol):
    def add_log(self, message: str, level: str = "info") -> None: ...


_profile: _ProfileSink | None = None


def set_profile(profile: _ProfileSink | None) -> None:
    """Attach (or detach with ``None``) a profile that mirrors log messages."""
    global _profile
    _profile = profile


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def progress(stage: str, percent: float) -> None:
    _emit({"event": "progress", "stage": stage, "percent": max(0.0, min(100.0, percent))})


def log(message: str, level: str = "info") -> None:
    _emit({"event": "log", "level": level, "message": message})
    sink = _profile
    if sink is not None:
        try:
            sink.add_log(message, level)
        except Exception:
            # Profiling is best-effort; never let it break the pipeline.
            pass


def done(
    txt_path: str,
    docx_path: str,
    profile_path: str | None = None,
    transcript: list[dict[str, Any]] | None = None,
    diarization_enabled: bool | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": "done",
        "txt_path": txt_path,
        "docx_path": docx_path,
    }
    if profile_path is not None:
        payload["profile_path"] = profile_path
    if transcript is not None:
        payload["transcript"] = transcript
    if diarization_enabled is not None:
        payload["diarization_enabled"] = diarization_enabled
    _emit(payload)


def cancelled() -> None:
    _emit({"event": "cancelled"})


def error(code: str, message: str) -> None:
    _emit({"event": "error", "code": code, "message": message})
