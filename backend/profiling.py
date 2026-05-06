"""Lightweight profiler for a single pipeline run.

The profiler records two things:

1. Per-stage wall-clock timings via a `stage(...)` context manager, plus an
   optional ``details`` dict for stage-specific metrics (e.g. realtime factor
   for transcription).
2. A timestamped copy of every ``events.log`` message emitted during the run,
   tagged with the currently-active stage.

At the end of a run we serialise everything to JSON next to the .docx output
so that we can profile the pipeline end-to-end after the fact.

Module name is ``profiling`` (not ``profile``) on purpose to avoid shadowing
the stdlib profiler when this directory is on ``sys.path``.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass
class _StageRecord:
    stage: str
    start_s: float
    end_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LogRecord:
    t_s: float
    level: str
    stage: str | None
    message: str


class RunProfile:
    """Thread-safe capture of stage timings and log messages for one run."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._finished_at: str | None = None
        self._stages: list[_StageRecord] = []
        self._logs: list[_LogRecord] = []
        self._stage_stack: list[_StageRecord] = []
        self._lock = threading.Lock()
        self.metadata: dict[str, Any] = {}

    def now(self) -> float:
        """Seconds elapsed since the profile was created."""
        return time.monotonic() - self._t0

    @contextmanager
    def stage(self, name: str, **details: Any) -> Iterator[_StageRecord]:
        """Time the enclosed block and tag it with ``name``.

        Stage records nest, so logs emitted inside a child stage are tagged
        with the innermost stage. Extra keyword arguments are stored as
        ``details`` and may be mutated after the fact (handy for stats like
        segment counts that are only known once the stage completes).
        """
        with self._lock:
            rec = _StageRecord(stage=name, start_s=self.now(), details=dict(details))
            self._stages.append(rec)
            self._stage_stack.append(rec)
        try:
            yield rec
        finally:
            with self._lock:
                rec.end_s = self.now()
                # Pop only if still on top; defensive against weird nesting.
                if self._stage_stack and self._stage_stack[-1] is rec:
                    self._stage_stack.pop()

    def add_log(self, message: str, level: str = "info") -> None:
        with self._lock:
            current = self._stage_stack[-1].stage if self._stage_stack else None
            self._logs.append(
                _LogRecord(t_s=self.now(), level=level, stage=current, message=message)
            )

    def finalize(self) -> None:
        if self._finished_at is None:
            self._finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        with self._lock:
            now = self.now()
            stages = [
                {
                    "stage": s.stage,
                    "start_s": round(s.start_s, 4),
                    "end_s": round(s.end_s if s.end_s is not None else now, 4),
                    "duration_s": round(
                        (s.end_s if s.end_s is not None else now) - s.start_s, 4
                    ),
                    "details": dict(s.details),
                }
                for s in self._stages
            ]
            logs = [
                {
                    "t_s": round(l.t_s, 4),
                    "level": l.level,
                    "stage": l.stage,
                    "message": l.message,
                }
                for l in self._logs
            ]
            return {
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "total_duration_s": round(now, 4),
                "metadata": dict(self.metadata),
                "stages": stages,
                "logs": logs,
            }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
