"""Silero VAD wrapper.

Produces a list of (start_seconds, end_seconds) speech segments from a 16 kHz
mono float32 numpy array. The Silero pip package bundles its own model weights,
so no manual download is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_io import SAMPLE_RATE


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech(audio: np.ndarray) -> list[Segment]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    waveform = torch.from_numpy(audio)
    raw = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200,
    )
    return [Segment(start=float(s["start"]), end=float(s["end"])) for s in raw]
