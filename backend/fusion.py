"""Fuse VAD-based transcriptions with Pyannote speaker segments.

For each transcribed VAD segment, we pick the speaker whose Pyannote segment
overlaps it the most, and group consecutive same-speaker segments into a single
turn. This produces ``Turn`` records that are the unit of output formatting.
"""

from __future__ import annotations

from dataclasses import dataclass

from asr import Transcription
from diarize import SpeakerSegment


UNKNOWN_SPEAKER = "SPEAKER_??"


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str
    text: str


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(
    seg_start: float,
    seg_end: float,
    speakers: list[SpeakerSegment],
) -> str:
    """Pick the speaker whose diarization span overlaps ``seg`` the most.

    Falls back to :data:`UNKNOWN_SPEAKER` when no diarization segment
    overlaps. ``speakers`` must be sorted by ``start`` (which `diarize.diarize`
    guarantees).
    """
    best_speaker = UNKNOWN_SPEAKER
    best_overlap = 0.0
    for sp in speakers:
        if sp.end < seg_start:
            continue
        if sp.start > seg_end:
            break
        ov = overlap(seg_start, seg_end, sp.start, sp.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = sp.speaker
    return best_speaker


def fuse(
    transcriptions: list[Transcription],
    speakers: list[SpeakerSegment],
) -> list[Turn]:
    if not transcriptions:
        return []

    labeled: list[Turn] = []
    for t in transcriptions:
        speaker = assign_speaker(t.start, t.end, speakers)
        labeled.append(Turn(start=t.start, end=t.end, speaker=speaker, text=t.text))

    merged: list[Turn] = []
    for turn in labeled:
        if merged and merged[-1].speaker == turn.speaker:
            prev = merged[-1]
            merged[-1] = Turn(
                start=prev.start,
                end=turn.end,
                speaker=prev.speaker,
                text=f"{prev.text} {turn.text}".strip(),
            )
        else:
            merged.append(turn)
    return merged
