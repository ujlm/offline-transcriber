"""Build ASR-friendly chunks from VAD speech segments.

Cohere Transcribe (like Whisper) has a fixed ~30-second receptive field, and
each ``model.generate(...)`` call carries roughly ~0.7-1.0s of fixed overhead
on CPU regardless of the clip length. Sending the model 47 short VAD segments
costs ~47 setup hits; sending it ~10 packed ~25-second chunks costs ~10. The
profile shows tiny segments dominated by overhead (e.g. a 0.5s clip taking
3.7s wall, RTF 7.4x), so packing is worth it.

We can't merge naively across speaker boundaries — that would smear two
speakers' words into one ``Turn``. So we use the diarization output to decide
where it's safe to merge: two adjacent VAD segments are joined iff their
dominant speaker (per diarization) matches and the silence gap between them
is small. The result is a list of ``Chunk`` objects, each carrying the
collapsed time span plus bookkeeping for the profile.
"""

from __future__ import annotations

from dataclasses import dataclass

from diarize import SpeakerSegment
from fusion import assign_speaker
from vad import Segment


@dataclass(frozen=True)
class Chunk:
    """A merged span of one or more VAD segments destined for a single ASR call.

    ``segment`` is the contract the ASR layer consumes (a ``vad.Segment``).
    The other fields are diagnostic / for the profile.
    """

    segment: Segment
    speaker: str
    source_segments: int


def merge_for_asr(
    vad_segments: list[Segment],
    speakers: list[SpeakerSegment],
    *,
    max_chunk_s: float = 28.0,
    max_gap_s: float = 1.5,
    min_isolated_chunk_s: float = 0.3,
) -> list[Chunk]:
    """Greedily pack ``vad_segments`` into chunks suitable for the ASR model.

    Args:
        vad_segments: speech segments from Silero VAD, sorted by start time.
        speakers: diarization segments, sorted by start time. Used only to
            decide where merging is safe.
        max_chunk_s: maximum duration of a merged chunk. Cohere/Whisper-class
            models have a 30s receptive field, so we cap below that to leave
            a small safety margin.
        max_gap_s: maximum silence gap (seconds) between two VAD segments that
            still allows them to be merged. Larger gaps imply real pauses /
            turn boundaries we'd rather preserve.
        min_isolated_chunk_s: minimum duration for a chunk made of a single
            VAD segment. Tiny isolated blips (often a cough or lip-smack
            mis-clustered into a fake speaker by pyannote) almost always
            transcribe to empty text but still cost ~1-2s of MPS time and
            can be near-unbounded when the model fails to emit EOS. Multi-
            segment chunks are never dropped because if two VAD segments
            agreed it was speech, it almost certainly is.

    A single VAD segment that already exceeds ``max_chunk_s`` is passed through
    as its own chunk; we don't split mid-segment because the ASR model handles
    its own internal chunking and we'd lose timing accuracy.
    """
    if not vad_segments:
        return []

    chunks: list[Chunk] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_speaker: str | None = None
    cur_count = 0

    def _flush() -> None:
        nonlocal cur_start, cur_end, cur_speaker, cur_count
        if cur_start is None or cur_end is None or cur_speaker is None:
            return
        duration = cur_end - cur_start
        if not (cur_count == 1 and duration < min_isolated_chunk_s):
            chunks.append(
                Chunk(
                    segment=Segment(start=cur_start, end=cur_end),
                    speaker=cur_speaker,
                    source_segments=cur_count,
                )
            )
        cur_start = cur_end = cur_speaker = None
        cur_count = 0

    for seg in vad_segments:
        sp = assign_speaker(seg.start, seg.end, speakers)

        if cur_start is None:
            cur_start, cur_end, cur_speaker, cur_count = seg.start, seg.end, sp, 1
            continue

        assert cur_end is not None and cur_speaker is not None
        gap = seg.start - cur_end
        merged_dur = seg.end - cur_start
        can_merge = (
            sp == cur_speaker
            and gap <= max_gap_s
            and merged_dur <= max_chunk_s
        )
        if can_merge:
            cur_end = seg.end
            cur_count += 1
        else:
            _flush()
            cur_start, cur_end, cur_speaker, cur_count = seg.start, seg.end, sp, 1

    _flush()
    return chunks
