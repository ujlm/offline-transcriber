"""Audio decoding and resampling.

Cohere Transcribe expects 16 kHz mono float32. We standardize on that here.

Two backends, in preference order:

1. **ffmpeg** (when on PATH) — decodes any container/codec ffmpeg supports
   (WAV, FLAC, OGG, MP3, M4A, MOV, MKV, ...) directly to f32 PCM via a single
   subprocess. ~3-5x faster than librosa's audioread fallback for MP3/M4A and
   roughly on par with soundfile for WAV/FLAC. We always go through ffmpeg
   when available so the fast path is the default.

2. **librosa** fallback — used only when ffmpeg isn't on PATH or the ffmpeg
   call fails for a specific file. Slow but pure-Python.

The returned ``DecodedAudio`` carries a ``method`` tag so the pipeline can
record which backend actually ran in the profile.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000


class AudioDecodeError(RuntimeError):
    """Raised when an audio file cannot be decoded."""


@dataclass(frozen=True)
class DecodedAudio:
    """Float32 mono audio at :data:`SAMPLE_RATE` plus a tag for telemetry."""

    samples: np.ndarray
    method: str


def load_audio(path: Path) -> DecodedAudio:
    """Decode any supported audio file to a 1-D float32 numpy array at 16 kHz mono."""
    if shutil.which("ffmpeg"):
        try:
            samples = _decode_via_ffmpeg(path)
            return DecodedAudio(samples=samples, method="ffmpeg")
        except (subprocess.CalledProcessError, OSError):
            # ffmpeg is on PATH but choked on this specific file; fall through
            # to librosa which sometimes handles odd containers differently.
            pass

    return DecodedAudio(samples=_decode_via_librosa(path), method="librosa")


def _decode_via_ffmpeg(path: Path) -> np.ndarray:
    """Stream-decode ``path`` to 16 kHz mono f32 via a single ffmpeg call.

    ``-nostdin`` keeps ffmpeg from grabbing the parent's stdin, ``-vn`` drops
    any video tracks, and ``-f f32le -`` writes raw little-endian float32 PCM
    to our stdout pipe. The whole audio is buffered in memory which is fine
    for typical inputs (a 1-hour 16 kHz mono f32 stream is ~230 MB).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    if not proc.stdout:
        raise AudioDecodeError(f"ffmpeg produced no audio output for {path.name}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioDecodeError(f"empty audio after decoding {path.name}")
    # frombuffer aliases ffmpeg's bytes (read-only); copy so downstream
    # in-place ops and slicing work without surprises.
    return np.array(audio, copy=True)


def _decode_via_librosa(path: Path) -> np.ndarray:
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioDecodeError(
            f"failed to decode '{path.name}'. "
            "WAV/FLAC/OGG are guaranteed; MP3/M4A require ffmpeg on PATH."
        ) from exc

    return audio.astype(np.float32, copy=False)
