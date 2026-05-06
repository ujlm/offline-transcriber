"""End-to-end pipeline orchestration.

Stages, in order, with the percent each contributes to the overall progress bar:

    decode      0 ->  10
    vad        10 ->  20
    diarize    20 ->  42
    chunk      42 ->  45
    transcribe 45 ->  90
    fusion     90 ->  95
    export     95 -> 100

These weights are coarse and assume CPU. They are good enough for a progress UI;
they are not used for any decision logic.

Every run also writes a JSON profile (``<stem>.profile.json``) next to the
.docx output capturing per-stage wall-clock timings and a copy of every log
message. The profile is written in a ``finally`` block so partial runs that
fail mid-pipeline still leave a profile behind.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import events
from asr_backends import AsrBackend, make_backend
from audio_io import SAMPLE_RATE, load_audio
from chunker import merge_for_asr
from diarize import DiarizePipeline, SpeakerSegment
from export import write_docx, write_txt
from fusion import fuse
from profiling import RunProfile
from vad import detect_speech


class _BackgroundLoader:
    """Run a single ``_ensure_loaded()``-style call in a daemon thread.

    The ASR weight load (~15 s) and the pyannote pipeline load (~1 s,
    plus ~9 s of MPS kernel compilation that happens later in the
    diarize forward pass on a cold sidecar) both happen at well-known
    points in the pipeline. Kicking each off in a background thread at
    run start lets them overlap with decode + vad / each other.

    By the time the consuming stage runs, ``wait()`` typically returns
    instantly and the whole load has been hidden. On a warm sidecar
    (where the underlying object is already loaded) ``_load_fn`` is a
    no-op and ``wait()`` returns ~0 timings.

    Note on MPS contention: an earlier version also called
    ``asr.warmup()`` from this thread to compile MPS kernels. That
    fought diarize for MPS — both slowed measurably, eating more time
    than the warmup saved. We deliberately keep ``_load_fn`` to "load
    weights into device memory" and leave kernel compilation to the
    first real forward pass on each model.
    """

    def __init__(self, name: str, load_fn) -> None:
        self._t_start = time.monotonic()
        self._t_end: float | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(load_fn,), name=name, daemon=True
        )
        self._thread.start()

    def _run(self, load_fn) -> None:
        try:
            load_fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised in wait()
            self._error = exc
        finally:
            self._t_end = time.monotonic()

    def wait(self) -> dict[str, float]:
        """Block until the load finishes; return a timing dict.

        Keys:
            ``load_s`` — wall time the background load actually took.
            ``blocked_s`` — how long the caller had to wait. Close to
                zero means the load was fully hidden behind earlier stages.
            ``overlap_s`` — ``load_s − blocked_s``, i.e. the saving.
        """
        t_wait_start = time.monotonic()
        self._thread.join()
        blocked_s = time.monotonic() - t_wait_start
        if self._error is not None:
            raise self._error
        end = self._t_end if self._t_end is not None else time.monotonic()
        load_s = end - self._t_start
        return {
            "load_s": load_s,
            "blocked_s": blocked_s,
            "overlap_s": max(0.0, load_s - blocked_s),
        }


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    models_dir: Path
    language: str = "en"
    diarization_enabled: bool = True


@dataclass(frozen=True)
class PipelineResult:
    txt_path: Path
    docx_path: Path
    profile_path: Path
    transcript: list[dict]
    diarization_enabled: bool


def run(
    cfg: PipelineConfig,
    *,
    asr: AsrBackend | None = None,
    diarize_pipeline: DiarizePipeline | None = None,
) -> PipelineResult:
    """Run the full pipeline once.

    If ``asr`` and/or ``diarize_pipeline`` are provided (warm-sidecar
    mode), they are reused across runs. A pre-loaded instance makes the
    corresponding background loader's ``wait()`` return instantly:
    ``model_load_s`` / ``diarize_load_s`` will both be ~0 in the profile,
    which is how we measure the warm-sidecar win. Beyond skipping the
    one-time disk load, caching also keeps MPS kernels warm — diarize
    typically gets ~9 s faster on the second run in the same process.

    The ASR model is multilingual, so ``cfg.language`` is honoured per
    run even when reusing a cached ``asr`` whose default language differs.
    Switching language across calls is free (no reload, no re-warm).

    If either argument is ``None`` (CLI / one-shot mode), a fresh
    instance is constructed and loaded on the background thread as before.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.input_path.stem
    txt_path = cfg.output_dir / f"{stem}.txt"
    docx_path = cfg.output_dir / f"{stem}.docx"
    profile_path = cfg.output_dir / f"{stem}.profile.json"

    profile = RunProfile()
    profile.metadata.update(
        {
            "input_path": str(cfg.input_path),
            "output_dir": str(cfg.output_dir),
            "models_dir": str(cfg.models_dir),
            "language": cfg.language,
            "diarization_enabled": cfg.diarization_enabled,
            "sample_rate": SAMPLE_RATE,
        }
    )
    events.set_profile(profile)

    try:
        # Kick off both the ASR weight load and the pyannote pipeline
        # construction in background threads so they overlap with
        # decode + vad. Each consuming stage will join() its loader;
        # when the warm sidecar passes us pre-loaded instances, those
        # joins return instantly.
        if asr is None:
            cohere_dir = cfg.models_dir / "cohere-transcribe-03-2026"
            asr = make_backend("transformers", model_dir=cohere_dir, language=cfg.language)
        if cfg.diarization_enabled and diarize_pipeline is None:
            diarize_pipeline = DiarizePipeline(cfg.models_dir)
        # ``device`` is set by ``_ensure_loaded`` and is ``None`` until
        # the underlying model is in memory. Use it as a more honest
        # "is the cache warm?" signal than just "was an instance passed in".
        asr_warm = asr.device is not None
        diarize_warm = (
            diarize_pipeline is not None and diarize_pipeline.device is not None
        )
        profile.metadata["asr_warm"] = asr_warm
        profile.metadata["diarize_warm"] = diarize_warm
        if asr_warm:
            events.log("Reusing warm ASR weights from sidecar cache")
        else:
            events.log("Loading Cohere Transcribe weights (background)")
        if cfg.diarization_enabled:
            if diarize_warm:
                events.log("Reusing warm pyannote pipeline from sidecar cache")
            else:
                events.log("Loading pyannote pipeline (background)")
        else:
            events.log("Diarization disabled; using a single speaker track")
        asr_loader = _BackgroundLoader("asr-loader", asr._ensure_loaded)
        diarize_loader = (
            _BackgroundLoader("diarize-loader", diarize_pipeline._ensure_loaded)
            if diarize_pipeline is not None
            else None
        )

        with profile.stage("decode") as stage_decode:
            events.progress("decode", 0)
            events.log(f"Loading audio: {cfg.input_path.name}")
            decoded = load_audio(cfg.input_path)
            audio = decoded.samples
            duration_s = len(audio) / float(SAMPLE_RATE)
            stage_decode.details["audio_duration_s"] = round(duration_s, 4)
            stage_decode.details["audio_samples"] = int(len(audio))
            stage_decode.details["decode_method"] = decoded.method
            profile.metadata["audio_duration_s"] = round(duration_s, 4)
            events.log(f"Decoded {duration_s:.1f}s of audio via {decoded.method}")
            events.progress("decode", 10)

        with profile.stage("vad") as stage_vad:
            events.progress("vad", 12)
            events.log("Running Silero VAD")
            speech = detect_speech(audio)
            speech_total_s = sum(seg.end - seg.start for seg in speech)
            stage_vad.details["segments"] = len(speech)
            stage_vad.details["speech_seconds"] = round(speech_total_s, 4)
            events.log(f"VAD found {len(speech)} speech segments")
            events.progress("vad", 20)

        if cfg.diarization_enabled:
            assert diarize_loader is not None
            assert diarize_pipeline is not None
            with profile.stage("diarize") as stage_diarize:
                events.progress("diarize", 22)
                events.log("Running Pyannote 3.1 diarization")
                diarize_timings = diarize_loader.wait()
                stage_diarize.details["pipeline_load_s"] = round(
                    diarize_timings["load_s"], 4
                )
                stage_diarize.details["pipeline_load_wait_s"] = round(
                    diarize_timings["blocked_s"], 4
                )
                stage_diarize.details["pipeline_load_overlap_s"] = round(
                    diarize_timings["overlap_s"], 4
                )
                stage_diarize.details["pipeline_warm"] = diarize_warm
                diarization = diarize_pipeline.diarize(audio)
                speakers = diarization.segments
                distinct = sorted({sp.speaker for sp in speakers})
                stage_diarize.details["device"] = diarization.device
                stage_diarize.details["min_speakers"] = diarization.min_speakers
                stage_diarize.details["max_speakers"] = diarization.max_speakers
                stage_diarize.details["speaker_segments"] = len(speakers)
                stage_diarize.details["speakers"] = distinct
                events.log(
                    f"Pyannote pipeline ready: load took "
                    f"{diarize_timings['load_s']:.1f}s, "
                    f"{diarize_timings['overlap_s']:.1f}s hidden behind earlier stages, "
                    f"{diarize_timings['blocked_s']:.1f}s blocked diarize"
                )
                events.log(f"Diarization device: {diarization.device}")
                events.log(
                    f"Diarization found {len(distinct)} speaker(s): "
                    f"{', '.join(distinct) or 'none'}"
                )
                events.progress("diarize", 42)
        else:
            with profile.stage("diarize") as stage_diarize:
                events.progress("diarize", 22)
                speakers = [SpeakerSegment(start=0.0, end=duration_s, speaker="Speaker")]
                stage_diarize.details["skipped"] = True
                stage_diarize.details["speaker_segments"] = len(speakers)
                stage_diarize.details["speakers"] = ["Speaker"]
                events.progress("diarize", 42)

        with profile.stage("chunk") as stage_chunk:
            events.progress("chunk", 43)
            try:
                max_chunk_s = float(os.environ.get("OFFLINE_TRANSCRIBER_MAX_CHUNK_S", "24.0"))
            except ValueError:
                max_chunk_s = 24.0
            try:
                max_gap_s = float(os.environ.get("OFFLINE_TRANSCRIBER_MAX_GAP_S", "1.0"))
            except ValueError:
                max_gap_s = 1.0
            try:
                min_isolated_chunk_s = float(
                    os.environ.get("OFFLINE_TRANSCRIBER_MIN_ISOLATED_CHUNK_S", "0.5")
                )
            except ValueError:
                min_isolated_chunk_s = 0.5
            chunks = merge_for_asr(
                speech,
                speakers,
                max_chunk_s=max_chunk_s,
                max_gap_s=max_gap_s,
                min_isolated_chunk_s=min_isolated_chunk_s,
            )
            chunk_durations = [c.segment.end - c.segment.start for c in chunks]
            stage_chunk.details["max_chunk_s"] = max_chunk_s
            stage_chunk.details["max_gap_s"] = max_gap_s
            stage_chunk.details["min_isolated_chunk_s"] = min_isolated_chunk_s
            stage_chunk.details["input_segments"] = len(speech)
            stage_chunk.details["output_chunks"] = len(chunks)
            stage_chunk.details["avg_chunk_s"] = (
                round(sum(chunk_durations) / len(chunk_durations), 4)
                if chunk_durations
                else 0.0
            )
            stage_chunk.details["max_chunk_duration_s"] = (
                round(max(chunk_durations), 4) if chunk_durations else 0.0
            )
            events.log(
                f"Chunker packed {len(speech)} VAD segments into {len(chunks)} ASR chunks "
                f"(<= {max_chunk_s:.0f}s, gap <= {max_gap_s:.1f}s)"
            )
            events.progress("chunk", 45)

        with profile.stage("transcribe") as stage_transcribe:
            events.progress("transcribe", 45)
            timings = asr_loader.wait()
            stage_transcribe.details["model_load_s"] = round(timings["load_s"], 4)
            stage_transcribe.details["model_load_wait_s"] = round(timings["blocked_s"], 4)
            stage_transcribe.details["model_load_overlap_s"] = round(timings["overlap_s"], 4)
            stage_transcribe.details["device"] = asr.device
            stage_transcribe.details["dtype"] = (
                str(asr.dtype).removeprefix("torch.") if asr.dtype is not None else None
            )
            stage_transcribe.details["encoder_attn_mask_fix"] = (
                asr.encoder_attn_mask_fix
            )
            stage_transcribe.details["encoder_attn_mask_fix_layers"] = (
                asr.encoder_attn_mask_fix_layers
            )
            stage_transcribe.details["encoder_fp32"] = asr.encoder_fp32
            events.log(
                f"ASR weights ready: load took {timings['load_s']:.1f}s, "
                f"{timings['overlap_s']:.1f}s hidden behind earlier stages, "
                f"{timings['blocked_s']:.1f}s blocked transcribe"
            )
            if asr.encoder_fp32:
                stem_note = " (whole encoder in fp32)"
            elif asr.encoder_attn_mask_fix:
                stem_note = (
                    f" (Parakeet attn mask -inf->-1e4 patched: "
                    f"{asr.encoder_attn_mask_fix_layers} layers)"
                )
            else:
                stem_note = ""
            events.log(
                f"ASR device: {asr.device} ({stage_transcribe.details['dtype']}){stem_note}"
            )
            stage_transcribe.details["language"] = cfg.language
            events.log(
                f"Transcribing {len(chunks)} ASR chunks (language={cfg.language})"
            )

            # Reset the ASR's CPU-fallback counters so the per-run profile
            # only reflects this run, not the cumulative total since the
            # warm sidecar started.
            asr.reset_retry_counters()

            transcripts: list = []
            per_chunk: list[dict] = []
            audio_seconds_processed = 0.0
            span = 90 - 45

            for i, chunk in enumerate(chunks, start=1):
                seg = chunk.segment
                seg_duration = seg.end - seg.start
                eager_attempts_before = asr.encoder_eager_attempts
                eager_recovered_before = asr.encoder_eager_recovered
                enc_attempts_before = asr.encoder_fallback_attempts
                enc_recovered_before = asr.encoder_fallback_recovered
                mps_attempts_before = asr.mps_retry_attempts
                mps_recovered_before = asr.mps_retry_recovered
                cpu_attempts_before = asr.cpu_retry_attempts
                cpu_recovered_before = asr.cpu_retry_recovered
                t0 = time.monotonic()
                texts = asr.transcribe_segments(audio, [seg], language=cfg.language)
                wall_s = time.monotonic() - t0
                produced = len(texts)
                transcripts.extend(texts)
                audio_seconds_processed += seg_duration
                eager_retried = asr.encoder_eager_attempts > eager_attempts_before
                eager_recovered = (
                    asr.encoder_eager_recovered > eager_recovered_before
                )
                cpu_enc_retried = asr.encoder_fallback_attempts > enc_attempts_before
                cpu_enc_recovered = (
                    asr.encoder_fallback_recovered > enc_recovered_before
                )
                mps_retried = asr.mps_retry_attempts > mps_attempts_before
                mps_recovered = asr.mps_retry_recovered > mps_recovered_before
                cpu_retried = asr.cpu_retry_attempts > cpu_attempts_before
                cpu_recovered = asr.cpu_retry_recovered > cpu_recovered_before
                per_chunk.append(
                    {
                        "index": i,
                        "start": round(seg.start, 4),
                        "end": round(seg.end, 4),
                        "audio_s": round(seg_duration, 4),
                        "wall_s": round(wall_s, 4),
                        "source_segments": chunk.source_segments,
                        "speaker": chunk.speaker,
                        "transcribed": produced > 0,
                        "eager_retried": eager_retried,
                        "eager_recovered": eager_recovered,
                        "cpu_encoder_retried": cpu_enc_retried,
                        "cpu_encoder_recovered": cpu_enc_recovered,
                        "mps_retried": mps_retried,
                        "mps_recovered": mps_recovered,
                        "cpu_retried": cpu_retried,
                        "cpu_recovered": cpu_recovered,
                        "rtf": round(wall_s / seg_duration, 4) if seg_duration > 0 else None,
                    }
                )
                events.progress(
                    "transcribe", 45 + (i / max(1, len(chunks))) * span
                )

            stage_transcribe.details["chunks"] = len(chunks)
            stage_transcribe.details["audio_seconds_processed"] = round(
                audio_seconds_processed, 4
            )
            stage_transcribe.details["transcribed_segments"] = len(transcripts)
            stage_transcribe.details["encoder_eager_attempts"] = (
                asr.encoder_eager_attempts
            )
            stage_transcribe.details["encoder_eager_recovered"] = (
                asr.encoder_eager_recovered
            )
            stage_transcribe.details["encoder_cpu_attempts"] = (
                asr.encoder_fallback_attempts
            )
            stage_transcribe.details["encoder_cpu_recovered"] = (
                asr.encoder_fallback_recovered
            )
            stage_transcribe.details["mps_retry_attempts"] = asr.mps_retry_attempts
            stage_transcribe.details["mps_retry_recovered"] = asr.mps_retry_recovered
            stage_transcribe.details["cpu_retry_attempts"] = asr.cpu_retry_attempts
            stage_transcribe.details["cpu_retry_recovered"] = asr.cpu_retry_recovered
            stage_transcribe.details["cpu_fallback_loaded"] = asr.cpu_fallback_loaded
            stage_transcribe.details["per_chunk"] = per_chunk
            if asr.encoder_eager_attempts:
                events.log(
                    f"Encoder eager retry ran on "
                    f"{asr.encoder_eager_attempts} chunk(s), "
                    f"recovered {asr.encoder_eager_recovered}"
                )
            if asr.encoder_fallback_attempts:
                events.log(
                    f"Encoder CPU fallback ran on "
                    f"{asr.encoder_fallback_attempts} chunk(s), "
                    f"recovered {asr.encoder_fallback_recovered}"
                )
            if asr.mps_retry_attempts:
                events.log(
                    f"MPS retry attempted on {asr.mps_retry_attempts} chunk(s), "
                    f"recovered {asr.mps_retry_recovered}"
                )
            if asr.cpu_retry_attempts:
                events.log(
                    f"Full-CPU fallback retried {asr.cpu_retry_attempts} chunk(s), "
                    f"recovered {asr.cpu_retry_recovered}"
                )
            events.log(f"Produced {len(transcripts)} transcribed segments")

        with profile.stage("fusion") as stage_fusion:
            events.progress("fusion", 92)
            events.log("Fusing speakers and transcripts")
            turns = fuse(transcripts, speakers)
            stage_fusion.details["turns"] = len(turns)
            events.log(f"Merged into {len(turns)} speaker turns")
            events.progress("fusion", 95)

        with profile.stage("export") as stage_export:
            events.progress("export", 96)
            events.log(f"Writing {txt_path.name}")
            write_txt(turns, txt_path)
            events.log(f"Writing {docx_path.name}")
            write_docx(turns, docx_path, title=cfg.input_path.stem)
            stage_export.details["txt_path"] = str(txt_path)
            stage_export.details["docx_path"] = str(docx_path)
            events.progress("export", 100)

        transcript = [
            {
                "start": round(turn.start, 4),
                "end": round(turn.end, 4),
                "speaker": turn.speaker,
                "text": turn.text,
            }
            for turn in turns
        ]

        # Real-time factor for the whole run, useful at a glance.
        if profile.metadata.get("audio_duration_s"):
            profile.metadata["realtime_factor"] = round(
                profile.now() / profile.metadata["audio_duration_s"], 4
            )

        return PipelineResult(
            txt_path=txt_path,
            docx_path=docx_path,
            profile_path=profile_path,
            transcript=transcript,
            diarization_enabled=cfg.diarization_enabled,
        )
    finally:
        try:
            profile.write_json(profile_path)
            events.log(f"Wrote profile to {profile_path.name}")
        except Exception as exc:
            events.log(f"Failed to write profile JSON: {exc}", level="warning")
        events.set_profile(None)
