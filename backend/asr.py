"""Cohere Transcribe wrapper using the HuggingFace transformers integration.

Hardware acceleration: the model is loaded on the fastest available device
(cuda -> mps -> cpu). Default dtypes:

    cuda → float16
    mps  → float16
    cpu  → float32

About MPS correctness: a small fraction of clips silently transcribe
to the empty string on Apple-Silicon MPS. The diagnostic in
``scripts/debug_empty_chunks.py`` (re-runs each chunk on MPS+fp16 and
CPU+fp32, dumping encoder hidden-state stats and token streams)
pinpointed it:

    [mps/float16] enc.shape=(1, 281, 1280) mean=+nan std=nan max_abs=nan nan=359680 inf=0
    [cpu/float32] enc.shape=(1, 281, 1280) mean=-0.0058 std=0.2441 max_abs=1.9637 nan=0 inf=0

The MPS encoder produces an all-NaN hidden state for these inputs.
After ruling out (in order) attention softmax precision (eager
recovered 0 / 4 chunks), dtype range (bf16 recovered 0 / 4),
conv + BatchNorm instability (Conv/BN-fp32 recovered 0 / 4) and
whole-encoder fp32 (correct but pushed the machine into swap),
a per-step trace inside ``ParakeetEncoderAttention.forward`` found
the actual culprit, which has nothing to do with precision:

    matrix_bd_masked: dtype=torch.float32 max_abs=3.4e38 nan=False inf=True
    scores_with_bd:                       max_abs=3.4e38 nan=False inf=True
    attn_weights:                         max_abs=0.31   nan=True  inf=False

For padded clips the per-row attention mask is all-False at the
padded query positions. The transformers Parakeet port writes
``float("-inf")`` into those rows via
``matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))``
(`modeling_parakeet.py:330`), then runs softmax. Softmax of a row of
all ``-inf`` is mathematically ``0/0`` and PyTorch's MPS softmax
returns ``NaN`` for it (CUDA/CPU happen to return zeros, hence the
divergence). One NaN row gets matmul'd into the next conv block and
contaminates the whole hidden state. The original NeMo reference
implementation that Parakeet was ported from uses ``-10000.0``
precisely to avoid this — see the comment right above the buggy line:

    # here the original codebase uses -10000.0 rather than float("-inf")
    # ...we rather went for a straight-forward approach with float("-inf")

Fix (``_patch_encoder_attention_mask_fix``): monkey-patch every
``ParakeetEncoderAttention`` instance with a near-copy of upstream's
forward, with one constant changed (``-1e4`` instead of ``-inf``).
Everything else stays in fp16, no extra memory, no measurable
perf cost. After the patch, NaN is gone on the four failing chunks
and the encoder hidden states match CPU+fp32 to ~5e-3 max abs.

Env vars:

* ``OFFLINE_TRANSCRIBER_ASR_DTYPE=fp16|bf16|fp32`` — primary dtype
  for the model.
* ``OFFLINE_TRANSCRIBER_ASR_ENCODER_MASK_FIX=auto|on|off`` — the
  default fix above. Default ``auto`` (= on whenever the encoder is
  Parakeet, regardless of device, since this is an upstream
  transformers bug, not just an MPS one). Toggle ``off`` to
  reproduce the buggy baseline for diagnostics.
* ``OFFLINE_TRANSCRIBER_ASR_ENCODER_FP32=on|off`` — heavy escape
  hatch: run the *whole* encoder in fp32. Default ``off``. Kept as
  a safety net for any future MPS-only numerics bug; the mask fix
  alone should be enough.
* ``OFFLINE_TRANSCRIBER_ASR_ENCODER_RETRY=eager|none`` — defensive
  retry: on encoder NaN, re-run with eager attention. Default
  ``eager``. With the mask fix on, this should never trigger.
* ``OFFLINE_TRANSCRIBER_ASR_ENCODER_FALLBACK=cpu|none`` — last-ditch
  CPU encoder fallback. Default ``none`` (slow per chunk, degrades
  MPS state — see profile 17).
* ``OFFLINE_TRANSCRIBER_ASR_RETRY=mps|none`` — extra decoder-side
  retries (``use_cache=False``, then sampling). Default ``none``.
* ``OFFLINE_TRANSCRIBER_ASR_FALLBACK=cpu|none`` — full-CPU retry as
  a last resort. Default ``none``.
* ``OFFLINE_TRANSCRIBER_ASR_FALLBACK_MIN_S=<float>`` — min clip
  length to bother retrying (default ``1.0``).

``PYTORCH_ENABLE_MPS_FALLBACK=1`` (set in ``main.py``) lets unsupported
ops fall back to CPU silently. If moving the model to the chosen device
fails entirely we fall back to CPU/fp32 rather than killing the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from audio_io import SAMPLE_RATE
from vad import Segment


@dataclass(frozen=True)
class Transcription:
    start: float
    end: float
    text: str


def _max_new_tokens_for(duration_s: float) -> int:
    """Bound the ASR decode budget to ~12 tokens per second of audio.

    Whisper-class models emit ~5-7 tokens/s of speech in practice; 12 tok/s
    is a comfortable safety margin. Floor of 32 covers very short clips,
    cap of 320 keeps a single pathological clip from burning unbounded time.
    """
    return max(32, min(320, int(duration_s * 12) + 16))


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_encoder_attention_mask_fix(model) -> int:
    """Replace ``-inf`` with ``-1e4`` in the Parakeet encoder attention mask.

    Root cause (see module docstring for the full diagnosis): for
    padded clips, the per-row attention mask is all-False at padded
    query positions. Transformers' ``ParakeetEncoderAttention.forward``
    fills those rows with ``float("-inf")`` and runs softmax. Softmax
    of a row of all ``-inf`` is mathematically ``0/0``; PyTorch's MPS
    softmax returns ``NaN`` (CUDA / CPU happen to return zeros, hence
    the "MPS-only" appearance), one NaN row gets matmul'd into the
    next conv block, and the entire encoder output is NaN.

    The original NeMo Parakeet implementation uses ``-10000.0`` to
    sidestep this — a value that's still effectively ``-inf`` after
    softmax (``exp(-1e4) ≈ 0``) but stays finite, so all-masked rows
    softmax to a uniform distribution instead of NaN. The transformers
    port "simplified" to ``float("-inf")`` and broke MPS in the
    process. We backport the NeMo behavior.

    Implementation: monkey-patch every ``ParakeetEncoderAttention``
    instance with a near-verbatim copy of upstream's forward, with
    that one constant changed. No precision casts, no extra memory,
    no measurable perf cost. We could in principle keep the upstream
    forward and just patch the constant, but Python doesn't expose a
    string/literal hook into a method body — the cheapest robust
    swap is to replace ``forward`` itself.

    Returns the number of attention modules patched. Zero means we
    couldn't find any ``ParakeetEncoderAttention`` (e.g. transformers
    refactored the class name) — caller logs that and falls back to
    the runtime eager / CPU encoder retries.
    """
    import torch

    try:
        from transformers.models.parakeet.modeling_parakeet import (
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )
    except Exception:
        return 0

    encoder = getattr(model, "get_encoder", lambda: None)()
    if encoder is None:
        return 0

    patched = 0
    for module in encoder.modules():
        if type(module).__name__ != "ParakeetEncoderAttention":
            continue

        def _make_forward(self_module):
            # Mirrors transformers/models/parakeet/modeling_parakeet.py::
            # ParakeetEncoderAttention.forward verbatim, except for the
            # one ``-1e4`` constant noted below. Keep this function in
            # sync with upstream when bumping transformers.
            def forward(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                batch_size, seq_length = input_shape
                hidden_shape = (batch_size, seq_length, -1, self_module.head_dim)

                query_states = (
                    self_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )
                key_states = (
                    self_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )
                value_states = (
                    self_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self_module.config._attn_implementation, eager_attention_forward
                )

                bias_u = self_module.bias_u.view(
                    1, self_module.config.num_attention_heads, 1, self_module.head_dim
                )
                bias_v = self_module.bias_v.view(
                    1, self_module.config.num_attention_heads, 1, self_module.head_dim
                )
                query_states_with_bias_u = query_states + bias_u
                query_states_with_bias_v = query_states + bias_v

                relative_key_states = self_module.relative_k_proj(position_embeddings)
                relative_key_states = relative_key_states.view(
                    batch_size,
                    -1,
                    self_module.config.num_attention_heads,
                    self_module.head_dim,
                )

                matrix_bd = query_states_with_bias_v @ relative_key_states.permute(
                    0, 2, 3, 1
                )
                matrix_bd = self_module._rel_shift(matrix_bd)
                matrix_bd = matrix_bd[..., :seq_length]
                matrix_bd = matrix_bd * self_module.scaling

                if attention_mask is not None:
                    # The single line that differs from upstream: -1e4
                    # instead of float("-inf"). exp(-1e4) ≈ 0 so the
                    # numerical effect on visible-row softmaxes is
                    # indistinguishable; the difference shows up only
                    # when an *entire* row is masked, which is exactly
                    # the case that NaN'd on MPS.
                    matrix_bd = matrix_bd.masked_fill(
                        attention_mask.logical_not(), -1e4
                    )

                attn_output, attn_weights = attention_interface(
                    self_module,
                    query=query_states_with_bias_u,
                    key=key_states,
                    value=value_states,
                    attention_mask=matrix_bd,
                    dropout=0.0
                    if not self_module.training
                    else self_module.attention_dropout,
                    scaling=self_module.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self_module.o_proj(attn_output)
                return attn_output, attn_weights

            return forward

        module.forward = _make_forward(module)  # type: ignore[assignment]
        patched += 1

    return patched


def _encoder_attn_mask_fix_enabled() -> bool:
    """Whether to apply the Parakeet attention-mask fix.

    Default ``auto`` resolves to ``on`` everywhere. The fix is a
    backport of the upstream NeMo behavior the transformers port
    dropped — it's correct on every device, not just MPS, and the
    cost is one method swap at load time. ``off`` exists so we can
    reproduce the buggy baseline for diagnostics.
    """
    setting = os.environ.get(
        "OFFLINE_TRANSCRIBER_ASR_ENCODER_MASK_FIX", "auto"
    ).lower()
    if setting == "off":
        return False
    return True  # "on" or "auto"


def _encoder_fp32_enabled(device: str, dtype) -> bool:
    """Whether to run the *whole* encoder in fp32 (heavy, opt-in).

    Kept as an escape hatch for the rare case where the attention-fp32
    patch is not enough. Default ``off``: ~+1.2 GB MPS memory and a
    ~30-40 % slower transcribe stage; on a 16 GB Mac with other apps
    competing for unified memory, this can push the run into swap.
    """
    del device, dtype  # auto resolves to off; only `on` enables it
    setting = os.environ.get("OFFLINE_TRANSCRIBER_ASR_ENCODER_FP32", "off").lower()
    return setting == "on"


def _patch_encoder_to_fp32(model, *, target_dtype) -> bool:
    """Cast the entire encoder (params + buffers) to fp32 and patch its forward.

    Heavy escape hatch behind ``OFFLINE_TRANSCRIBER_ASR_ENCODER_FP32=on``.
    The default fix is now ``_patch_encoder_attention_mask_fix``; this
    helper is only kept for the case where the mask fix ever leaks a
    NaN somewhere unforeseen.
    """
    import torch

    encoder = getattr(model, "get_encoder", lambda: None)()
    if encoder is None:
        return False

    encoder.float()
    original_forward = encoder.forward

    def patched(input_features, *args, **kwargs):
        if isinstance(input_features, torch.Tensor) and input_features.is_floating_point():
            input_features = input_features.to(torch.float32)
        new_kwargs = {
            k: (v.to(torch.float32) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in kwargs.items()
        }
        out = original_forward(input_features, *args, **new_kwargs)
        if hasattr(out, "last_hidden_state") and isinstance(
            out.last_hidden_state, torch.Tensor
        ):
            out.last_hidden_state = out.last_hidden_state.to(target_dtype)
        return out

    encoder.forward = patched  # type: ignore[assignment]
    return True


def _default_dtype_for(device: str):
    """Pick a dtype for the model on ``device``.

    Honours ``OFFLINE_TRANSCRIBER_ASR_DTYPE=fp16|bf16|fp32`` when set, so
    we can flip dtype without touching code if a future PyTorch fixes (or
    breaks) MPS fp16 numerics.
    """
    import torch

    override = os.environ.get("OFFLINE_TRANSCRIBER_ASR_DTYPE", "").lower()
    if override in ("fp16", "float16", "half"):
        return torch.float16
    if override in ("bf16", "bfloat16"):
        return torch.bfloat16
    if override in ("fp32", "float32", "full"):
        return torch.float32

    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


class TransformersCohereAsr:
    """Lazy-loading wrapper around CohereAsrForConditionalGeneration.

    The model is heavy (~8 GB FP32, ~4 GB FP16) so we only load it once per
    pipeline run. ``device`` / ``dtype`` may be passed explicitly for testing
    or to force a configuration; otherwise they're picked automatically.
    """

    def __init__(
        self,
        model_dir: Path,
        language: str = "en",
        *,
        punctuation: bool = True,
        device: str | None = None,
        dtype: object | None = None,
    ) -> None:
        self.model_dir = model_dir
        # Default language used when callers don't pass one explicitly.
        # The model itself is multilingual, so changing language across
        # calls does NOT require reloading weights; the warm sidecar can
        # serve different languages from one cached instance.
        self.language = language
        self.punctuation = punctuation
        self._processor = None
        self._model = None
        self._device: str | None = None
        self._dtype = None
        self._requested_device = device
        self._requested_dtype = dtype
        # Lazy-loaded CPU+fp16 copy of the model. Used only by the
        # opt-in CPU-encoder fallback and the opt-in full-CPU retry.
        # Stays unloaded on the happy path.
        self._cpu_model = None
        encoder_retry = os.environ.get(
            "OFFLINE_TRANSCRIBER_ASR_ENCODER_RETRY", "eager"
        ).lower()
        self._encoder_eager_retry_enabled = encoder_retry == "eager"
        encoder_fallback = os.environ.get(
            "OFFLINE_TRANSCRIBER_ASR_ENCODER_FALLBACK", "none"
        ).lower()
        self._encoder_cpu_fallback_enabled = encoder_fallback == "cpu"
        fallback = os.environ.get("OFFLINE_TRANSCRIBER_ASR_FALLBACK", "none").lower()
        self._cpu_fallback_enabled = fallback == "cpu"
        mps_retry = os.environ.get("OFFLINE_TRANSCRIBER_ASR_RETRY", "none").lower()
        self._mps_retry_enabled = mps_retry == "mps"
        try:
            self._fallback_min_s = float(
                os.environ.get("OFFLINE_TRANSCRIBER_ASR_FALLBACK_MIN_S", "1.0")
            )
        except ValueError:
            self._fallback_min_s = 1.0
        # Per-run counters surfaced via properties so the pipeline can
        # record them in the run profile.
        self._encoder_eager_attempts = 0
        self._encoder_eager_recovered = 0
        self._encoder_fallback_attempts = 0
        self._encoder_fallback_recovered = 0
        self._mps_retry_attempts = 0
        self._mps_retry_recovered = 0
        self._cpu_retry_attempts = 0
        self._cpu_retry_recovered = 0
        # Set in _load_model. ``encoder_attn_mask_fix`` is the default
        # fix for the encoder NaN bug (mask -inf -> -1e4 backport from
        # NeMo); ``encoder_fp32`` is the heavy escape hatch (whole
        # encoder).
        self._encoder_attn_mask_fix = False
        self._encoder_attn_mask_fix_layers = 0
        self._encoder_fp32 = False

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def encoder_eager_attempts(self) -> int:
        """Chunks where the primary encoder NaN'd and we re-ran with eager attention."""
        return self._encoder_eager_attempts

    @property
    def encoder_eager_recovered(self) -> int:
        """Of those, how many came back non-NaN (chunk then likely transcribed)."""
        return self._encoder_eager_recovered

    @property
    def encoder_fallback_attempts(self) -> int:
        """Chunks where eager retry also failed and we re-ran encoder on CPU."""
        return self._encoder_fallback_attempts

    @property
    def encoder_fallback_recovered(self) -> int:
        """Of those, how many produced non-empty text after the CPU encoder pass."""
        return self._encoder_fallback_recovered

    @property
    def mps_retry_attempts(self) -> int:
        """Number of empty-result chunks retried on the same MPS model."""
        return self._mps_retry_attempts

    @property
    def mps_retry_recovered(self) -> int:
        """Of the MPS retries, how many produced non-empty text."""
        return self._mps_retry_recovered

    @property
    def cpu_retry_attempts(self) -> int:
        """Number of empty-result chunks the full-CPU fallback has retried."""
        return self._cpu_retry_attempts

    @property
    def cpu_retry_recovered(self) -> int:
        """Of those retries, how many produced non-empty text."""
        return self._cpu_retry_recovered

    @property
    def cpu_fallback_loaded(self) -> bool:
        """Whether the lazy CPU model copy has been instantiated."""
        return self._cpu_model is not None

    @property
    def encoder_attn_mask_fix(self) -> bool:
        """True if the Parakeet attention mask fix (-inf -> -1e4) was applied."""
        return self._encoder_attn_mask_fix

    @property
    def encoder_attn_mask_fix_layers(self) -> int:
        """Number of ParakeetEncoderAttention modules patched."""
        return self._encoder_attn_mask_fix_layers

    @property
    def encoder_fp32(self) -> bool:
        """True if the entire encoder was cast to fp32 at load time."""
        return self._encoder_fp32

    def reset_retry_counters(self) -> None:
        """Zero the per-run retry counters before each pipeline run."""
        self._encoder_eager_attempts = 0
        self._encoder_eager_recovered = 0
        self._encoder_fallback_attempts = 0
        self._encoder_fallback_recovered = 0
        self._mps_retry_attempts = 0
        self._mps_retry_recovered = 0
        self._cpu_retry_attempts = 0
        self._cpu_retry_recovered = 0

    def warmup(self, language: str | None = None) -> None:
        """Run a tiny silent clip through ``generate()`` to compile MPS kernels.

        On Apple Silicon the very first ``generate()`` call after model load
        pays a one-time kernel compilation cost (~0.5-1s in practice). Doing
        it here from the background loader hides that cost behind diarize, so
        the first real chunk runs at steady-state RTF instead of 2-3x slower.
        """
        import torch

        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 s of silence
        inputs = self._processor(
            audio=dummy,
            language=language or self.language,
            punctuation=self.punctuation,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_device(inputs)
        with torch.inference_mode():
            self._model.generate(**inputs, max_new_tokens=4, do_sample=False, num_beams=1)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Cohere Transcribe weights not found at {self.model_dir}. "
                "Run `python scripts/download_models.py` first."
            )

        device = self._requested_device or _resolve_device()
        dtype = self._requested_dtype or _default_dtype_for(device)

        self._processor = AutoProcessor.from_pretrained(str(self.model_dir))
        model = CohereAsrForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=dtype,
        )
        try:
            model = model.to(device)
        except Exception:
            # Could not move to the chosen device (rare: MPS may reject some
            # tensor layouts at .to() time). Drop back to CPU / fp32 so the
            # run still completes, just slower.
            device = "cpu"
            dtype = torch.float32
            model = model.to(device, dtype=dtype)
        model.eval()

        self._encoder_attn_mask_fix = False
        self._encoder_attn_mask_fix_layers = 0
        if _encoder_attn_mask_fix_enabled():
            try:
                patched = _patch_encoder_attention_mask_fix(model)
                self._encoder_attn_mask_fix = patched > 0
                self._encoder_attn_mask_fix_layers = patched
            except Exception:
                # Patching is best-effort: if a future transformers refactor
                # renames ParakeetEncoderAttention or moves the attention
                # interface helpers, we silently fall back to the runtime
                # eager / CPU encoder retries.
                self._encoder_attn_mask_fix = False
                self._encoder_attn_mask_fix_layers = 0

        self._encoder_fp32 = False
        if _encoder_fp32_enabled(device, dtype):
            try:
                self._encoder_fp32 = _patch_encoder_to_fp32(
                    model, target_dtype=dtype
                )
            except Exception:
                self._encoder_fp32 = False

        self._model = model
        self._device = device
        self._dtype = dtype

    def transcribe_segments(
        self,
        audio: np.ndarray,
        segments: Iterable[Segment],
        *,
        language: str | None = None,
    ) -> list[Transcription]:
        """Transcribe each VAD segment of ``audio`` independently.

        ``language`` overrides ``self.language`` for this call only. The
        model is multilingual, so switching across calls is free (no
        reload, no re-warm).

        On CUDA, we optionally batch multiple chunks in one ``generate`` call
        (``OFFLINE_TRANSCRIBER_ASR_CUDA_BATCH=on``, default on). This is where
        batching tends to pay off. On MPS/CPU we keep the established sequential
        path that preserves the MPS-specific encoder safety nets.
        """
        self._ensure_loaded()
        lang = language or self.language
        segs = list(segments)
        if (
            self._device == "cuda"
            and os.environ.get("OFFLINE_TRANSCRIBER_ASR_CUDA_BATCH", "on").lower() != "off"
        ):
            return self._transcribe_segments_cuda_batched(audio, segs, language=lang)
        results: list[Transcription] = []
        for seg in segs:
            text = self._generate_one(audio, seg, language=lang)
            if not text:
                text = self._recover_empty(audio, seg, language=lang)
            if text:
                results.append(Transcription(start=seg.start, end=seg.end, text=text))
        return results

    def _transcribe_segments_cuda_batched(
        self, audio: np.ndarray, segments: list[Segment], *, language: str
    ) -> list[Transcription]:
        import torch

        assert self._model is not None and self._processor is not None
        if not segments:
            return []

        batch_size = max(
            1, int(os.environ.get("OFFLINE_TRANSCRIBER_ASR_BATCH_SIZE", "8"))
        )
        out: list[Transcription] = []
        idx = 0
        while idx < len(segments):
            group = segments[idx : idx + batch_size]
            idx += len(group)
            clips: list[np.ndarray] = []
            valid: list[Segment] = []
            for seg in group:
                start_idx = int(seg.start * SAMPLE_RATE)
                end_idx = int(seg.end * SAMPLE_RATE)
                clip = audio[start_idx:end_idx]
                if clip.size == 0:
                    continue
                clips.append(clip)
                valid.append(seg)
            if not clips:
                continue

            try:
                inputs = self._processor(
                    audio=clips,
                    language=language,
                    punctuation=self.punctuation,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                )
                audio_chunk_index = inputs.get("audio_chunk_index")
                inputs = self._move_inputs_to(
                    inputs, device="cuda", target_dtype=self._model.dtype
                )
                max_new_tokens = max(
                    _max_new_tokens_for(clip.size / float(SAMPLE_RATE))
                    for clip in clips
                )
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )
                decoded = self._processor.decode(
                    outputs,
                    skip_special_tokens=True,
                    audio_chunk_index=audio_chunk_index,
                    language=language,
                )
                texts = decoded if isinstance(decoded, list) else [decoded]
                for seg, text in zip(valid, texts):
                    clean = str(text).strip()
                    if not clean:
                        clean = self._recover_empty(audio, seg, language=language)
                    if clean:
                        out.append(
                            Transcription(start=seg.start, end=seg.end, text=clean)
                        )
            except Exception:
                # If batched generation fails for any reason, transparently
                # fall back to the established per-segment path.
                for seg in valid:
                    text = self._generate_one(audio, seg, language=language)
                    if not text:
                        text = self._recover_empty(audio, seg, language=language)
                    if text:
                        out.append(
                            Transcription(start=seg.start, end=seg.end, text=text)
                        )
        return out

    def _recover_empty(
        self, audio: np.ndarray, seg: Segment, *, language: str
    ) -> str:
        """Last-resort recovery for chunks that came back empty.

        The default encoder-on-CPU fallback in ``_generate_using``
        catches the common case (MPS encoder NaNs out). This method is
        only reached for the rare residual: clips where even a healthy
        encoder produces an empty decode (genuinely unintelligible
        audio, or some other obscure failure mode).

        Strategy ladder (all skipped if clip < ``self._fallback_min_s``):

        1. ``mps_retry``: same MPS model with ``use_cache=False``, then
           with light sampling. Off by default — empirically didn't
           recover anything on the one file we measured (profile 16),
           so we don't pay the cost. Toggle with
           ``OFFLINE_TRANSCRIBER_ASR_RETRY=mps``.
        2. ``cpu``: full CPU model retry (encoder + decoder). Off by
           default: ~25-80 s per chunk plus a one-off MPS-recovery stall
           on the next chunk. Toggle with
           ``OFFLINE_TRANSCRIBER_ASR_FALLBACK=cpu``.
        """
        clip_s = max(0.0, seg.end - seg.start)
        if clip_s < self._fallback_min_s:
            return ""

        if self._mps_retry_enabled and self._device != "cpu":
            for kwargs in (
                {"use_cache": False},
                {"do_sample": True, "temperature": 0.2, "top_p": 0.95},
            ):
                self._mps_retry_attempts += 1
                try:
                    text = self._generate_one(
                        audio, seg, language=language, override_kwargs=kwargs
                    )
                except Exception:
                    text = ""
                if text:
                    self._mps_retry_recovered += 1
                    return text

        if self._cpu_fallback_enabled and self._device != "cpu":
            return self._retry_on_cpu(audio, seg, language=language)

        return ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _move_inputs_to(self, inputs, *, device: str, target_dtype):
        """Move processor outputs to ``device`` with the right per-tensor dtype.

        Audio mel features (float) go to ``target_dtype`` (e.g. fp16 on
        MPS/CUDA). Token id / mask tensors must stay integer, so they're
        moved without a dtype change. Doing both with one
        ``.to(device, dtype=...)`` would silently cast int64 token ids to
        fp16 and break generation.
        """
        import torch

        moved = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                moved[k] = v
                continue
            if v.dtype.is_floating_point:
                moved[k] = v.to(device, dtype=target_dtype)
            else:
                moved[k] = v.to(device)
        return moved

    def _move_inputs_to_device(self, inputs):
        """Backwards-compatible helper: move to the primary model's device."""
        assert self._model is not None and self._device is not None
        return self._move_inputs_to(
            inputs, device=self._device, target_dtype=self._model.dtype
        )

    def _generate_one(
        self,
        audio: np.ndarray,
        seg: Segment,
        *,
        language: str,
        override_kwargs: dict | None = None,
    ) -> str:
        assert self._model is not None and self._device is not None
        return self._generate_using(
            audio,
            seg,
            language=language,
            model=self._model,
            device=self._device,
            override_kwargs=override_kwargs,
        )

    def _generate_using(
        self,
        audio: np.ndarray,
        seg: Segment,
        *,
        language: str,
        model,
        device: str,
        override_kwargs: dict | None = None,
    ) -> str:
        import torch

        assert self._processor is not None

        start_idx = int(seg.start * SAMPLE_RATE)
        end_idx = int(seg.end * SAMPLE_RATE)
        clip = audio[start_idx:end_idx]
        if clip.size == 0:
            return ""

        # The upstream ``CohereAsrProcessor.__call__`` (transformers >= 5.7)
        # builds the decoder prompt itself when given ``language=`` /
        # ``punctuation=`` and returns ``input_features`` plus ready-to-use
        # ``decoder_input_ids``. Earlier we called ``generate()`` with no
        # prompt at all, which silently dropped some clips (profile 10
        # lost 4 of 17 chunks). Letting the processor own the prompt also
        # avoids drift if Cohere ever changes the prompt schema.
        inputs = self._processor(
            audio=clip,
            language=language,
            punctuation=self.punctuation,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        inputs = self._move_inputs_to(inputs, device=device, target_dtype=model.dtype)
        decoder_input_ids = inputs.get("decoder_input_ids")
        max_new_tokens = _max_new_tokens_for(clip.size / float(SAMPLE_RATE))

        # Run the encoder explicitly (rather than letting ``generate``
        # do it internally). This costs the same in the happy case but
        # lets us inspect the encoder hidden state and recover from the
        # MPS NaN bug (see class docstring).
        encoder_outputs = self._compute_encoder_outputs(
            inputs, model=model, device=device
        )

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "encoder_outputs": encoder_outputs,
        }
        if override_kwargs:
            gen_kwargs.update(override_kwargs)

        # When ``encoder_outputs`` is supplied, ``input_features`` must
        # be omitted — otherwise ``generate`` re-runs the encoder and
        # ignores ours.
        decoder_inputs = {k: v for k, v in inputs.items() if k != "input_features"}

        with torch.inference_mode():
            outputs = model.generate(**decoder_inputs, **gen_kwargs)

        # ``outputs`` is shape [1, prompt_len + generated_len]. Strip the
        # echoed prompt prefix and stop at the first EOS so trailing
        # pad/EOS tokens don't end up decoded.
        token_ids = outputs[0].tolist()
        if decoder_input_ids is not None:
            prompt_ids = decoder_input_ids[0].tolist()
            prompt_len = len(prompt_ids)
            if len(token_ids) >= prompt_len and token_ids[:prompt_len] == prompt_ids:
                token_ids = token_ids[prompt_len:]

        eos_token_id = self._processor.tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(eos_token_id)]

        text = self._processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        if isinstance(text, list):
            text = " ".join(t.strip() for t in text if t and t.strip())
        return (text or "").strip()

    # ------------------------------------------------------------------
    # Encoder recovery (cheap MPS-side eager retry; opt-in CPU fallback)
    # ------------------------------------------------------------------

    def _run_encoder_once(self, model, inputs):
        """Single encoder forward, returning the BaseModelOutput."""
        import torch

        encoder = model.get_encoder()
        enc_kwargs = {"input_features": inputs["input_features"]}
        if inputs.get("attention_mask") is not None:
            enc_kwargs["attention_mask"] = inputs["attention_mask"]
        with torch.inference_mode():
            return encoder(**enc_kwargs)

    def _compute_encoder_outputs(self, inputs, *, model, device):
        """Run the encoder, recovering from MPS NaN bugs cheaply.

        The MPS encoder occasionally emits an all-NaN hidden state for
        certain inputs — almost certainly an fp16 softmax overflow in
        ``scaled_dot_product_attention``. The recovery ladder is:

        1. **Default**: re-run the encoder on the same MPS model with
           ``attn_implementation`` temporarily flipped to ``eager``.
           Eager attention computes ``softmax(x - x.max())`` with the
           subtraction in higher precision, so it doesn't blow up. No
           extra weights, no CPU↔MPS sync, decoder unaffected.
        2. **Opt-in**: if eager also returns NaN AND
           ``OFFLINE_TRANSCRIBER_ASR_ENCODER_FALLBACK=cpu``, lazy-load
           a CPU copy of the model and run the encoder there. This is
           slow (~5-15 s per chunk) and degrades MPS state for the
           next chunk, so it's off by default.

        Cost on the happy path: one ``isnan().any()`` check per chunk
        (a few microseconds).
        """
        import torch

        enc_out = self._run_encoder_once(model, inputs)
        if not bool(torch.isnan(enc_out.last_hidden_state).any().item()):
            return enc_out

        # Strategy 1: same model, eager attention.
        if self._encoder_eager_retry_enabled:
            self._encoder_eager_attempts += 1
            try:
                with self._temporarily_eager_attention(model):
                    eager_out = self._run_encoder_once(model, inputs)
                if not bool(torch.isnan(eager_out.last_hidden_state).any().item()):
                    self._encoder_eager_recovered += 1
                    return eager_out
            except Exception:
                # Best-effort; fall through to the next strategy.
                pass

        # Strategy 2: opt-in CPU encoder fallback.
        if self._encoder_cpu_fallback_enabled and device != "cpu":
            self._encoder_fallback_attempts += 1
            try:
                self._ensure_cpu_loaded()
                assert self._cpu_model is not None
                cpu_features = inputs["input_features"].to(
                    "cpu", dtype=self._cpu_model.dtype
                )
                cpu_kwargs = {"input_features": cpu_features}
                if inputs.get("attention_mask") is not None:
                    cpu_kwargs["attention_mask"] = inputs["attention_mask"].to("cpu")
                with torch.inference_mode():
                    cpu_enc_out = self._cpu_model.get_encoder()(**cpu_kwargs)
                recovered = cpu_enc_out.last_hidden_state.to(device, dtype=model.dtype)
                if not bool(torch.isnan(recovered).any().item()):
                    enc_out.last_hidden_state = recovered
                    self._encoder_fallback_recovered += 1
            except Exception:
                pass

        return enc_out

    def _temporarily_eager_attention(self, model):
        """Context manager: swap the model's attn implementation to eager.

        On exit, restores the original ``_attn_implementation`` on
        every config that we touched. Cohere ASR sets the flag on each
        sub-config (model, encoder, decoder), so we visit them all.
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            cfgs = []
            roots = [getattr(model, "config", None)]
            for root in list(roots):
                if root is None:
                    continue
                # Walk shallow sub-configs so we cover encoder/decoder
                # /generation_config etc. without going deep.
                for name in vars(root):
                    sub = getattr(root, name, None)
                    if sub is not None and hasattr(sub, "_attn_implementation"):
                        roots.append(sub)
            for cfg in roots:
                if cfg is None:
                    continue
                if hasattr(cfg, "_attn_implementation"):
                    cfgs.append((cfg, cfg._attn_implementation))
                    cfg._attn_implementation = "eager"
            try:
                yield
            finally:
                for cfg, original in cfgs:
                    cfg._attn_implementation = original

        return _ctx()

    # ------------------------------------------------------------------
    # Full-CPU fallback (opt-in last-resort retry)
    # ------------------------------------------------------------------

    def _ensure_cpu_loaded(self) -> None:
        """Lazy-load a CPU+fp16 copy of the model the first time it's needed.

        Used by both the encoder fallback (default on, just one forward
        pass) and the full-CPU retry (opt-in). Loaded on first need so
        users on hardware that never trips the MPS encoder bug — or
        non-MPS hardware — never pay the extra memory or load time.
        """
        if self._cpu_model is not None:
            return
        import torch
        from transformers import CohereAsrForConditionalGeneration

        cpu_dtype = torch.float16
        model = CohereAsrForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=cpu_dtype,
        )
        model = model.to("cpu")
        model.eval()
        self._cpu_model = model

    def _retry_on_cpu(
        self, audio: np.ndarray, seg: Segment, *, language: str
    ) -> str:
        """Retry an empty MPS/CUDA result on the CPU fallback model.

        Returns the recovered text (possibly still empty if the audio
        really is unintelligible). Updates the per-run counters so the
        pipeline can surface them in the profile.
        """
        self._cpu_retry_attempts += 1
        try:
            self._ensure_cpu_loaded()
            assert self._cpu_model is not None
            text = self._generate_using(
                audio,
                seg,
                language=language,
                model=self._cpu_model,
                device="cpu",
            )
        except Exception:
            # Never let a fallback failure kill the whole transcription
            # run — the original empty result will simply be dropped, as
            # it would have been without a fallback.
            return ""
        if text:
            self._cpu_retry_recovered += 1
        return text


# Backward-compatible alias while callers migrate to the backend factory.
CohereAsr = TransformersCohereAsr
