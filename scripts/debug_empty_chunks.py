"""Reproduce and inspect the empty-output chunks from rhcp.profile1{2,5,6}.

Runs each target chunk twice: once on the primary device (auto-picked,
typically MPS), once on CPU+fp32 as a known-good reference. For every
chunk it prints

* clip stats (rms, peak, sample count)
* encoder ``last_hidden_state`` stats (mean / std / max-abs / NaN/Inf
  counts) so we can tell whether MPS and CPU agree on the encoder
  output
* generation stats: number of tokens before EOS, the first 30 token
  ids, and the decoded text both with and without special tokens
* a side-by-side comparison block: cosine similarity + mean-abs-diff
  between the two encoder hidden states, and a diff of the generated
  token sequences

For every chunk on the primary device whose encoder output is NaN, it
*also* runs a layer-by-layer trace: forward-hooks every leaf module in
the encoder, records each one's input/output max-abs and NaN count,
and prints the **first** module whose output goes NaN. That answers
"which op is the actual source of the NaN?" in one shot, instead of
us guessing at suspect categories one at a time. Set
``DEBUG_TRACE_NAN=0`` to skip the trace pass.

The CPU pass is skipped when the primary device is already CPU. Set
``DEBUG_COMPARE_CPU=0`` to skip it explicitly. Set
``DEBUG_TARGETS=all|empty`` (default ``empty``) to control which
chunks the CPU pass re-runs.

Run as:

    .venv/bin/python scripts/debug_empty_chunks.py path/to/rhcp.mp3
                                                   [path/to/models]

Also writes the extracted chunks as 16 kHz mono WAVs into
``/tmp/asr_debug_chunks/`` so they can be inspected by ear.
"""

from __future__ import annotations

import os
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np  # noqa: E402

from audio_io import SAMPLE_RATE, load_audio  # noqa: E402

# Chunks 3, 4, 15, 17 from the rhcp profiles + a control that always works.
# (start_s, end_s, label)
TARGETS: list[tuple[float, float, str]] = [
    (27.8, 50.2, "chunk03_long_speaker01"),
    (51.1, 60.7, "chunk04_single_segment"),
    (187.8, 188.2, "chunk15_tiny_likely_silence"),
    (208.9, 229.3, "chunk17_tail_speaker02"),
    # control: chunk 5 worked in every profile, similar size to chunk 3
    (61.0, 84.7, "chunk05_control_works"),
]


@dataclass
class ChunkResult:
    """Per-chunk numbers gathered from one device pass."""

    label: str
    device: str
    dtype: str
    encoder_shape: tuple = ()
    encoder_mean: float = float("nan")
    encoder_std: float = float("nan")
    encoder_max_abs: float = float("nan")
    encoder_nan: int = 0
    encoder_inf: int = 0
    encoder_hidden_cpu_f32 = None  # torch.Tensor | None — kept for compare
    prompt_len: int = 0
    gen_len_pre_eos: int = 0
    hit_eos: bool = False
    first_30_gen_ids: list[int] = field(default_factory=list)
    text_skip: str = ""
    text_keep: str = ""


def _save_wav(out: Path, samples: np.ndarray) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _resolve_dtype(env_value: str):
    import torch

    v = (env_value or "").lower()
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("fp32", "float32", "full"):
        return torch.float32
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None


def _move_inputs_to(inputs, *, device, target_dtype):
    """Move processor outputs to ``device`` keeping integer tensors integer."""
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


def inspect_chunk(asr, audio: np.ndarray, target: tuple[float, float, str]) -> ChunkResult:
    """Run encoder + generate on one chunk and gather diagnostics."""
    import torch

    start, end, label = target
    device = asr.device or "cpu"
    dtype = str(asr.dtype).removeprefix("torch.") if asr.dtype is not None else "?"
    res = ChunkResult(label=label, device=device, dtype=dtype)

    s_idx = int(start * SAMPLE_RATE)
    e_idx = int(end * SAMPLE_RATE)
    clip = audio[s_idx:e_idx]
    if clip.size == 0:
        return res

    processor = asr._processor
    model = asr._model
    assert processor is not None and model is not None

    inputs = processor(
        audio=clip,
        language="en",
        punctuation=True,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    moved = _move_inputs_to(inputs, device=device, target_dtype=model.dtype)
    decoder_input_ids = moved.get("decoder_input_ids")
    res.prompt_len = (
        decoder_input_ids.shape[1] if decoder_input_ids is not None else 0
    )

    # ---- Encoder forward (separate from generate so we can inspect) ----
    encoder = model.get_encoder()
    enc_kwargs = {"input_features": moved["input_features"]}
    if "attention_mask" in moved:
        enc_kwargs["attention_mask"] = moved["attention_mask"]
    with torch.inference_mode():
        enc_out = encoder(**enc_kwargs)
    hidden = enc_out.last_hidden_state
    hidden_cpu_f32 = hidden.detach().to("cpu", dtype=torch.float32)
    res.encoder_shape = tuple(hidden.shape)
    res.encoder_mean = float(hidden_cpu_f32.mean().item())
    res.encoder_std = float(hidden_cpu_f32.std().item())
    res.encoder_max_abs = float(hidden_cpu_f32.abs().max().item())
    res.encoder_nan = int(torch.isnan(hidden_cpu_f32).sum().item())
    res.encoder_inf = int(torch.isinf(hidden_cpu_f32).sum().item())
    res.encoder_hidden_cpu_f32 = hidden_cpu_f32

    # ---- Greedy generate ----
    with torch.inference_mode():
        outputs = model.generate(
            **moved,
            max_new_tokens=320,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    out_ids = outputs[0].tolist()
    if (
        res.prompt_len
        and len(out_ids) >= res.prompt_len
        and decoder_input_ids is not None
        and out_ids[: res.prompt_len] == decoder_input_ids[0].tolist()
    ):
        gen_ids = out_ids[res.prompt_len :]
    else:
        gen_ids = out_ids

    eos_id = processor.tokenizer.eos_token_id
    res.hit_eos = eos_id is not None and eos_id in gen_ids
    stop_at = gen_ids.index(eos_id) if res.hit_eos else len(gen_ids)
    gen_pre_eos = gen_ids[:stop_at]
    res.gen_len_pre_eos = len(gen_pre_eos)
    res.first_30_gen_ids = gen_pre_eos[:30]
    res.text_skip = processor.tokenizer.decode(gen_pre_eos, skip_special_tokens=True)
    if isinstance(res.text_skip, list):
        res.text_skip = " ".join(t.strip() for t in res.text_skip if t and t.strip())
    res.text_keep = processor.tokenizer.decode(gen_pre_eos, skip_special_tokens=False)
    if isinstance(res.text_keep, list):
        res.text_keep = " ".join(t.strip() for t in res.text_keep if t and t.strip())
    return res


def _trace_encoder_first_nan(asr, audio: np.ndarray, target: tuple[float, float, str]):
    """Run the encoder with hooks on every leaf submodule; return per-module
    records ``(qualname, type, in_max_abs, in_nan, out_max_abs, out_nan)`` in
    *execution order*. The first record where ``out_nan and not in_nan`` is
    the source of the NaN.
    """
    import torch

    start, end, _label = target
    s_idx = int(start * SAMPLE_RATE)
    e_idx = int(end * SAMPLE_RATE)
    clip = audio[s_idx:e_idx]
    if clip.size == 0:
        return [], None

    processor = asr._processor
    model = asr._model
    assert processor is not None and model is not None

    inputs = processor(
        audio=clip,
        language="en",
        punctuation=True,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    moved = _move_inputs_to(inputs, device=asr.device or "cpu", target_dtype=model.dtype)

    # Stats on the encoder's input itself, in case the cast to fp16
    # already produced inf/NaN (ruling that out is two lines).
    feat = moved["input_features"]
    feat_cpu = feat.detach().to("cpu", dtype=torch.float32)
    feat_stats = {
        "shape": tuple(feat.shape),
        "dtype": str(feat.dtype).removeprefix("torch."),
        "max_abs": float(feat_cpu.abs().max().item()),
        "nan": int(torch.isnan(feat_cpu).sum().item()),
        "inf": int(torch.isinf(feat_cpu).sum().item()),
    }

    encoder = model.get_encoder()
    records: list[dict] = []

    def _flatten_floats(obj):
        """Pull every floating-point tensor out of a hook arg / output."""
        if isinstance(obj, torch.Tensor):
            return [obj] if obj.is_floating_point() else []
        if isinstance(obj, (list, tuple)):
            out = []
            for item in obj:
                out.extend(_flatten_floats(item))
            return out
        # ModelOutput, BaseModelOutput, etc. behave like dicts
        if hasattr(obj, "values") and callable(obj.values):
            out = []
            try:
                for item in obj.values():
                    out.extend(_flatten_floats(item))
            except Exception:
                pass
            return out
        return []

    def _stats(tensors):
        if not tensors:
            return float("nan"), False
        max_abs = 0.0
        any_nan = False
        for t in tensors:
            t = t.detach()
            if torch.isnan(t).any().item():
                any_nan = True
            # Use float() on a 0-dim cpu tensor so we don't materialise the
            # whole tensor for very large activations.
            local_max = t.abs().max().to("cpu", dtype=torch.float32).item()
            if local_max > max_abs and local_max == local_max:  # filter NaN
                max_abs = local_max
        return max_abs, any_nan

    def make_hook(qualname: str, module):
        type_name = type(module).__name__

        def hook(_m, args, kwargs, output):
            in_tensors = _flatten_floats(args) + _flatten_floats(kwargs)
            out_tensors = _flatten_floats(output)
            in_max, in_nan = _stats(in_tensors)
            out_max, out_nan = _stats(out_tensors)
            records.append(
                {
                    "name": qualname,
                    "type": type_name,
                    "in_max": in_max,
                    "in_nan": in_nan,
                    "out_max": out_max,
                    "out_nan": out_nan,
                }
            )

        return hook

    handles = []
    for name, module in encoder.named_modules():
        if name == "":
            continue
        if list(module.children()):
            continue  # only hook leaves
        handles.append(
            module.register_forward_hook(make_hook(name, module), with_kwargs=True)
        )

    enc_kwargs = {"input_features": moved["input_features"]}
    if "attention_mask" in moved:
        enc_kwargs["attention_mask"] = moved["attention_mask"]

    try:
        with torch.inference_mode():
            encoder(**enc_kwargs)
    finally:
        for h in handles:
            h.remove()

    return records, feat_stats


def _print_trace(records: list[dict], feat_stats: dict | None, label: str) -> None:
    print(f"    --- NaN trace for {label} ---")
    if feat_stats is not None:
        print(
            f"    input_features {feat_stats['dtype']} shape={feat_stats['shape']} "
            f"max_abs={feat_stats['max_abs']:.4f} nan={feat_stats['nan']} "
            f"inf={feat_stats['inf']}"
        )
    if not records:
        print("    (no leaf-module records — hook list was empty)")
        return

    nan_count = sum(1 for r in records if r["out_nan"])
    print(
        f"    leaves hooked: {len(records)} | leaves with NaN output: {nan_count}"
    )

    first_source = next(
        (r for r in records if r["out_nan"] and not r["in_nan"]), None
    )
    first_any = next((r for r in records if r["out_nan"]), None)

    if first_source is None and first_any is None:
        print("    no NaN observed — this chunk's encoder ran clean")
        return

    if first_source is not None:
        idx = records.index(first_source)
        print(
            f"    >>> first source of NaN at hook #{idx}: "
            f"{first_source['name']} ({first_source['type']})"
        )
        print(
            f"        input  max_abs={first_source['in_max']:.4f} (clean), "
            f"output max_abs={first_source['out_max']} nan=True"
        )
        # Print the 5 leaves immediately preceding it for chain context.
        ctx_start = max(0, idx - 5)
        print("    context (preceding leaves):")
        for r in records[ctx_start:idx]:
            print(
                f"        - {r['name']:50s} {r['type']:20s} "
                f"in={r['in_max']:.4f} out={r['out_max']:.4f} "
                f"in_nan={r['in_nan']} out_nan={r['out_nan']}"
            )
    else:
        # NaN appeared, but the first occurrence's input was already NaN —
        # the source is upstream of any hooked leaf (a functional op like
        # nn.functional.dropout / glu, or a scalar mul in the encoder body).
        idx = records.index(first_any)
        print(
            f"    !!! first NaN-out leaf is hook #{idx}: {first_any['name']} "
            f"({first_any['type']}) — but its INPUT is already NaN."
        )
        print(
            f"        input nan=True (max_abs={first_any['in_max']}), "
            f"output nan=True (max_abs={first_any['out_max']})"
        )
        print(
            "        => source is upstream of every hooked leaf — likely a "
            "functional op (dropout/glu/scalar mul) in the encoder body, "
            "or input_features themselves."
        )


def _print_chunk_result(res: ChunkResult) -> None:
    print(
        f"    [{res.device}/{res.dtype}] enc.shape={res.encoder_shape} "
        f"mean={res.encoder_mean:+.4f} std={res.encoder_std:.4f} "
        f"max_abs={res.encoder_max_abs:.4f} nan={res.encoder_nan} inf={res.encoder_inf}"
    )
    print(
        f"    [{res.device}/{res.dtype}] prompt_len={res.prompt_len} "
        f"gen_len_pre_eos={res.gen_len_pre_eos} hit_eos={res.hit_eos}"
    )
    print(f"    [{res.device}/{res.dtype}] first 30 gen ids: {res.first_30_gen_ids}")
    print(f"    [{res.device}/{res.dtype}] decoded skip=True : {res.text_skip!r}")
    print(f"    [{res.device}/{res.dtype}] decoded skip=False: {res.text_keep!r}")


def _print_compare(primary: ChunkResult, cpu: ChunkResult) -> None:
    """Side-by-side encoder + token-stream comparison."""
    import torch

    print(f"    --- compare {primary.device}/{primary.dtype} vs {cpu.device}/{cpu.dtype} ---")

    if (
        primary.encoder_hidden_cpu_f32 is not None
        and cpu.encoder_hidden_cpu_f32 is not None
        and primary.encoder_hidden_cpu_f32.shape == cpu.encoder_hidden_cpu_f32.shape
    ):
        a = primary.encoder_hidden_cpu_f32.flatten()
        b = cpu.encoder_hidden_cpu_f32.flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        mae = (a - b).abs().mean().item()
        max_abs_diff = (a - b).abs().max().item()
        rel = mae / max(1e-9, b.abs().mean().item())
        print(
            f"    encoder cos_sim={cos:.6f} mae={mae:.5f} max_abs_diff={max_abs_diff:.5f} "
            f"mae/|b|.mean={rel:.4f}"
        )
    else:
        print("    encoder shapes differ or missing — cannot compare")

    if primary.first_30_gen_ids == cpu.first_30_gen_ids:
        print("    first 30 token ids: IDENTICAL")
    else:
        # Find first divergence position
        common = 0
        for x, y in zip(primary.first_30_gen_ids, cpu.first_30_gen_ids):
            if x == y:
                common += 1
            else:
                break
        print(
            f"    first 30 token ids: diverge at pos {common} "
            f"(primary[{common}]={primary.first_30_gen_ids[common] if common < len(primary.first_30_gen_ids) else None} "
            f"vs cpu[{common}]={cpu.first_30_gen_ids[common] if common < len(cpu.first_30_gen_ids) else None})"
        )
    print(
        f"    gen_len_pre_eos: primary={primary.gen_len_pre_eos} cpu={cpu.gen_len_pre_eos} "
        f"hit_eos: primary={primary.hit_eos} cpu={cpu.hit_eos}"
    )


def _build_asr(cohere_dir: Path, *, device: str | None, dtype):
    from asr import CohereAsr

    asr = CohereAsr(cohere_dir, language="en", device=device, dtype=dtype)
    asr._ensure_loaded()
    return asr


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio_path = Path(sys.argv[1])
    models_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
    cohere_dir = models_dir / "cohere-transcribe-03-2026"

    out_dir = Path("/tmp/asr_debug_chunks")
    print(f"Loading audio {audio_path} ...")
    audio = load_audio(audio_path).samples
    duration = len(audio) / float(SAMPLE_RATE)
    print(f"  duration: {duration:.1f}s, samples: {len(audio)}")

    forced_dtype = _resolve_dtype(os.environ.get("DEBUG_DTYPE", ""))
    forced_device = os.environ.get("DEBUG_DEVICE")  # "cpu" / "mps" / unset

    print("\nLoading primary Cohere ASR ...")
    primary = _build_asr(cohere_dir, device=forced_device, dtype=forced_dtype)
    print(f"  primary device={primary.device}, dtype={primary.dtype}")
    eos_id = primary._processor.tokenizer.eos_token_id
    pad_id = primary._processor.tokenizer.pad_token_id
    print(f"  eos_token_id={eos_id}, pad_token_id={pad_id}")

    import torch

    primary_results: dict[str, ChunkResult] = {}
    print("\n=== Primary device pass ===\n")
    for target in TARGETS:
        start, end, label = target
        s_idx = int(start * SAMPLE_RATE)
        e_idx = int(end * SAMPLE_RATE)
        clip = audio[s_idx:e_idx]
        clip_dur = clip.size / float(SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
        peak = float(np.max(np.abs(clip))) if clip.size else 0.0
        print(f"--- {label}  start={start:.1f}s end={end:.1f}s dur={clip_dur:.2f}s")
        print(f"    rms={rms:.4f} peak={peak:.4f} samples={clip.size}")

        wav_path = out_dir / f"{label}.wav"
        _save_wav(wav_path, clip)
        print(f"    wrote {wav_path}")

        if clip.size == 0:
            print("    SKIP (empty clip)")
            continue

        res = inspect_chunk(primary, audio, target)
        primary_results[label] = res
        _print_chunk_result(res)

        # Auto-trace: if this chunk's encoder NaN'd, hook every leaf module
        # and report the first one that produced NaN. That answers "what's
        # the source op?" without us having to bisect categories by hand.
        if (
            os.environ.get("DEBUG_TRACE_NAN", "1") != "0"
            and (res.encoder_nan > 0 or not res.text_skip)
        ):
            try:
                records, feat_stats = _trace_encoder_first_nan(
                    primary, audio, target
                )
                _print_trace(records, feat_stats, label)
            except Exception as exc:  # noqa: BLE001
                print(f"    (NaN trace failed: {type(exc).__name__}: {exc})")
        print()

    do_cpu_compare = (
        os.environ.get("DEBUG_COMPARE_CPU", "1") != "0"
        and (primary.device or "cpu") != "cpu"
    )
    if not do_cpu_compare:
        print(
            "(Skipping CPU comparison pass: primary is already CPU "
            "or DEBUG_COMPARE_CPU=0)"
        )
        return 0

    which = os.environ.get("DEBUG_TARGETS", "empty").lower()
    if which == "all":
        compare_targets = list(TARGETS)
    else:
        compare_targets = [
            t for t in TARGETS
            if not primary_results.get(t[2], ChunkResult(label=t[2], device="", dtype="")).text_skip
        ]
    print(
        f"\n=== Comparison pass on CPU+fp32  ({len(compare_targets)} chunk(s)) ===\n"
    )

    if not compare_targets:
        print("(No empty chunks on the primary device — nothing to compare.)")
        return 0

    print("Loading CPU+fp32 Cohere ASR (this is a second copy of the weights) ...")
    cpu_asr = _build_asr(cohere_dir, device="cpu", dtype=torch.float32)
    print(f"  cpu device={cpu_asr.device}, dtype={cpu_asr.dtype}")

    for target in compare_targets:
        label = target[2]
        print(f"--- {label}")
        primary_res = primary_results.get(label)
        cpu_res = inspect_chunk(cpu_asr, audio, target)
        if primary_res is not None:
            _print_chunk_result(primary_res)
        _print_chunk_result(cpu_res)
        if primary_res is not None:
            _print_compare(primary_res, cpu_res)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
