#!/usr/bin/env python3
"""
Compact MLX / MLX-LM runtime probe.

Use this when a repo has no MLX validator of its own, or when you want a fast
current-behavior check before patching or reviewing code.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

DEPENDENCY_ERROR: Exception | None = None

try:
    import mlx.core as mx
    import mlx.nn as nn
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = exc
    mx = None
    nn = None

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if DEPENDENCY_ERROR is None:
        DEPENDENCY_ERROR = exc
    np = None


@dataclass
class Result:
    status: str
    name: str
    detail: str


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def run(name: str, fn: Callable[[], Result]) -> Result:
    try:
        result = fn()
    except Exception as exc:
        return Result("FAIL", name, f"{type(exc).__name__}: {exc}")
    result.name = name
    return result


def arrays_equal(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a, b).item())


def compare(actual: str, baseline: str) -> str:
    to_tuple = lambda s: tuple(int(p) for p in re.findall(r"\d+", s)[:3])
    a = to_tuple(actual)
    b = to_tuple(baseline)
    if a < b:
        return "older"
    if a > b:
        return "newer"
    return "equal"


def check_mlx_version() -> Result:
    baseline = "0.31.1"
    state = compare(mx.__version__, baseline)
    if state == "equal":
        return Result("PASS", "", f"mlx=={mx.__version__}")
    if state == "newer":
        return Result("WARN", "", f"mlx=={mx.__version__}; baseline is {baseline}")
    return Result("FAIL", "", f"mlx=={mx.__version__}; baseline requires >= {baseline}")


def check_mlx_core() -> Result:
    ints = mx.array([1, 2])
    floats = mx.array([1.0, 2.0])
    if ints.dtype != mx.int32 or floats.dtype != mx.float32:
        return Result("FAIL", "", f"unexpected default dtypes {ints.dtype}, {floats.dtype}")

    arr = mx.arange(6, dtype=mx.float32) - 2
    mask = arr > 0
    try:
        _ = arr[mask]
    except ValueError:
        pass
    else:
        return Result("FAIL", "", "boolean selection unexpectedly succeeded")

    tmp = mx.array([1.0, 2.0, 3.0])
    tmp[mx.array([True, False, True])] = mx.array([5.0, 6.0])
    if not arrays_equal(tmp, mx.array([5.0, 2.0, 6.0])):
        return Result("FAIL", "", f"unexpected boolean assignment result {tmp}")

    f16 = mx.zeros((2,), dtype=mx.float16)
    bf16 = mx.zeros((2,), dtype=mx.bfloat16)
    f16[mx.array([True, False])] = True
    bf16[mx.array([False, True])] = True
    mx.eval(f16, bf16)
    if f16.tolist() != [1.0, 0.0]:
        return Result("FAIL", "", f"unexpected float16 bool assignment result {f16.tolist()}")
    if [float(v) for v in bf16.tolist()] != [0.0, 1.0]:
        return Result("FAIL", "", f"unexpected bfloat16 bool assignment result {bf16.tolist()}")

    if not hasattr(mx, "bartlett"):
        return Result("FAIL", "", "mx.bartlett is missing")
    bartlett = np.array(mx.bartlett(5))
    if not np.allclose(bartlett, np.bartlett(5)):
        return Result("FAIL", "", f"unexpected bartlett result {bartlett}")

    sliced_source = mx.array([1, 2, 3])
    sliced = sliced_source[:]
    sliced[0] = 9
    if sliced_source[0].item() != 1:
        return Result("FAIL", "", "slice mutated the original array")

    return Result(
        "PASS",
        "",
        "default dtypes, bartlett(), boolean mask limits, low-precision bool assignment, and slice-copy behavior match baseline",
    )


def check_compile_rules() -> Result:
    counter = {"n": 0}

    def traced(x):
        counter["n"] += 1
        return x + 1

    compiled = mx.compile(traced)
    mx.eval(compiled(mx.array([1.0], dtype=mx.float32)))
    mx.eval(compiled(mx.array([2.0], dtype=mx.float32)))
    mx.eval(compiled(mx.array([[1.0]], dtype=mx.float32)))
    mx.eval(compiled(mx.array([1], dtype=mx.int32)))
    if counter["n"] != 3:
        return Result("FAIL", "", f"unexpected compile retrace count {counter['n']}")

    shapeless_counter = {"n": 0}

    def shapeless_traced(x):
        shapeless_counter["n"] += 1
        return x + 1

    shapeless = mx.compile(shapeless_traced, shapeless=True)
    mx.eval(shapeless(mx.array([1.0], dtype=mx.float32)))
    mx.eval(shapeless(mx.array([1.0, 2.0], dtype=mx.float32)))
    if shapeless_counter["n"] != 1:
        return Result("FAIL", "", f"shapeless retraced unexpectedly: {shapeless_counter['n']}")

    return Result("PASS", "", "compile retracing and shapeless behavior match baseline")


def check_training_layouts_streams() -> Result:
    import mlx.optimizers as optim

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 1, bias=False)

        def __call__(self, x):
            return self.lin(x)

    model = Tiny()
    optimizer = optim.SGD(learning_rate=0.1)
    mx.eval(model.parameters())

    if hasattr(mx.array([1.0]), "backward"):
        return Result("FAIL", "", "mx.array unexpectedly has backward()")
    if callable(getattr(optimizer, "step", None)):
        return Result("FAIL", "", "optimizer.step unexpectedly became callable")
    if not callable(getattr(optimizer, "update", None)):
        return Result("FAIL", "", "optimizer.update is missing")

    def loss_fn(m, x, y):
        return ((m(x) - y) ** 2).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(
        model,
        mx.array([[1.0, 2.0]], dtype=mx.float32),
        mx.array([[1.0]], dtype=mx.float32),
    )
    old_step = optimizer.step.item()
    old_weight = mx.array(model.lin.weight)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if optimizer.step.item() != old_step + 1:
        return Result("FAIL", "", "optimizer step state did not advance")
    if arrays_equal(old_weight, model.lin.weight):
        return Result("FAIL", "", "optimizer update did not change weights")

    y1 = nn.Conv1d(6, 8, 3)(mx.ones((2, 5, 6), dtype=mx.float32))
    y2 = nn.Conv2d(6, 8, (3, 5))(mx.ones((2, 7, 9, 6), dtype=mx.float32))
    y3 = nn.Conv3d(3, 4, 2)(mx.ones((2, 4, 5, 6, 3), dtype=mx.float32))
    mx.eval(y1, y2, y3)
    if y1.shape != (2, 3, 8) or y2.shape != (2, 5, 5, 8) or y3.shape != (2, 3, 4, 5, 4):
        return Result("FAIL", "", f"unexpected channels-last output shapes {y1.shape}, {y2.shape}, {y3.shape}")

    stream = mx.new_stream(mx.default_device())
    out = mx.add(mx.array([1.0]), mx.array([2.0]), stream=stream)
    mx.synchronize(stream)
    if out.item() != 3.0:
        return Result("FAIL", "", f"unexpected stream result {out}")
    if stream == mx.default_stream(mx.default_device()):
        return Result("FAIL", "", "new_stream() matched the default stream")

    return Result("PASS", "", "training flow, channels-last layouts, and stream APIs match baseline")


def check_metal_surface() -> Result:
    if not hasattr(mx, "custom_function"):
        return Result("FAIL", "", "mx.custom_function is missing")
    if not hasattr(mx, "metal"):
        return Result("FAIL", "", "mx.metal is missing")
    for name in ("is_available", "start_capture", "stop_capture"):
        if not hasattr(mx.metal, name):
            return Result("FAIL", "", f"mx.metal.{name} is missing")
    if not hasattr(mx.fast, "metal_kernel"):
        return Result("FAIL", "", "mx.fast.metal_kernel is missing")
    if not mx.metal.is_available():
        return Result("WARN", "", "Metal backend unavailable; only surface presence was checked")

    source = """
        uint elem = thread_position_in_grid.x;
        uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
        out[elem] = metal::exp(inp[loc]);
    """
    kernel = mx.fast.metal_kernel(
        name="probe_exp_strided",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )
    arr = (mx.arange(8, dtype=mx.float32).reshape(4, 2) - 3)[::2]
    out = kernel(
        inputs=[arr],
        template=[("T", mx.float32)],
        grid=(arr.size, 1, 1),
        threadgroup=(arr.size, 1, 1),
        output_shapes=[arr.shape],
        output_dtypes=[arr.dtype],
    )[0]
    mx.eval(out)
    if not bool(mx.allclose(out, mx.exp(arr)).item()):
        return Result("FAIL", "", f"unexpected metal kernel result {out}")
    return Result("PASS", "", "metal kernel path and Metal capture hooks match baseline")


def load_mlx_lm():
    try:
        import mlx_lm
        from mlx_lm import batch_generate, convert, generate, load, stream_generate
        from mlx_lm.generate import GenerationResponse
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import (
            ArraysCache,
            ConcatenateKVCache,
            KVCache,
            QuantizedKVCache,
            RotatingKVCache,
            load_prompt_cache,
            make_prompt_cache,
            save_prompt_cache,
        )
        from mlx_lm.utils import _transform_awq_weights
    except ImportError as exc:
        return exc, None

    return None, {
        "module": mlx_lm,
        "load": load,
        "generate": generate,
        "stream_generate": stream_generate,
        "batch_generate": batch_generate,
        "convert": convert,
        "GenerationResponse": GenerationResponse,
        "create_attention_mask": create_attention_mask,
        "KVCache": KVCache,
        "RotatingKVCache": RotatingKVCache,
        "QuantizedKVCache": QuantizedKVCache,
        "ConcatenateKVCache": ConcatenateKVCache,
        "ArraysCache": ArraysCache,
        "make_prompt_cache": make_prompt_cache,
        "save_prompt_cache": save_prompt_cache,
        "load_prompt_cache": load_prompt_cache,
        "_transform_awq_weights": _transform_awq_weights,
    }


MLX_LM_IMPORT_ERROR, MLX_LM = load_mlx_lm()


def check_mlx_lm_surface() -> Result:
    if MLX_LM_IMPORT_ERROR is not None:
        return Result("WARN", "", f"mlx_lm unavailable: {MLX_LM_IMPORT_ERROR}")

    version = MLX_LM["module"].__version__
    baseline = "0.31.0"
    state = compare(version, baseline)
    if state == "older":
        return Result("FAIL", "", f"mlx_lm=={version}; baseline requires >= {baseline}")

    exports = ("load", "generate", "stream_generate", "batch_generate", "convert")
    if not all(callable(MLX_LM[name]) for name in exports):
        return Result("FAIL", "", "mlx_lm exports are incomplete")

    batch_ann = str(inspect.signature(MLX_LM["batch_generate"]).parameters["prompts"].annotation)
    if "List[int]" not in batch_ann:
        return Result("FAIL", "", f"unexpected batch_generate prompts annotation {batch_ann}")

    fields = set(MLX_LM["GenerationResponse"].__annotations__)
    if "text" not in fields or "finish_reason" not in fields:
        return Result("FAIL", "", f"unexpected GenerationResponse fields {sorted(fields)}")

    h = mx.zeros((1, 4, 8), dtype=mx.float32)
    if MLX_LM["create_attention_mask"](h, cache=None) != "causal":
        return Result("FAIL", "", "attention mask fast path changed")

    kv = MLX_LM["KVCache"]()
    k = mx.ones((1, 2, 1, 4), dtype=mx.float16)
    v = mx.ones((1, 2, 1, 4), dtype=mx.float16)
    kv.update_and_fetch(k, v)
    if kv.offset != 1:
        return Result("FAIL", "", "KVCache offset did not advance")

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        MLX_LM["save_prompt_cache"](handle.name, [kv], metadata={"model": "probe"})
        loaded, meta = MLX_LM["load_prompt_cache"](handle.name, return_metadata=True)
    if loaded[0].offset != 1 or meta.get("model") != "probe":
        return Result("FAIL", "", "prompt-cache roundtrip changed metadata or offset")

    shifts = mx.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=mx.uint32) * 4
    unpacked = (mx.arange(64, dtype=mx.uint32).reshape(8, 8) % 16).astype(mx.uint32)
    packed = ((unpacked << shifts[None, :]).sum(axis=1, keepdims=True)).astype(mx.uint32)
    new_weights, qconf = MLX_LM["_transform_awq_weights"](
        {"layer.qweight": packed, "layer.scales": mx.ones((1, 8), dtype=mx.float16)},
        {"bits": 4, "group_size": 8},
    )
    if sorted(new_weights) != ["layer.biases", "layer.scales", "layer.weight"] or qconf != {"group_size": 8, "bits": 4}:
        return Result("FAIL", "", "AWQ/GPTQ transform helper changed")

    return Result("PASS", "", f"mlx_lm=={version}; surface, caches, and quantization helpers match baseline")


def check_local_model(model_path: str) -> Result:
    if MLX_LM_IMPORT_ERROR is not None:
        return Result("WARN", "", "skipping local model checks because mlx_lm is unavailable")

    model, tokenizer = MLX_LM["load"](model_path, lazy=True)
    text = MLX_LM["generate"](model, tokenizer, "Say hi.", max_tokens=2, verbose=False)
    response = next(MLX_LM["stream_generate"](model, tokenizer, "Say hi.", max_tokens=1))
    if not isinstance(text, str):
        return Result("FAIL", "", f"generate() returned {type(text).__name__}")
    if not isinstance(response, MLX_LM["GenerationResponse"]):
        return Result("FAIL", "", f"stream_generate() yielded {type(response).__name__}")

    cache = MLX_LM["make_prompt_cache"](model, max_kv_size=16)
    prompt = mx.array(tokenizer.encode("Hello"), dtype=mx.int32)[None]
    _ = model(prompt, cache=cache)
    mx.eval([c.state for c in cache])
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        MLX_LM["save_prompt_cache"](handle.name, cache, metadata={"model": "local"})
        loaded, meta = MLX_LM["load_prompt_cache"](handle.name, return_metadata=True)
    if len(loaded) != len(cache) or meta.get("model") != "local":
        return Result("FAIL", "", "local prompt-cache roundtrip changed length or metadata")

    genmod = importlib.import_module("mlx_lm.generate")
    if genmod.generation_stream == mx.default_stream(mx.default_device()):
        return Result("FAIL", "", "mlx_lm generation_stream unexpectedly matched default stream")

    counts = {"async_eval": 0, "clear_cache": 0}
    original_async_eval = mx.async_eval
    original_clear_cache = mx.clear_cache

    def wrapped_async_eval(*args, **kwargs):
        counts["async_eval"] += 1
        return original_async_eval(*args, **kwargs)

    def wrapped_clear_cache(*args, **kwargs):
        counts["clear_cache"] += 1
        return original_clear_cache(*args, **kwargs)

    try:
        mx.async_eval = wrapped_async_eval
        mx.clear_cache = wrapped_clear_cache
        _ = MLX_LM["generate"](model, tokenizer, "Say hi.", max_tokens=2, verbose=False)
    finally:
        mx.async_eval = original_async_eval
        mx.clear_cache = original_clear_cache

    detail = f"local model ok; async_eval={counts['async_eval']}, clear_cache={counts['clear_cache']}"
    try:
        _ = MLX_LM["batch_generate"](
            model,
            tokenizer,
            [tokenizer.encode('Hello'), tokenizer.encode('Goodbye')],
            max_tokens=1,
            verbose=False,
        )
    except ZeroDivisionError:
        return Result("WARN", "", detail + "; batch_generate(max_tokens=1) reproduced ZeroDivisionError")
    return Result("PASS", "", detail)


def summarize(results: list[Result]) -> int:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    worst = "PASS"
    for result in results:
        counts[result.status] += 1
        if STATUS_ORDER[result.status] > STATUS_ORDER[worst]:
            worst = result.status
        print(f"[{result.status}] {result.name}: {result.detail}")
    print()
    print(f"Summary: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    return 1 if worst == "FAIL" else 0


def main() -> int:
    if DEPENDENCY_ERROR is not None:
        missing = getattr(DEPENDENCY_ERROR, "name", str(DEPENDENCY_ERROR))
        print(
            "[FAIL] dependency: "
            f"missing Python package {missing!r} for interpreter {sys.executable}. "
            "Activate an MLX-capable environment or use mlx_probe.sh.",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description="Probe current MLX / MLX-LM runtime behavior.")
    parser.add_argument("--model-path", default=os.environ.get("MLX_LM_LOCAL_MODEL"))
    args = parser.parse_args()

    checks: list[tuple[str, Callable[[], Result]]] = [
        ("mlx version", check_mlx_version),
        ("mlx core", check_mlx_core),
        ("compile rules", check_compile_rules),
        ("training/layouts/streams", check_training_layouts_streams),
        ("metal surface", check_metal_surface),
        ("mlx-lm surface", check_mlx_lm_surface),
    ]
    if args.model_path:
        checks.append(("local model", lambda: check_local_model(args.model_path)))

    return summarize([run(name, fn) for name, fn in checks])


if __name__ == "__main__":
    raise SystemExit(main())
