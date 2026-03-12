"""
Source-backed MLX / MLX-LM verifier.

This script checks the high-signal claims in cheatsheet.md against a compact set
of local runtime assertions. It is intentionally small and correctness-focused:
no model downloads, no giant benchmarks, and no undocumented conclusions passed
off as contracts.

Baseline used for this repo revision:
- MLX v0.31.1
- MLX-LM v0.31.0

Runtime notes:
- A newer version than the baseline is reported as a warning, not an outright
  failure, because some checks may still hold while the docs drift.
- Out-of-bounds indexing is printed as an observed behavior note only. Upstream
  documents it as undefined behavior.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

BASELINE_MLX = "0.31.1"
BASELINE_MLX_LM = "0.31.0"

STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class Result:
    status: str
    name: str
    detail: str


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def compare_version(actual: str, baseline: str) -> str:
    actual_t = version_tuple(actual)
    baseline_t = version_tuple(baseline)
    if actual_t < baseline_t:
        return "older"
    if actual_t > baseline_t:
        return "newer"
    return "equal"


def arrays_equal(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def run_check(name: str, fn: Callable[[], Result]) -> Result:
    try:
        result = fn()
    except Exception as exc:  # pragma: no cover - defensive harness code
        return Result("FAIL", name, f"{type(exc).__name__}: {exc}")
    result.name = name
    return result


def check_mlx_version() -> Result:
    state = compare_version(mx.__version__, BASELINE_MLX)
    if state == "equal":
        return Result("PASS", "", f"mlx=={mx.__version__}")
    if state == "newer":
        return Result(
            "WARN",
            "",
            f"mlx=={mx.__version__}; repo baseline is {BASELINE_MLX}",
        )
    return Result(
        "FAIL",
        "",
        f"mlx=={mx.__version__}; repo baseline requires at least {BASELINE_MLX}",
    )


def check_default_dtypes() -> Result:
    ints = mx.array([1, 2])
    floats = mx.array([1.0, 2.0])
    if ints.dtype != mx.int32 or floats.dtype != mx.float32:
        return Result(
            "FAIL",
            "",
            f"got int dtype {ints.dtype} and float dtype {floats.dtype}",
        )
    return Result("PASS", "", "integer literals -> int32, float literals -> float32")


def check_boolean_selection_and_assignment() -> Result:
    arr = mx.array([1.0, 2.0, 3.0])
    mask = mx.array([True, False, True])
    try:
        _ = arr[mask]
    except ValueError as exc:
        if "boolean indices" not in str(exc):
            return Result("FAIL", "", f"unexpected boolean selection error: {exc}")
    else:
        return Result("FAIL", "", "boolean selection unexpectedly succeeded")

    arr[mask] = mx.array([5.0, 6.0])
    if not arrays_equal(arr, mx.array([5.0, 2.0, 6.0])):
        return Result("FAIL", "", f"boolean assignment result was {arr}")
    return Result("PASS", "", "selection unsupported, assignment supported")


def check_where_nonzero_argwhere_absence() -> Result:
    mask = mx.array([True, False, True])
    try:
        _ = mx.where(mask)
    except TypeError:
        pass
    else:
        return Result("FAIL", "", "single-argument mx.where unexpectedly succeeded")

    missing = [name for name in ("nonzero", "argwhere") if hasattr(mx, name)]
    if missing:
        return Result("FAIL", "", f"unexpected helpers present: {', '.join(missing)}")
    return Result("PASS", "", "single-argument where/nonzero/argwhere absent")


def check_helper_creation_apis() -> Result:
    arr = mx.arange(4)
    try:
        _ = mx.ones_like(arr, dtype=mx.int32)
    except TypeError:
        pass
    else:
        return Result("FAIL", "", "ones_like unexpectedly accepted dtype=")

    try:
        _ = mx.zeros_like(arr, dtype=mx.int32)
    except TypeError:
        pass
    else:
        return Result("FAIL", "", "zeros_like unexpectedly accepted dtype=")

    if hasattr(mx, "full_like"):
        return Result("FAIL", "", "mx.full_like unexpectedly exists")

    if not hasattr(mx, "bartlett"):
        return Result("FAIL", "", "mx.bartlett is missing")
    bartlett = np.array(mx.bartlett(5))
    if not np.allclose(bartlett, np.bartlett(5)):
        return Result("FAIL", "", f"mx.bartlett(5) did not match NumPy: {bartlett}")

    if arr.nbytes != arr.size * arr.itemsize:
        return Result(
            "FAIL",
            "",
            f"nbytes mismatch: nbytes={arr.nbytes}, size={arr.size}, itemsize={arr.itemsize}",
        )
    return Result(
        "PASS",
        "",
        "nbytes exists; ones_like dtype/full_like match baseline; bartlett() matches NumPy",
    )


def check_indexing_surface() -> Result:
    arr = mx.arange(12, dtype=mx.int32).reshape(3, 4)

    if arr[-1, -1].item() != 11:
        return Result("FAIL", "", f"negative indexing returned {arr[-1, -1]}")

    expanded = arr[None, ..., 1:3]
    if expanded.shape != (1, 3, 2):
        return Result("FAIL", "", f"unexpected None/ellipsis shape {expanded.shape}")

    take = mx.take(arr, mx.array([2, 0], dtype=mx.int32), axis=0)
    if take.shape != (2, 4) or take[0, 0].item() != 8 or take[1, 0].item() != 0:
        return Result("FAIL", "", f"unexpected take() result {take}")

    gather_idx = mx.array([[0, 2], [1, 0], [3, 1]], dtype=mx.int32)
    gathered = mx.take_along_axis(arr, gather_idx, axis=1)
    expected = mx.array([[0, 2], [5, 4], [11, 9]], dtype=mx.int32)
    if not arrays_equal(gathered, expected):
        return Result("FAIL", "", f"unexpected take_along_axis() result {gathered}")

    return Result("PASS", "", "negative indices, None/ellipsis, take(), and take_along_axis() work")


def check_slice_copy_and_aliasing() -> Result:
    alias_source = mx.array([1, 2, 3])
    alias = alias_source
    alias[2] = 0
    if not arrays_equal(alias_source, mx.array([1, 2, 0])):
        return Result("FAIL", "", "alias update did not affect original array")

    sliced_source = mx.array([1, 2, 3])
    sliced = sliced_source[:]
    sliced[2] = 0
    if not arrays_equal(sliced_source, mx.array([1, 2, 3])):
        return Result("FAIL", "", "slice unexpectedly mutated original array")
    return Result("PASS", "", "aliases share storage; slices are copies")


def check_duplicate_updates() -> Result:
    idx = mx.array([0, 2, 0, 2, 4], dtype=mx.int32)
    standard = mx.zeros((5,), dtype=mx.int32)
    standard[idx] = mx.ones_like(idx)

    accumulated = mx.zeros((5,), dtype=mx.int32)
    accumulated = accumulated.at[idx].add(1)
    mx.eval(accumulated)

    if not arrays_equal(standard, mx.array([1, 0, 1, 0, 1], dtype=mx.int32)):
        return Result("FAIL", "", f"unexpected direct duplicate-write result: {standard}")
    if not arrays_equal(accumulated, mx.array([2, 0, 2, 0, 1], dtype=mx.int32)):
        return Result("FAIL", "", f"unexpected .at.add result: {accumulated}")
    return Result("PASS", "", "direct duplicate writes differ from .at accumulations")


def check_bool_assignment_to_low_precision_floats() -> Result:
    f16 = mx.zeros((2,), dtype=mx.float16)
    bf16 = mx.zeros((2,), dtype=mx.bfloat16)
    f16[mx.array([True, False])] = True
    bf16[mx.array([False, True])] = True
    mx.eval(f16, bf16)

    if f16.tolist() != [1.0, 0.0]:
        return Result("FAIL", "", f"unexpected float16 bool assignment result {f16.tolist()}")

    bf16_list = [float(v) for v in bf16.tolist()]
    if bf16_list != [0.0, 1.0]:
        return Result("FAIL", "", f"unexpected bfloat16 bool assignment result {bf16_list}")

    return Result(
        "PASS",
        "",
        "bool assignment into float16/bfloat16 stores numeric 1.0/0.0",
    )


def check_index_dtype_and_dynamic_slice() -> Result:
    arr = mx.arange(10, dtype=mx.float32) - 5
    picked = arr[mx.array([6, 7, 8, 9], dtype=mx.int32)]
    if not arrays_equal(picked, mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)):
        return Result("FAIL", "", f"unexpected integral indexing result: {picked}")

    try:
        _ = mx.take(arr, mx.array([0.5, 1.5]))
    except ValueError:
        pass
    else:
        return Result("FAIL", "", "mx.take unexpectedly accepted float indices")

    if not arrays_equal(arr[:5], mx.array([-5, -4, -3, -2, -1], dtype=mx.float32)):
        return Result("FAIL", "", "dynamic Python slicing returned the wrong values")
    return Result("PASS", "", "integral indexing and dynamic slicing behave as expected")


def check_compile_and_shapeless() -> Result:
    compiled = mx.compile(lambda x: x + 1)
    shapeless = mx.compile(lambda x: x + 1, shapeless=True)

    one = compiled(mx.array([1.0], dtype=mx.float32))
    two = shapeless(mx.array([1.0, 2.0], dtype=mx.float32))
    three = shapeless(mx.array([1.0, 2.0, 3.0], dtype=mx.float32))
    mx.eval(one, two, three)

    if one.shape != (1,) or two.shape != (2,) or three.shape != (3,):
        return Result(
            "FAIL",
            "",
            f"unexpected compiled shapes: {one.shape}, {two.shape}, {three.shape}",
        )
    return Result("PASS", "", "compile() and shapeless compile() both work")


def check_compile_and_fast_surface() -> Result:
    compile_doc = (mx.compile.__doc__ or "").splitlines()[0]
    if "shapeless" not in compile_doc:
        return Result("FAIL", "", f"unexpected compile doc signature: {compile_doc!r}")
    if not hasattr(mx, "disable_compile"):
        return Result("FAIL", "", "mx.disable_compile is missing")
    if not hasattr(mx.fast, "scaled_dot_product_attention"):
        return Result("FAIL", "", "mx.fast.scaled_dot_product_attention is missing")
    if not hasattr(mx.fast, "rms_norm"):
        return Result("FAIL", "", "mx.fast.rms_norm is missing")
    return Result("PASS", "", "compile debug hooks and mx.fast kernels are present")


def check_compile_recompilation_rules() -> Result:
    retrace_counter = {"n": 0}

    def traced(x):
        retrace_counter["n"] += 1
        return x + 1

    compiled = mx.compile(traced)
    mx.eval(compiled(mx.array([1.0], dtype=mx.float32)))
    mx.eval(compiled(mx.array([2.0], dtype=mx.float32)))
    mx.eval(compiled(mx.array([[1.0]], dtype=mx.float32)))
    mx.eval(compiled(mx.array([1], dtype=mx.int32)))

    if retrace_counter["n"] != 3:
        return Result(
            "FAIL",
            "",
            f"expected 3 traces for same/rank/dtype changes, got {retrace_counter['n']}",
        )

    shapeless_counter = {"n": 0}

    def shapeless_traced(x):
        shapeless_counter["n"] += 1
        return x + 1

    shapeless_compiled = mx.compile(shapeless_traced, shapeless=True)
    mx.eval(shapeless_compiled(mx.array([1.0], dtype=mx.float32)))
    mx.eval(shapeless_compiled(mx.array([1.0, 2.0], dtype=mx.float32)))
    if shapeless_counter["n"] != 1:
        return Result(
            "FAIL",
            "",
            f"shapeless compile retraced on shape-only change: {shapeless_counter['n']}",
        )

    arity_counter = {"n": 0}

    def variadic(*xs):
        arity_counter["n"] += 1
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out

    arity_compiled = mx.compile(variadic)
    mx.eval(arity_compiled(mx.array([1.0], dtype=mx.float32)))
    mx.eval(arity_compiled(mx.array([2.0], dtype=mx.float32)))
    mx.eval(
        arity_compiled(
            mx.array([1.0], dtype=mx.float32),
            mx.array([2.0], dtype=mx.float32),
        )
    )
    mx.eval(
        arity_compiled(
            mx.array([3.0], dtype=mx.float32),
            mx.array([4.0], dtype=mx.float32),
        )
    )
    if arity_counter["n"] != 2:
        return Result(
            "FAIL",
            "",
            f"expected arity change to retrace once, got {arity_counter['n']}",
        )

    return Result(
        "PASS",
        "",
        "compile retraces on rank/dtype/arity changes, while shapeless avoids shape-only retraces",
    )


def check_numpy_interop() -> Result:
    arr = mx.arange(3)
    view = np.array(arr, copy=False)
    view[0] = 7
    if arr[0].item() != 7:
        return Result("FAIL", "", "NumPy copy=False did not reflect back into MLX")

    bf16 = mx.array([1.5, 2.5], dtype=mx.bfloat16)
    try:
        _ = np.array(bf16)
    except RuntimeError:
        pass
    else:
        return Result("FAIL", "", "bfloat16 unexpectedly converted to NumPy without cast")

    casted = np.array(bf16.astype(mx.float32))
    if casted.dtype != np.float32 or casted.tolist() != [1.5, 2.5]:
        return Result("FAIL", "", f"unexpected casted bfloat16 conversion: {casted}")

    np64 = np.array([1.0, 2.0], dtype=np.float64)
    if mx.array(np64).dtype != mx.float32:
        return Result("FAIL", "", "NumPy float64 no longer defaults to MLX float32")
    return Result("PASS", "", "NumPy view and dtype-conversion caveats match baseline")


def check_random_contract() -> Result:
    key_a = mx.random.key(0)
    key_b = mx.random.key(0)
    if not arrays_equal(key_a, key_b):
        return Result("FAIL", "", "mx.random.key(0) was not reproducible")

    mx.random.seed(42)
    first = mx.random.uniform(shape=(3,))
    mx.random.seed(42)
    second = mx.random.uniform(shape=(3,))
    if not arrays_equal(first, second):
        return Result("FAIL", "", "mx.random.seed(...) was not reproducible")
    return Result("PASS", "", "global seed and explicit keys are reproducible")


def check_oob_observation() -> Result:
    arr = mx.arange(10, dtype=mx.float32) - 5
    single = arr[10]
    multi = arr[mx.array([10, 11], dtype=mx.int32)]
    return Result(
        "WARN",
        "",
        f"observed OOB result single={single}, multi={multi}; upstream docs still call this undefined behavior",
    )


def check_at_surface_and_weight_layouts() -> Result:
    arr = mx.array([1, 2, 3])
    updater = arr.at[mx.array([0], dtype=mx.int32)]
    methods = sorted(name for name in dir(updater) if not name.startswith("_"))
    expected_methods = ["add", "divide", "maximum", "minimum", "multiply", "subtract"]
    if methods != expected_methods:
        return Result("FAIL", "", f"unexpected .at methods: {methods}")

    linear = nn.Linear(3, 4)
    conv1d = nn.Conv1d(6, 8, 3, groups=2)
    conv2d = nn.Conv2d(6, 8, (3, 5), groups=2)
    if linear.weight.shape != (4, 3):
        return Result("FAIL", "", f"unexpected Linear weight shape {linear.weight.shape}")
    if conv1d.weight.shape != (8, 3, 3):
        return Result("FAIL", "", f"unexpected Conv1d weight shape {conv1d.weight.shape}")
    if conv2d.weight.shape != (8, 3, 5, 3):
        return Result("FAIL", "", f"unexpected Conv2d weight shape {conv2d.weight.shape}")
    return Result("PASS", "", ".at surface and MLX linear/conv weight layouts match baseline")


def check_training_surface_and_optimizer_flow() -> Result:
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
        return Result("FAIL", "", "mx.array unexpectedly exposes backward()")
    if callable(getattr(optimizer, "step", None)):
        return Result("FAIL", "", "optimizer.step unexpectedly became a method")
    if not callable(getattr(optimizer, "update", None)):
        return Result("FAIL", "", "optimizer.update is missing or not callable")

    def loss_fn(m, x, y):
        return ((m(x) - y) ** 2).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(
        model,
        mx.array([[1.0, 2.0]], dtype=mx.float32),
        mx.array([[1.0]], dtype=mx.float32),
    )
    if loss.shape != ():
        return Result("FAIL", "", f"unexpected scalar loss shape {loss.shape}")
    if "lin" not in grads or "weight" not in grads["lin"]:
        return Result("FAIL", "", f"unexpected gradient tree {grads.keys()}")

    old_step = optimizer.step.item()
    old_weight = mx.array(model.lin.weight)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if optimizer.step.item() != old_step + 1:
        return Result(
            "FAIL",
            "",
            f"optimizer step counter did not advance: {old_step} -> {optimizer.step.item()}",
        )
    if arrays_equal(old_weight, model.lin.weight):
        return Result("FAIL", "", "optimizer.update did not change model weights")
    return Result(
        "PASS",
        "",
        "training uses nn.value_and_grad + optimizer.update; optimizer.step is state, and eval(model.parameters(), optimizer.state) materializes the update",
    )


def check_channels_last_surface() -> Result:
    conv1d = nn.Conv1d(6, 8, 3)
    conv2d = nn.Conv2d(6, 8, (3, 5))
    conv3d = nn.Conv3d(3, 4, 2)

    y1 = conv1d(mx.ones((2, 5, 6), dtype=mx.float32))
    y2 = conv2d(mx.ones((2, 7, 9, 6), dtype=mx.float32))
    y3 = conv3d(mx.ones((2, 4, 5, 6, 3), dtype=mx.float32))
    mx.eval(y1, y2, y3)

    if y1.shape != (2, 3, 8):
        return Result("FAIL", "", f"unexpected Conv1d output shape {y1.shape}")
    if y2.shape != (2, 5, 5, 8):
        return Result("FAIL", "", f"unexpected Conv2d output shape {y2.shape}")
    if y3.shape != (2, 3, 4, 5, 4):
        return Result("FAIL", "", f"unexpected Conv3d output shape {y3.shape}")
    return Result(
        "PASS",
        "",
        "MLX convolution inputs are channels-last: NLC, NHWC, and NDHWC",
    )


def check_stream_surface() -> Result:
    for name in ("default_stream", "new_stream", "stream", "synchronize"):
        if not hasattr(mx, name):
            return Result("FAIL", "", f"mx.{name} is missing")

    stream = mx.new_stream(mx.default_device())
    if stream == mx.default_stream(mx.default_device()):
        return Result("FAIL", "", "mx.new_stream returned the default stream")

    out = mx.add(mx.array([1.0]), mx.array([2.0]), stream=stream)
    mx.synchronize(stream)
    if out.item() != 3.0:
        return Result("FAIL", "", f"unexpected stream result {out}")
    return Result(
        "PASS",
        "",
        "stream= works on core ops, new_stream() creates a non-default stream, and synchronize() is available",
    )


def check_memory_profiling_surface() -> Result:
    required = (
        "get_active_memory",
        "get_peak_memory",
        "reset_peak_memory",
        "get_cache_memory",
        "clear_cache",
        "device_info",
    )
    missing = [name for name in required if not hasattr(mx, name)]
    if missing:
        return Result("FAIL", "", f"missing top-level memory helpers: {', '.join(missing)}")

    mx.reset_peak_memory()
    if int(mx.get_peak_memory()) != 0:
        return Result("FAIL", "", f"peak memory did not reset cleanly: {mx.get_peak_memory()}")

    before_active = int(mx.get_active_memory())
    arr = mx.ones((1024, 1024), dtype=mx.float32)
    mx.eval(arr)
    after_active = int(mx.get_active_memory())
    peak = int(mx.get_peak_memory())
    info = mx.device_info()

    if after_active < before_active:
        return Result(
            "FAIL",
            "",
            f"active memory unexpectedly decreased across an evaluated allocation: {before_active} -> {after_active}",
        )
    if peak < after_active:
        return Result("FAIL", "", f"peak memory {peak} was smaller than active memory {after_active}")
    if "max_recommended_working_set_size" not in info:
        return Result("FAIL", "", f"device_info() keys changed: {sorted(info)}")

    del arr
    gc.collect()
    mx.clear_cache()
    cache_bytes = int(mx.get_cache_memory())
    if cache_bytes != 0:
        return Result("FAIL", "", f"clear_cache() did not empty cached bytes: {cache_bytes}")

    return Result(
        "PASS",
        "",
        "top-level memory helpers exist, peak tracking works, clear_cache() empties cached bytes, and device_info() exposes max_recommended_working_set_size",
    )


def check_metal_kernel_surface() -> Result:
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
        return Result(
            "WARN",
            "",
            "Metal backend unavailable, so only metal-kernel and capture hook surface was checked",
        )

    source = """
        uint elem = thread_position_in_grid.x;
        uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
        out[elem] = metal::exp(inp[loc]);
    """
    kernel = mx.fast.metal_kernel(
        name="validation_exp_strided",
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
    return Result(
        "PASS",
        "",
        "mx.fast.metal_kernel works on a strided-input exp kernel; custom_function and metal capture hooks are present",
    )


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
            can_trim_prompt_cache,
            load_prompt_cache,
            make_prompt_cache,
            save_prompt_cache,
            trim_prompt_cache,
        )
        from mlx_lm.utils import _transform_awq_weights
    except ImportError as exc:
        return exc, None

    exports = {
        "_transform_awq_weights": _transform_awq_weights,
        "ArraysCache": ArraysCache,
        "ConcatenateKVCache": ConcatenateKVCache,
        "GenerationResponse": GenerationResponse,
        "module": mlx_lm,
        "batch_generate": batch_generate,
        "can_trim_prompt_cache": can_trim_prompt_cache,
        "convert": convert,
        "trim_prompt_cache": trim_prompt_cache,
        "generate": generate,
        "load": load,
        "stream_generate": stream_generate,
        "create_attention_mask": create_attention_mask,
        "KVCache": KVCache,
        "QuantizedKVCache": QuantizedKVCache,
        "RotatingKVCache": RotatingKVCache,
        "load_prompt_cache": load_prompt_cache,
        "make_prompt_cache": make_prompt_cache,
        "save_prompt_cache": save_prompt_cache,
    }
    return None, exports


MLX_LM_IMPORT_ERROR, MLX_LM = load_mlx_lm()


def require_mlx_lm() -> Result:
    if MLX_LM_IMPORT_ERROR is not None:
        return Result("FAIL", "", f"mlx_lm import failed: {MLX_LM_IMPORT_ERROR}")
    version = MLX_LM["module"].__version__
    state = compare_version(version, BASELINE_MLX_LM)
    if state == "equal":
        return Result(
            "WARN",
            "",
            "mlx_lm==0.31.0 loaded; note the PyPI 0.31.0 wheel was yanked, so prefer the GitHub tag or a newer non-yanked release",
        )
    if state == "newer":
        return Result(
            "WARN",
            "",
            f"mlx_lm=={version}; repo baseline is {BASELINE_MLX_LM}",
        )
    return Result(
        "FAIL",
        "",
        f"mlx_lm=={version}; repo baseline requires at least {BASELINE_MLX_LM}",
    )


def check_mlx_lm_exports_and_signature() -> Result:
    exports = (
        MLX_LM["load"],
        MLX_LM["convert"],
        MLX_LM["generate"],
        MLX_LM["stream_generate"],
        MLX_LM["batch_generate"],
    )
    if not all(callable(fn) for fn in exports):
        return Result("FAIL", "", "mlx_lm top-level exports are incomplete")

    signature = inspect.signature(MLX_LM["load"])
    for name in ("lazy", "revision"):
        if name not in signature.parameters:
            return Result("FAIL", "", f"mlx_lm.load is missing parameter {name!r}")
    return Result("PASS", "", "top-level mlx_lm exports and load() signature match baseline")


def check_mlx_lm_generation_surface() -> Result:
    stream_signature = inspect.signature(MLX_LM["stream_generate"])
    batch_signature = inspect.signature(MLX_LM["batch_generate"])

    prompt_annotation = str(stream_signature.parameters["prompt"].annotation)
    batch_prompts_annotation = str(batch_signature.parameters["prompts"].annotation)
    if "str" not in prompt_annotation:
        return Result(
            "FAIL",
            "",
            f"stream_generate prompt annotation changed: {prompt_annotation}",
        )
    if "List[int]" not in batch_prompts_annotation:
        return Result(
            "FAIL",
            "",
            f"batch_generate prompts annotation changed: {batch_prompts_annotation}",
        )

    fields = set(MLX_LM["GenerationResponse"].__annotations__)
    expected = {
        "text",
        "token",
        "logprobs",
        "from_draft",
        "prompt_tokens",
        "prompt_tps",
        "generation_tokens",
        "generation_tps",
        "peak_memory",
        "finish_reason",
    }
    if fields != expected:
        return Result("FAIL", "", f"GenerationResponse fields changed: {sorted(fields)}")
    return Result(
        "PASS",
        "",
        "generate/stream_generate/batch_generate prompt surfaces and GenerationResponse match baseline",
    )


def check_mlx_lm_quantization_transform() -> Result:
    transform_awq = MLX_LM["_transform_awq_weights"]
    bits = 4
    shifts = mx.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=mx.uint32) * bits
    unpacked = (mx.arange(64, dtype=mx.uint32).reshape(8, 8) % 16).astype(mx.uint32)
    packed = ((unpacked << shifts[None, :]).sum(axis=1, keepdims=True)).astype(mx.uint32)
    weights = {
        "layer.qweight": packed,
        "layer.scales": mx.ones((1, 8), dtype=mx.float16),
    }
    new_weights, qconf = transform_awq(weights, {"bits": 4, "group_size": 8})

    if sorted(new_weights) != ["layer.biases", "layer.scales", "layer.weight"]:
        return Result("FAIL", "", f"unexpected AWQ transform keys {sorted(new_weights)}")
    if qconf != {"group_size": 8, "bits": 4}:
        return Result("FAIL", "", f"unexpected AWQ quantization config {qconf}")
    if new_weights["layer.weight"].shape != (8, 1):
        return Result(
            "FAIL",
            "",
            f"unexpected transformed AWQ weight shape {new_weights['layer.weight'].shape}",
        )
    if new_weights["layer.scales"].shape != (8, 1):
        return Result(
            "FAIL",
            "",
            f"unexpected transformed AWQ scales shape {new_weights['layer.scales'].shape}",
        )
    if new_weights["layer.biases"].shape != (8, 1):
        return Result(
            "FAIL",
            "",
            f"unexpected transformed AWQ biases shape {new_weights['layer.biases'].shape}",
        )
    return Result(
        "PASS",
        "",
        "synthetic AutoAWQ/GPTQ packed weights transform into MLX quantized layout",
    )


def check_mlx_lm_attention_mask() -> Result:
    create_attention_mask = MLX_LM["create_attention_mask"]
    h = mx.zeros((1, 4, 8), dtype=mx.float32)
    single = mx.zeros((1, 1, 8), dtype=mx.float32)

    causal = create_attention_mask(h, cache=None)
    explicit = create_attention_mask(h, cache=None, return_array=True)
    none_case = create_attention_mask(single, cache=None)

    if causal != "causal":
        return Result("FAIL", "", f"expected 'causal', got {causal!r}")
    if explicit.shape != (4, 4):
        return Result("FAIL", "", f"unexpected explicit mask shape {explicit.shape}")
    if none_case is not None:
        return Result("FAIL", "", f"expected None for single-token mask, got {none_case!r}")
    return Result("PASS", "", "mlx_lm attention mask helper matches current contract")


def check_mlx_lm_caches() -> Result:
    ArraysCache = MLX_LM["ArraysCache"]
    ConcatenateKVCache = MLX_LM["ConcatenateKVCache"]
    KVCache = MLX_LM["KVCache"]
    QuantizedKVCache = MLX_LM["QuantizedKVCache"]
    RotatingKVCache = MLX_LM["RotatingKVCache"]
    can_trim_prompt_cache = MLX_LM["can_trim_prompt_cache"]
    make_prompt_cache = MLX_LM["make_prompt_cache"]
    trim_prompt_cache = MLX_LM["trim_prompt_cache"]

    kv = KVCache()
    k = mx.ones((1, 2, 1, 4), dtype=mx.float16)
    v = mx.ones((1, 2, 1, 4), dtype=mx.float16)
    k_up, v_up = kv.update_and_fetch(k, v)
    if kv.offset != 1 or k_up.shape != (1, 2, 1, 4) or v_up.shape != (1, 2, 1, 4):
        return Result(
            "FAIL",
            "",
            f"unexpected KVCache state: offset={kv.offset}, k={k_up.shape}, v={v_up.shape}",
        )

    rotating = RotatingKVCache(max_size=8)
    rotating.update_and_fetch(
        mx.ones((1, 2, 2, 4), dtype=mx.float16),
        mx.ones((1, 2, 2, 4), dtype=mx.float16),
    )
    if rotating.offset != 2:
        return Result("FAIL", "", f"unexpected RotatingKVCache offset {rotating.offset}")

    conc = ConcatenateKVCache()
    conc_k, conc_v = conc.update_and_fetch(k, v)
    if conc.offset != 1 or conc_k.shape != (1, 2, 1, 4) or conc_v.shape != (1, 2, 1, 4):
        return Result("FAIL", "", "ConcatenateKVCache did not track concatenated state")

    quant = QuantizedKVCache(bits=8, group_size=32)
    qk, qv = quant.update_and_fetch(
        mx.ones((1, 2, 1, 32), dtype=mx.float16),
        mx.ones((1, 2, 1, 32), dtype=mx.float16),
    )
    if quant.offset != 1 or qk[0].shape[-2] != 1 or qv[0].shape[-2] != 1:
        return Result("FAIL", "", "QuantizedKVCache did not produce quantized state")

    arrays = ArraysCache(size=2)
    arrays[0] = mx.ones((1, 3, 4), dtype=mx.float16)
    arrays[1] = mx.ones((1, 3, 5), dtype=mx.float16)
    if arrays.nbytes <= 0 or arrays.is_trimmable():
        return Result("FAIL", "", "ArraysCache surface no longer matches baseline")

    class DummyModel:
        def __init__(self):
            self.layers = [object(), object(), object()]

    prompt_cache = make_prompt_cache(DummyModel(), max_kv_size=8)
    if len(prompt_cache) != 3 or type(prompt_cache[0]).__name__ != "RotatingKVCache":
        return Result(
            "FAIL",
            "",
            f"unexpected prompt cache fallback: len={len(prompt_cache)}, first={type(prompt_cache[0]).__name__}",
        )
    if not can_trim_prompt_cache(prompt_cache):
        return Result("FAIL", "", "RotatingKVCache prompt cache unexpectedly became non-trimmable")
    if trim_prompt_cache(prompt_cache, 1) != 0:
        return Result("FAIL", "", "trimming an empty RotatingKVCache should trim 0 tokens")
    return Result("PASS", "", "KV cache helpers and prompt cache fallback are current")


def check_mlx_lm_prompt_cache_roundtrip() -> Result:
    KVCache = MLX_LM["KVCache"]
    load_prompt_cache = MLX_LM["load_prompt_cache"]
    save_prompt_cache = MLX_LM["save_prompt_cache"]

    cache = KVCache()
    keys = mx.ones((1, 2, 1, 4), dtype=mx.float16)
    values = mx.full((1, 2, 1, 4), 3, dtype=mx.float16)
    cache.update_and_fetch(keys, values)

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        save_prompt_cache(handle.name, [cache], metadata={"model": "dummy"})
        loaded, metadata = load_prompt_cache(handle.name, return_metadata=True)

    loaded_cache = loaded[0]
    if loaded_cache.offset != 1:
        return Result("FAIL", "", f"unexpected loaded cache offset {loaded_cache.offset}")
    if metadata.get("model") != "dummy":
        return Result("FAIL", "", f"unexpected prompt cache metadata {metadata}")
    if not arrays_equal(loaded_cache.keys, keys) or not arrays_equal(loaded_cache.values, values):
        return Result("FAIL", "", "prompt cache tensors changed across save/load")
    return Result("PASS", "", "prompt caches round-trip through safetensors")


def check_local_model_load_and_decode(model_path: str) -> Result:
    model, tokenizer = MLX_LM["load"](model_path, lazy=True)
    text = MLX_LM["generate"](
        model, tokenizer, "Say hi in one word.", max_tokens=1, verbose=False
    )
    response = next(
        MLX_LM["stream_generate"](
            model, tokenizer, "Say hi in one word.", max_tokens=1
        )
    )
    if not isinstance(text, str):
        return Result("FAIL", "", f"generate() returned {type(text).__name__}")
    if not isinstance(response, MLX_LM["GenerationResponse"]):
        return Result(
            "FAIL", "", f"stream_generate() yielded {type(response).__name__}"
        )
    return Result(
        "PASS",
        "",
        "local model loads with lazy=True, generate() returns text, and stream_generate() yields GenerationResponse",
    )


def check_local_model_generation_loop_internals(model_path: str) -> Result:
    genmod = importlib.import_module("mlx_lm.generate")
    default_stream = mx.default_stream(mx.default_device())
    if genmod.generation_stream == default_stream:
        return Result("FAIL", "", "mlx_lm generation_stream unexpectedly matches the default stream")

    counts = {"async_eval": 0, "clear_cache": 0}
    original_async_eval = mx.async_eval
    original_clear_cache = mx.clear_cache

    def wrapped_async_eval(*args, **kwargs):
        counts["async_eval"] += 1
        return original_async_eval(*args, **kwargs)

    def wrapped_clear_cache(*args, **kwargs):
        counts["clear_cache"] += 1
        return original_clear_cache(*args, **kwargs)

    model, tokenizer = MLX_LM["load"](model_path, lazy=True)
    try:
        mx.async_eval = wrapped_async_eval
        mx.clear_cache = wrapped_clear_cache
        _ = MLX_LM["generate"](model, tokenizer, "Say hi.", max_tokens=2, verbose=False)
    finally:
        mx.async_eval = original_async_eval
        mx.clear_cache = original_clear_cache

    if counts["async_eval"] <= 0:
        return Result("FAIL", "", "generate() did not exercise mx.async_eval in the local model check")
    if counts["clear_cache"] <= 0:
        return Result("FAIL", "", "generate() did not exercise mx.clear_cache in the local model check")
    return Result(
        "PASS",
        "",
        f"mlx_lm generate() used a dedicated stream and exercised async_eval={counts['async_eval']} / clear_cache={counts['clear_cache']}",
    )


def check_local_model_prompt_cache(model_path: str) -> Result:
    model, tokenizer = MLX_LM["load"](model_path, lazy=True)
    prompt = mx.array(tokenizer.encode("Hello"), dtype=mx.int32)[None]
    cache = MLX_LM["make_prompt_cache"](model, max_kv_size=16)
    _ = model(prompt, cache=cache)
    mx.eval([c.state for c in cache])

    counts: dict[str, int] = {}
    for c in cache:
        counts[type(c).__name__] = counts.get(type(c).__name__, 0) + 1
    if not counts:
        return Result("FAIL", "", "local model returned an empty prompt cache")
    if sum(c.nbytes for c in cache) <= 0:
        return Result("FAIL", "", "local model prompt cache did not materialize state")

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
        MLX_LM["save_prompt_cache"](handle.name, cache, metadata={"model": "local"})
        loaded, metadata = MLX_LM["load_prompt_cache"](
            handle.name, return_metadata=True
        )

    loaded_counts: dict[str, int] = {}
    for c in loaded:
        loaded_counts[type(c).__name__] = loaded_counts.get(type(c).__name__, 0) + 1
    if loaded_counts != counts:
        return Result(
            "FAIL",
            "",
            f"prompt cache class mix changed across round-trip: {counts} -> {loaded_counts}",
        )
    if metadata.get("model") != "local":
        return Result("FAIL", "", f"unexpected local prompt cache metadata {metadata}")
    return Result(
        "PASS",
        "",
        f"local model prompt cache round-trips with class mix {counts}",
    )


def check_local_model_batch_generate_edge(model_path: str) -> Result:
    model, tokenizer = MLX_LM["load"](model_path, lazy=True)
    prompts = [tokenizer.encode("Hello"), tokenizer.encode("Goodbye")]

    try:
        MLX_LM["batch_generate"](model, tokenizer, prompts, max_tokens=1, verbose=False)
    except ZeroDivisionError:
        pass
    else:
        return Result(
            "FAIL",
            "",
            "expected current batch_generate(max_tokens=1) edge case did not reproduce",
        )

    response = MLX_LM["batch_generate"](
        model, tokenizer, prompts, max_tokens=2, verbose=False
    )
    if len(response.texts) != 2:
        return Result("FAIL", "", f"unexpected batch_generate result length {len(response.texts)}")
    return Result(
        "WARN",
        "",
        "local model reproduced the current batch_generate(max_tokens=1) ZeroDivisionError edge case; max_tokens=2 succeeded",
    )


def print_results(results: Iterable[Result]) -> int:
    results = list(results)
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    worst = "PASS"

    for result in results:
        counts[result.status] += 1
        if STATUS_ORDER[result.status] > STATUS_ORDER[worst]:
            worst = result.status
        print(f"[{result.status}] {result.name}: {result.detail}")

    print()
    print(
        "Summary: "
        f"{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail"
    )
    return 1 if worst == "FAIL" else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the current MLX / MLX-LM cheat sheet claims.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MLX_LM_LOCAL_MODEL"),
        help="Optional local MLX model path for live mlx-lm load/decode checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = [
        ("mlx version", check_mlx_version),
        ("default dtypes", check_default_dtypes),
        ("boolean selection and assignment", check_boolean_selection_and_assignment),
        ("where/nonzero/argwhere", check_where_nonzero_argwhere_absence),
        ("helper creation APIs", check_helper_creation_apis),
        ("indexing surface", check_indexing_surface),
        ("slice copy and aliasing", check_slice_copy_and_aliasing),
        ("duplicate updates", check_duplicate_updates),
        ("bool assignment to low-precision floats", check_bool_assignment_to_low_precision_floats),
        ("index dtype and dynamic slice", check_index_dtype_and_dynamic_slice),
        ("compile and shapeless", check_compile_and_shapeless),
        ("compile and fast surface", check_compile_and_fast_surface),
        ("compile recompilation rules", check_compile_recompilation_rules),
        ("NumPy interop", check_numpy_interop),
        ("random contract", check_random_contract),
        ("out-of-bounds note", check_oob_observation),
        ("at surface and weight layouts", check_at_surface_and_weight_layouts),
        ("training surface and optimizer flow", check_training_surface_and_optimizer_flow),
        ("channels-last surface", check_channels_last_surface),
        ("stream surface", check_stream_surface),
        ("memory profiling surface", check_memory_profiling_surface),
        ("metal kernel surface", check_metal_kernel_surface),
        ("mlx-lm availability/version", require_mlx_lm),
        ("mlx-lm exports and load signature", check_mlx_lm_exports_and_signature),
        ("mlx-lm generation surface", check_mlx_lm_generation_surface),
        ("mlx-lm quantization transform", check_mlx_lm_quantization_transform),
        ("mlx-lm attention mask", check_mlx_lm_attention_mask),
        ("mlx-lm caches", check_mlx_lm_caches),
        ("mlx-lm prompt cache roundtrip", check_mlx_lm_prompt_cache_roundtrip),
    ]
    if args.model_path:
        checks.extend(
            [
                (
                    "local model load and decode",
                    lambda: check_local_model_load_and_decode(args.model_path),
                ),
                (
                    "local model prompt cache roundtrip",
                    lambda: check_local_model_prompt_cache(args.model_path),
                ),
                (
                    "local model generation loop internals",
                    lambda: check_local_model_generation_loop_internals(args.model_path),
                ),
                (
                    "local model batch_generate edge",
                    lambda: check_local_model_batch_generate_edge(args.model_path),
                ),
            ]
        )
    results = [run_check(name, fn) for name, fn in checks]
    return print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
