# MLX + MLX-LM Cheat Sheet

Source-backed baseline:

- `mlx` source tag: `v0.31.1`
- `mlx-lm` source tag: `v0.31.0`
- Runtime spot checks: March 12, 2026

Use documented contracts first. If runtime behavior is not documented upstream,
treat it as unstable unless you verify it again.

## High-Signal Rules

- MLX is lazy. Many operations build graphs until `mx.eval(...)` or an implicit
  evaluation trigger happens.
- MLX arrays support in-place updates.
- Aliases share updates, but slices are copies, not views.
- Module parameters and optimizer updates are lazy; `mx.eval(...)` is part of
  correct training and benchmarking flow.
- Boolean mask selection is unsupported. Boolean mask assignment is supported.
- Out-of-bounds indexing is undefined behavior. Do not rely on current
  zero-filled results.
- Channels-last is the default layout family for MLX conv and related
  channel-wise layers.
- Streams are first-class. If you benchmark without `mx.eval(...)` or
  `mx.synchronize(...)`, you are often timing enqueue/dispatch, not compute.
- `mlx-lm` generation code is built around explicit KV caches, prompt caching,
  and `mx.fast.scaled_dot_product_attention(...)`.

## Array Creation and Dtypes

- `mx.array([1, 2])` defaults to `int32`.
- `mx.array([1.0, 2.0])` defaults to `float32`.
- Supported floating types include `bfloat16`, `float16`, `float32`, and
  `float64`.
- `float64` is CPU-only. Using it on GPU-backed operations is not supported.
- `array.nbytes` exists and is useful for model/cache sizing.
- `mx.ones_like(x)` and `mx.zeros_like(x)` exist, but do not accept a `dtype=`
  keyword.
- There is no `mx.full_like`. Use `mx.full(x.shape, value, dtype)` instead.
- `mx.bartlett(M)` now exists alongside `mx.hanning`, `mx.hamming`, and
  `mx.blackman`.

```python
import mlx.core as mx

x = mx.array([1, 2])          # int32
y = mx.array([1.0, 2.0])      # float32
z = mx.full(x.shape, 7, mx.int32)
```

## Indexing

- Supported: integers, negative integers, slices, `...`, `None`, and integral
  MLX array indices.
- `mx.take(...)` and `mx.take_along_axis(...)` are the official helper
  functions for gather-style indexing.
- Dynamic Python slicing works: `arr[:n]` is valid.
- Boolean selection is unsupported:
  `arr[arr > 0]` raises `ValueError`.
- Boolean assignment is supported:
  `arr[mask] = updates`.
- Single-argument `mx.where(mask)` is unsupported.
- `mx.nonzero` and `mx.argwhere` are not available.
- Index arrays must be integral.
- Out-of-bounds indexing is documented as undefined behavior.

```python
idx = mx.array([5, 7], dtype=mx.int32)
out = arr[idx]

rows = mx.array([1, 3], dtype=mx.int32)
cols = mx.array([2, 0], dtype=mx.int32)
block = matrix[rows[:, None], cols[None, :]]
```

### Boolean Mask Caveat

Do not use this:

```python
indices = mx.arange(arr.size, dtype=mx.int32)[:mx.sum(mask).item()]
```

That selects the first `N` positions, not the positions where `mask` is `True`.

If you truly need indices from a mask, you need a documented workaround built
from three-argument `mx.where(...)` plus compaction logic, or a reformulation
that avoids data-dependent output shapes entirely.

## Updates, Aliasing, and `.at`

- In-place updates are supported: `a[2] = 0`.
- Aliases see the same updates:
  `b = a; b[2] = 0` also changes `a`.
- Slices are copies, not views:
  `b = a[:]` can be mutated without changing `a`.
- Assigning booleans into floating arrays stores numeric `1.0` / `0.0`.
  This is fixed for `float16` and `bfloat16` in current MLX.
- Repeated direct writes to the same index are nondeterministic.
- Use `.at[...]` when all updates to duplicate indices must be applied.

```python
a = mx.array([0, 0])
idx = mx.array([0, 1, 0, 1], dtype=mx.int32)

a[idx] += 1           # one update per location survives, not an accumulation
a = a.at[idx].add(1)  # accumulates all updates
```

Supported `.at` operations:

- `.add(...)`
- `.subtract(...)`
- `.multiply(...)`
- `.divide(...)`
- `.maximum(...)`
- `.minimum(...)`

## Lazy Evaluation

- Most MLX ops build a graph and defer execution.
- Explicit triggers: `mx.eval(...)`, `mx.async_eval(...)`
- Implicit triggers include:
  - `print(array)`
  - `np.array(array)`
  - `memoryview(array)`
  - `array.item()`
  - save/load paths that need materialized data
- Good default: evaluate at iteration boundaries, not after every small op.

```python
loss, grad = value_and_grad_fn(model, batch)
optimizer.update(model, grad)
mx.eval(loss, model.parameters(), optimizer.state)
```

## Modules, Training, and Optimizers

- There is no tensor `loss.backward()` pattern on `mx.array`.
- The MLX-native training flow is:
  `nn.value_and_grad(model, loss_fn)`, then `optimizer.update(model, grads)`,
  then `mx.eval(model.parameters(), optimizer.state)`.
- `optimizer.step` is optimizer state, not a `.step()` method.
- `nn.value_and_grad(model, loss_fn)` differentiates with respect to
  `model.trainable_parameters()`.
- `Module.freeze()` changes what `nn.value_and_grad(...)` differentiates.
- Module parameters are created lazily; `mx.eval(model.parameters())`
  allocates and initializes them.
- Do that explicit parameter evaluation before timing, export, checkpoint
  sizing, or first-token measurements.

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

model = MyModel()
optimizer = optim.Adam(learning_rate=1e-4)
mx.eval(model.parameters())

def loss_fn(model, batch):
    logits = model(batch["x"])
    return cross_entropy(logits, batch["y"])

loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, batch)
optimizer.update(model, grads)
mx.eval(loss, model.parameters(), optimizer.state)
```

## Function Transforms and Compilation

- `mx.grad(fn)` computes gradients.
- `mx.value_and_grad(fn)` returns both value and gradient.
- `mx.vmap(fn, in_axes=..., out_axes=...)` vectorizes functions.
- `mx.compile(fun, inputs=None, outputs=None, shapeless=False)` compiles a
  function.

Compilation details that matter:

- First call compiles and is slower.
- Recompilation is triggered by changes in dtype, rank, or input arity.
- `shapeless=True` avoids recompiling on shape changes, but shape-dependent code
  can break.
- Compiled functions should avoid side effects unless state is captured via
  `inputs=` and `outputs=`.
- Use `mx.disable_compile()` or `MLX_DISABLE_COMPILE=1` for debugging.

## NumPy and PyTorch Interop

- `np.array(mx_array)` copies by default.
- `np.array(mx_array, copy=False)` creates a NumPy view onto MLX memory.
- Changes through a NumPy view affect the MLX array.
- `bfloat16` must be cast before NumPy conversion:
  `np.array(x.astype(mx.float32))`.
- NumPy `float64` arrays are converted to MLX `float32` by default when passed
  through `mx.array(...)`.
- PyTorch support through `memoryview` exists, but upstream documents it as
  experimental for multi-dimensional arrays. Going through NumPy is safer.

```python
view = np.array(a, copy=False)
view[0] = 1
assert a[0].item() == 1
```

## Layout Conventions

- MLX `Conv1d` inputs are `NLC`.
- MLX `Conv2d` inputs are `NHWC`.
- MLX `Conv3d` inputs are `NDHWC`.
- `BatchNorm` also uses `NC` / `NLC` / `NHWC`.
- Channel-wise dropout layers also assume channels-last:
  `Dropout2d` expects `WHC` / `NHWC`, and `Dropout3d` expects `DHWC` /
  `NDHWC`.
- Porting conv or image/audio code from PyTorch often requires transposing
  activations, not just weights.

```python
x1 = mx.ones((batch, length, channels))        # Conv1d
x2 = mx.ones((batch, height, width, channels)) # Conv2d
x3 = mx.ones((batch, depth, height, width, channels))  # Conv3d
```

## Streams and Timing

- Most MLX ops accept `stream=...`.
- Core stream APIs: `mx.default_stream(...)`, `mx.new_stream(...)`,
  `with mx.stream(s):`, `mx.synchronize(...)`
- `mlx-lm` generation uses a dedicated non-default stream.
- Accurate timings need an evaluation or synchronization boundary:
  `mx.eval(...)` or `mx.synchronize(stream)`.
- This matters for token/s throughput measurements and microbenchmarks.

```python
s = mx.new_stream(mx.default_device())
y = mx.matmul(a, b, stream=s)
mx.synchronize(s)
```

## Custom Metal Kernels and Debugging

- `mx.fast.metal_kernel(...)` is the Python-level fused-kernel escape hatch for
  custom Metal code.
- Build the kernel once and reuse it. Kernel creation can JIT compile a new
  Metal library.
- `ensure_row_contiguous=True` by default can hide input copies. Disable it
  only if your kernel handles `*_shape`, `*_strides`, and `*_ndim` correctly.
- `@mx.custom_function` is the MLX path for pairing a custom forward kernel
  with custom differentiation logic.
- Use standard MLX ops first, then existing `mx.fast.*` kernels, then
  `mx.fast.metal_kernel(...)`, and only then C++ `Primitive` extensions if you
  need deeper integration.
- `mx.metal.start_capture(path)` / `mx.metal.stop_capture()` can record a
  `.gputrace` for Xcode. Capture requires `MTL_CAPTURE_ENABLED=1`.
- Building MLX with `MLX_METAL_DEBUG` improves capture readability.

```python
source = """
    uint elem = thread_position_in_grid.x;
    uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
    out[elem] = metal::exp(inp[loc]);
"""

exp_kernel = mx.fast.metal_kernel(
    name="exp_strided",
    input_names=["inp"],
    output_names=["out"],
    source=source,
    ensure_row_contiguous=False,
)

x = mx.arange(8, dtype=mx.float32).reshape(4, 2)[::2]
y = exp_kernel(
    inputs=[x],
    template=[("T", mx.float32)],
    grid=(x.size, 1, 1),
    threadgroup=(x.size, 1, 1),
    output_shapes=[x.shape],
    output_dtypes=[x.dtype],
)[0]
mx.eval(y)
```

## Performance and Correctness Habits

- Use `mx.fast.*` kernels when the model stack already expects them, especially
  `mx.fast.scaled_dot_product_attention(...)` and `mx.fast.rms_norm(...)`.
- Do not assume `mx.take(...)` is faster than `arr[idx]`; benchmark on the
  target machine.
- Treat undocumented behavior as unstable, especially out-of-bounds indexing
  and silent type casts during assignment.
- Prefer explicit dtype control for index tensors and low-precision paths.
- Use `array.nbytes` and tree reductions for memory accounting.

## MLX-LM: What Actually Shows Up in Real Code

### Top-Level API

`mlx_lm` exports:

- `load(...)`
- `generate(...)`
- `stream_generate(...)`
- `batch_generate(...)`
- `convert(...)`

`load(...)` matters because:

- `lazy=False` eagerly evaluates parameters with `mx.eval(model.parameters())`
- `lazy=True` defers parameter materialization
- `revision=` is supported for loading a specific Hub revision

Prompt handling details that matter:

- `generate(...)` accepts a string prompt or token IDs
- `stream_generate(...)` accepts a string prompt or token IDs
- `batch_generate(...)` takes `List[List[int]]` token IDs, not raw strings

`stream_generate(...)` yields `GenerationResponse` objects with:

- `text`
- `token`
- `logprobs`
- `from_draft`
- `prompt_tokens`
- `prompt_tps`
- `generation_tokens`
- `generation_tps`
- `peak_memory`
- `finish_reason`

Observed runtime caveat on `mlx-lm==0.31.0`:

- `batch_generate(..., max_tokens=1)` hit a `ZeroDivisionError` in a local
  Qwen3.5 spot check because `BatchGenerator.stats()` divides by
  `generation_time` without a zero guard. Re-test tiny batch decode paths on
  the target runtime.

### Attention Masks

`mlx_lm.models.base.create_attention_mask(...)` does not always return an array.

It can return:

- `"causal"` for the fast path
- `None` when sequence length is `1`
- an explicit boolean array when requested or when a windowed mask is needed

This is easy to get wrong if you assume every mask is already materialized.

### Attention Tensor Layout

Most model implementations reshape Q/K/V into:

- `(B, heads, L, head_dim)` for queries
- `(B, kv_heads, L, head_dim)` for keys and values

Typical pattern:

```python
queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
```

### Caches

Important cache types in `mlx-lm` include:

- `KVCache`
- `RotatingKVCache`
- `QuantizedKVCache`
- `ConcatenateKVCache`
- `ArraysCache`

KV-style invariants:

- `cache.offset` tracks processed tokens
- key/value sequence length lives on axis `-2`
- `RotatingKVCache` is the bounded-memory path for long generation
- prompt caches can be saved and reloaded with safetensors

Not every cache is KV-style:

- hybrid and state-space models can use `ArraysCache` or mixed cache lists
- `ArraysCache` stores opaque per-layer state and does not expose
  `keys`/`values`/`offset`
- current example: `qwen3_5.Model.make_cache()` returns `ArraysCache(size=2)`
  for linear-attention layers and `KVCache()` for full-attention layers
- `can_trim_prompt_cache(...)` is model-dependent and can be `False` for mixed
  caches

Prompt cache helpers:

- `make_prompt_cache(...)`
- `can_trim_prompt_cache(...)`
- `save_prompt_cache(...)`
- `load_prompt_cache(...)`
- `trim_prompt_cache(...)`

### Generation Loop Patterns

`mlx_lm.generate` uses several MLX-specific patterns that are worth knowing:

- a dedicated generation stream: `mx.new_stream(mx.default_device())`
- chunked prompt prefill
- `mx.async_eval(...)` for overlapping next-step work
- periodic `mx.clear_cache()`
- optional KV-cache quantization after a threshold

These are good references when writing your own efficient decode loop.

### Attention Kernel Choice

`mlx-lm` uses two main attention paths:

- default: `mx.fast.scaled_dot_product_attention(...)`
- quantized cache path: `mx.quantized_matmul(...)` based SDPA

Do not assume every attention path is the same once KV cache quantization is
enabled.

### Weight Layouts That Matter

- `nn.Linear.weight` is `[out_dim, in_dim]`
- MLX `Conv1d` weight is `[out, kernel, in // groups]`
- MLX `Conv2d` weight is `[out, kh, kw, in // groups]`

For PyTorch interop:

- 1D conv: MLX `[out, kernel, in]` -> PyTorch `[out, in, kernel]`
- 2D conv: MLX `[out, kh, kw, in]` -> PyTorch `[out, in, kh, kw]`

Grouped convolutions still use `in // groups` in the MLX weight tensor, so the
simple permutation rule is not the whole story.

### Quantization and Conversion

`mlx_lm.utils` handles several quantization paths directly:

- native MLX quantization configs
- activation quantization for supported quantized linears
- legacy config handling for `bitnet`, `mxfp4`, and compressed tensors
- AWQ/GPTQ transform path that unpacks packed weights, transposes to MLX
  layout, then repacks

If a model was originally authored for another stack, layout conversion is often
the real bug source, not the math itself.

### Memory Sizing

`mlx-lm` uses `tree_reduce(... x.nbytes ...)` to estimate model size and compares
it against `mx.device_info()["max_recommended_working_set_size"]`.

That is the right mental model for "will this fit comfortably" checks on Apple
silicon.

## Missing or Easy to Hallucinate Today

- No boolean mask selection
- No `mx.nonzero`
- No `mx.argwhere`
- No single-argument `mx.where`
- No `mx.full_like`
- No `dtype=` kwarg for `mx.ones_like` / `mx.zeros_like`
- Out-of-bounds behavior is not a stable contract
- Duplicate direct writes are not safe accumulation

## Primary Sources

- [MLX indexing docs](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/indexing.rst)
- [MLX lazy evaluation docs](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst)
- [MLX compile docs](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/compile.rst)
- [MLX function transforms docs](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/function_transforms.rst)
- [MLX NumPy and framework interop docs](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/numpy.rst)
- [MLX data types docs](https://github.com/ml-explore/mlx/blob/main/docs/src/python/data_types.rst)
- [MLX custom Metal kernels docs](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst)
- [MLX Metal debugger docs](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/metal_debugger.rst)
- [MLX custom extensions docs](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/extensions.rst)
- [MLX indexing implementation](https://github.com/ml-explore/mlx/blob/main/python/src/indexing.cpp)
- [MLX array implementation](https://github.com/ml-explore/mlx/blob/main/python/src/array.cpp)
- [MLX-LM README](https://github.com/ml-explore/mlx-lm/blob/main/README.md)
- [MLX-LM base attention helpers](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py)
- [MLX-LM cache helpers](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- [MLX-LM generation loop](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
- [MLX-LM loader and quantization utils](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py)
