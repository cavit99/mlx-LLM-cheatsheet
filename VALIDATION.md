# Validation Coverage

Baseline for this repo revision:

- `mlx`: `v0.31.0`
- `mlx-lm`: `v0.31.0`
- Runtime spot checks: March 11, 2026

The cheat sheet mixes three kinds of statements:

- runtime-validated: checked directly by `validation.py`
- source-validated: confirmed from current upstream docs or source, but not
  asserted by the default runtime suite
- advice / observed caveat: guidance or a current observation, not a stable API
  contract

## Runtime-Validated by Default

- MLX version floor and current `0.31.0` baseline
- Literal default dtypes: integer lists -> `int32`, float lists -> `float32`
- `ones_like` / `zeros_like` reject `dtype=`
- `full_like` is absent
- `array.nbytes` matches `size * itemsize`
- Negative indexing, `None`, `...`, `take`, and `take_along_axis`
- Boolean mask selection fails and boolean mask assignment succeeds
- Single-argument `mx.where` is absent
- `mx.nonzero` and `mx.argwhere` are absent
- Integral indexing works and float indices are rejected
- Dynamic Python slicing works
- Aliases share in-place updates
- Slices are copies, not views
- Duplicate direct writes differ from `.at.add(...)`
- `.at` exposes `add`, `subtract`, `multiply`, `divide`, `maximum`, `minimum`
- `mx.array` does not expose a `backward()` training API
- `optimizer.update(...)` is the MLX model-update entry point and `optimizer.step` is optimizer state, not a `.step()` method
- `nn.value_and_grad(...)` plus `mx.eval(model.parameters(), optimizer.state)` drives a working optimizer update
- `mx.compile(...)` works
- `shapeless=True` works across shape changes
- `mx.compile(...)` retraces on rank, dtype, and input-arity changes
- `mx.disable_compile()` exists
- `mx.fast.scaled_dot_product_attention(...)` exists
- `mx.fast.rms_norm(...)` exists
- `mx.fast.metal_kernel(...)` runs a simple fused Metal kernel correctly, including a strided-input path with `ensure_row_contiguous=False`
- `mx.custom_function` exists
- `mx.metal.is_available()`, `mx.metal.start_capture(...)`, and `mx.metal.stop_capture()` are present
- NumPy `copy=False` returns a view onto MLX memory
- `bfloat16` fails direct NumPy conversion and works after cast
- NumPy `float64` defaults to MLX `float32`
- `mx.random.key(...)` and `mx.random.seed(...)` are reproducible
- MLX `nn.Linear`, `Conv1d`, and `Conv2d` weight layouts match the cheat sheet
- MLX convolution inputs are channels-last: `NLC`, `NHWC`, and `NDHWC`
- Core stream APIs exist, `stream=` works on ops, and `mx.new_stream(...)` returns a non-default stream
- `mlx_lm` exports `load`, `generate`, `stream_generate`, `batch_generate`, `convert`
- `mlx_lm.load(...)` supports `lazy=` and `revision=`
- `stream_generate(...)` and `batch_generate(...)` signatures match the current API
- `GenerationResponse` fields match the cheat sheet
- Synthetic AutoAWQ / GPTQ packed weights transform into MLX quantized layout
- `create_attention_mask(...)` returns `"causal"`, `None`, or an explicit array depending on inputs
- `KVCache`, `RotatingKVCache`, `QuantizedKVCache`, `ConcatenateKVCache`, and `ArraysCache` match the checked surface
- Default prompt-cache fallback with `max_kv_size` uses `RotatingKVCache`
- Prompt caches round-trip through safetensors

## Source-Validated Only

- MLX is lazy by default
- Implicit evaluation triggers include `print(array)`, `np.array(array)`, `memoryview(array)`, `array.item()`, and save/load paths
- Out-of-bounds indexing is undefined behavior
- `float64` is CPU-only
- Module parameters are created lazily and `mx.eval(model.parameters())` allocates / initializes them
- Compiled functions should avoid side effects unless state is captured via `inputs=` / `outputs=`
- Metal capture needs `MTL_CAPTURE_ENABLED=1`, and `MLX_METAL_DEBUG` improves capture readability
- C++ `Primitive` extensions are the deeper path when `mx.fast.metal_kernel(...)` is not enough
- Attention-mask and attention-kernel logic in `mlx-lm` uses `"causal"` fast paths and quantized SDPA paths
- Typical Q/K/V reshape pattern in transformer attention implementations is `(B, heads, L, head_dim)`
- `mlx-lm.generate` uses chunked prefill and optional KV-cache quantization
- `mlx_lm.utils.load_model(...)` eagerly evaluates parameters when `lazy=False`
- `mlx_lm.utils` handles legacy quantization configs for `bitnet`, `mxfp4`, and compressed tensors
- Memory sizing in `mlx-lm` uses `tree_reduce(... x.nbytes ...)` and `mx.device_info()["max_recommended_working_set_size"]`
- PyTorch `memoryview` support is documented as experimental for multi-dimensional arrays
- BatchNorm and channel-wise dropout docs follow channels-last layout conventions

## Advice or Observed Caveats

- Do not use `mx.arange(arr.size)[:mx.sum(mask).item()]` as a boolean-mask workaround; it selects the first `N` positions, not the `True` positions
- Prefer explicit index dtypes in indexing-heavy or low-precision code
- Benchmark `mx.take(...)` against `arr[idx]` on the target machine instead of assuming one is faster
- Treat undocumented behavior as unstable, especially out-of-bounds indexing and silent assignment casts
- Grouped convolution interop needs more than a simple axis permutation because MLX stores `in // groups`
- Benchmarking without `mx.eval(...)` or `mx.synchronize(...)` often measures enqueue / dispatch rather than full compute

## Optional Live-Model Validation

Run:

```bash
MLX_LM_LOCAL_MODEL=/path/to/mlx-model python validation.py
```

When a local model path is provided, the validator additionally checks:

- `load(..., lazy=True)` on a real local model
- `generate(...)` on a real local model
- `stream_generate(...)` yielding a real `GenerationResponse`
- Prompt-cache save/load on the real model cache
- `generate(...)` uses a dedicated non-default `generation_stream`
- `generate(...)` exercises `mx.async_eval(...)` and `mx.clear_cache()`
- The current `batch_generate(max_tokens=1)` edge case

Local spot checks for this repo revision used a Qwen3.5 35B A3B 4-bit MLX model.
Observed results:

- real prompt cache mix: `30 x ArraysCache`, `10 x KVCache`
- `batch_generate` rejects raw string prompts because it expects token ID lists
- `batch_generate(..., max_tokens=1)` raised `ZeroDivisionError`
- `batch_generate(..., max_tokens=2)` succeeded on the same model
