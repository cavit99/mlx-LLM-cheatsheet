# Current Facts

Validated baseline for this skill:

- `mlx`: `v0.31.1`
- `mlx-lm`: `v0.31.0`
- Validation date: March 12, 2026

Re-check release status with:

```bash
"$CODEX_HOME/skills/mlx/scripts/mlx_release_info.sh"
```

## MLX

- Integer literals default to `int32`
- Float literals default to `float32`
- Boolean mask selection is unsupported
- Boolean mask assignment is supported
- `mx.nonzero`, `mx.argwhere`, and single-argument `mx.where` are absent
- Slices are copies, not views
- Duplicate direct indexed writes are not safe accumulation; use `.at[...]`
- `mx.compile(...)` retraces on dtype, rank, and input arity
- `shapeless=True` avoids shape-only retracing
- Training uses `nn.value_and_grad(...)`, `optimizer.update(...)`, and explicit `mx.eval(...)`
- Conv inputs are channels-last: `NLC`, `NHWC`, `NDHWC`
- Stream APIs are normal MLX surface, not a niche internal detail
- `mx.bartlett(...)` now exists and matches NumPy semantics
- `mx.fast.metal_kernel(...)` exists and works as the Python-level fused-kernel path
- `mx.custom_function` is the custom-gradient hook that pairs with bespoke kernels
- `mx.metal.start_capture(...)` / `mx.metal.stop_capture()` exist for GPU traces
- Bool assignment into `float16` / `bfloat16` arrays now stores numeric `1.0` / `0.0`

## MLX-LM

- Top-level exports include `load`, `generate`, `stream_generate`,
  `batch_generate`, `convert`
- `batch_generate(...)` takes token ID lists, not raw strings
- `stream_generate(...)` yields `GenerationResponse`
- `create_attention_mask(...)` can return `"causal"`, `None`, or an explicit array
- Prompt caches can mix `ArraysCache` and `KVCache`
- Synthetic AutoAWQ/GPTQ transforms map packed weights into MLX quantized layout

## Current Caveats

- Out-of-bounds indexing is undefined behavior even if the current runtime
  returns zero-like or garbage values
- PyPI `mlx-lm==0.31.0` is yanked; prefer the GitHub `v0.31.0` tag or a newer
  non-yanked release
- On a real local MLX model check, `batch_generate(..., max_tokens=1)` reproduced
  a `ZeroDivisionError`, while `max_tokens >= 2` succeeded

## Source-Only Facts

Keep these source-backed unless you need a dedicated probe:

- implicit lazy-eval triggers such as `print(array)`, `np.array(array)`,
  `memoryview(array)`, `array.item()`, and save/load
- parameter materialization wording around allocation and initialization
- deeper generation-loop implementation details beyond the validated helpers
- `MTL_CAPTURE_ENABLED=1` and `MLX_METAL_DEBUG` capture workflow details
