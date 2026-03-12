---
name: mlx
description: "Use when working with Apple's MLX or MLX-LM: fact-checking current behavior against upstream source/runtime, patching MLX-based repos, porting PyTorch/JAX code to MLX, validating lazy evaluation, indexing, compilation, streams, channels-last layouts, Metal kernels, quantization, caches, and local MLX model loading or generation on Apple silicon."
---

# MLX

Use this skill for MLX or MLX-LM engineering work where correctness depends on
current upstream behavior, not model memory.

## When to Use

- Auditing or patching MLX or MLX-LM repos
- Fact-checking "latest" MLX or MLX-LM behavior
- Porting PyTorch or JAX code to MLX
- Debugging MLX indexing, lazy evaluation, compilation, or stream behavior
- Deciding when to use stock ops, `mx.fast.*`, `mx.fast.metal_kernel(...)`, or a deeper extension path
- Profiling or debugging MLX GPU execution with Metal capture hooks
- Profiling MLX memory usage or allocator/cache behavior on Apple silicon
- Reviewing MLX-LM model load, cache, prompt-cache, quantization, or generation code
- Validating local MLX model paths on Apple silicon

## Core Rules

- If the user asks for current or latest MLX facts, verify releases and source first.
- Prefer upstream docs/source plus runtime checks over memory.
- Treat undocumented runtime behavior as unstable.
- Distinguish documented contracts from observed caveats.
- Keep MLX-LM inference checks local and minimal: `lazy=True`, short prompts, small
  `max_tokens`.

## Quick Start

Set the skill path once:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export MLX_SKILL="$CODEX_HOME/skills/mlx"
```

Check latest upstream releases:

```bash
"$MLX_SKILL/scripts/mlx_release_info.sh"
```

Run the bundled runtime probe:

```bash
"$MLX_SKILL/scripts/mlx_probe.sh"
```

The launcher checks both `python3` and `python` and picks one that can import
`mlx`.

Run the probe with a local MLX model:

```bash
MLX_LM_LOCAL_MODEL=/path/to/model "$MLX_SKILL/scripts/mlx_probe.sh"
```

## Workflow

### 1. Classify the task

- **Current facts**: verify latest `mlx` / `mlx-lm` releases, then inspect source
- **Repo validation**: run the repo's own validator if it exists; otherwise use the bundled probe
- **Porting or debugging**: check the current facts reference, then validate the specific behavior locally
- **Local model inference**: use a local MLX model path and keep decode checks short

### 2. For current upstream facts

Use authenticated GitHub workflows when possible:

```bash
"$MLX_SKILL/scripts/mlx_release_info.sh"
gh repo clone ml-explore/mlx /tmp/mlx-upstream -- --depth 1
gh repo clone ml-explore/mlx-lm /tmp/mlx-lm-upstream -- --depth 1
```

Inspect only the files relevant to the question. Typical targets:

- MLX: `docs/src/usage/indexing.rst`, `lazy_evaluation.rst`, `compile.rst`,
  `numpy.rst`, `python/data_types.rst`, `python/memory_management.rst`,
  `python/mlx/nn/layers/convolution.py`,
  `docs/src/dev/custom_metal_kernels.rst`, `docs/src/dev/metal_debugger.rst`,
  `docs/src/dev/extensions.rst`
- MLX-LM: `mlx_lm/generate.py`, `mlx_lm/utils.py`, `mlx_lm/models/base.py`,
  `mlx_lm/models/cache.py`

### 3. For runtime validation

If the repo already has an MLX validator, prefer that first.

Otherwise run:

```bash
"$MLX_SKILL/scripts/mlx_probe.sh"
```

The bundled probe checks high-signal MLX and MLX-LM behavior:

- indexing and mask limitations
- slice-copy vs aliasing
- compile and retracing rules
- training flow and optimizer semantics
- channels-last inputs
- stream APIs
- custom Metal kernel and capture-hook surface
- MLX-LM API surface, attention mask, caches, prompt-cache roundtrip
- AutoAWQ/GPTQ transform helpers

### 4. For local model checks

Use a local MLX model path when load/generate behavior matters:

```bash
MLX_LM_LOCAL_MODEL=/path/to/model "$MLX_SKILL/scripts/mlx_probe.sh"
```

This adds:

- real `load(..., lazy=True)`
- one-step `generate(...)`
- `stream_generate(...)` response validation
- prompt-cache save/load on the actual model cache
- generation-stream / `async_eval` / `clear_cache` checks

### 5. For porting or reviews

Check [current-facts.md](./references/current-facts.md) first.

Then use [porting-checklist.md](./references/porting-checklist.md) for the
common MLX-specific failure modes:

- boolean mask selection unsupported
- slices are copies, not views
- no tensor `backward()` pattern
- explicit `mx.eval(...)` required in training and timing
- channels-last activations
- stream-aware benchmarking
- MLX-LM cache and generation API differences

## Kernel Escalation Path

- Start with stock MLX ops.
- If there is already a tuned kernel in `mx.fast.*`, prefer that first.
- Use `mx.fast.metal_kernel(...)` for Apple-only fused hot paths when the stock
  op graph is the bottleneck.
- Be explicit about contiguity: `ensure_row_contiguous=True` can hide copies.
- Use `@mx.custom_function` when the custom kernel also needs custom gradient
  logic.
- Move to C++ `Primitive` extensions only when Python-level Metal kernels are
  not enough.
- For serious GPU profiling, capture a `.gputrace` with
  `mx.metal.start_capture(...)` / `mx.metal.stop_capture()` and inspect it in
  Xcode.

## High-Signal MLX Differences

- Training is `nn.value_and_grad(...)` plus `optimizer.update(...)` plus
  `mx.eval(model.parameters(), optimizer.state)`.
- Module parameters are created lazily; explicit `mx.eval(model.parameters())`
  matters before timing and export.
- Conv inputs are channels-last: `NLC`, `NHWC`, `NDHWC`.
- `mx.compile(...)` retraces on dtype, rank, and input-arity changes.
- `shapeless=True` avoids shape-only retracing but can break shape-dependent code.
- Streams are first-class, and timing without `mx.eval(...)` or
  `mx.synchronize(...)` is often wrong.
- Memory profiling should use the top-level `mx.get_*_memory()` helpers and
  `mx.device_info()`, not deprecated `mx.metal.*` aliases.
- MLX has a real Python-level fused-kernel escape hatch in
  `mx.fast.metal_kernel(...)`.

## High-Signal MLX-LM Differences

- `generate(...)` and `stream_generate(...)` accept strings or token IDs.
- `batch_generate(...)` expects token ID lists, not raw strings.
- `stream_generate(...)` yields `GenerationResponse` objects.
- Prompt caches are not always pure KV caches; hybrid models can mix
  `ArraysCache` and `KVCache`.
- Current `mlx-lm==0.31.0` caveat: `batch_generate(..., max_tokens=1)` can hit a
  `ZeroDivisionError`.

## References

- Current validated facts and caveats: [current-facts.md](./references/current-facts.md)
- Porting and review checklist: [porting-checklist.md](./references/porting-checklist.md)

## Helpers

- Release helper: [scripts/mlx_release_info.sh](./scripts/mlx_release_info.sh)
- Runtime probe launcher: [scripts/mlx_probe.sh](./scripts/mlx_probe.sh)
- Runtime probe implementation: [scripts/mlx_probe.py](./scripts/mlx_probe.py)
