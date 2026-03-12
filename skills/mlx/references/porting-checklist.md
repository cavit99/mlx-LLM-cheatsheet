# Porting Checklist

Use this when moving PyTorch or JAX code into MLX or reviewing MLX code that
feels "Torch-shaped".

## Data and Indexing

- Replace boolean mask selection patterns like `x[x > 0]`
- Keep index tensors integral
- Assume slices are copies, not views
- Treat out-of-bounds indexing as undefined behavior

## Training

- Replace `loss.backward()` with `nn.value_and_grad(...)`
- Use `optimizer.update(model, grads)`
- Materialize updates with `mx.eval(model.parameters(), optimizer.state)`
- If using partial finetuning, check `trainable_parameters()` / `freeze()`

## Layouts

- Conv activations are channels-last: `NLC`, `NHWC`, `NDHWC`
- Do not only permute weights; verify activation layout too
- BatchNorm and channel-wise dropout follow the same layout family

## Compilation and Performance

- Benchmark with `mx.eval(...)` or `mx.synchronize(...)`
- Expect first-call compile cost
- Expect recompilation on dtype, rank, or input-arity changes
- Use `shapeless=True` only when shape variability is the real problem
- Prefer existing `mx.fast.*` kernels before reaching for custom Metal code
- If you use `mx.fast.metal_kernel(...)`, check whether `ensure_row_contiguous=True` is hiding copies
- For GPU bottlenecks, capture a `.gputrace` instead of guessing from dispatch timings

## MLX-LM

- `generate(...)` and `stream_generate(...)` can take strings
- `batch_generate(...)` takes token ID lists
- Prompt caches may be mixed-cache structures, not pure KV tensors
- Keep live model checks minimal: `lazy=True`, short prompt, low `max_tokens`

## Review Questions

- Is the code assuming boolean mask selection exists?
- Is it assuming NumPy-like slice views?
- Is it benchmarking without synchronization?
- Is it using NCHW-style tensors where MLX expects NHWC?
- Is a hidden contiguous copy inside a custom Metal kernel undoing the expected speedup?
- Is it assuming prompt cache entries all expose `keys`, `values`, and `offset`?
