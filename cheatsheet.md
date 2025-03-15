This guide is tailored for MLX, a machine learning framework optimized for Apple Silicon with Metal GPU acceleration. MLX uses lazy evaluation and differs from NumPy and PyTorch, especially in indexing and array operations. Use this to write correct, efficient MLX code and avoid common pitfalls.

---

### Array Creation
- **From Data**: Use `mx.array(data)` to create an array from a list or iterable.  
  Example: `mx.array([1, 2, 3])`.
- **Filled Arrays**:  
  - `mx.ones(shape, dtype)`: Array of ones. Example: `mx.ones((3, 2), mx.float32)`.  
  - `mx.zeros(shape, dtype)`: Array of zeros. Example: `mx.zeros((2, 2), mx.int32)`.  
  - `mx.full(shape, value, dtype)`: Fill with a value. Use for `full_like`/`ones_like` workarounds.  
    Examples: `mx.full((2, 3), 5, mx.float32)`, `mx.full(x.shape, -1, mx.float32)`, `mx.full(x.shape, 1, mx.bool_)`.
- **Default Dtype**: `mx.float32` unless specified. Example: `mx.array([1, 2])` → `float32`.
- **Supported Dtypes**: See "Key Differences: Dtypes". Example: `mx.array([1], dtype=mx.int32)`.
- There is no ones_like with dtype support; use mx.full(x.shape, 1, dtype) instead.

---

### Array Indexing
- **Basic Indexing**: Supports integers, static slices, ellipsis (`...`), `None`, and Python variables in slices (e.g., `arr[:n]`).  
  Examples: `arr[3]`, `arr[2:8:2]`, `arr[:5]`, `arr[..., 0]`, `arr[None]`.
- **Array Indexing**: Use `mx.array` with integral dtype (e.g., `mx.int32`) for indexing or `mx.take`. Always specify dtype explicitly to prevent mismatches.  
  Examples: `arr[mx.array([5, 7], dtype=mx.int32)]`, `mx.take(arr, mx.arange(3, dtype=mx.int32))`.  
  Invalid: `mx.take(arr, mx.array([0.5, 1.5]))` → `ValueError: Indices must be integral`.
- **No Boolean Indexing**: `arr[arr > 0]` fails (`ValueError: boolean indices not supported`). Use `mx.where` with three arguments:  
  - **Reliable Workaround**: Compute valid indices with `mx.where` and `mx.sum`.  
    Example: 
    ```python
	arr = mx.array([1.5, -2.0, 0.0, 3.7, -1.0, 4.2], dtype=mx.float32) 
	mask = arr > 0 indices_all = mx.arange(arr.size, dtype=mx.int32) 
	num_valid = mx.sum(mask).item() 
	temp_indices = mx.where(mask, indices_all, arr.size) 
	sorted_indices = mx.argsort(temp_indices) 
	valid_indices = mx.take(temp_indices, sorted_indices[:num_valid]) 
	filtered = arr[valid_indices] 
	print(valid_indices, filtered)
    ```
  - **Avoid**: `mx.where(mask, indices, -1)[:-1]`—unreliable, may overcount elements (e.g., 9999 vs. 4999 valid) or include invalid indices.
  - **Empty Masks**: May fail with `ValueError: Shapes cannot be broadcast` if mask shape mismatches (e.g., `(0,)` vs. `(10000,)`). Ensure compatible shapes.
- **Multi-Axis Indexing**: Use `mx.take(arr, indices, axis)` for 1D or N-D extraction (faster than `[]` in some cases).  
  Example: `mx.take(arr, mx.array([0, 2], dtype=mx.int32), axis=0)`.
- **Batch Operations**: Combine small index steps into single `mx.take` calls to reduce overhead.  
  Example: `mx.take(arr, mx.array([0, 2, 4], dtype=mx.int32))`.
- **Reuse Indices**: Cache frequent index arrays (e.g., `sorted_indices`) to avoid recomputation.  
  Example: `sorted_indices = mx.argsort(arr); top_3 = arr[sorted_indices[:3]]`.
- **Index Buffer Pre-allocation**: Pre-allocate writeable arrays for frequent index updates. Size must match or exceed assigned indices.  
  Example: 
  ```python
  max_indices = 10
  index_buffer = mx.zeros((max_indices,), dtype=mx.int32)
  index_buffer[:3] = mx.array([0, 2, 4], dtype=mx.int32)  # Succeeds
  # index_buffer[:5] = mx.array([0, 1, 2, 3, 4, 5])  # Fails: Shape mismatch
  result = arr[index_buffer[:3]]
  ```
  - **`.at` API**: Use `arr.at[indices].operation(value)` (e.g., `.add`, `.multiply`) for operations like duplicate index updates. Returns a new array (not in-place); reassign and evaluate:  
    Example: `arr = arr.at[mx.array([0, 0], dtype=mx.int32)].add(1); mx.eval(arr)` → `[2, 0, ...]`.
  - Float Indices Behavior: Float indices (e.g., mx.array([1.5, 2.5])) are rejected in indexing operations (ValueError: Indices must be integral), but in assignments (e.g., arr[:n] = float_indices), they are silently cast to integers (e.g., 1.5 → 1), risking truncation errors. Always use integral dtypes for indexing safety.
- **Bounds Behavior**: Out-of-bounds single index returns `0.0` (scalar); multi-index returns an array of zeros. No error raised.
- No argwhere or nonzero: Use mx.where or workarounds above.

---

### Array Operations
- **Math Ops**: `mx.add`, `mx.multiply`, etc., work like NumPy.  
  Example: `mx.add(arr1, arr2)`.
- **Conditional Selection**: `mx.where(condition, x, y)` picks values, requires 3 arguments, must be broadcast-compatible.  
  Example: `mx.where(arr > 0, arr, mx.full(arr.shape, 0, arr.dtype))`.  
  Note: Unlike NumPy’s `np.where(condition)`, it doesn’t return indices.
- **Sorting**: `mx.argsort(arr)` sorts ascending. For descending: `mx.argsort(arr)[::-1]`.  
  Example: `top_indices = mx.argsort(arr)[::-1][:5]`.
- **Rounding**: Use `mx.round(arr)`, not Python’s `round()`.  
  Example: `n_un = mx.round(mx.array(3.7)).item()` → `4`.
- **Dtype as String**: Use `str(dtype)`. Example: `str(mx.float32)` → `"float32"`. Avoid `dtype.__name__`.

---

### Lazy Evaluation
- **Graph Building**: Operations are deferred until `mx.eval()`.  
  Example: `a = arr + 1; mx.eval(a)`.
- **Trigger Points**: Call `mx.eval()` after key updates or before scalar extraction (e.g., `.item()`).  
  Example: `loss, grad = fn(x); mx.eval(loss, grad)`.
- **Implicit Evaluation**: Printing, NumPy conversion, or `memoryview` triggers computation.  
  Example: `print(arr)` or `np.array(arr)`.

---

### Function Transformations
- **Gradients**: `mx.grad(fn)` computes gradients.  
  Example: `grad_fn = mx.grad(loss_fn); dloss = grad_fn(x)`.
- **Value and Grad**: `mx.value_and_grad(fn)` returns both.  
  Example: `loss, grad = mx.value_and_grad(loss_fn)(x)`.
- **Vectorization**: `mx.vmap(fn, in_axes, out_axes)` vectorizes functions.  
  Example: `vmap_fn = mx.vmap(add, in_axes=(0, 1))`.
- **Nesting**: Transformations can combine.  
  Example: `d2f = mx.grad(mx.grad(fn))`.

---

### Compilation in MLX
- MLX uses just-in-time (JIT) compilation via mx.compile() for graph optimization, fusing operations (e.g., exp, erf, sqrt) into single kernels. See "Making MLX Go Fast: Compilation" for details. 

---

### Conversion to Other Frameworks
- **To NumPy**: `np.array(mx_array)`. Cast `bfloat16` to `float32` first to avoid precision issues.  
  Example: `np.array(mx_arr.astype(mx.float32))`.
- **From NumPy**: `mx.array(np_array)`.  
  Example: `mx.array(np_arr)`.
- **To PyTorch**: Use `torch.tensor(memoryview(mx_arr))`. For 1D conv weights, permute MLX `[out, kernel, in]` to PyTorch `[out, in, kernel]`.  
  Example: `torch.from_numpy(np.array(mlx_weights)).permute(0, 2, 1)`.  
  For 2D: MLX `[out, kh, kw, in]` to PyTorch `[out, in, kh, kw]` with `.permute(0, 3, 1, 2)`.
- Generally best to avoid casting to other frameworks unless there is a specific need.

---

### Key Differences to other frameworks
- **Immutability**: Arrays are immutable; no explicit copying needed. Modifications affect all references.  
  Example: `a = mx.array([1, 2]); b = a; b[0] = 0; print(a)` → `[0, 2]`.
- **Conv Weights**: MLX: `[out_channels, kernel_size, in_channels]` (1D); PyTorch: `[out_channels, in_channels, kernel_size]`.
- **Dtypes**:  
  - `mx.bfloat16`: GPU/CPU, half memory, good range.  
  - `mx.float32`: Default, GPU/CPU.  
  - `mx.float64`: CPU-only, errors on GPU.  
  - `mx.int32`: Indexing, GPU/CPU.
- **Random**:  
  - Implicit global PRNG: `mx.random.uniform()` varies each call.  
  - Explicit keys: `mx.random.uniform(key=mx.random.key(0))` for reproducibility.  
  - Global seed: `mx.random.seed(42)` (function, not state).

---

### Core Workarounds
1. **Dynamic Position Indexing**: Use `mx.arange` or Python variables directly.  
   Example: `n = 5; arr[:n]` or `arr[mx.arange(n, dtype=mx.int32)]`.
2. **Boolean Mask Conversion**: Use `mx.where` with sum-based slicing (see "Array Indexing: No Boolean Indexing").  
   Example: 
   ```python
   mask = arr > 0
   indices = mx.arange(arr.size, dtype=mx.int32)[:mx.sum(mask).item()]
   filtered = arr[indices]
   ```
3. **Multi-Axis Coordinate Indexing**: Use broadcast shaping for N-D indexing.  
   Example: 
   ```python
   row_coords = mx.array([1, 3], dtype=mx.int32)
   col_coords = mx.array([2, 0], dtype=mx.int32)
   matrix_selection = arr[row_coords[:, None], col_coords[None, :]]
   ```

---

### Making MLX Go Fast
- **Graph Evaluation**: Avoid frequent `mx.eval()` calls; evaluate at iteration boundaries. Use `mx.async_eval` for latency-sensitive tasks.  
  Example: `mx.eval(loss)` after a training step.
- **Type Promotion**: Avoid unintended upcasting (e.g., `mx.array(1.0, mx.float32) * mx.array(2.0, mx.bfloat16)` → `float32`). Use Python scalars to preserve type.  
  Example: `arr * 2` keeps `bfloat16`, unlike `arr * mx.array(2)`.
- **Fast Operations**: Use `mx.fast` (e.g., `mx.fast.rms_norm`, `mx.fast.scaled_dot_product_attention`) and efficient patterns (e.g., `x @ W.T` for vector-matrix ops).
- **Compilation**: Use `mx.compile()` for JIT compilation to accelerate graph execution.  
  - First-Run Compilation: Triggers on first call with unique shape/dtype, fuses ops into single kernels.  
  - Caching: Reuses compiled graphs for identical inputs.  
  - Examples:  
    - `@mx.compile def model(x): return mx.exp(-x) * 2`.  
    - `@mx.compile def indexed_operation(arr, indices): return arr[indices] * 2 + 1` (e.g., ~1.5-2x speedup).  
  - Tips: Use `shapeless=True` for dynamic shapes (may recompile), disable with `MLX_DISABLE_COMPILE=1` for debugging, use `mx.where` for optimizable control flow.
- **Memory Use**: Pass file paths to `mx.load`, cast to `mx.float16` before evaluation, release temps early in loops.  
  Example: `arr = mx.load("data.npz")["arr"].astype(mx.float16); mx.eval(arr)`.
- **Profiling**: Monitor GPU with mactop, use Metal debugger for bottlenecks.
- **Indexing Optimizations**: `mx.take` may be 0.9-2x faster than `[]` on some hardware but slower on others (e.g., 1.1x slower on certain setups); test on your system. Batch operations and reuse indices for efficiency.

---

### Notes
- MLX indexing is stricter than NumPy, requiring integral types (e.g., `mx.int32`) for array indices and explicit tensor construction for Metal GPU optimization.  
- Avoid NumPy assumptions: no direct boolean indexing, no `nonzero`.  
- Dynamic slicing with Python variables (e.g., `arr[:n]`) works as of MLX 0.23.2, unlike earlier limitations.  
- Trigger `mx.eval()` before scalar extraction (e.g., `.item()`) or conversions, and after `.at` operations if immediate results are needed.