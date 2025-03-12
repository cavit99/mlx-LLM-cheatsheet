This guide is tailored for working with MLX, a machine learning framework optimized for Apple Silicon with Metal GPU acceleration. MLX uses lazy evaluation and has distinct differences from NumPy and PyTorch, especially in indexing and array operations. Use this to write correct, efficient MLX code and avoid common pitfalls.

---

### Array Creation
- **From Data**: Use `mx.array(data)` to create an array from a list or iterable.  
  Example: `mx.array([1, 2, 3])`.
- **Filled Arrays**:
  - `mx.ones(shape, dtype)`: Array of ones. Example: `mx.ones((3, 2), mx.float32)`.
  - `mx.zeros(shape, dtype)`: Array of zeros. Example: `mx.zeros((2, 2), mx.int32)`.
  - `mx.full(shape, value, dtype)`: Fill with a value. Example: `mx.full((2, 3), 5, mx.float32)`.
- **No `full_like`**: Use `mx.full(x.shape, value, dtype)`. Example: `mx.full(x.shape, -1, mx.float32)`.
- **No `ones_like` with dtype**: Use `mx.full(x.shape, 1, dtype)`. Example: `mx.full(x.shape, 1, mx.bool_)`.
- **Supported Dtypes**: 
  - `mx.bfloat16` (GPU/CPU, half memory, good range).
  - `mx.float32` (default, GPU/CPU).
  - `mx.float64` (CPU-only, high precision).
  - `mx.int32` (integral, required for indexing).
  Example: `mx.array([1], dtype=mx.int32)`.

---

### Array Indexing
- **Basic Indexing**: Supports integers, static slices, ellipsis (`...`), and `None`.  
  Examples: `arr[3]`, `arr[2:8:2]`, `arr[..., 0]`, `arr[None]`.
- **Array Indexing**: Use an `mx.array` with integral dtype (e.g., `mx.int32`) to index another array.  
  Example: `arr[mx.array([5, 7], dtype=mx.int32)]`.
- **Dynamic Slicing Limitation**: Slice bounds must be static integers or `None`, not Python variables or scalars.  
  - **Invalid**: `arr[:n]` where `n` is a Python `int`.  
    Error: `ValueError: Slice indices must be integers or None`.
  - **Workaround**: Use `mx.arange` to create dynamic bounds as an `mx.array`.  
    Example: 
    ```python
    n = 5
    indices = mx.arange(n, dtype=mx.int32)
    dynamic_slice = arr[indices]
    ```
- **No Boolean Indexing**: `arr[arr > 0]` fails. Use `mx.argsort` and `mx.take` instead.  
  Example: 
  ```python
  sorted_indices = mx.argsort(arr)[::-1]
  top_values = mx.take(arr, sorted_indices[:3])
  ```
- **No `argwhere` or `nonzero`**: Replace with manual index construction or `mx.where` with scalar logic.  
  Example: 
  ```python
  mask = arr > 0
  indices = mx.arange(arr.size)
  filtered_indices = mx.take(indices, mx.where(mask, indices, -1))[:-1]  # Exclude -1
  ```
- **Integral Indices Requirement**: Functions like `mx.take` require indices to be `mx.array` with integral dtype (e.g., `mx.int32`).  
  - **Invalid**: `mx.take(arr, mx.array([0.5, 1.5]))`.  
    Error: `ValueError: [gather] Got indices with invalid dtype`.
  - **Fix**: Ensure indices are integral.  
    Example: `mx.take(arr, mx.arange(3, dtype=mx.int32))`.
- **No Bounds Checking**: Out-of-bounds indexing breaks silently. Verify ranges manually.
- **Efficient Extraction**: Use `mx.take(arr, indices, axis)` for 1D or multi-axis indexing.  
  Example: `fast_extract = mx.take(arr, mx.array([0, 2], dtype=mx.int32), axis=0)`.

---

### Array Operations
- **Math Ops**: `mx.add`, `mx.multiply`, etc., work like NumPy.  
  Example: `mx.add(arr1, arr2)`.
- **Conditional Selection**: `mx.where(condition, x, y)` picks values, requires 3 arguments, and must be broadcast-compatible.  
  Example: `mx.where(arr > 0, arr, mx.full(arr.shape, 0, arr.dtype))`.
  - Note: Unlike NumPy's `np.where(condition)`, it doesn't return indices.
- **Sorting**: `mx.argsort(arr)` sorts ascending. For descending, use `mx.argsort(arr)[::-1]`.  
  Example: `top_indices = mx.argsort(arr)[::-1][:5]`.
- **Rounding**: Use `mx.round(arr)` for array rounding, not Python's `round()`.  
  Example: `n_un = mx.round(mx.array(3.7)).item()` → `4`.
- **Dtype as String**: Use `str(dtype)`. Example: `str(mx.float32)` → `"float32"`. Avoid `dtype.__name__`.

---

### Lazy Evaluation
- **Graph Building**: Operations are deferred until `mx.eval()`.  
  Example: `a = arr + 1; mx.eval(a)`.
- **Trigger Points**: Call `mx.eval()` after key updates or before scalar extraction (e.g., `.item()`).  
  Example: `loss, grad = fn(x); mx.eval(loss, grad)`.
- **Implicit Evaluation**: Printing, NumPy conversion, or `memoryview` triggers computation.

---

### Function Transformations
- **Gradients**: `mx.grad(fn)` computes gradients. Example: `grad_fn = mx.grad(loss_fn); dloss = grad_fn(x)`.
- **Value and Grad**: `mx.value_and_grad(fn)`. Example: `loss, grad = mx.value_and_grad(loss_fn)(x)`.
- **Vectorization**: `mx.vmap(fn, in_axes, out_axes)`. Example: `vmap_fn = mx.vmap(add, in_axes=(0, 1))`.
- **Nesting**: Transformations can combine. Example: `d2f = mx.grad(mx.grad(fn))`.

---

### Compilation in MLX
MLX uses **just-in-time (JIT) compilation** via `mx.compile()` for graph optimization, compiling functions at runtime on their first invocation.

- **JIT with `mx.compile()`**:
  - **First-Run Compilation**: Triggers on the first function call with a unique input configuration (shape, dtype), tracing the computation graph, optimizing it, and fusing operations into single kernels.
  - **Operation Fusion**: Merges operations (e.g., `exp`, `erf`, `sqrt` in GELU) into `Compiled` primitives, reducing GPU kernel launches.
  - **Caching**: Stores compiled graphs in memory for reuse with identical input shapes and dtypes, avoiding recompilation on subsequent calls.
  - **Example**:
    ```python
    @mx.compile  # JIT compiles on first call
    def model(x):
        return mx.exp(-x) * 2
    output = model(x)  # Compiles here, reuses cache on subsequent calls
    ```

- **Workflow Tips**:
  - Use `shapeless=True` for dynamic shapes, but expect recompilation if shapes change.
  - Set `MLX_DISABLE_COMPILE=1` in the environment to disable compilation for debugging.
  - Use `mx.where` for dynamic control flow to maintain graph optimizability instead of Python conditionals.

---

### Conversion to Other Frameworks
- **To NumPy**: `np.array(mx_array)`. Example: `np_arr = np.array(mx_arr)`.  
  - **Warning**: Cast `bfloat16` to `float32` first (e.g., `mx_arr.astype(mx.float32)`), as NumPy doesn't support `bfloat16` natively, to avoid precision issues.
- **From NumPy**: `mx.array(np_array)`. Example: `mx_arr = mx.array(np_arr)`.
- **To PyTorch**: Use `torch.tensor(memoryview(mx_arr))` or via NumPy if necessary.  
  - **Conv Weight Reshaping**: For 1D convolutions, MLX weights `[out_channels, kernel_size, in_channels]` need permuting to PyTorch's `[out_channels, in_channels, kernel_size]`:
    ```python
    mlx_weights = mlx_conv.weight
    torch_weights = torch.from_numpy(np.array(mlx_weights)).permute(0, 2, 1)
    ```
  - For 2D: MLX `[out, kh, kw, in]` to PyTorch `[out, in, kh, kw]` with `.permute(0, 3, 1, 2)`.
  - Generally best to avoid casting to other frameworks unless mission critical.

---

### Key Differences
- **Immutability**: Arrays are immutable; no explicit copying needed.
- **Conv Weights**: MLX: `[out_channels, kernel_size, in_channels]` (1D); PyTorch: `[out_channels, in_channels, kernel_size]`.
- **In-Place Updates**: Affect all references. Example: `a = mx.array([1, 2]); b = a; b[0] = 0; print(a)` → `[0, 2]`.
- **Dtypes**: 
  - `bfloat16` (GPU/CPU, low memory).
  - `float32` (default, GPU/CPU).
  - `float64` (CPU-only, errors on GPU).
  - `int32` (indexing, GPU/CPU).

---

### Core Workarounds (MLX-Only)
1. **Dynamic Position Indexing**:
   - For non-static slice bounds, use `mx.arange` with integral dtype.  
     Example: 
     ```python
     n = 5
     indices = mx.arange(n, dtype=mx.int32)
     dynamic_slice = arr[indices]
     ```
2. **Boolean Mask Conversion**:
   - Replace masks with positional indices using `mx.where` or `mx.argsort`.  
     Example: 
     ```python
     mask = arr > 0.5
     indices = mx.arange(arr.size, dtype=mx.int32)
     filtered_indices = mx.take(indices, mx.where(mask, indices, -1))[:-1]
     filtered = arr[filtered_indices]
     ```
3. **Multi-Axis Coordinate Indexing**:
   - Use broadcast shaping for N-D indexing.  
     Example: 
     ```python
     row_coords = mx.array([1, 3], dtype=mx.int32)
     col_coords = mx.array([2, 0], dtype=mx.int32)
     matrix_selection = arr[row_coords[:, None], col_coords[None, :]]
     ```

---

### Advanced Techniques
- **Index Buffer Pre-allocation**:
  - Use writeable arrays for frequent updates.  
    Example: 
    ```python
    max_indices = 10
    index_buffer = mx.zeros((max_indices,), dtype=mx.int32)
    index_buffer[:3] = mx.array([0, 2, 4], dtype=mx.int32)
    result = arr[index_buffer[:3]]
    ```
- **Compiled Index Functions**:
  - Accelerate with `mx.compile`.  
    Example: 
    ```python
    @mx.compile
    def indexed_operation(arr, indices):
        return arr[indices] * 2 + 1
    result = indexed_operation(arr, mx.array([0, 1], dtype=mx.int32))
    ```

---

### Performance Pro Tips
- **Use `mx.take`**: 2-3x faster than basic `[]` indexing for 1D extraction.  
  Example: `fast_extract = mx.take(arr, mx.array([0, 2], dtype=mx.int32), axis=0)`.
- **Batch Operations**: Combine small index steps into single kernels to reduce overhead.
- **Reuse Indices**: Cache frequent index arrays (e.g., `sorted_indices`) to avoid recomputation.
- **Explicit Dtypes**: Always specify `dtype=mx.int32` for indices to prevent dtype mismatches.

---

### Making MLX Go Fast
- Graph Evaluation: Avoid overly frequent evaluations to prevent expensive graph rebuilds. Evaluate your computation graphs at logical iteration boundaries (e.g., after a full training iteration). For latency-sensitive tasks, consider using `mx.async_eval` to pipeline computations.
- Type Promotion: Be cautious of unintended upcasting. For example, operations like `mx.array(1.0, mx.float32) * mx.array(2.0, mx.float16)` result in `mx.float32`. Use Python scalars when multiplying to preserve the array's type.
- Fast Operations: Leverage optimized ops from the `mx.fast` namespace (e.g., `mx.fast.rms_norm`, `mx.fast.layer_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`) and use efficient computation patterns (e.g., prefer `x @ W.T` for vector-matrix multiplications).
- Compilation: Use `mx.compile()` for JIT compilation to accelerate graph execution. Be mindful that changes in input shapes or constant inputs trigger recompilations; for closures, explicitly pass arrays as inputs or use partial decoration.
- Memory Use: Utilize lazy loading by passing file paths to `mx.load` and cast arrays to lower precision (like `mx.float16`) before evaluation to reduce peak memory usage. Also, release temporary variables early in loops to free memory.
- Profiling: Monitor GPU utilization (e.g., via mactop) and use the Metal debugger to identify and resolve performance bottlenecks.

### Notes
MLX's indexing is stricter than NumPy's flexible approach, requiring explicit tensor construction with integral types. This aligns with Metal's GPU optimization but demands careful dtype and indexing management. Avoid assumptions about NumPy-like behavior (e.g., dynamic slicing, `argwhere`, `nonzero`, `dtype.__name__`)—they're not supported yet. Use this updated guide to ensure accurate, performant MLX code.
