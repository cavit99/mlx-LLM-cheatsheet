---

## MLX Cheat Sheet for Coding Assistance

This is a guide for working with MLX, a machine learning framework for Apple silicon with Metal GPU acceleration. MLX uses lazy evaluation and differs from NumPy and PyTorch in important ways. Use this to write correct MLX code and avoid common mistakes.

## Array Creation  
Use mx.array(data) to create an array from a list or iterable. Example: mx.array([1, 2, 3]).  
Use mx.ones(shape, dtype) for an array of ones. Example: mx.ones((3, 2), mx.float32).  
Use mx.zeros(shape, dtype) for an array of zeros. Example: mx.zeros((2, 2), mx.int32).  
Use mx.full(shape, value, dtype) to fill an array with a value. Example: mx.full((2, 3), 5, mx.float32).  
No full_like: Use mx.full(x.shape, value, dtype) instead. Example: mx.full(x.shape, -1, mx.float32).  
No dtype in ones_like: Use mx.full(x.shape, 1, dtype). Example: mx.full(x.shape, 1, mx.bool_).  
Supported dtypes: mx.bfloat16 (gpu/cpu), mx.float32 (default, gpu/cpu), mx.float64 (cpu-only). Example: mx.array([1.0], dtype=mx.bfloat16).  

## Array Indexing  
Supports integers, slices, ellipsis (...), and None like NumPy. Examples: arr[3], arr[2:8:2], arr[..., 0], arr[None].  
Use an MLX array to index another. Example: arr[mx.array([5, 7])].  
No boolean indexing (arr[arr > 0] fails). Instead, use mx.argsort and mx.take. Example: valid_indices = mx.argsort(arr)[::-1]; arr = mx.take(arr, valid_indices[:5]).  
No bounds checking: Out-of-bounds indexing breaks silently. Be careful.  
Use mx.take(arr, indices) or mx.take_along_axis(arr, indices, axis) for indexing.

## Array Operations  
Math ops like mx.add, mx.multiply work like NumPy. Example: mx.add(arr1, arr2).  
mx.where(condition, x, y) picks values, not indices (unlike NumPy’s np.where(condition)), and requires 3 arguments. Example: mx.where(arr > 0, arr, mx.full(arr.shape, 0, arr.dtype)). Condition, x, and y must be broadcast-compatible.  
For index-like logic, build conditions manually or use mx.argwhere with a boolean mask. Example: mask = mx.zeros(arr.shape, mx.bool_); for i in range(arr.size): mask = mx.where(arr[i] > 0, True, mask); indices = mx.argwhere(mask).  
mx.argsort(arr) sorts ascending only. For descending, use mx.argsort(arr)[::-1]. Example: top_indices = mx.argsort(arr)[::-1][:5].  
To get dtype as string, use str(dtype). Example: str(mx.float32) gives "float32". Do not use dtype.__name__—it fails.

## Lazy Evaluation  
Operations build a graph; nothing computes until mx.eval(). Example: a = arr + 1; mx.eval(a).  
Call mx.eval() at logical points, like after a loop iteration. Example: loss, grad = fn(x); mx.eval(loss, grad).  
Printing, converting to NumPy, or accessing memoryview triggers evaluation automatically.

## Function Transformations  
mx.grad(fn) computes the gradient of fn. Example: grad_fn = mx.grad(loss_fn); dloss = grad_fn(x).  
mx.value_and_grad(fn) gives value and gradient. Example: loss, grad = mx.value_and_grad(loss_fn)(x).  
mx.vmap(fn, in_axes, out_axes) vectorizes fn. Example: vmap_fn = mx.vmap(add, in_axes=(0, 1)).  
Transformations can nest. Example: d2f = mx.grad(mx.grad(fn)).

## Conversion to Other Frameworks  
To NumPy: np.array(mx_array). Example: np_arr = np.array(mx_arr). NumPy lacks bfloat16; cast to float32 first if needed: np.array(mx_arr.astype(mx.float32)).  
From NumPy: mx.array(np_array). Example: mx_arr = mx.array(np_arr).  
For PyTorch, JAX, TensorFlow, use buffer protocol or DLPack. PyTorch may need memoryview or NumPy. Example: torch.tensor(memoryview(mx_arr)).  
Do not cast unless asked. If absolutely needed, ask first.

## Key Differences  
Arrays are immutable; no need to copy them explicitly.  
Conv layer weights: MLX uses [out_channels, kernel_size, in_channels]; PyTorch uses [out_channels, in_channels, kernel_size].  
In-place updates affect all references. Example: a = mx.array([1, 2, 3]); b = a; b[2] = 0; print(a) shows [1, 2, 0].  
Dtypes: bfloat16 (GPU/CPU, half memory, good range), float32 (default, GPU/CPU), float64 (CPU-only, high precision). float64 on GPU raises an error.

## Common Workarounds  
Replace boolean indexing with mx.argsort and mx.take. Example: mx.take(arr, mx.argsort(arr)[::-1][:3]).  
For index extraction, use manual logic or mx.argwhere since mx.where doesn’t return indices. Example: mask = mx.full(x.shape, False); mask = mx.where(mx.arange(len(x)) == 2, True, mask); indices = mx.argwhere(mask).  
Descending sort: mx.argsort(arr)[::-1].  
Type-specific arrays: Use mx.full instead of ones_like with dtype. Example: mx.full(x.shape, 1, mx.bool_).  
Use this guide to generate accurate MLX code. Avoid assuming NumPy or PyTorch behavior like boolean indexing, nonzero functions, or dtype.__name__—MLX doesn’t support them yet.

---
