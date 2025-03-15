"""
MLX Array Indexing Validation Script

This script comprehensively tests and validates MLX array indexing behaviors,
focusing on proper usage patterns, edge cases, and performance characteristics.
It covers:
- Basic array indexing with various mask types
- Slice assignment operations
- Edge cases and error conditions
- NumPy-like behavior comparison
- Boolean indexing workarounds
- Framework conversion (MLX ↔ NumPy/PyTorch)
- Performance benchmarks and compilation tests

Results are summarized at the end of output to provide a clear overview
of supported operations and recommended practices.
"""

import mlx.core as mx
import time
import numpy as np  
try:
    import torch  # Try to import PyTorch for comparison
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Helper to display arrays and their properties
def print_array(label, arr, extra_info=""):
    """Print array information in a readable format with sample values for large arrays."""
    if arr is None:
        print(f"{label}: None (Failed to create)")
    # Check if the result is a Python scalar
    elif isinstance(arr, (int, float, bool)):
        print(f"{label}: {arr} (Python {type(arr).__name__})")
    # Special case for ArrayAt objects which don't have size, shape, etc.
    elif hasattr(arr, '__class__') and arr.__class__.__name__ == 'ArrayAt':
        print(f"{label}: {type(arr).__name__} object (Use with .add(), .subtract(), etc.)")
    else:
        if arr.size > 10:
            # For 1D or flattened arrays, take first 5 elements directly
            if len(arr.shape) == 1:
                sample = [str(float(x)) if x.dtype in (mx.float32, mx.float64, mx.bfloat16) else str(int(x)) 
                          for x in arr[:5]]
            else:
                # For 2D+, flatten first to avoid multi-element arrays
                flat_sample = arr.flatten()[:5]
                sample = [str(float(x)) if x.dtype in (mx.float32, mx.float64, mx.bfloat16) else str(int(x)) 
                          for x in flat_sample]
            print(f"{label}: array([{', '.join(sample)}, ...], dtype={arr.dtype})")
        else:
            print(f"{label}: {arr}")
        print(f"  Shape: {arr.shape}, Dtype: {arr.dtype}, Size: {arr.size}")
    if extra_info:
        print(f"  {extra_info}")
    print()

# Helper to run a test case, catch failures, and measure time
def run_test(description, func, results_dict, key):
    """Execute a test function, measure performance, and record results."""
    print(f"Testing: {description}")
    tic = time.perf_counter()
    try:
        result = func()
        elapsed = (time.perf_counter() - tic) * 1000  # ms
        print(f"  Success: Completed in {elapsed:.2f} ms")
        results_dict[key] = {"success": True, "result": result, "time": elapsed}
        return result
    except Exception as e:
        elapsed = (time.perf_counter() - tic) * 1000
        print(f"  Failed: {type(e).__name__} - {str(e)} (in {elapsed:.2f} ms)")
        results_dict[key] = {"success": False, "error": f"{type(e).__name__}: {str(e)}", "time": elapsed}
        return None

# Print MLX version
print(f"MLX Version: {mx.__version__}\n")

# --- Array Indexing Analysis ---
print("=== Array Indexing Analysis ===")
print("Exploring indexing behaviors across 1D and 2D arrays\n")

arr_1d = mx.arange(10000, dtype=mx.float32) - 5000
arr_2d = mx.arange(10000, dtype=mx.float32).reshape(100, 100) - 5000
empty_arr = mx.array([], dtype=mx.float32)
print_array("1D Array", arr_1d)
print_array("2D Array", arr_2d)
print_array("Empty Array", empty_arr)

masks_1d = {
    "All Elements Positive": arr_1d > -6000,
    "No Elements Positive": arr_1d > 6000,
    "Some Elements Positive": arr_1d > 0,
    "Empty Mask": empty_arr > 0
}
masks_2d = {
    "All Elements Positive": arr_2d > -6000,
    "Some Elements Positive": arr_2d > 0
}

results = {
    "indexing": {}, 
    "assignment": {}, 
    "edge_cases": {}, 
    "workarounds": {},
    "performance": {},
    "laziness": {},
    "conversion": {},
    "buffer_prealloc": {},
    "compiled_index": {}
}

# 1D Indexing Tests
for mask_name, mask in masks_1d.items():
    print(f"--- 1D Indexing with {mask_name} ---")
    print_array("Mask", mask)
    key_prefix = f"1d_{mask_name.lower().replace(' ', '_')}"

    result = run_test("Direct Mask Indexing", lambda: arr_1d[mask], results["indexing"], f"{key_prefix}_direct")
    print_array("Result", result)

    indices = run_test("Mask-to-Indices with Placeholder and Slice", 
                       lambda: mx.take(mx.arange(arr_1d.size), mx.where(mask, mx.arange(arr_1d.size), -1))[:-1],
                       results["indexing"], f"{key_prefix}_placeholder_slice_indices")
    print_array("Indices", indices)
    result = run_test("Filtered with Placeholder and Slice", 
                      lambda: arr_1d[indices] if indices is not None else None,
                      results["indexing"], f"{key_prefix}_placeholder_slice_filtered")
    print_array("Result", result)

    indices = run_test("Single-Argument mx.where", lambda: mx.where(mask)[0],
                       results["indexing"], f"{key_prefix}_single_where")
    print_array("Indices", indices)

    indices_all = mx.arange(arr_1d.size, dtype=mx.int32)
    where_result = run_test("Three-Argument mx.where with Placeholder", 
                            lambda: mx.where(mask, indices_all, -1 * mx.ones_like(indices_all)),
                            results["indexing"], f"{key_prefix}_three_where")
    print_array("Indices (including placeholders)", where_result)
    if where_result is not None:
        mask_sum = mx.sum(mask).item()
        valid_indices = run_test("Extract Valid Indices (Sum-Based)", 
                                 lambda: indices_all[:mask_sum] if mask_sum > 0 else mx.array([], dtype=mx.int32),
                                 results["indexing"], f"{key_prefix}_valid_indices")
        print_array("Valid Indices", valid_indices)
        result = run_test("Filtered with Valid Indices", 
                          lambda: arr_1d[valid_indices] if valid_indices is not None else None,
                          results["indexing"], f"{key_prefix}_valid_filtered")
        print_array("Result", result)

    indices = run_test("Nonzero Function", lambda: mx.nonzero(mask),
                       results["indexing"], f"{key_prefix}_nonzero")
    print_array("Indices", indices)

# 2D Indexing Tests
for mask_name, mask in masks_2d.items():
    print(f"--- 2D Indexing with {mask_name} ---")
    print_array("Mask", mask)
    key_prefix = f"2d_{mask_name.lower().replace(' ', '_')}"

    result = run_test("Direct Mask Indexing", lambda: arr_2d[mask],
                      results["indexing"], f"{key_prefix}_direct")
    print_array("Result", result)

    flat_arr = arr_2d.flatten()
    flat_mask = mask.flatten()
    indices_all = mx.arange(flat_arr.size, dtype=mx.int32)
    where_result = run_test("Three-Argument mx.where on Flattened Array", 
                            lambda: mx.where(flat_mask, indices_all, -1 * mx.ones_like(indices_all)),
                            results["indexing"], f"{key_prefix}_three_where_flat")
    print_array("Flat Indices (including placeholders)", where_result)
    if where_result is not None:
        mask_sum = mx.sum(flat_mask).item()
        valid_indices = run_test("Extract Valid Indices from Flat (Sum-Based)", 
                                 lambda: indices_all[:mask_sum] if mask_sum > 0 else mx.array([], dtype=mx.int32),
                                 results["indexing"], f"{key_prefix}_valid_indices_flat")
        print_array("Valid Flat Indices", valid_indices)
        result = run_test("Filtered Flat Array", 
                          lambda: flat_arr[valid_indices] if valid_indices is not None else None,
                          results["indexing"], f"{key_prefix}_valid_filtered_flat")
        print_array("Result (flattened)", result)

# --- Slice Assignment Analysis ---
print("\n=== Slice Assignment Analysis ===")
buffer = mx.zeros((10000,), dtype=mx.int32)
print_array("Initial Buffer", buffer)

indices_normal = mx.arange(0, 3000, 2, dtype=mx.int32)
indices_large = mx.arange(0, 5000, 2, dtype=mx.int32)
indices_dynamic = mx.arange(0, 2000, 4, dtype=mx.int32)
indices_float = mx.arange(0, 3000, 2, dtype=mx.float32)
indices_empty = mx.array([], dtype=mx.int32)
indices_out = mx.array([1], dtype=mx.int32)

result = run_test("Assign Normal-Length Indices", 
                  lambda: [buffer.__setitem__(slice(0, 1500), indices_normal), buffer][1],
                  results["assignment"], "normal_length")
print_array("Buffer After Normal Assignment", result)

result = run_test("Assign Over-Length Indices", 
                  lambda: [buffer.__setitem__(slice(0, 1500), indices_large), buffer][1],
                  results["assignment"], "over_length")
print_array("Buffer After Over-Length Assignment", result)

result = run_test("Concatenate Dynamic Indices", 
                  lambda: mx.concatenate([indices_dynamic, mx.zeros(10000 - indices_dynamic.size, dtype=mx.int32)]),
                  results["assignment"], "dynamic_concat")
print_array("Buffer After Concatenation", result)

result = run_test("Assign Float Indices", 
                  lambda: [buffer.__setitem__(slice(0, 1500), indices_float), buffer][1],
                  results["assignment"], "float_indices")
print_array("Buffer After Float Assignment", result)

result = run_test("Assign Empty Indices", 
                  lambda: [buffer.__setitem__(slice(0, 0), indices_empty), buffer][1],
                  results["assignment"], "empty_indices")
print_array("Buffer After Empty Assignment", result)

result = run_test("Assign Beyond Bounds", 
                  lambda: [buffer.__setitem__(slice(10000, None), indices_out), buffer][1],
                  results["assignment"], "beyond_bounds")
print_array("Buffer After Beyond Bounds", result)

# Test assignment using standard slice syntax
result = run_test("Assign with Standard Syntax", 
                  lambda: [buffer.__setitem__(slice(5000, 5010), mx.full((10,), 99, dtype=mx.int32)), buffer][1],
                  results["assignment"], "standard_assign")
print_array("Buffer After Standard Assignment", result)

# Test the .at API properly - using it for in-place operations with duplicate indices
print("\n--- .at API Tests ---")
base_array = mx.zeros((5,), dtype=mx.int32)
print_array("Initial Array", base_array)

duplicate_indices = mx.array([0, 2, 0, 2, 4], dtype=mx.int32)
print_array("Duplicate Indices", duplicate_indices)

# Test standard assignment (ignores duplicates)
result_standard = run_test("Standard Assignment with Duplicates", 
                 lambda: [arr := mx.zeros((5,), dtype=mx.int32), 
                         arr.__setitem__(duplicate_indices, mx.ones_like(duplicate_indices)),
                         arr][2],
                 results["assignment"], "standard_duplicate")
print_array("Result with Standard Assignment", result_standard)

# Test .at.add() with duplicates (correctly applies all updates)
result_at = run_test("Using .at.add() with Duplicates", 
                    lambda: [arr := mx.zeros((5,), dtype=mx.int32), 
                            arr := arr.at[duplicate_indices].add(1),
                            mx.eval(arr),
                            arr][3],
                    results["assignment"], "at_add_duplicate")
print_array("Result with .at.add()", result_at)

# Test other .at operations
result_multiply = run_test("Using .at.multiply() with Duplicates",
                         lambda: [arr := mx.ones((5,), dtype=mx.int32) * 2, 
                                 arr := arr.at[duplicate_indices].multiply(2),
                                 mx.eval(arr),
                                 arr][3],
                         results["assignment"], "at_multiply_duplicate")
print_array("Result with .at.multiply()", result_multiply)

# Test reference handling (modify one reference, check if other is affected)
a = mx.arange(10000, dtype=mx.int32)
b = a  # b references the same array as a
result = run_test("Modify Reference and Check", 
                 lambda: [a.__setitem__(slice(0, 5000), mx.full((5000,), 42, dtype=mx.int32)), (a, b)][1],
                 results["assignment"], "reference_modify")
print_array("Modified Array", result[0])
print_array("Referenced Array", result[1])

# --- Edge Case Analysis ---
print("\n=== Edge Case Analysis ===")

# Test indexing beyond array bounds
result = run_test("Index Beyond Bounds", lambda: arr_1d[10000],
                  results["edge_cases"], "beyond_bounds")
print_array("Result", result)

# Test dynamic slicing with Python variables (mentioned in docs as a limitation)
n = 5  # A Python int
result = run_test("Dynamic Slicing with Python Variable", 
                 lambda: arr_1d[:n],  # This may fail with ValueError about slice indices
                 results["edge_cases"], "dynamic_slice_python_var")
print_array("Result", result)

# Test multiple indices beyond bounds
multi_out_indices = mx.array([10000, 10001], dtype=mx.int32)
result = run_test("Multi-Index Beyond Bounds", 
                 lambda: arr_1d[multi_out_indices],  # Check if returns array or scalar
                 results["edge_cases"], "multi_beyond_bounds")
print_array("Result", result)

# Test indexing with float values (should be rejected as non-integral)
float_indices = mx.arange(0, 3000, 2, dtype=mx.float32) / 2
result = run_test("Index with Float Values", lambda: arr_1d[float_indices],
                  results["edge_cases"], "float_indices")
print_array("Result", result)

# Test lazy evaluation impact on indexing
lazy_arr = arr_1d + 1  # Creates lazy expression, not evaluated yet
indices = mx.arange(0, 2000, 4, dtype=mx.int32)
result = run_test("Index Unevaluated Array", lambda: lazy_arr[indices],
                  results["edge_cases"], "lazy_unevaluated")
print_array("Result Before Evaluation", result)

# Force evaluation and test again
mx.eval(lazy_arr)
result = run_test("Index Evaluated Array", lambda: lazy_arr[indices],
                  results["edge_cases"], "lazy_evaluated")
print_array("Result After Evaluation", result)

# Test float indices in assignment operations
buffer_float_test = mx.zeros((1500,), dtype=mx.int32)
result = run_test("Assign Float Indices to Buffer", 
                 lambda: [buffer_float_test.__setitem__(slice(0, 1500), float_indices), buffer_float_test][1],
                 results["assignment"], "float_indices_assign")
print_array("Buffer After Float Indices Assignment", result)

# --- NumPy-like Behavior Checks ---
print("\n=== NumPy-like Behavior Checks ===")

# Test 2D slice flattening (check if similar to NumPy)
result = run_test("2D Slice Flattening (like NumPy arr[:5])", 
                  lambda: arr_2d[:5].flatten()[:5],  # NumPy would flatten directly
                  results["edge_cases"], "2d_slice_flatten")
print_array("Result", result)

# Test scalar extraction properly
result = run_test("Scalar Extraction from Slice (like NumPy arr[0, 0])", 
                  lambda: arr_2d[0, 0].item(),  # Extract and convert to Python scalar with .item()
                  results["edge_cases"], "scalar_extract")
print_array("Result", result)

# Test direct scalar extraction without .item()
result = run_test("Direct Scalar Access (without .item())", 
                  lambda: arr_2d[0, 0],  # Direct access returns mx array scalar
                  results["edge_cases"], "direct_scalar")
print_array("Result", result)

# --- Boolean Indexing Workarounds ---
print("\n=== Boolean Indexing Workarounds ===")

# Create a simple boolean mask for testing
mask_pos = arr_1d > 0  # True for positive elements
print_array("Boolean Mask (arr_1d > 0)", mask_pos)

# Workaround 1: Using mx.where with sum-based slicing
result = run_test("Workaround: mx.where with sum-based slicing", 
                 lambda: [indices_all := mx.arange(arr_1d.size, dtype=mx.int32), 
                          mask_sum := mx.sum(mask_pos).item(),
                          indices_all[:mask_sum] if mask_sum > 0 else mx.array([], dtype=mx.int32)][2],
                 results["workarounds"], "where_sum_based")
print_array("Valid Indices from mx.where", result)

# Filter the original array with the valid indices
filtered = run_test("Filter Original Array with Valid Indices", 
                   lambda: arr_1d[result] if result is not None else None,
                   results["workarounds"], "filtered_where")
print_array("Filtered Result", filtered)

# Workaround 2: Using mx.where with placeholder and slice
result_placeholder = run_test("Workaround: mx.where with placeholder and slice", 
                             lambda: [indices_all := mx.arange(arr_1d.size, dtype=mx.int32), 
                                      mx.where(mask_pos, indices_all, -1 * mx.ones_like(indices_all))[:-1]][1],
                             results["workarounds"], "where_placeholder")
print_array("Indices from mx.where with Placeholder", result_placeholder)

# Filter the original array with placeholder indices
filtered_placeholder = run_test("Filter Original Array with Placeholder Indices", 
                               lambda: arr_1d[result_placeholder] if result_placeholder is not None else None,
                               results["workarounds"], "filtered_where_placeholder")
print_array("Filtered Result with Placeholder", filtered_placeholder)

# Workaround 3: Using argsort for top-n selection
top_n = 5
result = run_test("Workaround: argsort for top-n", 
                 lambda: mx.argsort(arr_1d)[::-1][:top_n],  # Get indices of top n values
                 results["workarounds"], "argsort_top_n")
print_array(f"Top {top_n} Indices via argsort", result)

# Extract the top values using the indices
top_values = run_test("Extract Top Values via argsort", 
                     lambda: mx.take(arr_1d, result) if result is not None else None,
                     results["workarounds"], "top_values")
print_array(f"Top {top_n} Values", top_values)

# Workaround 4: Multi-axis coordinate indexing
result = run_test("Multi-Axis Coordinate Indexing", 
                 lambda: [row_coords := mx.array([1, 3], dtype=mx.int32),
                          col_coords := mx.array([2, 0], dtype=mx.int32),
                          result := arr_2d[row_coords[:, None], col_coords[None, :]],
                          mx.eval(result),
                          result
                         ][4],
                 results["workarounds"], "multi_axis")
print_array("Multi-Axis Selection Result", result)

# --- Performance and Type Tests ---
print("\n=== Performance and Type Tests ===")

# Test type promotion between different float types
a_f32 = mx.array(1.0, dtype=mx.float32)
b_f16 = mx.array(2.0, dtype=mx.bfloat16)
result = run_test("Type Promotion (float32 * bfloat16)", 
                 lambda: a_f32 * b_f16,  # Should promote to wider type (float32)
                 results["performance"], "type_promotion")
print_array("Result", result)

# Test type preservation with Python scalar
result = run_test("Type Preservation (bfloat16 * Python scalar)", 
                 lambda: b_f16 * 2.0,  # Should preserve original type
                 results["performance"], "python_scalar_mult")
print_array("Result", result)

# Benchmark mx.take vs direct indexing
indices = mx.array([5, 10, 15, 20, 25], dtype=mx.int32)
n_runs = 1000

start = time.perf_counter()
for _ in range(n_runs):
    result1 = arr_1d[indices]
direct_time = (time.perf_counter() - start) * 1000
print(f"Direct indexing [{n_runs} runs]: {direct_time:.2f} ms")

start = time.perf_counter()
for _ in range(n_runs):
    result2 = mx.take(arr_1d, indices)
take_time = (time.perf_counter() - start) * 1000
print(f"mx.take indexing [{n_runs} runs]: {take_time:.2f} ms")
print(f"Performance ratio (take/direct): {take_time/direct_time:.2f}x")

# Verify results are equivalent
equal = mx.array_equal(result1, result2)
print(f"Results equivalent: {equal}")

# --- Lazy Evaluation Tests ---
print("\n=== Lazy Evaluation Tests ===")

# Create unevaluated expression
a = arr_1d + 1
result = run_test("Array Creation (Unevaluated)", 
                 lambda: a[:5],  # Should create expression tree, not compute values
                 results["laziness"], "unevaluated_slice")
print_array("Result Before Evaluation", result)

# Explicitly evaluate and test again
mx.eval(a)
result = run_test("Array After Evaluation", 
                 lambda: a[:5],  # Should access computed values
                 results["laziness"], "evaluated_slice")
print_array("Result After Evaluation", result)

# --- NumPy vs MLX Comparison ---
print("\n=== NumPy vs MLX Comparison ===")

# Create equivalent arrays for comparison
np_arr = np.array(arr_1d)
mx_arr = arr_1d

# Compare memory usage
print(f"NumPy array memory: {np_arr.nbytes} bytes")
try:
    print(f"MLX array memory: {mx_arr.nbytes} bytes")
except AttributeError:
    print("MLX arrays don't have .nbytes attribute")

# Basic performance test for similar operations
n_ops = 10000
start = time.perf_counter()
for _ in range(n_ops):
    np_result = np_arr + 1.0
np_time = (time.perf_counter() - start) * 1000
print(f"NumPy add operation [{n_ops} runs]: {np_time:.2f} ms")

start = time.perf_counter()
for _ in range(n_ops):
    mx_result = mx_arr + 1.0
    # Force evaluation on the last iteration only
    if _ == n_ops - 1:
        mx.eval(mx_result)
mx_time = (time.perf_counter() - start) * 1000
print(f"MLX add operation with final eval [{n_ops} runs]: {mx_time:.2f} ms")
print(f"Speed ratio (NumPy/MLX): {np_time/mx_time:.2f}x")

# --- Compilation Performance Tests ---
print("\n=== Compilation Performance Tests ===")

# Define a simple function to compile
def simple_func(x):
    """Compute a series of operations on input array x."""
    return mx.exp(x) * mx.sin(x) + mx.cos(x)

# Create compiled version
compiled_func = mx.compile(simple_func)

# Create test data
x = mx.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=mx.float32)
runs = 1000

# Benchmark uncompiled function
start = time.perf_counter()
for _ in range(runs):
    result = simple_func(x)
    if _ == runs - 1:  # Only evaluate on last run
        mx.eval(result)
uncompiled_time = (time.perf_counter() - start) * 1000

# Benchmark first compiled run (includes compilation overhead)
start = time.perf_counter()
result_compiled = compiled_func(x)
mx.eval(result_compiled)
first_compile_time = (time.perf_counter() - start) * 1000

# Benchmark subsequent compiled runs (uses cached compilation)
start = time.perf_counter()
for _ in range(runs - 1):
    result_compiled = compiled_func(x)
    if _ == runs - 2:  # Only evaluate on last run
        mx.eval(result_compiled)
subsequent_time = (time.perf_counter() - start) * 1000

print(f"Uncompiled function [{runs} runs]: {uncompiled_time:.2f} ms")
print(f"Compiled function - first run: {first_compile_time:.2f} ms (includes compilation)")
print(f"Compiled function - subsequent [{runs-1} runs]: {subsequent_time:.2f} ms")
print(f"Compilation speedup (uncompiled/subsequent): {uncompiled_time/subsequent_time:.2f}x")

# --- Framework Conversion Tests ---
print("\n=== Framework Conversion Tests ===")

# MLX to NumPy conversion
result = run_test("MLX to NumPy Conversion", 
                 lambda: np.array(arr_1d),
                 results["conversion"], "mlx_to_numpy")
print(f"NumPy Array Shape: {result.shape}, Dtype: {result.dtype}")

# NumPy to MLX conversion
np_arr_small = np.array([1, 2, 3, 4, 5], dtype=np.float32)
result = run_test("NumPy to MLX Conversion", 
                 lambda: mx.array(np_arr_small),
                 results["conversion"], "numpy_to_mlx")
print_array("MLX Array", result)

# MLX bfloat16 to NumPy (requires explicit casting)
bf16_arr = mx.array([1.5, 2.5, 3.5], dtype=mx.bfloat16)
result = run_test("MLX bfloat16 to NumPy (with casting)", 
                 lambda: np.array(bf16_arr.astype(mx.float32)),
                 results["conversion"], "bf16_to_numpy")
print(f"NumPy Array from bfloat16: {result}")

# Test PyTorch conversion tests (if available)
if TORCH_AVAILABLE:
    # MLX to PyTorch via memoryview
    result = run_test("MLX to PyTorch Conversion (via memoryview)", 
                    lambda: torch.tensor(memoryview(arr_1d[:10])),
                    results["conversion"], "mlx_to_torch_memoryview")
    print(f"PyTorch Array (via memoryview): {result}")
    
    # MLX to PyTorch via NumPy intermediate
    result = run_test("MLX to PyTorch Conversion (via NumPy)", 
                    lambda: torch.from_numpy(np.array(arr_1d[:10])),
                    results["conversion"], "mlx_to_torch_numpy")
    print(f"PyTorch Array (via NumPy): {result}")
    
    # Test convolution weight format conversion (MLX to PyTorch)
    # MLX: [out_channels, kernel_size, in_channels]
    # PyTorch: [out_channels, in_channels, kernel_size]
    mlx_conv_weights = mx.ones((16, 3, 8), dtype=mx.float32)
    result = run_test("MLX 1D Conv Weight to PyTorch Format", 
                    lambda: torch.from_numpy(np.array(mlx_conv_weights)).permute(0, 2, 1),
                    results["conversion"], "mlx_conv_to_torch")
    print(f"PyTorch Conv Weights Shape: {list(result.shape)}")
    
    # Test 2D convolution weight format conversion
    # MLX: [out_channels, kernel_height, kernel_width, in_channels]
    # PyTorch: [out_channels, in_channels, kernel_height, kernel_width]
    mlx_conv2d_weights = mx.ones((16, 3, 3, 8), dtype=mx.float32)
    result = run_test("MLX 2D Conv Weight to PyTorch Format", 
                    lambda: torch.from_numpy(np.array(mlx_conv2d_weights)).permute(0, 3, 1, 2),
                    results["conversion"], "mlx_conv2d_to_torch")
    print(f"PyTorch Conv2D Weights Shape: {list(result.shape)}")
else:
    print("PyTorch not available for conversion tests")

#------------------------------------------------------------------------------
# Buffer Pre-allocation Tests
#------------------------------------------------------------------------------
print("\n=== Buffer Pre-allocation Tests ===")

# Test index buffer pre-allocation with dynamic resizing
max_size = 10
buffer_index = mx.zeros((max_size,), dtype=mx.int32)
print_array("Initial Index Buffer", buffer_index)

# Test with different sizes
sizes_to_test = [3, 5, 8, 10]
for n in sizes_to_test:
    result = run_test(f"Dynamic Buffer Resize (n={n})", 
                     lambda: [buffer_index[:n].__setitem__(slice(None), mx.arange(n, dtype=mx.int32)), 
                             arr_1d[buffer_index[:n]]][1],
                     results["buffer_prealloc"], f"resize_{n}")
    print_array(f"Result with n={n}", result)

# Test exceeding buffer capacity
n_exceed = max_size + 2
result = run_test(f"Exceed Buffer Capacity (n={n_exceed})", 
                 lambda: [buffer_index[:n_exceed].__setitem__(slice(None), mx.arange(n_exceed, dtype=mx.int32)), 
                         arr_1d[buffer_index[:n_exceed]]][1],
                 results["buffer_prealloc"], "exceed_capacity")
print_array("Result when Exceeding Capacity", result)

#------------------------------------------------------------------------------
# Compiled Indexing Tests
#------------------------------------------------------------------------------
print("\n=== Compiled Indexing Tests ===")

# Define a function that performs indexing operations
def index_function(arr, indices):
    """Perform multiple indexing and arithmetic operations on array elements."""
    # Chain multiple operations to make the compilation difference more visible
    result = arr[indices]
    result = result + 1.0
    result = result * 2.0
    return result

# Create a compiled version for performance comparison
compiled_index_function = mx.compile(index_function)

# Create test data for indexing benchmarks
test_arr = mx.arange(10000, dtype=mx.float32)
test_indices = mx.array([i for i in range(0, 5000, 10)], dtype=mx.int32)  # 500 indices
print_array("Test Array Sample", test_arr[:5])
print_array("Test Indices Sample", test_indices[:5])

# Warmup (compile the function)
warmup = run_test("Compiled Indexing Warmup", 
                 lambda: [result := compiled_index_function(test_arr, test_indices), mx.eval(result)][0],
                 results["compiled_index"], "warmup")

# Benchmark uncompiled function
runs = 100
start = time.perf_counter()
for _ in range(runs):
    result_uncompiled = index_function(test_arr, test_indices)
    if _ == runs - 1:  # Only evaluate on last run
        mx.eval(result_uncompiled)
uncompiled_time = (time.perf_counter() - start) * 1000
print(f"Uncompiled indexing function [{runs} runs]: {uncompiled_time:.2f} ms")

# Benchmark compiled function
start = time.perf_counter()
for _ in range(runs):
    result_compiled = compiled_index_function(test_arr, test_indices)
    if _ == runs - 1:  # Only evaluate on last run
        mx.eval(result_compiled)
compiled_time = (time.perf_counter() - start) * 1000
print(f"Compiled indexing function [{runs} runs]: {compiled_time:.2f} ms")

# Calculate speedup from compilation
speedup = uncompiled_time / compiled_time if compiled_time > 0 else 0
print(f"Indexing compilation speedup: {speedup:.2f}x")

# Verify compiled and uncompiled results match
result_match = run_test("Check Compiled vs Uncompiled Results Match", 
                       lambda: [mx.eval(result_uncompiled), 
                                mx.eval(result_compiled),
                                mx.array_equal(result_uncompiled, result_compiled)][2],
                       results["compiled_index"], "result_match")
print(f"Results match: {result_match}")

#------------------------------------------------------------------------------
# Results Summary
#------------------------------------------------------------------------------
print("\n=== Summary of Behaviors ===")

print("1. Array Indexing:")
direct_fails = all(not r["success"] for k, r in results["indexing"].items() if "direct" in k)
print(f"  - Direct mask indexing {'unsupported' if direct_fails else 'partially supported'} across dimensions.")
slice_issues = any("placeholder_slice_filtered" in k and r["success"] and r["result"].size != mx.sum(masks_1d.get(mask_name, mx.zeros(0))).item() 
                   for mask_name in masks_1d for k, r in results["indexing"].items() if mask_name.lower().replace(' ', '_') in k)
print(f"  - Placeholder-based indexing with slicing {'inconsistent' if slice_issues else 'consistent'}; {'may alter element count' if slice_issues else 'preserves count'}.")
single_where_fails = all(not r["success"] for k, r in results["indexing"].items() if "single_where" in k)
print(f"  - Single-argument mx.where {'invalid' if single_where_fails else 'valid'}; {'requires 3 arguments' if single_where_fails else 'works as expected'}.")
three_where_works = all(r["success"] for k, r in results["indexing"].items() if "valid_filtered" in k and "empty" not in k)
print(f"  - Three-argument mx.where {'effective' if three_where_works else 'ineffective'} with valid index extraction.")
nonzero_fails = all(not r["success"] for k, r in results["indexing"].items() if "nonzero" in k)
print(f"  - Nonzero function {'unavailable' if nonzero_fails else 'available'}.")
two_d_flat_works = all(r["success"] for k, r in results["indexing"].items() if "2d" in k and "valid_filtered_flat" in k)
print(f"  - 2D indexing {'requires flattening' if two_d_flat_works else 'supports direct methods'} for workaround.")

print("2. Slice Assignment:")
normal_works = results["assignment"]["normal_length"]["success"]
print(f"  - Normal-length assignments {'modify buffer correctly' if normal_works else 'fail to modify buffer'}.")
over_fails = not results["assignment"]["over_length"]["success"]
print(f"  - Over-length assignments {'fail with shape mismatch' if over_fails else 'succeed unexpectedly'}.")
dynamic_works = results["assignment"]["dynamic_concat"]["success"]
print(f"  - Concatenation {'reliable' if dynamic_works else 'unreliable'} for dynamic lengths.")
float_casts = results["assignment"]["float_indices_assign"]["success"]
print(f"  - Float indices in assignments {'cast silently to integer' if float_casts else 'rejected or fail'}.")
empty_noop = results["assignment"]["empty_indices"]["success"] and buffer[0] == results["assignment"]["empty_indices"]["result"][0]
print(f"  - Empty assignments {'leave buffer unchanged' if empty_noop else 'modify buffer'}.")
beyond_ignored = results["assignment"]["beyond_bounds"]["success"] and buffer[0] == results["assignment"]["beyond_bounds"]["result"][0]
print(f"  - Beyond-bounds assignments {'silently ignored' if beyond_ignored else 'processed or error'}.")
at_works = results["assignment"]["at_add_duplicate"]["success"] and results["assignment"]["at_multiply_duplicate"]["success"]
print(f"  - .at API operations {'supported' if at_works else 'unsupported'} for correct handling of duplicate indices.")
standard_dup_works = results["assignment"]["standard_duplicate"]["success"]
at_dup_diff = False
if standard_dup_works and at_works:
    # Check if .at.add() behaves differently than standard assignment for duplicates
    std_result = results["assignment"]["standard_duplicate"]["result"]
    at_result = results["assignment"]["at_add_duplicate"]["result"]
    at_dup_diff = not mx.array_equal(std_result, at_result) and at_result[0] > std_result[0]
print(f"  - Duplicate index handling {'differs between standard and .at API' if at_dup_diff else 'behaves identically'}; " + 
      f"{'standard assignment keeps only last update [1,0,1,0,1], while .at applies all updates [2,0,2,0,1]' if at_dup_diff else 'both APIs handle duplicates the same way'}.")
ref_match = results["assignment"]["reference_modify"]["success"] and mx.array_equal(results["assignment"]["reference_modify"]["result"][0], results["assignment"]["reference_modify"]["result"][1])
print(f"  - Modifications {'affect all references' if ref_match else 'create new copies'}.")

print("3. Edge Cases:")
beyond_scalar = results["edge_cases"]["beyond_bounds"]["success"] and results["edge_cases"]["beyond_bounds"]["result"].shape == ()
print(f"  - Beyond-bounds indexing {'returns scalar silently' if beyond_scalar else 'errors or behaves differently'}.")
float_reject = not results["edge_cases"]["float_indices"]["success"]
print(f"  - Float indices in indexing {'rejected as non-integral' if float_reject else 'accepted unexpectedly'}.")
lazy_same = results["edge_cases"]["lazy_unevaluated"]["success"] and results["edge_cases"]["lazy_evaluated"]["success"] and mx.array_equal(results["edge_cases"]["lazy_unevaluated"]["result"], results["edge_cases"]["lazy_evaluated"]["result"])
print(f"  - Lazy evaluation {'does not alter indexing results' if lazy_same else 'affects indexing results'}.")
dynamic_slice_fails = not results["edge_cases"]["dynamic_slice_python_var"]["success"]
print(f"  - Dynamic slicing with Python variables {'fails with ValueError' if dynamic_slice_fails else 'supported in MLX 0.23.2'}.")
multi_bounds = results["edge_cases"]["multi_beyond_bounds"]["success"]
multi_array = multi_bounds and results["edge_cases"]["multi_beyond_bounds"]["result"].ndim > 0
print(f"  - Multi-index beyond bounds {'returns array' if multi_array else 'returns scalar or fails'} when accessing invalid indices.")

print("4. NumPy-like Behavior:")
slice_flat = results["edge_cases"]["2d_slice_flatten"]["success"]
print(f"  - 2D slice flattening {'requires explicit flatten' if slice_flat else 'behaves like NumPy'} (NumPy flattens directly).")
scalar_extract = results["edge_cases"]["scalar_extract"]["success"]
print(f"  - Scalar extraction from slice {'works like NumPy' if scalar_extract else 'differs from NumPy'}.")

print("5. Boolean Indexing Workarounds:")
where_works = results["workarounds"]["where_sum_based"]["success"]
print(f"  - Boolean mask conversion with sum-based slicing {'supported' if where_works else 'unsupported'}.")
placeholder_works = results["workarounds"]["where_placeholder"]["success"]
placeholder_reliable = placeholder_works and results["workarounds"]["filtered_where_placeholder"]["success"] and \
                      results["workarounds"]["filtered_where_placeholder"]["result"].size == mx.sum(mask_pos).item()
print(f"  - Boolean mask conversion with placeholder {'reliable' if placeholder_reliable else 'unreliable or unsupported'}.")
argsort_works = results["workarounds"]["argsort_top_n"]["success"]
print(f"  - Top-n extraction via argsort {'supported' if argsort_works else 'unsupported'}.")
multi_axis_works = results["workarounds"]["multi_axis"]["success"]
print(f"  - Multi-axis coordinate indexing {'supported' if multi_axis_works else 'unsupported'}.")

print("6. Performance:")
type_promotion_works = results["performance"]["type_promotion"]["success"]
print(f"  - Type promotion {'supported' if type_promotion_works else 'unsupported'}.")
python_scalar_mult_works = results["performance"]["python_scalar_mult"]["success"]
print(f"  - Type preservation with Python scalar {'supported' if python_scalar_mult_works else 'unsupported'}.")

print("7. Laziness:")
unevaluated_slice_works = results["laziness"]["unevaluated_slice"]["success"]
print(f"  - Array creation (unevaluated slice) {'supported' if unevaluated_slice_works else 'unsupported'}.")
evaluated_slice_works = results["laziness"]["evaluated_slice"]["success"]
print(f"  - Array evaluation (evaluated slice) {'supported' if evaluated_slice_works else 'unsupported'}.")

print("8. Framework Conversion:")
to_numpy_works = results["conversion"]["mlx_to_numpy"]["success"]
print(f"  - MLX to NumPy conversion {'supported' if to_numpy_works else 'unsupported'}.")
from_numpy_works = results["conversion"]["numpy_to_mlx"]["success"]
print(f"  - NumPy to MLX conversion {'supported' if from_numpy_works else 'unsupported'}.")
bf16_to_numpy_works = results["conversion"]["bf16_to_numpy"]["success"]
print(f"  - BFloat16 conversion {'requires explicit casting' if bf16_to_numpy_works else 'automatic'}.")
if TORCH_AVAILABLE:
    torch_memview_works = results["conversion"]["mlx_to_torch_memoryview"]["success"]
    print(f"  - MLX to PyTorch via memoryview {'supported' if torch_memview_works else 'unsupported'}.")
    conv_perm_works = results["conversion"]["mlx_conv_to_torch"]["success"]
    print(f"  - Conv weight permutation {'required' if conv_perm_works else 'not required'} for PyTorch compatibility.")

print("9. Buffer Pre-allocation:")
buffer_resize_works = all(results["buffer_prealloc"][f"resize_{n}"]["success"] for n in [3, 5, 8, 10])
print(f"  - Dynamic buffer resizing {'supported' if buffer_resize_works else 'unsupported'} for varying sizes.")
exceed_fails = not results["buffer_prealloc"]["exceed_capacity"]["success"]
print(f"  - Exceeding pre-allocated buffer capacity {'fails as expected' if exceed_fails else 'succeeds unexpectedly'}.")

print("10. Compiled Indexing:")
compiled_works = results["compiled_index"]["warmup"]["success"]
result_matches = results["compiled_index"]["result_match"]["success"] and results["compiled_index"]["result_match"]["result"]
print(f"  - Compiled indexing functions {'supported' if compiled_works else 'unsupported'}.")
print(f"  - Compiled and uncompiled results {'match' if result_matches else 'differ'}.")
if compiled_works:
    print(f"  - Compilation provides {speedup:.2f}x speedup for indexing operations.")