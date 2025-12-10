<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVBM Kernel Testing Architecture

This document explains how CUDA kernels are tested through PyTorch bindings using a multi-layer FFI architecture.

## What is FFI?

**FFI (Foreign Function Interface)** is a mechanism that allows code written in one programming language to call functions written in another language. In this project, we use FFI in two places:

1. **Rust ↔ CUDA**: Rust calls C/C++ CUDA functions via the C ABI
2. **Python ↔ Rust**: Python calls Rust functions via PyO3 (a Rust-Python bridge)

This enables us to write performance-critical code in CUDA while making it accessible from high-level Python/PyTorch for testing and integration.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Python Tests (test_tensor_kernels.py)             │
│ • Creates PyTorch tensors as test data                      │
│ • Uses pure PyTorch ops as reference implementation         │
│ • Validates CUDA kernels match PyTorch reference            │
└──────────────────┬──────────────────────────────────────────┘
                   │ import
                   │ from kvbm import kernels
┌──────────────────▼──────────────────────────────────────────┐
│ Layer 3: PyO3 Bindings (src/kernels.rs)                    │
│ • Exposes Rust/CUDA functions to Python                     │
│ • Extracts GPU pointers from PyTorch tensors                │
│ • Validates shapes, dtypes, device placement                │
│ • Functions: block_to_universal(), universal_to_block()     │
└──────────────────┬──────────────────────────────────────────┘
                   │ calls (Rust FFI)
                   │ use kvbm_kernels
┌──────────────────▼──────────────────────────────────────────┐
│ Layer 2: Rust FFI (lib/kvbm-kernels/src/tensor_kernels.rs) │
│ • Wraps CUDA kernels with safe Rust API                     │
│ • Manages CUDA contexts, streams, memory                    │
│ • Exports: universal_from_block(), block_from_universal()   │
└──────────────────┬──────────────────────────────────────────┘
                   │ extern "C" calls
                   │ unsafe { cuda_function(...) }
┌──────────────────▼──────────────────────────────────────────┐
│ Layer 1: CUDA Kernels (lib/kvbm-kernels/cuda/*.cu)         │
│ • Raw CUDA kernel implementations                           │
│ • Converts between KV cache layouts on GPU                  │
│ • Files: tensor_kernels.cu, vectorized_copy.cu             │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Layer Breakdown

### Layer 1: CUDA Kernels (`lib/kvbm-kernels/cuda/`)

**Purpose**: Implements the actual GPU computation

**Files**:
- `tensor_kernels.cu` - Converts between Stacked (vLLM), Operational (TensorRT-LLM), and Universal (Dynamo) layouts
- `vectorized_copy.cu` - Optimized memory copy operations

**Example**:
```cuda
// CUDA kernel that does the actual work
__global__ void universal_from_block_kernel(
    void** universal_ptrs,
    const void** block_ptrs,
    size_t nb, size_t nh, size_t nl, size_t no, size_t nt, size_t hd
) {
    // GPU code that rearranges memory layouts
}
```

### Layer 2: Rust FFI (`lib/kvbm-kernels/src/`)

**Purpose**: Wraps CUDA kernels with type-safe Rust API

**Files**:
- `lib.rs` - Module initialization, loads CUDA fatbin files
- `tensor_kernels.rs` - Rust wrappers for CUDA functions

**Example**:
```rust
// Rust function that calls CUDA kernel via FFI
pub unsafe extern "C" fn universal_from_block(
    universal_ptrs: *const *mut c_void,
    block_ptrs: *const *const c_void,
    nb: usize, nh: usize, nl: usize, no: usize, nt: usize, hd: usize,
    dtype: TensorDataType,
    layout: BlockLayout,
    stream: cudaStream_t,
) -> cudaError_t {
    // Call CUDA kernel
    // Return CUDA status code
}
```

**Why Rust?**
- Memory safety without runtime overhead
- Strong type system catches errors at compile time
- Excellent FFI support for calling C/CUDA code

### Layer 3: PyO3 Bindings (`lib/bindings/kvbm/src/kernels.rs`)

**Purpose**: Exposes Rust/CUDA functions to Python

**Example**:
```rust
#[pyfunction]
unsafe fn block_to_universal(
    py: Python<'_>,
    blocks: &Bound<'_, PyAny>,      // PyTorch tensors
    universals: &Bound<'_, PyAny>,  // PyTorch tensors
    layout: &str,
) -> PyResult<()> {
    // 1. Extract GPU pointers from PyTorch tensors
    let ptr: usize = tensor.call_method0("data_ptr")?;
    let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;

    // 2. Validate tensor properties
    if !tensor.getattr("is_cuda")?.extract()? {
        return Err(PyValueError::new_err("Tensor must be on CUDA"));
    }

    // 3. Call Rust/CUDA function
    let status = universal_from_block(
        universal_ptrs, block_ptrs, nb, nh, nl, no, nt, hd,
        dtype, layout_enum, stream
    );

    // 4. Handle errors and synchronize
    if status != cudaSuccess {
        return Err(PyRuntimeError::new_err("CUDA error"));
    }
    stream.synchronize()?;
    Ok(())
}
```

**What PyO3 Does**:
- Converts Python objects to Rust types
- Extracts raw GPU memory pointers from PyTorch tensors
- Validates input (shapes, dtypes, device placement)
- Handles errors and converts them to Python exceptions
- Manages CUDA stream synchronization

### Layer 4: Python Tests (`lib/bindings/kvbm/tests/`)

**Purpose**: Validates kernel correctness using PyTorch as reference

**Files**:
- `test_tensor_kernels.py` - Comprehensive kernel tests

**Testing Strategy**:
1. Create random PyTorch CUDA tensors
2. Generate **reference output** using pure PyTorch operations (slicing, permuting)
3. Run **CUDA kernel** through PyO3 bindings
4. Compare kernel output vs PyTorch reference with appropriate tolerances

**Example Test**:
```python
import torch
from kvbm import kernels as ctk

def test_block_universal_roundtrip():
    # 1. Create test data
    device = torch.device("cuda:0")
    universals = [torch.randn(3, 2, 2, 4, 5, device=device)]  # [nh, nl, no, nt, hd]

    # 2. Reference: Convert using pure PyTorch
    def _make_blocks(universal, layout="NHD"):
        nh, nl, no, nt, hd = universal.shape
        blocks = []
        for layer in range(nl):
            for outer in range(no):
                slice_ = universal[:, layer, outer, :, :]  # [nh, nt, hd]
                block = slice_.permute(1, 0, 2)             # [nt, nh, hd] for NHD
                blocks.append(block)
        return blocks

    blocks = _make_blocks(universals[0], "NHD")

    # 3. Test: Run CUDA kernel
    outputs = [torch.empty_like(universals[0])]
    ctk.block_to_universal(blocks, outputs, "NHD")  # ← Calls CUDA via PyO3!
    torch.cuda.synchronize()

    # 4. Validate: CUDA output should match PyTorch reference
    assert torch.allclose(outputs[0], universals[0], atol=1e-5, rtol=1e-5)
```

## Why This Architecture?

### Separation of Concerns
- **CUDA**: Performance-critical GPU code
- **Rust**: Safe systems programming, manages memory/contexts
- **Python**: High-level testing, integration with ML frameworks

### Testing Benefits
1. **Ground Truth**: Pure PyTorch operations are easy to understand and verify
2. **Black Box**: Tests don't need to know CUDA implementation details
3. **Comprehensive**: Can test many configurations (dtypes, layouts, backends)
4. **Debuggable**: When tests fail, can compare intermediate results

### Development Workflow
```bash
# 1. Modify CUDA kernel
vim lib/kvbm-kernels/cuda/tensor_kernels.cu

# 2. Rebuild (compiles CUDA, Rust, and Python bindings)
cd lib/bindings/kvbm
cargo build --release

# 3. Run tests
pytest tests/test_tensor_kernels.py -v

# 4. If tests fail, debug with PyTorch
python -c "import torch; from kvbm import kernels; ..."
```

## Running the Tests

### Prerequisites
```bash
# CUDA toolkit (for GPU tests)
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install kvbm package with dev dependencies
cd lib/bindings/kvbm
pip install -e ".[dev]"
```

### Run All Tests
```bash
cd lib/bindings/kvbm
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_tensor_kernels.py::test_block_universal_roundtrip -v
```

### Run with Specific Parameters
```bash
# Test only NHD layout with float32
pytest tests/test_tensor_kernels.py::test_block_universal_roundtrip[NHD-torch.float32] -v
```

## Test Coverage

### `test_block_universal_roundtrip`
- **Tests**: `block_to_universal()` and `universal_to_block()`
- **Layouts**: NHD (vLLM), HND
- **Dtypes**: float16, bfloat16, float32, float64
- **Validates**: Lossless round-trip conversion

### `test_operational_roundtrip`
- **Tests**: `block_to_operational()` and `operational_to_block()`
- **Validates**: Correct flattening/unflattening of block data

### `test_operational_backends`
- **Tests**: Different memcpy backends (kernel, async, batch, auto)
- **Validates**: All backends produce correct results

### Error Handling Tests
- `test_universal_shape_mismatch` - Rejects incorrect shapes
- `test_dtype_mismatch_error` - Rejects mixed dtypes
- `test_non_cuda_tensor_error` - Rejects CPU tensors
- `test_empty_batch_noop` - Handles empty inputs gracefully

## Debugging Tips

### Enable CUDA Error Checking
```python
import torch
torch.cuda.set_sync_debug_mode(1)  # Synchronous CUDA calls for debugging
```

### Compare Intermediate Results
```python
# Get reference
blocks_ref = _make_blocks(universal, "NHD")

# Run kernel
outputs = [torch.empty_like(universal)]
ctk.block_to_universal(blocks, outputs, "NHD")

# Compare specific elements
print(f"Max diff: {(outputs[0] - universal).abs().max()}")
print(f"Mean diff: {(outputs[0] - universal).abs().mean()}")
```

### Check Tensor Properties
```python
def inspect_tensor(t, name):
    print(f"{name}:")
    print(f"  shape: {t.shape}")
    print(f"  dtype: {t.dtype}")
    print(f"  device: {t.device}")
    print(f"  is_contiguous: {t.is_contiguous()}")
    print(f"  data_ptr: 0x{t.data_ptr():x}")
```

## Common Issues

### Issue: "Tensor must be contiguous"
**Solution**: Call `.contiguous()` before passing to kernel
```python
tensor = tensor.contiguous()
ctk.block_to_universal(blocks, [tensor], "NHD")
```

### Issue: "Mixed dtype error"
**Solution**: Ensure all tensors in a batch have the same dtype
```python
# Bad: mixed dtypes
blocks = [torch.randn(..., dtype=torch.float32), torch.randn(..., dtype=torch.float16)]

# Good: consistent dtype
blocks = [torch.randn(..., dtype=torch.float32) for _ in range(n)]
```

### Issue: "Shape mismatch"
**Solution**: Verify dimensions match expected layout
```python
# For NHD layout, each block should be [nt, nh, hd]
# For universal, should be [nh, nl, no, nt, hd]
```

## Further Reading

- [KVBM Kernels README](../../../lib/kvbm-kernels/README.md) - Detailed explanation of kernel layouts
- [PyO3 Documentation](https://pyo3.rs/) - Python-Rust FFI framework
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - CUDA fundamentals
