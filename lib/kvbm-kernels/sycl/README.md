# SYCL Kernels for XPU

SYCL C++ sources compiled into `libkvbm_kernels_xpu.so` and called from Rust
via `extern "C"` FFI through `tensor_kernels_xpu.rs`.

## Files

- `tensor_permute_kernel.cpp` — Block/universal permute kernels (`universal_from_block`, `block_from_universal`).
- `vectorized_copy_kernel.cpp` — Vectorized memory copy between device pointer pairs.

## Build

Compiled automatically by `build.rs` when `KVBM_ENABLE_XPU_KERNELS` is set.
Requires Intel oneAPI DPC++ compiler (`icpx`).

```bash
# Automatic (via cargo):
KVBM_ENABLE_XPU_KERNELS=1 cargo build --features xpu_permute_kernels

# Manual (equivalent):
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -shared -fPIC -O2 -o libkvbm_kernels_xpu.so \
     tensor_permute_kernel.cpp vectorized_copy_kernel.cpp
```

The resulting `libkvbm_kernels_xpu.so` is linked dynamically (`-lkvbm_kernels_xpu -lsycl`).

## Extern "C" API

### vectorized_copy_kernel.cpp

```c
int kvbm_kernels_xpu_launch_vectorized_copy(
    void** src_ptrs,          // device pointer array of sources
    void** dst_ptrs,          // device pointer array of destinations
    size_t copy_size_bytes,   // bytes per pair (uniform)
    int    num_pairs,         // number of pointer pairs
    void*  queue_ptr          // opaque sycl::queue*
);
```

- Work group size: 128
- Per-pair alignment detection: 16B / 8B / 4B / 1B vectorized copy

### tensor_permute_kernel.cpp

```c
int kvbm_kernels_xpu_launch_universal_from_block(
    void* const* universal_ptrs, const void* const* block_ptrs,
    size_t num_blocks, size_t nh, size_t nl, size_t no, size_t nt, size_t hd,
    size_t elem_size, int layout_value, void* queue_ptr);

int kvbm_kernels_xpu_launch_block_from_universal(
    const void* const* universal_ptrs, void* const* block_ptrs,
    size_t num_blocks, size_t nh, size_t nl, size_t no, size_t nt, size_t hd,
    size_t elem_size, int layout_value, void* queue_ptr);
```

- `layout_value`: 0 = NHD, 1 = HND
- `elem_size`: bytes per element (2 = f16/bf16, 4 = f32, 8 = f64)
- All pointer arrays and queue must be valid for the duration of the kernel
