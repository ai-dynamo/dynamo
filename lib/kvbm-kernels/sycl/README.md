# SYCL Kernels for XPU (Level-Zero)

## Files

- `vectorized_copy_kernel.cpp` — SYCL free-function kernel source for vectorized memory copy between arbitrary device pointer pairs.
- `vectorized_copy_kernel.spv` — Pre-compiled SPIR-V binary, embedded at build time via `include_bytes!` in `kvbm-physical/src/device/ze/mod.rs`.

## Building the SPIR-V Binary

Requires Intel oneAPI DPC++ compiler (`icpx`) and `llvm-spirv` translator.

```bash
# 1. Source oneAPI environment (if not already active)
source /opt/intel/oneapi/setvars.sh

# 2. Compile SYCL source to LLVM bitcode
icpx -fsycl -fsycl-device-only -fsycl-targets=spir64 -O2 \
     -o vectorized_copy_kernel.bc vectorized_copy_kernel.cpp

# 3. Convert LLVM bitcode to SPIR-V
llvm-spirv vectorized_copy_kernel.bc -o vectorized_copy_kernel.spv

# 4. Verify SPIR-V magic (should print: 03 02 23 07)
od -A x -t x1z -N 4 vectorized_copy_kernel.spv

# 5. Clean up intermediate file
rm -f vectorized_copy_kernel.bc
```

If `llvm-spirv` is not on `$PATH`, it is typically at:
```
/opt/intel/oneapi/compiler/<version>/bin/compiler/llvm-spirv
```

## Runtime Behavior

- **Valid `.spv`**: Loaded via `zeModuleCreate` at device init. The kernel `kvbm_vectorized_copy` is launched on the compute engine via `zeCommandListAppendLaunchKernel`.
- **Placeholder/invalid `.spv`**: Module creation fails gracefully. A warning is logged and the fallback path (host-readback + `batch_copy`) is used instead.

## Kernel Interface

```
void kvbm_vectorized_copy(
    void** src_ptrs,          // device pointer array of sources
    void** dst_ptrs,          // device pointer array of destinations
    size_t copy_size_bytes,   // bytes per pair (uniform)
    int    num_pairs          // number of pointer pairs
)
```

- Work group size: 128
- Max work groups: min(num_pairs, 65535)
- Per-pair alignment detection: 16B / 8B / 4B / 1B vectorized copy
