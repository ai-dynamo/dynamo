# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

kvbm-kernels is a high-performance CUDA transfer library for batched H2D, D2H, and D2D block copies used by the Dynamo KV cache system. The core API (`vectorized_copy`, `memcpy_batch`) is always available and handles the common case of moving KV cache blocks between host and device without layout changes. Fused permute-and-copy kernels for layout conversion between **Block Stack** (vLLM) and **Universal** (Dynamo storage) formats are feature-gated behind `permute_kernels`.

## Build Commands

```bash
# Default build (auto-detects nvcc -> source build; no nvcc -> stubs)
cargo build

# Build from source with custom GPU architectures
CUDA_ARCHS="80,86,89,90,100" cargo build

# Static linking (embed kernels into binary instead of .so)
cargo build --features static-kernels

# Check compilation without linking
cargo check

# Run CUDA integration tests for core transfer APIs (requires GPU + nvcc)
cargo test --features testing-cuda

# Run all CUDA integration tests including permute kernels
cargo test --features testing-cuda,permute_kernels

# Run KVTC quantization tests
cargo test --features testing-cuda,kvtc_kernels

# Run all CUDA tests (all features)
cargo test --features testing-cuda,permute_kernels,kvtc_kernels

# Run a specific test
cargo test --features testing-cuda,permute_kernels fused_copy_roundtrip -- --nocapture --test-threads=1

# Run transfer benchmarks (Llama 3.1 70B KV cache profile)
cargo run --example kvbench --features kvbench

# Run KVTC compression benchmarks
cargo run --example kvbench --features kvbench -- --kvtc

# Run KVTC benchmarks with specific quantization types
cargo run --example kvbench --features kvbench -- --kvtc --kvtc-bits fp8,int4
```

**Environment variables**: `CUDA_ARCHS` (comma-separated SM versions), `CUDA_PTX_ARCHS` (PTX targets), `KVBM_REQUIRE_CUDA` (fail if nvcc missing), `CUDA_PATH`/`CUDA_HOME`.

## Architecture

### Two-tier build system (`build.rs`)

The build script selects one of two modes: **FromSource** (nvcc available, compiles CUDA, requires CUDA >= 12.0) or **Stubs** (no nvcc, C stubs that abort on call). Stubs set the `stub_kernels` cfg flag so tests can be conditionally skipped.

### Core transfer API (always available)

These live in `src/tensor_kernels.rs` and work on any device-visible memory (device allocations or pinned host via unified addressing):

- **`vectorized_copy`** — Batched copy of `(src, dst)` pointer pairs. Per-pair runtime alignment detection selects the widest safe vector width (int4/int2/int/char for 16/8/4/1-byte loads).
- **`memcpy_batch`** — Takes HOST arrays of src/dst pointers. Dispatches to `cudaMemcpyBatchAsync` (CUDA 12.9+) with fallback to individual `cudaMemcpyAsync` loop. Three modes: `BatchedWithFallback`, `FallbackOnly`, `BatchWithoutFallback`.
- **`is_using_stubs`** / **`is_memcpy_batch_available`** — Runtime capability queries.

### KVTC kernels (feature-gated: `kvtc_kernels`)

These implement KV Cache Transform Coding (KVTC) compression — a full GPU pipeline for KV cache compression with PCA projection, mixed-precision quantization, and scattered block gather/scatter:

**Quantization kernels:**
- **`quantize_fp8`** / **`dequantize_fp8`** — Convert float32 ↔ fp8_e4m3fn (1 byte/value). Native `cuda_fp8.h` intrinsics on SM89+ (Ada/Hopper); software emulation fallback on older architectures.
- **`minmax_reduce`** — Per-row min/max reduction for IntX scaling parameters.
- **`quantize_intx`** / **`dequantize_intx`** — Min/max scaled quantization with bit-packing (1/2/4/8-bit). Packs multiple values per byte LSB-first.
- **`kvtc_quantize_ranges`** / **`kvtc_dequantize_ranges`** — Range-iterating orchestrators that produce/consume a self-describing compressed format with per-range headers.

**Gather/scatter kernels:**
- **`gather_mean_subtract`** — Read from scattered block pointers (typed: F16/BF16/F32/F64), cast to float32, subtract mean, write contiguous output.
- **`mean_add_scatter`** — Add mean to float32 input, cast to target dtype, scatter-write to block pointers.

**Full pipeline (PCA + quantization via cuBLAS):**
- **`kvtc_compress`** — Gather from scattered blocks → mean-subtract → cuBLAS SGEMM (PCA projection) → quantize ranges → self-describing compressed output (can write directly to pinned host).
- **`kvtc_decompress`** — Dequantize ranges → cuBLAS SGEMM (inverse PCA) → mean-add → scatter to block pointers.
- **`KvtcConfig`** — Configuration struct holding device pointers to mean vector and projection matrix, plus quantization range definitions.
- **`kvtc_workspace_size`** / **`kvtc_compressed_size`** — Buffer size helpers.
- **`kvtc_create_cublas_handle`** / **`kvtc_destroy_cublas_handle`** — cuBLAS handle management (direct FFI, not cudarc's dlopen loader).

### Permute kernels (feature-gated: `permute_kernels`)

These fuse layout permutation with copy for non-standard transfer paths:

- **`universal_from_block`** / **`block_from_universal`** — Permute between block stack layout (`nl*no` separate allocations, each `[nt, nh, hd]` NHD or `[nh, nt, hd]` HND) and universal layout (contiguous `[nh, nl, no, nt, hd]`).

### Source organization

- `cuda/tensor_kernels.cu` — Transfer/permute CUDA kernels. C++ templates on dtype (F16/BF16/F32/F64) and layout (NHD/HND), exposed via `extern "C"` functions prefixed `kvbm_kernels_launch_*` / `kvbm_kernels_memcpy_batch`.
- `cuda/kvtc_kernels.cu` — KVTC CUDA kernels: FP8 E4M3FN conversion (native intrinsics on SM89+, software fallback otherwise), per-row min/max reduction, IntX quantize/dequantize with bit-packing, typed gather/scatter with mean subtract/add. Extern "C" functions prefixed `kvbm_kernels_kvtc_*`.
- `cuda/stubs.c` — Abort-on-call fallbacks for all `extern "C"` symbols.
- `src/tensor_kernels.rs` — Rust FFI wrappers, enums (`TensorDataType`, `BlockLayout`, `MemcpyBatchMode`), and integration tests.
- `src/kvtc_kernels.rs` — KVTC Rust FFI wrappers, types (`KvtcQuantType`, `KvtcTypeRange`, `KvtcRangeHeader`, `KvtcConfig`, `TensorDataType`), size helpers, range-iterating orchestrators, and full compress/decompress pipeline with cuBLAS SGEMM (direct FFI).
- `examples/kvbench.rs` — Benchmark harness: transfer bandwidth (Llama 3.1 70B profile) and KVTC compress/decompress throughput (`--kvtc` mode). CSV output.
- `scripts/plot_roofline.py` — Roofline bandwidth plots from kvbench output.

### Dimension conventions

`nl` = layers, `no` = outer chunks (2: K and V), `nh` = attention heads, `nt` = tokens per block, `hd` = head dimension.

### Pointer conventions

All pointer-list parameters (e.g. `universal_ptrs`, `src_ptrs`) must be device-accessible: allocated via `cudaMalloc` (device memory) or `cudaMallocHost` / `cuMemHostRegister` (pinned/registered/page-locked host memory).

### Cargo features

| Feature | Purpose |
|---------|---------|
| `permute_kernels` | Enable fused permute-and-copy kernels (block<->universal) |
| `kvtc_kernels` | Enable KVTC quantization kernels (FP8/IntX) |
| `testing-cuda` | Enable CUDA integration tests |
| `static-kernels` | Link as `.a` instead of `.so` |
| `kvbench` | Enable benchmark example (pulls in `clap` + `kvtc_kernels`); supports `--kvtc` mode |

### Test organization

- `tests/stub_build.rs` — Verifies stub behavior (gated on `stub_kernels`).
- `tests/memcpy_batch.rs` — Core transfer API roundtrip tests (H2D + D2H via pinned host memory). Gated on `testing-cuda`.
- `tests/kernel_roundtrip.rs` — Permute kernel roundtrip tests across all dtypes and layouts. Gated on `testing-cuda` + `permute_kernels`.
- `tests/kvtc_roundtrip.rs` — KVTC roundtrip tests: FP8, IntX 1/2/4/8-bit, min/max, multi-range orchestrator, gather/scatter with mean, identity projection compress/decompress, orthogonal projection with dimensionality reduction. Gated on `testing-cuda` + `kvtc_kernels`.
- Inline tests in `src/tensor_kernels.rs` — Integration tests including `universal_roundtrip`. Gated on `testing-cuda` + `permute_kernels`.
