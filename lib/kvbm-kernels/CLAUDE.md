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

# Run a specific test
cargo test --features testing-cuda,permute_kernels fused_copy_roundtrip -- --nocapture --test-threads=1

# Run benchmarks (Llama 3.1 70B KV cache profile)
cargo run --example kvbench --features kvbench
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

### Permute kernels (feature-gated: `permute_kernels`)

These fuse layout permutation with copy for non-standard transfer paths:

- **`universal_from_block`** / **`block_from_universal`** — Permute between block stack layout (`nl*no` separate allocations, each `[nt, nh, hd]` NHD or `[nh, nt, hd]` HND) and universal layout (contiguous `[nh, nl, no, nt, hd]`).

### Source organization

- `cuda/tensor_kernels.cu` — All CUDA kernels. C++ templates on dtype (F16/BF16/F32/F64) and layout (NHD/HND), exposed via `extern "C"` functions prefixed `kvbm_kernels_launch_*` / `kvbm_kernels_memcpy_batch`.
- `cuda/stubs.c` — Abort-on-call fallbacks for all `extern "C"` symbols.
- `src/tensor_kernels.rs` — Rust FFI wrappers, enums (`TensorDataType`, `BlockLayout`, `MemcpyBatchMode`), and integration tests.
- `examples/kvbench.rs` — Benchmark harness (Llama 3.1 70B profile, CSV output).
- `scripts/plot_roofline.py` — Roofline bandwidth plots from kvbench output.

### Dimension conventions

`nl` = layers, `no` = outer chunks (2: K and V), `nh` = attention heads, `nt` = tokens per block, `hd` = head dimension.

### Pointer conventions

All pointer-list parameters (e.g. `universal_ptrs`, `src_ptrs`) must be device-accessible: allocated via `cudaMalloc` (device memory) or `cudaMallocHost` / `cuMemHostRegister` (pinned/registered/page-locked host memory).

### Cargo features

| Feature | Purpose |
|---------|---------|
| `permute_kernels` | Enable fused permute-and-copy kernels (block<->universal) |
| `testing-cuda` | Enable CUDA integration tests |
| `static-kernels` | Link as `.a` instead of `.so` |
| `kvbench` | Enable benchmark example (pulls in `clap`) |

### Test organization

- `tests/stub_build.rs` — Verifies stub behavior (gated on `stub_kernels`).
- `tests/memcpy_batch.rs` — Core transfer API roundtrip tests (H2D + D2H via pinned host memory). Gated on `testing-cuda`.
- `tests/kernel_roundtrip.rs` — Permute kernel roundtrip tests across all dtypes and layouts. Gated on `testing-cuda` + `permute_kernels`.
- Inline tests in `src/tensor_kernels.rs` — Integration tests including `universal_roundtrip`. Gated on `testing-cuda` + `permute_kernels`.
