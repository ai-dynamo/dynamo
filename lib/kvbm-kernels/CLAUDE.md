# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

kvbm-kernels is a CUDA kernel library for GPU-accelerated KV cache layout conversions between three formats used by LLM inference frameworks: **Block Stack** (vLLM), **Operational** (TensorRT-LLM), and **Universal** (Dynamo storage). Rust FFI wrappers expose the kernels to the Dynamo ecosystem.

## Build Commands

```bash
# Default build (auto-detects nvcc → source build; no nvcc → stubs)
cargo build

# Build from source with custom GPU architectures
CUDA_ARCHS="80,86,89,90,100" cargo build

# Static linking (embed kernels into binary instead of .so)
cargo build --features static-kernels

# Check compilation without linking
cargo check

# Run CUDA integration tests (requires real GPU + nvcc)
cargo test --features testing-cuda,permute_kernels

# Run a specific test
cargo test --features testing-cuda,permute_kernels fused_copy_roundtrip -- --nocapture --test-threads=1
```

**Environment variables**: `CUDA_ARCHS` (comma-separated SM versions), `KVBM_REQUIRE_CUDA` (fail if nvcc missing), `CUDA_PATH`/`CUDA_HOME`.

## Architecture

### Two-tier build system (`build.rs`)

The build script selects one of two modes: **FromSource** (nvcc available → compile CUDA) or **Stubs** (no CUDA at all → C stubs that abort on call). Stubs set the `stub_kernels` cfg flag so tests can be conditionally skipped.

### Kernel organization

- `cuda/tensor_kernels.cu` — All permutation kernels (block↔universal, block↔operational). C++ templates on dtype (`F16`/`BF16`/`F32`/`F64`) and layout (`NHD`/`HND`), exposed via `extern "C"` dispatch functions prefixed `kvbm_kernels_launch_*`.
- `cuda/vectorized_copy.cu` — Adaptive vectorized memory copy (4/8/16-byte units).
- `cuda/stubs.c` — Abort-on-call fallbacks for all `extern "C"` symbols.

### Rust FFI layer (`src/tensor_kernels.rs`)

- Always-available: `vectorized_copy`, `memcpy_batch`, `is_using_stubs`, `is_memcpy_batch_available`
- Feature-gated (`permute_kernels`): `universal_from_block`, `block_from_universal`, `operational_copy`
- Integration tests live at the bottom of this file, gated by `#[cfg(all(test, feature = "testing-cuda", feature = "permute_kernels", not(stub_kernels)))]`

### Dimension conventions

`nl` = layers, `no` = outer chunks (2: K and V), `nh` = attention heads, `nt` = tokens per block, `hd` = head dimension.

### Cargo features

| Feature | Purpose |
|---------|---------|
| `permute_kernels` | Enable layout permutation kernels (block↔universal↔operational) |
| `testing-cuda` | Enable CUDA integration tests |
| `static-kernels` | Link as `.a` instead of `.so` |
