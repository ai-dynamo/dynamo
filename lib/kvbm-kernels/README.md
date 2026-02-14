## Dynamo KV Block Manager Kernels

GPU kernels for converting KV cache blocks between three memory layouts used by LLM inference frameworks. All conversions run entirely on-device via fused CUDA kernels.

### Dimensions

| Symbol | Meaning                        | Example          |
|--------|--------------------------------|------------------|
| `nl`   | Number of layers               | 32 (Llama-70B)   |
| `no`   | Outer chunks (K and V)         | 2                |
| `nh`   | Number of attention heads      | 32 or 64         |
| `nt`   | Tokens per block               | 128 or 256       |
| `hd`   | Head dimension                 | 128              |

### Layouts

#### Block Stack (NHD or HND)

`nl * no` separate GPU allocations per block. Each allocation holds one layer's keys or values.

- **NHD shape**: `[nt, nh, hd]` — index: `(nt_idx * nh + nh_idx) * hd + hd_idx`
- **HND shape**: `[nh, nt, hd]` — index: `(nh_idx * nt + nt_idx) * hd + hd_idx`

Passed to kernels as a flat pointer table of length `nb * nl * no`.

#### Operational

Single contiguous buffer per block: `[nl, no, inner]` where `inner = nt * nh * hd`.

The three innermost dimensions (`nt`, `nh`, `hd`) are fused into one `inner` dimension. When no layout permutation is needed (same TP config, same head layout), block-to-operational is a flat copy — the cheapest conversion. Transforming to/from other layouts requires knowing the constituent dimensions.

#### Universal

Single contiguous buffer per block: `[nh, nl, no, nt, hd]`.

Heads are the outermost dimension so that tensor-parallelism resharding is a contiguous slice along `nh`. A block saved from a TP=4 deployment can be loaded into TP=8 by slicing the head dimension differently.

### Layout Cheat Sheet

| Layout              | Logical Shape              | Stored As                          | Notes                         |
|---------------------|----------------------------|------------------------------------|-------------------------------|
| NHD block stack     | `[nl][no][nt, nh, hd]`     | list of `nl * no` pointers         | Inner layout = NHD            |
| HND block stack     | `[nl][no][nh, nt, hd]`     | list of `nl * no` pointers         | Inner layout = HND            |
| Operational block   | `[nl, no, inner]`          | contiguous buffer per block        | `inner = nt * nh * hd`        |
| Universal block     | `[nh, nl, no, nt, hd]`     | contiguous buffer per block        | Heads outermost for TP slicing |

### Kernel Functions

All kernels are batched: a single launch processes `nb` blocks from flat pointer tables prepared by host code.

#### Layout permutation kernels

| C API                                        | Conversion                  |
|----------------------------------------------|-----------------------------|
| `kvbm_kernels_launch_universal_from_block`   | Block stack → Universal     |
| `kvbm_kernels_launch_block_from_universal`   | Universal → Block stack     |

Both accept `layout_value` (NHD=0, HND=1) and `dtype_value` (F16=0, BF16=1, F32=2, F64=3). Internally dispatched to C++ template kernels specialized on dtype and layout.

#### Operational copy

| C API                                        | Conversion                              |
|----------------------------------------------|-----------------------------------------|
| `kvbm_kernels_launch_operational_copy`       | Block stack ↔ Operational (either direction) |

Since block-to-operational is a flat copy (no permutation), multiple backends are available:

| Backend              | Value | Description                                                  |
|----------------------|-------|--------------------------------------------------------------|
| `Auto`               | 0     | Tries vectorized → memcpy batch → memcpy async               |
| `VectorizedKernel`   | 1     | Custom kernel using `int64_t` loads; requires 8-byte alignment |
| `KernelOnly`         | 2     | Dtype-specific CUDA kernel (256 threads/block)               |
| `MemcpyAsync`        | 3     | Per-chunk `cudaMemcpyAsync` loop                             |
| `MemcpyBatch`        | 4     | `cudaMemcpyBatchAsync` (CUDA 12.9+)                         |

Direction is controlled by `direction_value` (BlockToOperational=0, OperationalToBlock=1).

#### Standalone copy utilities

| C API                                    | Description                                              |
|------------------------------------------|----------------------------------------------------------|
| `kvbm_kernels_launch_vectorized_copy`    | Adaptive vectorized copy (16/8/4-byte or scalar) across `num_pairs` pointer pairs |
| `kvbm_kernels_memcpy_batch`              | Batched `cudaMemcpyAsync` from host pointer arrays       |
| `kvbm_kernels_has_memcpy_batch_async`    | Returns `true` if `cudaMemcpyBatchAsync` is available    |
| `kvbm_kernels_is_stub_build`             | Returns `true` if built without CUDA (stub mode)         |

### Repository Structure

```text
.
├── Cargo.toml              # Rust lib/bin targets
├── build.rs                # NVCC build script (sm80+sm90 by default)
├── cuda/
│   ├── tensor_kernels.cu   # Batched CUDA kernels + memcpy fallback
│   ├── vectorized_copy.cu  # Adaptive vectorized memory copy
│   └── stubs.c             # Abort-on-call fallbacks when CUDA unavailable
├── src/
│   ├── lib.rs              # Rust facade for the kernels
│   └── tensor_kernels.rs   # FFI wrappers + integration tests
```

### Python Bindings

Python bindings live in `lib/bindings/kvbm/` and are built as the `kvbm` wheel via maturin.

```python
import torch
from kvbm import kernels

blocks = [...]         # list[list[torch.Tensor]] — nb x (nl*no)
universals = [...]     # list[torch.Tensor] — nb
operationals = [...]   # list[torch.Tensor] — nb

kernels.block_to_universal(blocks, universals, layout="NHD")
kernels.universal_to_block(universals, blocks, layout="NHD")

kernels.block_to_operational(blocks, operationals, backend="auto")
kernels.operational_to_block(operationals, blocks, backend="auto")
```

All tensors must be CUDA-resident and contiguous. The bindings validate shapes and dtypes, stage pointer tables on-device, and launch the appropriate kernel.

### Development

```bash
# Default build (auto-detects nvcc → source; no nvcc → stubs)
cargo build

# Custom GPU architectures
CUDA_ARCHS="80,86,89,90,100" cargo build

# Static linking
cargo build --features static-kernels

# Run CUDA integration tests (requires GPU + nvcc)
cargo test --features testing-cuda,permute_kernels

# Specific test with output
cargo test --features testing-cuda,permute_kernels fused_copy_roundtrip -- --nocapture

# Python bindings
cd lib/bindings/kvbm
uv pip install -e ".[dev]"
pytest tests/
```

**Environment variables**: `CUDA_ARCHS` (comma-separated SM versions, default `80,86,89,90,100,120`), `CUDA_PATH`/`CUDA_HOME` (toolkit root), `KVBM_REQUIRE_CUDA` (fail build if nvcc missing).

### Troubleshooting

| Symptom                               | Likely Cause / Fix                                                 |
|---------------------------------------|--------------------------------------------------------------------|
| `cudaErrorInvalidValue` on launch     | Pointer counts mismatch (`nb`, `nl`, `no`) or non-contiguous input |
| Wrong values when using HND layout    | Inner tensors not shaped as `[nh, nt, hd]` before passing in       |
| Python bindings complain about dtype  | Mixed precision in a batch; convert tensors to a common dtype      |
| Kernels take unexpected time          | Verify that `CUDA_ARCHS` matches your GPU to avoid JIT at runtime  |
