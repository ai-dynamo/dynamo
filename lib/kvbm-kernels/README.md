## Dynamo KV Block Manager Kernels

GPU kernels for converting KV cache blocks between three memory layouts used by LLM inference frameworks. All conversions run entirely on-device via fused CUDA kernels.

### Dimensions

| Symbol | Meaning                        | Example          |
|--------|--------------------------------|------------------|
| `nb`   | Number of blocks in the batch  | 1–128            |
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

#### Standalone copy utilities

| C API                                    | Description                                              |
|------------------------------------------|----------------------------------------------------------|
| `kvbm_kernels_launch_vectorized_copy`    | Adaptive vectorized copy (16/8/4-byte or scalar) across `num_pairs` pointer pairs |
| `kvbm_kernels_memcpy_batch`              | Batched `cudaMemcpyAsync` from host pointer arrays       |
| `kvbm_kernels_has_memcpy_batch_async`    | Returns `true` if `cudaMemcpyBatchAsync` is available    |
| `kvbm_kernels_is_stub_build`             | Returns `true` if built without CUDA (stub mode)         |


### Python Bindings (Planned)

Python kernel bindings are not yet implemented. The `lib/bindings/kvbm/` crate currently exposes block manager functionality only. Future work will add Python wrappers for the permute and copy kernels.

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

### Benchmarking

```text
root@9eb240f7ded8:/workspace/lib/kvbm-kernels# cargo run --release --example kvbench --features testing-cuda,kvbench -- --num-blocks=1,128 --tokens-per-block=16,64 --
backend vectorized,batched --direction h2d
...
     Running `/workspace/target/release/examples/kvbench --num-blocks=1,128 --tokens-per-block=16,64 --backend vectorized,batched --direction h2d`
KV Cache Transfer Benchmark
  Model: Llama 3.1 70B (bf16)
  Layers: 80, KV heads: 8, Head dim: 128, Outer dim: 2
  Warmup: 10, Timed: 100
  Batch API available: true
  tokens_per_block: [16, 64]
  num_blocks: [1, 128]
  directions: [h2d]
  patterns: [fc_to_fc, lw_to_fc]
  backends: [vectorized, batched]
  Total tests: 16

tokens_per_block,num_blocks,pattern,direction,backend,total_bytes,inner_bytes,copy_size,num_copies,median_ms,bandwidth_gbps
--- tokens_per_block=16, inner=32768 bytes (32 KB), block=5242880 bytes (5.0 MB) ---
  [1/16] tpb=16 N=  1 fc_to_fc h2d    vectorized   ... 16,1,fc_to_fc,h2d,vectorized,5242880,32768,5242880,1,1.8686,2.81
2.81 GB/s (1.8686 ms)
  [2/16] tpb=16 N=  1 fc_to_fc h2d    batched      ... 16,1,fc_to_fc,h2d,batched,5242880,32768,5242880,1,0.2105,24.91
24.91 GB/s (0.2105 ms)
  [3/16] tpb=16 N=  1 lw_to_fc h2d    vectorized   ... 16,1,lw_to_fc,h2d,vectorized,5242880,32768,32768,160,0.2171,24.15
24.15 GB/s (0.2171 ms)
  [4/16] tpb=16 N=  1 lw_to_fc h2d    batched      ... 16,1,lw_to_fc,h2d,batched,5242880,32768,32768,160,0.2775,18.89
18.89 GB/s (0.2775 ms)
  [5/16] tpb=16 N=128 fc_to_fc h2d    vectorized   ... 16,128,fc_to_fc,h2d,vectorized,671088640,32768,5242880,128,26.6097,25.22
25.22 GB/s (26.6097 ms)
  [6/16] tpb=16 N=128 fc_to_fc h2d    batched      ... 16,128,fc_to_fc,h2d,batched,671088640,32768,5242880,128,26.6180,25.21
25.21 GB/s (26.6180 ms)
  [7/16] tpb=16 N=128 lw_to_fc h2d    vectorized   ... 16,128,lw_to_fc,h2d,vectorized,671088640,32768,32768,20480,26.6034,25.23
25.23 GB/s (26.6034 ms)
  [8/16] tpb=16 N=128 lw_to_fc h2d    batched      ... 16,128,lw_to_fc,h2d,batched,671088640,32768,32768,20480,30.3346,22.12
22.12 GB/s (30.3346 ms)
--- tokens_per_block=64, inner=131072 bytes (128 KB), block=20971520 bytes (20.0 MB) ---
  [9/16] tpb=64 N=  1 fc_to_fc h2d    vectorized   ... 64,1,fc_to_fc,h2d,vectorized,20971520,131072,20971520,1,7.5837,2.77
2.77 GB/s (7.5837 ms)
  [10/16] tpb=64 N=  1 fc_to_fc h2d    batched      ... 64,1,fc_to_fc,h2d,batched,20971520,131072,20971520,1,0.8334,25.16
25.16 GB/s (0.8334 ms)
  [11/16] tpb=64 N=  1 lw_to_fc h2d    vectorized   ... 64,1,lw_to_fc,h2d,vectorized,20971520,131072,131072,160,0.8407,24.95
24.95 GB/s (0.8407 ms)
  [12/16] tpb=64 N=  1 lw_to_fc h2d    batched      ... 64,1,lw_to_fc,h2d,batched,20971520,131072,131072,160,0.9020,23.25
23.25 GB/s (0.9020 ms)
  [13/16] tpb=64 N=128 fc_to_fc h2d    vectorized   ... 64,128,fc_to_fc,h2d,vectorized,2684354560,131072,20971520,128,106.3677,25.24
25.24 GB/s (106.3677 ms)
  [14/16] tpb=64 N=128 fc_to_fc h2d    batched      ... 64,128,fc_to_fc,h2d,batched,2684354560,131072,20971520,128,106.3199,25.25
25.25 GB/s (106.3199 ms)
  [15/16] tpb=64 N=128 lw_to_fc h2d    vectorized   ... 64,128,lw_to_fc,h2d,vectorized,2684354560,131072,131072,20480,106.3158,25.25
25.25 GB/s (106.3158 ms)
  [16/16] tpb=64 N=128 lw_to_fc h2d    batched      ... 64,128,lw_to_fc,h2d,batched,2684354560,131072,131072,20480,110.0665,24.39
24.39 GB/s (110.0665 ms)

Done.
```

### Troubleshooting

| Symptom                               | Likely Cause / Fix                                                 |
|---------------------------------------|--------------------------------------------------------------------|
| `cudaErrorInvalidValue` on launch     | Pointer counts mismatch (`nb`, `nl`, `no`) or non-contiguous input |
| Wrong values when using HND layout    | Inner tensors not shaped as `[nh, nt, hd]` before passing in       |
| Python bindings complain about dtype  | Mixed precision in a batch; convert tensors to a common dtype      |
| Kernels take unexpected time          | Verify that `CUDA_ARCHS` matches your GPU to avoid JIT at runtime  |
