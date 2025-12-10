## Dynamo KV Block Manager Kernels

### Overview

In LLM inference, we cache attention keys and values (KV cache) to avoid
recomputing them for each token. Different serving frameworks store this cache
in different memory layouts, and converting between them on CPU is prohibitively
slow for production workloads.

This workspace provides high-performance CUDA kernels that convert KV cache data
between three layouts **directly on the GPU**, enabling efficient interoperability
between vLLM, TensorRT-LLM, and Dynamo's storage format.

**The Problem**: vLLM generates cache as 64 separate GPU allocations (32 layers × 2 for K/V),
TensorRT-LLM wants one flat buffer, and Dynamo's storage needs a format that works
across different tensor parallelism configurations. Copying to CPU, rearranging in
Python, and copying back would add seconds of latency per request.

**The Solution**: These CUDA kernels perform all conversions directly in GPU memory,
typically completing in microseconds instead of seconds.

---

### Dimension Reference

Before diving into layouts, here's what each dimension represents:

- **nl** = number of layers (e.g., 32 for Llama-70B)
- **no** = number of outer chunks (typically 2: one for keys, one for values)
- **nh** = number of attention heads (e.g., 32 or 64 heads)
- **nt** = number of tokens per block (e.g., 128 or 256 tokens)
- **hd** = head dimension (e.g., 128 for most models)

For a 32-layer model with 32 heads and 128-element head dimension, a single
128-token block contains: **32 layers × 2 (K/V) × 32 heads × 128 tokens × 128 dims
= 33.5 million float16 values = 67 MB**.

---

### The Three Layouts

This workspace houses CUDA + Rust + Python tooling for shuttling attention
blocks between three commonly used layouts:

#### 1. Stacked NHD / HND blocks (vLLM's format)

**Structure**: `nl * no` separate GPU memory allocations per block
**Each allocation**: `[nt, nh, hd]` (NHD) or `[nh, nt, hd]` (HND)

**Example** (32 layers, 128 tokens, 32 heads, 128 head_dim):
```text
64 separate pointers:
  Layer 0, Keys   → GPU buffer at ptr[0]:  [128, 32, 128]
  Layer 0, Values → GPU buffer at ptr[1]:  [128, 32, 128]
  Layer 1, Keys   → GPU buffer at ptr[2]:  [128, 32, 128]
  ...
  Layer 31, Values → GPU buffer at ptr[63]: [128, 32, 128]
```

**Why?** vLLM processes attention layer-by-layer and benefits from having
each layer's K/V in separate allocations for efficient memory management.

#### 2. Operational blocks (TensorRT-LLM's format)

**Structure**: Single contiguous GPU buffer per block
**Shape**: `[nl, no, inner]` where `inner = nt * nh * hd`

**Example** (same 32-layer model):
```text
1 contiguous buffer: [32, 2, 524288]
  where 524288 = 128 tokens × 32 heads × 128 head_dim

All data packed: [L0_K | L0_V | L1_K | L1_V | ... | L31_V]
```

**Why?** TensorRT-LLM treats blocks as opaque blobs and doesn't need the
internal structure exposed. Single contiguous buffers enable efficient bulk
memory operations and DMA transfers. Also used by Dynamo when no TP
resharding is needed.

#### 3. Universal blocks (Dynamo's storage format)

**Structure**: Single contiguous GPU buffer per block
**Shape**: `[nh, nl, no, nt, hd]` (heads in the outermost dimension)

**Example** (same 32-layer model):
```text
1 contiguous buffer: [32, 32, 2, 128, 128]

Data organized with heads first:
  [Head_0 data | Head_1 data | ... | Head_31 data]
```

**Why?** The head dimension is outermost to support **tensor parallelism (TP)
resharding**. With 32 heads:
- TP=4 setup: slice `[0:8, ...]`, `[8:16, ...]`, `[16:24, ...]`, `[24:32, ...]`
- TP=8 setup: slice `[0:4, ...]`, `[4:8, ...]`, etc.

Cache saved from a TP=4 run can be loaded into a TP=8 deployment (or vice versa)
by simply slicing the head dimension differently. This enables **efficient KV
cache transfer between deployments with different parallelism configurations**.

All kernels are batch aware: a single launch can process `nb` blocks by
walking flattened pointer tables that the host code prepares ahead of time.
Bindings are provided for both Rust and PyTorch so you can slot the kernels
into existing pipelines without living in CUDA all day.

---

### Example: Cross-Framework KV Cache Transfer

Here's how these kernels enable efficient KV cache sharing between frameworks:

```
1. vLLM generates KV cache during inference
   Format: Stacked NHD (64 pointers per block)

2. Save to Dynamo storage:
   kernel: block_to_universal(NHD → Universal)
   Result: Single buffer per block, head dimension first

3. Later, load into TensorRT-LLM (different TP config):
   kernel: universal_to_operational(Universal → Operational)
   Result: Single flat buffer ready for TRT-LLM

Alternative: Direct vLLM → TRT-LLM conversion
   kernel: block_to_operational(NHD → Operational)
   Bypasses storage format entirely
```

**Performance**: These conversions typically complete in **microseconds** on modern
GPUs (A100/H100), compared to **seconds** for CPU-based approaches. For a 70B model
with 128 blocks, that's the difference between 50μs and 5s of added latency per request.

---

### Layout Cheat Sheet

| Term                | Logical Shape              | Stored As                          | Notes                         |
|---------------------|----------------------------|------------------------------------|-------------------------------|
| NHD block stack     | `[nl][no][nt, nh, hd]`     | list of `nl * no` pointers         | Inner layout = NHD            |
| HND block stack     | `[nl][no][nh, nt, hd]`     | list of `nl * no` pointers         | Inner layout = HND            |
| Operational block   | `[nl, no, inner]`          | contiguous buffer per block        | `inner = nt * nh * hd`        |
| Universal block     | `[nh, nl, no, nt, hd]`     | contiguous buffer per block        | Ideal when all dims are fixed |

> **Pointer prep**
> For each logical block you provide:
> - one universal pointer,
> - `nl * no` pointers for either NHD or HND chunks, and
> - one operational pointer (when needed).

---

### Repository Structure

```text
.
├── Cargo.toml              # Rust lib/bin targets
├── build.rs                # NVCC build script (sm80+sm90 by default)
├── cuda/
│   ├── tensor_kernels.cu   # Batched CUDA kernels + memcpy fallback
│   └── prebuilt/           # Prebuilt .fatbin files with MD5 checksums
├── src/
│   ├── lib.rs              # Rust facade for the kernels
│   └── tensor_kernels.rs   # FFI wrappers + integration tests
```

> **Note:** Python bindings (`python.rs`) and tests have been moved to
> `lib/bindings/kvbm/` as part of the integrated `kvbm` wheel.

---

### Development Environment

The recommended way to develop the CUDA kernels is using the **Dynamo dev container**,
which includes all necessary CUDA development tools (`nvcc`, `nvlink`, headers, etc.).

#### Using the Dev Container

```bash
# From the repository root
# Open in VS Code with Dev Container extension, or:
cd .devcontainer/vllm
# Follow instructions in .devcontainer/README.md
```

The dev container includes:
- ✅ CUDA 12.9 toolkit with full development tools
- ✅ Rust toolchain
- ✅ Python + PyTorch + vLLM
- ✅ All necessary build dependencies

#### Building the CUDA Library

From within the dev container (or with local CUDA toolkit installed):

```bash
# Navigate to kernels directory
cd lib/kvbm-kernels

# Default build (sm_80, sm_90) - uses prebuilt kernels if nvcc not found
cargo build

# Build from source with broader GPU compatibility
CUDA_ARCHS="80,86,89,90,100" cargo build

# Common architectures:
# 80  = Ampere (A100)
# 86  = Ampere (RTX 30xx)
# 89  = Ada Lovelace (RTX 40xx, L4, L40)
# 90  = Hopper (H100, H200)
# 100 = Blackwell (B100, B200, GB200)
```

#### Running Tests

```bash
# Quick syntax check
cargo check

# Run integration tests
cargo test

# Run specific test with output
cargo test fused_copy_roundtrip -- --nocapture
```

The unit test synthesizes two blocks on-device, exercises every conversion
path (block ⇄ universal ⇄ operational), and asserts lossless round-trips.

#### Prebuilt Kernels

By default, the build system uses prebuilt `.fatbin` files from `cuda/prebuilt/`
if `nvcc` is not available. To force building from source:

```bash
# Disable prebuilt kernels
export DYNAMO_USE_PREBUILT_KERNELS=false
cargo build
```

After modifying CUDA source, regenerate prebuilt kernels and update checksums:

```bash
# This rebuilds tensor_kernels.cu and updates MD5 hashes
cargo build --release
# Commit the updated cuda/prebuilt/tensor_kernels.{fatbin,md5}
```

**Important:** If you change `CUDA_ARCHS` or update your nvcc version, you need to
force regeneration by deleting the checksums:

```bash
# Force regeneration after changing CUDA_ARCHS or nvcc version
rm cuda/prebuilt/*.md5
cargo build --release
# Commit the updated files
```

The build system only checks if the `.cu` source has changed, not build configuration.
This prevents CI from regenerating non-reproducible `.a` files unnecessarily.

---

### Python Bindings & Tests

> **Note:** The Python bindings and tests have been migrated to the `kvbm` wheel
> at `lib/bindings/kvbm/`. Install and test using that package instead.

#### Install locally

```bash
cd lib/bindings/kvbm
uv pip install -e ".[dev]"
```

This installs the `kvbm` package with all development dependencies including
the CUDA tensor kernels, pytest, and build tools.

#### Validate against PyTorch baselines

```bash
cd lib/bindings/kvbm
pytest tests/
```

Each test synthesizes random CUDA tensors, permutes them using native PyTorch
ops, then compares the kernel output with tolerances tuned per dtype.

#### Python API Sketch

```python
import torch
from kvbm import kernels

blocks = [...]         # list[list[torch.Tensor]] sized nb x (nl*no)
universals = [...]     # list[torch.Tensor] sized nb
operationals = [...]   # list[torch.Tensor] sized nb

kernels.block_to_universal(blocks, universals, layout="NHD")
kernels.universal_to_block(universals, blocks, layout="NHD")

kernels.block_to_operational(blocks, operationals, backend="batch")  # or "async" / "kernel" / "auto"
kernels.operational_to_block(operationals, blocks, backend="auto")
```

All tensors must be CUDA accessible by the specified device and match the expected
shapes and be contiguous in those shapes. The bindings validate shapes/dtypes, stage
pointer tables on-device, and launch the appropriate CUDA kernel.

---

---

### Troubleshooting

| Symptom                               | Likely Cause / Fix                                                 |
|---------------------------------------|--------------------------------------------------------------------|
| `cudaErrorInvalidValue` on launch     | Pointer counts mismatch (`nb`, `nl`, `no`) or non-contiguous input |
| Wrong values when using HND layout    | Inner tensors not permuted to `[nh, nt, hd]` before passing in     |
| Python bindings complain about dtype  | Mixed precision in a batch; convert tensors to a common dtype      |
| Kernels take unexpected time          | Verify that `CUDA_ARCHS` matches your GPU to avoid JIT at runtime  |

- `backend="auto"` defaults to the fused kernel, then `cudaMemcpyBatchAsync`, then `cudaMemcpyAsync`. Override if you want to benchmark a specific path.
