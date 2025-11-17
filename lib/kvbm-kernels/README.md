## Dynamo KV Block Manager Kernels

This workspace houses CUDA + Rust + Python tooling for shuttling attention
blocks between three commonly used layouts:

1. **Stacked NHD / HND blocks** – `nl * no` tensors per block, each shaped
   `[nt, nh, hd]` (NHD) or `[nh, nt, hd]` (HND).
   - primarily used by vLLM
2. **Operational blocks** – flattened buffers shaped `[nl, no, inner]`,
   where `inner = nt * nh * hd`.
   - primarily used by TensorRT LLM
   - used by Dynamo's KVBM for non-device storage when no adjustments to
     the layout is need to translate to/from different TP world sizes
3. **Universal blocks** – contiguous buffers shaped `[nh, nl, no, nt, hd]`.
   - move the head dimension to the front
   - excellent format for storage blocks that can be used by different tp
     world sizes by scattering/gathering on slices of the leading dimension
     allowing for large contiguous transfers.

All kernels are batch aware: a single launch can process `nb` blocks by
walking flattened pointer tables that the host code prepares ahead of time.
Bindings are provided for both Rust and PyTorch so you can slot the kernels
into existing pipelines without living in CUDA all day.

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

```
.
├── Cargo.toml              # Rust lib/bin targets
├── build.rs                # NVCC build script (sm80+sm90 by default)
├── cuda/
│   └── tensor_kernels.cu   # Batched CUDA kernels + memcpy fallback
├── src/
│   ├── lib.rs              # Rust facade for the kernels
│   ├── main.rs             # Legacy cudaMemcpyBatchAsync demo (bin)
│   └── tensor_kernels.rs   # FFI wrappers + integration tests
└── run.sh / Dockerfile     # Optional CUDA 12.9 container harness
```

> **Note:** Python bindings (`python.rs`) and tests have been moved to
> `lib/bindings/kvbm/` as part of the integrated `kvbm` wheel.

---

### Building the CUDA Library

The CUDA code is compiled via `nvcc` in `build.rs`. Supported architectures
default to `sm_80` (Ampere) and `sm_90` (Hopper). Override with `CUDA_ARCHS`
for broader compatibility:

```bash
# Default build (sm_80, sm_90)
cargo build

# Broader compatibility across GPU generations
CUDA_ARCHS="80,86,89,90,100" cargo build

# Common architectures:
# 80  = Ampere (A100)
# 86  = Ampere (RTX 30xx)
# 89  = Ada Lovelace (RTX 40xx, L4, L40)
# 90  = Hopper (H100, H200)
# 100 = Blackwell (B100, B200, GB200)
```

> **Prerequisites**
> - CUDA 12.1+ toolkit on PATH
> - `nvcc` and compatible driver
> - Rust stable (1.70+) with `cargo`

For rapid iteration without the Python bindings:

```bash
cargo check
cargo test fused_copy_roundtrip -- --nocapture
```

The unit test synthesizes two blocks on-device, exercises every conversion
path (block ⇄ universal ⇄ operational), and asserts lossless round-trips.

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

All tensors must be CUDA accessible by the specificed device and match the expected
shapes and be contiguous in those shapes. The bindings validate shapes/dtypes, stage
pointer tables on-device, and launch the appropriate CUDA kernel.

---

### Docker Workflow (Optional)

Need a reproducible environment? The repo includes a CUDA 12.9 container that
installs Rust and builds the project.

```bash
# Build and run the demo binary inside the container
./run.sh

# Or build manually
# Or build manually
docker build -t kvbm-kernels .
docker run --rm --gpus all kvbm-kernels
```

To develop interactively with Python, extend the Dockerfile with your preferred
Python distribution and PyTorch wheel.

---

### Troubleshooting

| Symptom                               | Likely Cause / Fix                                                 |
|---------------------------------------|--------------------------------------------------------------------|
| `cudaErrorInvalidValue` on launch     | Pointer counts mismatch (`nb`, `nl`, `no`) or non-contiguous input |
| Wrong values when using HND layout    | Inner tensors not permuted to `[nh, nt, hd]` before passing in     |
| Python bindings complain about dtype  | Mixed precision in a batch; convert tensors to a common dtype      |
| Kernels take unexpected time          | Verify that `CUDA_ARCHS` matches your GPU to avoid JIT at runtime  |
- `backend="auto"` defaults to the fused kernel, then `cudaMemcpyBatchAsync`, then `cudaMemcpyAsync`. Override if you want to benchmark a specific path.
