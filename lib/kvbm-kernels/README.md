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
├── Cargo.toml              # Rust lib/bin targets + PyO3 feature
├── build.rs                # NVCC build script (sm80+sm90 by default)
├── cuda/
│   └── tensor_kernels.cu   # Batched CUDA kernels + memcpy fallback
├── pyproject.toml          # maturin config for Python wheel builds
├── python/tests/           # pytest suites using PyTorch baselines
├── src/
│   ├── lib.rs              # Rust facade for the kernels
│   ├── main.rs             # Legacy cudaMemcpyBatchAsync demo (bin)
│   ├── python.rs           # PyO3 bindings
│   └── tensor_kernels.rs   # FFI wrappers + integration tests
└── run.sh / Dockerfile     # Optional CUDA 12.9 container harness
```

---

### Building the CUDA Library

The CUDA code is compiled via `nvcc` in `build.rs`. Supported architectures
default to `sm_80` and `sm_90`. Override with `CUDA_ARCHS` if required:

```bash
CUDA_ARCHS=80,90,102 cargo build --features python-bindings
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

We ship PyTorch-friendly shims using PyO3. The interface mirrors what most
attention kernels expect in production code.

#### Install locally

```bash
pip install maturin
maturin develop --features python-bindings
```

This builds the shared library in-place and installs the `cuda_tensor_kernels`
module into your active virtual environment.

#### Validate against PyTorch baselines

```bash
pytest python/tests --maxfail=1 -q
```

Each test synthesizes random CUDA tensors, permutes them using native PyTorch
ops, then compares the kernel output with tolerances tuned per dtype.

#### Python API Sketch

```python
import torch
import cuda_tensor_kernels as ctk

blocks = [...]         # list[list[torch.Tensor]] sized nb x (nl*no)
universals = [...]     # list[torch.Tensor] sized nb
operationals = [...]   # list[torch.Tensor] sized nb

ctk.block_to_universal(blocks, universals, layout="NHD")
ctk.universal_to_block(universals, blocks, layout="NHD")

ctk.block_to_operational(blocks, operationals, backend="batch")  # or "async" / "kernel" / "auto"
ctk.operational_to_block(operationals, blocks, backend="auto")
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
docker build -t kvbm-kernel
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
