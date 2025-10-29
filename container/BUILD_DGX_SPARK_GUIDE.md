# Building Dynamo for DGX-SPARK (vLLM)

## How `build.sh` Chooses the Dockerfile

The `build.sh` script automatically selects the correct Dockerfile based on the platform and optional flags:

### Dockerfile Selection Logic

```
IF framework == "VLLM":
    IF --dgx-spark flag is set OR platform is linux/arm64:
        Use: Dockerfile.vllm.dgx-spark  (NVIDIA's pre-built vLLM with Blackwell support)
    ELSE:
        Use: Dockerfile.vllm            (Build from source)
ELSE IF framework == "TRTLLM":
    Use: Dockerfile.trtllm
ELSE IF framework == "SGLANG":
    Use: Dockerfile.sglang
ELSE:
    Use: Dockerfile
```

### How to Use

#### For DGX-SPARK (Blackwell GPUs)

**Automatic detection (recommended):**
```bash
./container/build.sh --framework VLLM --platform linux/arm64
```

**Explicit flag:**
```bash
./container/build.sh --framework VLLM --dgx-spark
```

#### For x86_64 (standard GPUs)

```bash
./container/build.sh --framework VLLM
# or explicitly
./container/build.sh --framework VLLM --platform linux/amd64
```

## Key Differences

### Standard vLLM Dockerfile (`Dockerfile.vllm`)
- Builds vLLM from source
- Uses CUDA 12.8
- Supports: Ampere, Ada, Hopper GPUs
- **Does NOT support Blackwell (compute_121)**

### DGX-SPARK Dockerfile (`Dockerfile.vllm.dgx-spark`)
- Uses NVIDIA's pre-built vLLM container (`nvcr.io/nvidia/vllm:25.09-py3`)
- Uses CUDA 13.0
- Supports: **Blackwell GPUs (compute_121)** via DGX-SPARK
- Skips building vLLM from source (avoids nvcc errors)
- **Builds UCX v1.19.0 from source** with CUDA 13 support
- **Builds NIXL 0.7.0 from source** with CUDA 13 support (self-contained, no cache dependency)
- **Builds NIXL Python wheel** with CUDA 13 support
- Adds Dynamo's runtime customizations and integrations

## Why DGX-SPARK Needs Special Handling

DGX-SPARK systems use **Blackwell GPUs** with architecture `compute_121`. When trying to build vLLM from source with older CUDA toolchains:

```
ERROR: nvcc fatal : Unsupported gpu architecture 'compute_121a'
```

**Solution:** Use NVIDIA's pre-built vLLM container that already includes:
- CUDA 13.0 support
- Blackwell GPU architecture support
- DGX Spark functional support
- NVFP4 format optimization

### Why Build UCX and NIXL from Source?

The DGX-SPARK Dockerfile builds UCX v1.19.0 and NIXL 0.7.0 **from source** instead of copying from the base image:

**Reason 1: CUDA 13 Compatibility**
- NIXL 0.7.0 is the first version with native CUDA 13.0 support
- Building from source ensures proper linkage against `libcudart.so.13` (not `libcudart.so.12`)
- Avoids runtime errors: `libcudart.so.12: cannot open shared object file`

**Reason 2: Cache Independence**
- The base image (`dynamo_base`) may have cached NIXL 0.6.x built with CUDA 12
- Building fresh in the DGX-SPARK Dockerfile ensures we always get NIXL 0.7.0 with CUDA 13
- Self-contained build = predictable results

**Reason 3: ARM64 Optimization**
- UCX and NIXL are built specifically for `aarch64` architecture
- GDS backend is disabled (`-Ddisable_gds_backend=true`) as it's not supported on ARM64

## Build Arguments

When using the `--dgx-spark` flag, `build.sh` automatically:
- Selects `Dockerfile.vllm.dgx-spark`
- Sets `PLATFORM=linux/arm64` (forced)
- Sets `NIXL_REF=0.7.0` (for CUDA 13 support)
- Sets `ARCH=arm64` and `ARCH_ALT=aarch64`

The DGX-SPARK Dockerfile itself hardcodes:
- `BASE_IMAGE=nvcr.io/nvidia/vllm`
- `BASE_IMAGE_TAG=25.09-py3`

All other build arguments work the same way.

## Troubleshooting

### Error: `exec /bin/sh: exec format error`
- **Cause:** Building with wrong platform
- **Fix:** Use `--platform linux/arm64` for DGX-SPARK

### Error: `nvcc fatal : Unsupported gpu architecture 'compute_121a'`
- **Cause:** Building from source without Blackwell support
- **Fix:** Use `--dgx-spark` or `--platform linux/arm64` to use pre-built container

### Error: `libcudart.so.12: cannot open shared object file`
- **Cause:** NIXL was built with CUDA 12 but container has CUDA 13
- **Fix:** Rebuild with `--dgx-spark` flag to ensure NIXL 0.7.0 with CUDA 13 support
- **Verify:** Inside container: `ldd /opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu/plugins/libplugin_UCX_MO.so | grep cudart` should show `libcudart.so.13` (not `.so.12`)

## References

- [NVIDIA vLLM Release 25.09 Documentation](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-25-09.html)
- [NVIDIA NGC Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [NIXL 0.7.0 Release Notes](https://github.com/ai-dynamo/nixl/releases/tag/0.7.0) - CUDA 13.0 support
- [DGX-SPARK README](../docs/backends/vllm/DGX-SPARK_README.md) - Complete deployment guide

