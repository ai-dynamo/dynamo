# Vendored NVTX v3 headers

These are NVIDIA's header-only NVTX v3 C headers, copied unmodified from the
`nvtx-sys` crate (`nvtx` 1.3.0, `nvtx-sys/NVTX/c/include/nvtx3/`). Each file keeps its
original NVIDIA license header. Upstream: https://github.com/NVIDIA/NVTX

License: Apache-2.0 WITH LLVM-exception (https://llvm.org/LICENSE.txt).

They are compiled into `src/nvtx/shim.c` by `build.rs` when the `nvtx` feature is on.
