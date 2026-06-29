# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-side CRC32 over a device pointer (slice-by-16).

Used by the GDS-direct demote path to compute the integrity stamp
WITHOUT a D2H copy of the payload. Engine's HBM bytes go to
storage via GPUDirect Storage; CRC is computed in situ via this
kernel.

Implementation:
  - Parallel chunked CRC32 kernel compiled via NVRTC on first use.
    Each thread processes a contiguous chunk using SLICE-BY-16:
    16 bytes consumed per inner-loop iteration via 16 precomputed
    256-entry tables (T0..T15) in __constant__ memory. The
    last 1-15 bytes of each chunk fall through to a byte-by-byte
    tail loop.
  - Per-chunk CRCs land in a device buffer; we D2H the small output
    (one u32 per thread) and combine via the Rust-side
    `crc32_combine_chunks` helper (GF(2) matrix math, zlib-compat).
  - Up to 256 threads × ⌈N/256⌉ bytes per thread by default. The
    chunk_bytes is rounded up to 16 so per-thread `start = tid *
    chunk_bytes` is 16-byte aligned (assuming the data pointer
    itself is 16-byte aligned, which holds for torch CUDA tensors
    and cuMemAlloc'd ranges). Aligned u32 loads compile to one
    LDG.32 per word; without alignment the compiler emits the
    byte-by-byte fallback that's many times slower.

Tables baked into the source by `_build_slice_tables(16)`. Total
constant memory used: 16 × 256 × 4 = 16 KB.

PERF NOTE (honest):
  Slice-by-16 gives ~2x over the original byte-by-byte kernel,
  but on this SM86 GPU still measures slower than the D2H+CPU CRC
  path for typical block sizes (1.8 MB: ~3 ms GPU vs ~660 µs
  D2H+zlib). The table-lookup parallel-CRC algorithm is bottle-
  necked by per-thread serial dependencies on the CRC state and
  constant-memory bandwidth — adding threads helps occupancy but
  pushes host-side combine cost up at the same rate, capping the
  achievable speedup. To actually beat D2H+CPU, the next-level
  optimization is PCLMUL-style polynomial folding (sm_75+ inline
  PTX `mad.s32.lo.lo` or similar), which is a different
  algorithm. GDS restore nevertheless uses this kernel because a
  host round-trip defeats the GPU-direct data path.

If NVRTC compilation, module load, or kernel launch fails,
`is_available()` returns False. GPU-direct callers fail closed rather
than copying the payload through host memory."""

from __future__ import annotations

import logging
import threading
from importlib.util import find_spec
from typing import Optional

logger = logging.getLogger(__name__)


# ----- CRC32 lookup table (zlib polynomial, reversed) -----


def _build_slice_tables(n_slices: int = 16) -> list[list[int]]:
    """Build slice-by-N CRC32 tables. T[k][b] = state after
    processing byte b followed by k zero bytes (from initial state
    0). T[0] is the standard zlib CRC32 byte table; T[1]..T[N-1]
    are derived by advancing T[k-1][b] through 8 zero bits.

    Reversed IEEE polynomial 0xEDB88320 (zlib compatible)."""
    poly = 0xEDB88320
    T = [[0] * 256 for _ in range(n_slices)]
    for b in range(256):
        c = b
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        T[0][b] = c & 0xFFFFFFFF
    for k in range(1, n_slices):
        for b in range(256):
            prev = T[k - 1][b]
            T[k][b] = ((prev >> 8) ^ T[0][prev & 0xFF]) & 0xFFFFFFFF
    return T


_SLICE_TABLES = _build_slice_tables(16)


# ----- CUDA C source (compiled by NVRTC on first use) -----


def _kernel_source() -> str:
    """Generate CUDA C with the 16 slice-by-N tables baked as a
    flat __constant__ array. The kernel processes 16 bytes per
    iteration via 16 table lookups, then falls through to byte-by-
    byte for the 0-15 byte tail. NVRTC compiles this once per
    process — the source is ~80 KB of hex literals but compile is
    fast (one shot)."""
    flat = ", ".join(f"0x{v:08x}u" for table in _SLICE_TABLES for v in table)
    # T is laid out as T[16][256] in C — flat array of 4096 u32s,
    # row-major so T[k][b] = T[k*256 + b].
    return f"""
extern "C" {{

__constant__ unsigned int T[16 * 256] = {{ {flat} }};

#define TBL(k, b) T[(k) * 256 + (b)]

__global__ void crc32_chunks_kernel(
    const unsigned char* __restrict__ data,
    unsigned long long n_bytes,
    unsigned long long chunk_bytes,
    unsigned int* __restrict__ out_crcs)
{{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    unsigned long long start = tid * chunk_bytes;
    if (start >= n_bytes) {{
        // Threads with no work leave out_crcs[tid] at the
        // zero-init value. The host-side combine loop uses
        // remaining-bytes tracking and ignores them.
        return;
    }}
    unsigned long long end = start + chunk_bytes;
    if (end > n_bytes) end = n_bytes;

    unsigned int crc = 0xFFFFFFFFu;
    unsigned long long i = start;

    // Slice-by-16 main loop: consume 16 bytes per iteration.
    // The host-side wrapper rounds chunk_bytes up to a multiple of
    // 16 and torch / cuMemAlloc base pointers are 256-byte aligned,
    // so `data + i` is 16-byte aligned for every iteration. Direct
    // u32 reinterpret-cast reads compile to vectorized aligned
    // loads (one LDG.32 per word).
    while (i + 16 <= end) {{
        const unsigned int* p = (const unsigned int*)(data + i);
        unsigned int w0 = p[0];
        unsigned int w1 = p[1];
        unsigned int w2 = p[2];
        unsigned int w3 = p[3];
        unsigned int v  = crc ^ w0;
        crc = TBL(15,  v        & 0xFFu)
            ^ TBL(14, (v  >> 8) & 0xFFu)
            ^ TBL(13, (v  >> 16) & 0xFFu)
            ^ TBL(12, (v  >> 24))
            ^ TBL(11,  w1       & 0xFFu)
            ^ TBL(10, (w1 >> 8) & 0xFFu)
            ^ TBL( 9, (w1 >> 16) & 0xFFu)
            ^ TBL( 8, (w1 >> 24))
            ^ TBL( 7,  w2       & 0xFFu)
            ^ TBL( 6, (w2 >> 8) & 0xFFu)
            ^ TBL( 5, (w2 >> 16) & 0xFFu)
            ^ TBL( 4, (w2 >> 24))
            ^ TBL( 3,  w3       & 0xFFu)
            ^ TBL( 2, (w3 >> 8) & 0xFFu)
            ^ TBL( 1, (w3 >> 16) & 0xFFu)
            ^ TBL( 0, (w3 >> 24));
        i += 16;
    }}

    // Tail: 0-15 leftover bytes byte-by-byte using T[0].
    while (i < end) {{
        crc = (crc >> 8) ^ TBL(0, (crc ^ (unsigned int)data[i]) & 0xFFu);
        i += 1;
    }}

    out_crcs[tid] = crc ^ 0xFFFFFFFFu;
}}

}}  // extern "C"
"""


# ----- Compiled module cache (one per process) -----

_lock = threading.Lock()
_kernel: Optional[object] = None  # cuda.bindings.driver.CUfunction
_compile_attempted = False
_compile_error: Optional[str] = None


_TLS = threading.local()


def _ensure_cuda_context(device_ordinal: int = 0) -> None:
    """Retain + push the primary CUDA context on this thread, once.
    Mirrors the daemon's helper; we duplicate here to avoid a hot
    import from gms_kv_ring.daemon (we live in common/)."""
    if getattr(_TLS, "cuda_pushed", False):
        return
    from cuda.bindings import driver as drv

    err = drv.cuInit(0)[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")
    err, dev = drv.cuDeviceGet(device_ordinal)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")
    err, ctx = drv.cuDevicePrimaryCtxRetain(dev)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")
    err = drv.cuCtxPushCurrent(ctx)[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxPushCurrent failed: {err}")
    _TLS.cuda_pushed = True


def _nvrtc_arch_option(drv, nvrtc, device_ordinal: int = 0) -> bytes:
    """Choose the newest NVRTC target supported by both compiler and GPU.

    CUDA 13 no longer accepts ``compute_70``.  Conversely, an older NVRTC may
    not know a newer GPU's native target yet.  Querying both sides lets newer
    GPUs run compatible PTX while refusing to target an architecture newer
    than the device.
    """
    err, dev = drv.cuDeviceGet(device_ordinal)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed while selecting NVRTC arch: {err}")
    err, major = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        dev,
    )
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot read CUDA compute-capability major: {err}")
    err, minor = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        dev,
    )
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot read CUDA compute-capability minor: {err}")
    device_arch = int(major) * 10 + int(minor)

    target_arch = device_arch
    get_archs = getattr(nvrtc, "nvrtcGetSupportedArchs", None)
    if get_archs is not None:
        arch_err, supported = get_archs()
        if arch_err == nvrtc.nvrtcResult.NVRTC_SUCCESS and supported:
            compatible = [int(arch) for arch in supported if int(arch) <= device_arch]
            if not compatible:
                raise RuntimeError(
                    "NVRTC has no target compatible with device "
                    f"compute_{device_arch}; supported={list(supported)}"
                )
            target_arch = max(compatible)
    return f"--gpu-architecture=compute_{target_arch}".encode("ascii")


def _compile_once() -> bool:
    """Lazy NVRTC compile + module load. Returns True on success.
    Subsequent calls are O(1) — the kernel handle is cached."""
    global _kernel, _compile_attempted, _compile_error
    with _lock:
        if _compile_attempted:
            return _kernel is not None
        _compile_attempted = True
        if find_spec("gms_rust_ring") is None:
            _compile_error = "gms_rust_ring extension is not installed"
            logger.warning(
                "crc32_gpu: kernel unavailable (%s)",
                _compile_error,
            )
            return False
        try:
            _ensure_cuda_context()
            from cuda.bindings import driver as drv
            from cuda.bindings import nvrtc

            src = _kernel_source().encode("utf-8")
            err, prog = nvrtc.nvrtcCreateProgram(
                src,
                b"crc32_chunks.cu",
                0,
                [],
                [],
            )
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"nvrtcCreateProgram failed: {err}")
            opts = [_nvrtc_arch_option(drv, nvrtc)]
            err = nvrtc.nvrtcCompileProgram(
                prog,
                len(opts),
                opts,
            )[0]
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                # Read the log for the error.
                lsz_err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
                log = b"\x00" * log_size
                nvrtc.nvrtcGetProgramLog(prog, log)
                raise RuntimeError(f"nvrtcCompileProgram failed: {err} log={log!r}")
            err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"nvrtcGetPTXSize failed: {err}")
            ptx = b"\x00" * ptx_size
            err = nvrtc.nvrtcGetPTX(prog, ptx)[0]
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"nvrtcGetPTX failed: {err}")
            nvrtc.nvrtcDestroyProgram(prog)

            err, module = drv.cuModuleLoadData(ptx)
            if err != drv.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuModuleLoadData failed: {err}")
            err, func = drv.cuModuleGetFunction(
                module,
                b"crc32_chunks_kernel",
            )
            if err != drv.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuModuleGetFunction failed: {err}")
            _kernel = func
            logger.info("crc32_gpu: NVRTC kernel compiled OK")
            return True
        except Exception as exc:  # noqa: BLE001
            _compile_error = str(exc)
            logger.warning(
                "crc32_gpu: kernel unavailable (%s)",
                exc,
            )
            return False


def is_available() -> bool:
    """True iff the GPU CRC32 kernel can be used. Lazy-compiles on
    first call. Subsequent calls are O(1)."""
    return _compile_once()


def crc32_gpu(device_ptr: int, n_bytes: int) -> int:
    """Compute CRC32 (IEEE polynomial, zlib-compatible) over
    `n_bytes` starting at the GPU `device_ptr`. The pointer must be
    a valid CUdeviceptr (e.g., from `cuMemAlloc` or a torch CUDA
    tensor's data_ptr()).

    Returns the 32-bit CRC, matching `zlib.crc32(host_bytes)`.

    Raises if the kernel is unavailable or the launch fails. GDS
    callers must fail the restore rather than route payload bytes
    through host memory."""
    if not is_available():
        raise RuntimeError(f"crc32_gpu kernel not available: {_compile_error}")
    if n_bytes == 0:
        return 0
    from cuda.bindings import driver as drv
    from gms_rust_ring import crc32_combine_chunks

    # Choose parallelism. We want enough threads to hide latency,
    # but not so many that the host-side combine dominates. 64
    # threads × ~28KB chunks for a 1.8MB block, kernel ~50 µs.
    # For very small payloads we use fewer threads.
    # Parallelism: target enough threads to keep an SM busy. Above
    # ~256 the host-side combine cost dominates (each combine is
    # O(log chunk) GF(2) matrix-squarings in Rust, ~1 µs).
    num_threads = min(
        max(1, (n_bytes + 1023) // 1024),  # ≥1 thread per 1 KB
        256,
    )
    chunk_bytes = (n_bytes + num_threads - 1) // num_threads
    # Round chunk_bytes UP to 16 so each thread's `start = tid *
    # chunk_bytes` is 16-byte aligned (assuming the data ptr is
    # itself ≥16-byte aligned, which holds for torch CUDA tensors
    # and cuMemAlloc'd ranges). Aligned loads avoid the
    # byte-by-byte fallback the compiler emits for
    # potentially-unaligned addresses.
    chunk_bytes = (chunk_bytes + 15) & ~15
    if chunk_bytes == 0:
        chunk_bytes = 16

    # Output buffer: one u32 per thread.
    err, d_out = drv.cuMemAlloc(num_threads * 4)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAlloc({num_threads*4}) failed: {err}")
    try:
        # Zero the output buffer so unused thread slots stay 0
        # (they won't write anything; combine ignores them via
        # remaining-bytes tracking, but defensive zero anyway).
        drv.cuMemsetD8(d_out, 0, num_threads * 4)

        # Launch. CUDA kernel arg packing via ctypes.
        import ctypes

        ptr_arg = ctypes.c_void_p(int(device_ptr))
        n_arg = ctypes.c_ulonglong(int(n_bytes))
        chunk_arg = ctypes.c_ulonglong(int(chunk_bytes))
        out_arg = ctypes.c_void_p(int(d_out))
        args = (ctypes.c_void_p * 4)(
            ctypes.addressof(ptr_arg),
            ctypes.addressof(n_arg),
            ctypes.addressof(chunk_arg),
            ctypes.addressof(out_arg),
        )

        block_dim = min(num_threads, 256)
        grid_dim = (num_threads + block_dim - 1) // block_dim
        err = drv.cuLaunchKernel(
            _kernel,
            grid_dim,
            1,
            1,  # grid
            block_dim,
            1,
            1,  # block
            0,
            0,  # shared mem, stream (default)
            args,
            0,
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel failed: {err}")

        # D2H the per-chunk CRCs. cuMemcpy synchronizes implicitly.
        h_out = (ctypes.c_uint * num_threads)()
        err = drv.cuMemcpyDtoH(
            ctypes.addressof(h_out),
            d_out,
            num_threads * 4,
        )[0]
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyDtoH failed: {err}")
    finally:
        drv.cuMemFree(d_out)

    return crc32_combine_chunks(
        list(h_out),
        int(chunk_bytes),
        int(n_bytes),
    )
