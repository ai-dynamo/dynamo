"""Build scripts for GPU Memory Service C/CUDA extensions.

This directory contains small native extensions that are imported directly from
the source tree (e.g. `gpu_memory_service.core.csrc._rpc_cumem_ext`).
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def get_cuda_arch_flags():
    """Get CUDA architecture flags for compilation."""
    # Default to common architectures if not specified
    return [
        "-gencode",
        "arch=compute_70,code=sm_70",  # V100
        "-gencode",
        "arch=compute_80,code=sm_80",  # A100
        "-gencode",
        "arch=compute_89,code=sm_89",  # L40, Ada
        "-gencode",
        "arch=compute_90,code=sm_90",  # H100
    ]


def get_cuda_library_dirs():
    """Get CUDA library directories including stubs for build-time linking."""
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    return [
        # Standard CUDA toolkit stubs location
        str(cuda_home / "lib64" / "stubs"),
        # NVIDIA container stubs location
        str(cuda_home / "targets" / "x86_64-linux" / "lib" / "stubs"),
        # Standard lib64 path
        str(cuda_home / "lib64"),
    ]


setup(
    name="gpu_memory_service_ext",
    ext_modules=[
        CUDAExtension(
            name="_rpc_cumem_ext",
            sources=["rpc_cumem.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-fPIC"],
                "nvcc": ["-O3"] + get_cuda_arch_flags(),
            },
            libraries=["cuda"],
            library_dirs=get_cuda_library_dirs(),
        ),
        # Used by the import-only loader to construct tensors that alias mapped
        # CUDA memory at known virtual addresses.
        CppExtension(
            name="_tensor_from_pointer",
            sources=["tensor_from_pointer.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-fPIC"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
