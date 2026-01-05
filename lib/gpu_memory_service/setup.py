# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for GPU Memory Service with CUDA extensions.

This setup.py builds the C/CUDA extensions as part of pip install.
Extensions require PyTorch and CUDA to be available at build time.

Following the torch_memory_saver pattern of using pure setuptools for extension building.
"""

import logging
import os
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)


def _find_cuda_home():
    """Find CUDA installation path."""
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home is None:
        compiler_path = shutil.which("nvcc")
        if compiler_path is not None:
            home = os.path.dirname(os.path.dirname(compiler_path))
        else:
            home = "/usr/local/cuda"
    return home


def _get_torch_include_dirs():
    """Get PyTorch include directories for building extensions that use torch headers."""
    try:
        from torch.utils.cpp_extension import include_paths

        return include_paths()
    except ImportError:
        return []


def _get_torch_library_dirs():
    """Get PyTorch library directories."""
    try:
        import torch

        return [os.path.join(os.path.dirname(torch.__file__), "lib")]
    except ImportError:
        return []


class CUDAExtension(Extension):
    """Extension class for CUDA/C++ modules."""

    pass


class BuildExtension(build_ext):
    """Custom build extension that handles CUDA compilation."""

    def build_extensions(self):
        # Check if we have a CUDA-capable compiler
        cuda_home = _find_cuda_home()
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")

        if not os.path.exists(nvcc_path):
            # Check if extensions are actually requested
            if self.extensions:
                raise RuntimeError(
                    f"NVCC not found at {nvcc_path}. Cannot build CUDA extensions. "
                    f"Ensure CUDA is installed and CUDA_HOME or CUDA_PATH is set. "
                    f"If using pip wheel, use --no-build-isolation to inherit environment."
                )
            return

        # Check if PyTorch is available (needed for _tensor_from_pointer)
        try:
            import torch  # noqa: F401
        except ImportError:
            if self.extensions:
                raise RuntimeError(
                    "PyTorch not found. Cannot build CUDA extensions. "
                    "Install PyTorch first, or use --no-build-isolation with pip wheel."
                )
            return

        # Configure compiler
        self.compiler.set_executable("compiler_so", "g++")
        self.compiler.set_executable("compiler_cxx", "g++")
        self.compiler.set_executable("linker_so", "g++ -shared")

        build_ext.build_extensions(self)


def _create_ext_modules():
    """Create extension modules for gpu_memory_service."""
    cuda_home = _find_cuda_home()
    torch_include = _get_torch_include_dirs()
    torch_lib = _get_torch_library_dirs()

    # Common compile arguments
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC"]

    # CUDA include/library paths
    cuda_include = os.path.join(cuda_home, "include")
    cuda_lib_dirs = [
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "lib64", "stubs"),
    ]

    ext_modules = []

    # _allocator_ext: CUDAPluggableAllocator using CUDA driver API and Python C API
    ext_modules.append(
        CUDAExtension(
            name="gpu_memory_service.client.torch.extensions._allocator_ext",
            sources=["client/torch/extensions/allocator.cpp"],
            include_dirs=[cuda_include],
            library_dirs=cuda_lib_dirs,
            libraries=["cuda"],
            extra_compile_args=extra_compile_args,
        )
    )

    # _tensor_from_pointer: Uses PyTorch C++ API (requires torch headers)
    if torch_include:
        ext_modules.append(
            CUDAExtension(
                name="gpu_memory_service.client.torch.extensions._tensor_from_pointer",
                sources=["client/torch/extensions/tensor_from_pointer.cpp"],
                include_dirs=torch_include + [cuda_include],
                library_dirs=torch_lib + cuda_lib_dirs,
                libraries=["c10", "torch", "torch_python"],
                extra_compile_args=extra_compile_args,
                # Define for PyTorch extension compatibility
                define_macros=[("TORCH_EXTENSION_NAME", "_tensor_from_pointer")],
            )
        )
    else:
        logger.warning(
            "PyTorch not available. _tensor_from_pointer extension will not be built."
        )

    return ext_modules


setup(
    name="gpu-memory-service",
    version="0.8.0",
    description="GPU Memory Service for Dynamo - CUDA VMM-based GPU memory allocation and sharing",
    author="NVIDIA Inc.",
    author_email="sw-dl-dynamo@nvidia.com",
    license="Apache-2.0",
    python_requires=">=3.10",
    install_requires=[
        "msgpack>=1.0",
        "uvloop>=0.21.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.4",
            "pytest-asyncio",
        ],
    },
    # Package directory mapping: the current directory IS the gpu_memory_service package
    packages=[
        "gpu_memory_service",
        "gpu_memory_service.common",
        "gpu_memory_service.server",
        "gpu_memory_service.client",
        "gpu_memory_service.client.torch",
        "gpu_memory_service.client.torch.extensions",
    ],
    package_dir={
        "gpu_memory_service": ".",
        "gpu_memory_service.common": "common",
        "gpu_memory_service.server": "server",
        "gpu_memory_service.client": "client",
        "gpu_memory_service.client.torch": "client/torch",
        "gpu_memory_service.client.torch.extensions": "client/torch/extensions",
    },
    package_data={
        "gpu_memory_service.client.torch.extensions": ["*.cpp"],
    },
    ext_modules=_create_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)
