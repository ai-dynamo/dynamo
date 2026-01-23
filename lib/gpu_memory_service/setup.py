# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for GPU Memory Service with C++ extensions.

This setup.py builds the C++ extensions as part of pip install.
The _allocator_ext extension only requires Python headers (no CUDA or PyTorch needed).

Following the torch_memory_saver pattern of using pure setuptools for extension building.
"""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtension(build_ext):
    """Custom build extension for C++ modules."""

    def build_extensions(self):
        import os

        # Use CXX environment variable if set, otherwise default to g++
        cxx = os.environ.get("CXX", "g++")
        self.compiler.set_executable("compiler_so", cxx)
        self.compiler.set_executable("compiler_cxx", cxx)
        self.compiler.set_executable("linker_so", f"{cxx} -shared")

        build_ext.build_extensions(self)


def _create_ext_modules():
    """Create extension modules for gpu_memory_service."""
    # Common compile arguments
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC"]

    # _allocator_ext: CUDAPluggableAllocator shim using only Python C API
    # No CUDA or PyTorch dependency - just provides my_malloc/my_free that call Python callbacks
    return [
        Extension(
            name="gpu_memory_service.client.torch.extensions._allocator_ext",
            sources=["client/torch/extensions/allocator.cpp"],
            extra_compile_args=extra_compile_args,
        )
    ]


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
        "gpu_memory_service.common.protocol",
        "gpu_memory_service.server",
        "gpu_memory_service.client",
        "gpu_memory_service.client.torch",
        "gpu_memory_service.client.torch.extensions",
    ],
    package_dir={
        "gpu_memory_service": ".",
        "gpu_memory_service.common": "common",
        "gpu_memory_service.common.protocol": "common/protocol",
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
