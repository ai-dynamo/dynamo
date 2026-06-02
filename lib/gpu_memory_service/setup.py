# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build script for GPU Memory Service with C++ extensions.

This setup.py builds the C++ extensions as part of pip install.
The _allocator_ext extension only requires Python headers (no CUDA or PyTorch needed).

Following the torch_memory_saver pattern of using pure setuptools for extension building.
"""

import os
import pathlib
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExtension(build_ext):
    """Custom build extension for C++ and optional Rust helper modules."""

    def initialize_options(self):
        super().initialize_options()
        self._rust_outputs = []

    def build_extensions(self):
        self._build_rust_posix_helper()

        # Use CXX environment variable if set, otherwise default to g++
        cxx = os.environ.get("CXX", "g++")
        self.compiler.set_executable("compiler_so", cxx)
        self.compiler.set_executable("compiler_cxx", cxx)
        self.compiler.set_executable("linker_so", f"{cxx} -shared")

        build_ext.build_extensions(self)

    def get_outputs(self):
        return build_ext.get_outputs(self) + self._rust_outputs

    def _build_rust_posix_helper(self):
        """Build the optional in-process Rust NIXL/POSIX helper.

        Runtime images have Rust in the wheel builder, but local editable
        installs may not.  Missing rustc leaves the Python fallback path intact.
        """
        rustc = os.environ.get("RUSTC") or shutil.which("rustc")
        if not rustc:
            self.announce(
                "rustc not found; skipping optional GMS Rust NIXL POSIX helper",
                level=3,
            )
            return

        source = (
            pathlib.Path(__file__).parent
            / "snapshot"
            / "backends"
            / "rust_src"
            / "gms_nixl_posix.rs"
        )
        if not source.exists():
            self.announce(
                f"{source} not found; skipping optional GMS Rust NIXL POSIX helper",
                level=3,
            )
            return

        output_dir = (
            pathlib.Path(self.build_lib)
            / "gpu_memory_service"
            / "snapshot"
            / "backends"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "libgms_nixl_posix.so"
        cmd = [
            rustc,
            "--crate-type",
            "cdylib",
            "-C",
            "opt-level=3",
            "-C",
            "debuginfo=0",
            str(source),
            "-o",
            str(output),
        ]
        self.announce(
            "building optional GMS Rust NIXL POSIX helper: " + " ".join(cmd),
            level=3,
        )
        try:
            subprocess.check_call(cmd)
            self._rust_outputs.append(str(output))
        except subprocess.CalledProcessError as exc:
            if os.environ.get("GMS_REQUIRE_RUST_NIXL", "0") == "1":
                raise
            self.announce(
                "failed to build optional GMS Rust NIXL POSIX helper "
                f"({exc}); Python fallback remains available",
                level=3,
            )


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
    version="0.9.0",
    description="GPU Memory Service for Dynamo - CUDA VMM-based GPU memory allocation and sharing",
    author="NVIDIA Inc.",
    author_email="sw-dl-dynamo@nvidia.com",
    license="Apache-2.0",
    python_requires=">=3.10",
    install_requires=[
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
        "gpu_memory_service.cli",
        "gpu_memory_service.cli.snapshot",
        "gpu_memory_service.common",
        "gpu_memory_service.common.protocol",
        "gpu_memory_service.server",
        "gpu_memory_service.client",
        "gpu_memory_service.client.torch",
        "gpu_memory_service.client.torch.extensions",
        "gpu_memory_service.failover_lock",
        "gpu_memory_service.failover_lock.flock",
        "gpu_memory_service.integrations",
        "gpu_memory_service.integrations.common",
        "gpu_memory_service.integrations.sglang",
        "gpu_memory_service.integrations.trtllm",
        "gpu_memory_service.integrations.vllm",
        "gpu_memory_service.snapshot",
        "gpu_memory_service.snapshot.backends",
    ],
    package_dir={
        "gpu_memory_service": ".",
        "gpu_memory_service.cli": "cli",
        "gpu_memory_service.cli.snapshot": "cli/snapshot",
        "gpu_memory_service.common": "common",
        "gpu_memory_service.common.protocol": "common/protocol",
        "gpu_memory_service.server": "server",
        "gpu_memory_service.client": "client",
        "gpu_memory_service.client.torch": "client/torch",
        "gpu_memory_service.client.torch.extensions": "client/torch/extensions",
        "gpu_memory_service.failover_lock": "failover_lock",
        "gpu_memory_service.failover_lock.flock": "failover_lock/flock",
        "gpu_memory_service.integrations": "integrations",
        "gpu_memory_service.integrations.common": "integrations/common",
        "gpu_memory_service.integrations.sglang": "integrations/sglang",
        "gpu_memory_service.integrations.trtllm": "integrations/trtllm",
        "gpu_memory_service.integrations.vllm": "integrations/vllm",
        "gpu_memory_service.snapshot": "snapshot",
        "gpu_memory_service.snapshot.backends": "snapshot/backends",
    },
    package_data={
        "gpu_memory_service.client.torch.extensions": ["*.cpp"],
        "gpu_memory_service.snapshot.backends": ["libgms_nixl_posix.so"],
    },
    entry_points={
        "console_scripts": [
            "gpu-memory-service=gpu_memory_service.cli.runner:main",
            "gms-storage-client=gpu_memory_service.cli.storage_runner:main",
        ]
    },
    ext_modules=_create_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)
