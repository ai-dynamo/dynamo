#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import platform
import subprocess
import sys

import distro
import pkg_resources


def get_os_version() -> str:
    """Get OS version."""
    # TODO: Revisit once we need to support Windows based systems
    return f"{distro.name()} {distro.version()}"


def execute_subprocess_output(command: str) -> str:
    """Execute a subprocess command and return the output."""
    try:
        out = subprocess.check_output(command, shell=True, stderr=subprocess.DEVNULL)
        if not out.strip():
            return "N/A"
        return out.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "N/A"


def get_glibc_version() -> str:
    """Get GLIBC version."""
    return execute_subprocess_output("ldd --version | head -n 1 | awk '{print $NF}'")


def get_gcc_version() -> str:
    """Get GCC version."""
    return execute_subprocess_output("gcc --version | head -n 1 | awk '{print $NF}'")


def get_cmake_version() -> str:
    """Get Cmake version."""
    return execute_subprocess_output("cmake --version | head -n 1 | awk '{print $NF}'")


def get_rust_version() -> str:
    """Get Rust version."""
    return execute_subprocess_output(
        "rustc --version | head -n 1 | awk '{print $(NF-2)}'"
    )


def get_docker_version() -> str:
    """Get Docker version."""
    return execute_subprocess_output("docker --version | awk '{print $3}' | tr -d ','")


def get_cpu_architecture() -> str:
    """Get CPU architecture."""
    return execute_subprocess_output("lscpu")


def query_nvidia_smi(param: str) -> str:
    """Get GPU information from nvidia-smi if available"""
    return execute_subprocess_output(
        f"nvidia-smi --query-gpu={param} --format=csv,noheader"
    )


def get_gpu_topo() -> str:
    """Get GPU topology if available"""
    return execute_subprocess_output("nvidia-smi topo -m")


def get_cuda_version() -> str:
    """Get CUDA version if available."""
    return execute_subprocess_output(r"nvcc --version | grep -Po 'release \K\d+\.\d+'")


def get_python_platform():
    return platform.platform()


def get_installed_packages() -> list[tuple[str, str]]:
    """Get list of installed Python packages and their versions."""
    return [(pkg.key, pkg.version) for pkg in pkg_resources.working_set]


def get_python_packages() -> str:
    """Get list of specified Python packages and their versions."""
    installed_packages = get_installed_packages()
    out = []
    search_python_packages = [
        "ai-dynamo",
        "ai-dynamo-runtime",
        "ai-dynamo-vllm",
        "genai-perf",
        "nixl",
        "numpy",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-ml-py",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvtx-cu12",
        "pyzmq",
        "tensorrt_llm",
        "torch",
        "torchaudio",
        "transformers",
        "tritonclient",
    ]
    for pkg_name in search_python_packages:
        version = next(
            (version for name, version in installed_packages if name == pkg_name), None
        )
        if version:
            out.append(f"{pkg_name}: {version}")
        else:
            out.append(f"{pkg_name}: Not installed")
    return "\n".join(out)


def env() -> None:
    """Display information about the current environment."""
    print("System Information:")
    print(f"OS: {get_os_version()}")
    print(f"Glibc Version: {get_glibc_version()}")
    print(f"GCC Version: {get_gcc_version()}")
    print(f"Cmake Version: {get_cmake_version()}")
    print(f"Rust Version: {get_rust_version()}")
    print(f"Docker Version: {get_docker_version()}")

    print("\nCPU Information:")
    print(f"{get_cpu_architecture()}")

    # Python Environment
    py_version = sys.version.split()[0]
    print(f"\nPython Version: {py_version}")
    print(f"Python Platform: {get_python_platform()}")
    print("\nPython Packages:")
    print(f"{get_python_packages()}")
