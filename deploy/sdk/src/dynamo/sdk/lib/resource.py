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

# TODO: this should be used for planner as well and should leverage proper nvml bindings

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import psutil

try:
    import pynvml

    PYNVML_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
NVIDIA_GPU = "nvidia.com/gpu"


class ResourceError(Exception):
    """Base exception for resource-related errors."""

    pass


@dataclass
class GPUProcess:
    """Information about a process running on a GPU."""

    pid: int
    used_memory: int  # in bytes
    name: str = ""

    def __post_init__(self):
        """Get process name if available."""
        try:
            self.name = psutil.Process(self.pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class GPUInfo:
    """Information about a specific GPU device."""

    def __init__(self, index: int, total_memory: int, name: str, uuid: str):
        self.index = index
        self.total_memory = total_memory  # in bytes
        self.name = name
        self.uuid = uuid
        self.available = True  # Can be set to False if GPU is reserved/in use
        self.utilization = 0  # in percent (0-100)
        self.processes: list[GPUProcess] = []

    def __repr__(self) -> str:
        return f"GPUInfo(index={self.index}, name='{self.name}', total_memory={self.total_memory/1024/1024:.0f}MB, available={self.available})"


class GPUManager:
    """
    Manages GPU resources using NVML.

    This class provides methods to:
    - Discover available GPUs
    - Query GPU properties and status
    - Track GPU processes
    - Allocate and release GPUs
    - Generate CUDA_VISIBLE_DEVICES environment variables
    """

    def __init__(self):
        """Initialize the GPU manager."""
        self.gpus: list[GPUInfo] = []
        self._initialized = False
        # List to track fractional GPU allocations
        # Each item is (gpu_index, fraction_used, fraction_size)
        # E.g. (0, 0.5, 0.5) means GPU 0 has 0.5 used with fraction size of 0.5
        self._gpu_fractions: list[tuple[int, float, float]] = []
        self._init_nvml()

    def _init_nvml(self):
        """Initialize NVML and discover GPUs."""
        if not PYNVML_AVAILABLE:
            logger.warning("PyNVML not available. GPU functionality will be limited.")
            return

        try:
            pynvml.nvmlInit()
            self._initialized = True
            self._discover_gpus()
        except (
            pynvml.NVMLError_LibraryNotFound,
            pynvml.NVMLError_DriverNotLoaded,
            OSError,
        ) as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self._initialized = False

    def __del__(self):
        """Clean up NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pylint: disable=broad-except
                pass

    def _discover_gpus(self):
        """Discover available GPUs and their properties."""
        if not self._initialized:
            return

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpus = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                uuid = pynvml.nvmlDeviceGetUUID(handle)

                gpu_info = GPUInfo(
                    index=i, total_memory=memory_info.total, name=name, uuid=uuid
                )

                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info.utilization = utilization.gpu
                except pynvml.NVMLError:
                    logger.debug(f"Could not get utilization for GPU {i}")

                # Get processes running on GPU
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    gpu_info.processes = [
                        GPUProcess(pid=p.pid, used_memory=p.usedGpuMemory)
                        for p in processes
                    ]
                except pynvml.NVMLError:
                    logger.debug(f"Could not get processes for GPU {i}")

                self.gpus.append(gpu_info)

            logger.info(f"Discovered {len(self.gpus)} GPUs")
        except pynvml.NVMLError as e:
            logger.warning(f"Error discovering GPUs: {e}")

    def update_gpu_stats(self):
        """Update GPU statistics (utilization, memory etc.)."""
        if not self._initialized:
            return

        for gpu in self.gpus:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)

                # Update memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu.total_memory = memory_info.total

                # Update utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu.utilization = utilization.gpu
                except pynvml.NVMLError:
                    pass

                # Update processes
                try:
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    gpu.processes = [
                        GPUProcess(pid=p.pid, used_memory=p.usedGpuMemory)
                        for p in processes
                    ]
                except pynvml.NVMLError:
                    pass

            except pynvml.NVMLError as e:
                logger.warning(f"Error updating GPU {gpu.index} stats: {e}")

    def get_gpu_count(self) -> int:
        """Return the number of available GPUs."""
        return len(self.gpus)

    def get_available_gpus(self) -> list[int]:
        """Return a list of available GPU indices."""
        return [gpu.index for gpu in self.gpus if gpu.available]

    def get_gpu_memory(self, index: int) -> tuple[int, int]:
        """
        Return (total memory, free memory) in bytes for a specific GPU.

        Args:
            index: GPU index

        Returns:
            Tuple of (total memory, free memory) in bytes
        """
        if not self._initialized or index >= len(self.gpus):
            return (0, 0)

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (memory_info.total, memory_info.free)
        except pynvml.NVMLError as e:
            logger.warning(f"Error getting GPU memory for GPU {index}: {e}")
            return (0, 0)

    def reset_allocations(self):
        """Reset all GPU allocations."""
        self._gpu_fractions = []
        for gpu in self.gpus:
            gpu.available = True

    def get_gpu_stats(self) -> list[dict[str, t.Any]]:
        """
        Get detailed statistics for all GPUs.

        Returns:
            List of dictionaries with GPU statistics
        """
        self.update_gpu_stats()

        stats = []
        for gpu in self.gpus:
            total_memory, free_memory = self.get_gpu_memory(gpu.index)
            stats.append(
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "uuid": gpu.uuid,
                    "total_memory": total_memory,
                    "free_memory": free_memory,
                    "used_memory": total_memory - free_memory,
                    "memory_utilization": (total_memory - free_memory)
                    / total_memory
                    * 100
                    if total_memory > 0
                    else 0,
                    "gpu_utilization": gpu.utilization,
                    "process_count": len(gpu.processes),
                    "processes": [
                        {
                            "pid": process.pid,
                            "name": process.name,
                            "used_memory": process.used_memory,
                        }
                        for process in gpu.processes
                    ],
                    "available": gpu.available,
                }
            )

        return stats


def system_resources() -> dict[str, t.Any]:
    """
    Get available GPU resources

    Returns:
        Dictionary of resources with keys 'nvidia.com/gpu'
    """
    resources = {}

    # Get GPU resources
    gpu_manager = GPUManager()
    resources[NVIDIA_GPU] = gpu_manager.get_available_gpus()

    return resources
