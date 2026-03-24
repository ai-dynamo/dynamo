# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
GPU discovery utilities for fault tolerance testing.

Provides functions to discover GPU information from Kubernetes pods,
including mapping processes to GPUs and handling CUDA_VISIBLE_DEVICES remapping.
"""

import logging
from typing import List, Optional

from kr8s.objects import Pod

logger = logging.getLogger(__name__)


def get_available_gpu_ids(pod: Pod) -> List[int]:
    """
    Get list of actual GPU IDs available in the pod.

    Handles non-sequential GPU IDs correctly (e.g., [0, 1, 3, 7] with gaps).

    Args:
        pod: Kubernetes pod object (kr8s pod with exec() method)

    Returns:
        List of GPU IDs (e.g., [0, 1, 2, 3]) or empty list if no GPUs found

    Example:
        >>> gpu_ids = get_available_gpu_ids(pod)
        >>> print(gpu_ids)
        [0, 1, 2, 3]
    """
    try:
        result = pod.exec(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])

        # Parse GPU indices from output
        gpu_ids = []
        for line in result.stdout.decode().splitlines():
            line = line.strip()
            if line.isdigit():
                gpu_ids.append(int(line))

        if not gpu_ids:
            logger.warning(f"No GPUs found in pod {pod.name}")
            return []

        logger.debug(f"Available GPU IDs in pod {pod.name}: {gpu_ids}")
        return gpu_ids

    except Exception as e:
        logger.error(f"Failed to get GPU IDs from pod {pod.name}: {e}")
        return []


def get_gpu_id_for_process(pod: Pod, process_pid: int) -> int:
    """
    Find which GPU a process is using.

    Queries nvidia-smi to determine the primary GPU for a given process.
    This correctly handles:
    - Non-sequential GPU IDs
    - CUDA_VISIBLE_DEVICES remapping
    - Multi-GPU processes (returns primary GPU)

    Args:
        pod: Kubernetes pod object (kr8s pod with exec() method)
        process_pid: Process ID to find GPU for

    Returns:
        GPU ID (0-N) where the process is running, or 0 if not found

    Example:
        >>> gpu_id = get_gpu_id_for_process(pod, 603)
        >>> print(gpu_id)
        1  # Process 603 is running on GPU 1
    """
    try:
        # Get actual GPU IDs available in pod (handles non-sequential IDs)
        gpu_ids = get_available_gpu_ids(pod)

        if not gpu_ids:
            logger.error(f"No GPUs found in pod {pod.name}!")
            return 0

        logger.debug(
            f"Searching for PID {process_pid} across {len(gpu_ids)} GPUs: {gpu_ids}"
        )

        # Check each GPU for our process
        for gpu_id in gpu_ids:
            result = pod.exec(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_id),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ]
            )

            # Parse PIDs running on this GPU
            pids_output = result.stdout.decode().strip()

            # Handle both single PID and multiple PIDs
            # Output can be:
            # "602" (single PID)
            # "602\n603\n604" (multiple PIDs)
            # " 602 " (with spaces)
            pids_on_gpu = [p.strip() for p in pids_output.split("\n") if p.strip()]

            # Check if our PID is in the list
            if str(process_pid) in pids_on_gpu:
                logger.info(
                    f"PID {process_pid} found on GPU {gpu_id} in pod {pod.name}"
                )
                return gpu_id

        # Process not found on any GPU
        logger.warning(
            f"PID {process_pid} not found on any GPU in pod {pod.name}. "
            f"This may happen if the process hasn't initialized CUDA yet or "
            f"if nvidia-smi doesn't track multi-process CUDA apps. "
            f"Defaulting to first GPU: {gpu_ids[0]}"
        )
        return gpu_ids[0]

    except Exception as e:
        logger.error(
            f"GPU discovery failed for PID {process_pid} in pod {pod.name}: {e}"
        )
        return 0


def get_gpu_pci_address(pod: Pod, gpu_id: int) -> Optional[str]:
    """
    Get PCI bus address for a GPU.

    The PCI address is used in kernel XID messages and identifies
    the physical hardware location of the GPU.

    Args:
        pod: Kubernetes pod object
        gpu_id: GPU index (0-N) as shown by nvidia-smi

    Returns:
        PCI address (e.g., "00000000:8D:00.0") or None if failed

    Example:
        >>> pci_addr = get_gpu_pci_address(pod, 1)
        >>> print(pci_addr)
        00000000:91:00.0
    """
    try:
        result = pod.exec(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
            ]
        )

        pci_addr = result.stdout.decode().strip()

        if not pci_addr:
            logger.error(f"Empty PCI address for GPU {gpu_id}")
            return None

        logger.debug(f"GPU {gpu_id} in pod {pod.name} has PCI address: {pci_addr}")
        return pci_addr

    except Exception as e:
        logger.error(
            f"Failed to get PCI address for GPU {gpu_id} in pod {pod.name}: {e}"
        )
        return None


def get_gpu_info(pod: Pod, gpu_id: int) -> Optional[dict]:
    """
    Get comprehensive information about a GPU.

    Args:
        pod: Kubernetes pod object
        gpu_id: GPU index (0-N)

    Returns:
        Dict with keys: index, name, pci_bus_id, memory_total, driver_version
        or None if failed

    Example:
        >>> info = get_gpu_info(pod, 0)
        >>> print(info)
        {
            'index': 0,
            'name': 'NVIDIA H200',
            'pci_bus_id': '00000000:8D:00.0',
            'memory_total': '143771 MiB',
            'driver_version': '550.163.01'
        }
    """
    try:
        result = pod.exec(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=index,name,pci.bus_id,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )

        output = result.stdout.decode().strip()
        parts = [p.strip() for p in output.split(",")]

        if len(parts) < 5:
            logger.error(f"Unexpected nvidia-smi output format: {output}")
            return None

        return {
            "index": int(parts[0]),
            "name": parts[1],
            "pci_bus_id": parts[2],
            "memory_total": parts[3],
            "driver_version": parts[4],
        }

    except Exception as e:
        logger.error(f"Failed to get GPU info for GPU {gpu_id}: {e}")
        return None


def get_processes_on_gpu(pod: Pod, gpu_id: int) -> List[int]:
    """
    Get list of process IDs running on a specific GPU.

    Args:
        pod: Kubernetes pod object
        gpu_id: GPU index (0-N)

    Returns:
        List of PIDs running on this GPU, or empty list if none/error

    Example:
        >>> pids = get_processes_on_gpu(pod, 1)
        >>> print(pids)
        [602, 603]
    """
    try:
        result = pod.exec(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )

        pids_output = result.stdout.decode().strip()

        if not pids_output:
            logger.debug(f"No processes found on GPU {gpu_id} in pod {pod.name}")
            return []

        # Parse PIDs (handle multiple PIDs on same GPU)
        pids = []
        for line in pids_output.split("\n"):
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))

        logger.debug(f"GPU {gpu_id} in pod {pod.name} has processes: {pids}")
        return pids

    except Exception as e:
        logger.error(f"Failed to get processes for GPU {gpu_id}: {e}")
        return []
