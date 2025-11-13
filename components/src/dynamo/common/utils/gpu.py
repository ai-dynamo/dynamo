# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU utilities for collecting comprehensive hardware metadata.

This module provides utilities for querying GPU information using NVML (NVIDIA Management Library).
Used to expose GPU hardware specs and identifiers in worker metrics for:
    - Correlation with DCGM telemetry (when available)
    - Hardware inventory and capacity planning (without DCGM)
    - Compatibility verification (driver versions, CUDA versions)
    - Debugging and troubleshooting

Comprehensive Metadata Collected:
    - Identifiers: UUID, PCI Bus ID, GPU index
    - Hardware: Model name, memory capacity
    - Compute: CUDA compute capability, driver version, CUDA version
    - Configuration: Power limit, MIG mode status

For multi-GPU workers (tensor parallelism), all GPUs are collected and exposed as an info metric.
This allows external tools like AIPerf to correlate DCGM metrics (per-GPU) with worker metrics (per-worker)
using standard Prometheus joins, or to understand hardware configuration without DCGM.

Info Metric Pattern:
    dynamo_worker_gpu_info{
        gpu_uuid="GPU-abc",
        gpu_index="0",
        gpu_model="NVIDIA RTX 6000 Ada Generation",
        compute_capability="8.9",
        cuda_version="12.2",
        model="Qwen3"
    } 1

    # Prometheus join example with DCGM:
    dynamo_component_kvstats_total_blocks
      * on(model) group_left(gpu_uuid, compute_capability)
    dynamo_worker_gpu_info
      * on(gpu_uuid) group_left(utilization)
    label_replace(DCGM_FI_DEV_GPU_UTIL, "gpu_uuid", "$1", "UUID", "(.*)")

    # Hardware inventory without DCGM:
    count by (gpu_model, compute_capability) (dynamo_worker_gpu_info)

Note on tensor parallelism:
    - Single-process TP (vLLM, TRT-LLM): All GPUs visible to one process
    - Multi-process TP (SGLang): Each rank sees subset of GPUs via CUDA_VISIBLE_DEVICES
    - This utility returns metadata for all GPUs visible to the current process
    - Each GPU gets its own metric series with unique gpu_index label
"""

import logging
from typing import Dict, List, Optional

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Prometheus info metric for GPU metadata
# This metric exposes GPU UUIDs with labels for easy correlation with DCGM metrics
# Each GPU visible to the worker gets one series with value=1
_gpu_info_metric: Optional[Gauge] = None


def get_gpu_uuids() -> List[str]:
    """Get GPU UUIDs for all GPUs visible to this process.

    Queries NVML to retrieve the UUID for each GPU accessible via CUDA_VISIBLE_DEVICES.
    GPU UUIDs are persistent identifiers that can be correlated with DCGM telemetry data.

    Returns:
        List of GPU UUIDs (e.g., ["GPU-abc12345-...", "GPU-def67890-..."])
        Returns empty list if NVML is unavailable or query fails.

    Example:
        # Single GPU worker
        >>> get_gpu_uuids()
        ["GPU-ef6ef310-6bb8-8bfa-1b66-9fb6e8479ee8"]

        # Multi-GPU worker (tensor parallel)
        >>> get_gpu_uuids()
        ["GPU-abc12345-...", "GPU-def67890-..."]
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        uuids = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            # Convert bytes to str if needed (pynvml returns bytes in some versions)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            uuids.append(uuid)

        pynvml.nvmlShutdown()
        return uuids

    except ImportError:
        logger.warning(
            "pynvml not available - GPU UUIDs will not be included in metrics. "
            "Install with: pip install nvidia-ml-py"
        )
        return []
    except Exception as e:
        logger.warning(f"Failed to query GPU UUIDs via NVML: {e!r}")
        return []


def get_gpu_info() -> List[Dict[str, str]]:
    """Get comprehensive GPU hardware information for all visible GPUs.

    Queries NVML for GPU metadata including hardware specs, capabilities, and identifiers.
    All values are static hardware characteristics (not dynamic runtime metrics).
    Each GPU is returned as a dictionary with metadata fields.

    Returns:
        List of dictionaries, one per GPU, with keys:
            - gpu_uuid: GPU UUID (e.g., "GPU-ef6ef310-...")
            - gpu_index: GPU index in local process (0, 1, 2, ...)
            - gpu_model: GPU model name (e.g., "NVIDIA RTX 6000 Ada Generation")
            - gpu_memory_gb: Total GPU memory in GB (e.g., "48.0")
            - compute_capability: CUDA compute capability (e.g., "8.9")
            - pci_bus_id: PCI Bus ID (e.g., "00000000:01:00.0")
            - driver_version: NVIDIA driver version (e.g., "535.129.03")
            - cuda_version: CUDA version (e.g., "12.2")
            - power_limit_w: GPU power limit in watts (e.g., "300")
            - mig_mode: Multi-Instance GPU mode status ("Enabled" or "Disabled")

        Returns empty list if NVML unavailable or query fails.

    Example:
        >>> get_gpu_info()
        [
            {
                "gpu_uuid": "GPU-ef6ef310-...",
                "gpu_index": "0",
                "gpu_model": "NVIDIA RTX 6000 Ada Generation",
                "gpu_memory_gb": "48.0",
                "compute_capability": "8.9",
                "pci_bus_id": "00000000:01:00.0",
                "driver_version": "535.129.03",
                "cuda_version": "12.2",
                "power_limit_w": "300",
                "mig_mode": "Disabled"
            }
        ]
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # Get driver and CUDA versions once (same for all GPUs)
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
        except Exception:
            driver_version = "Unknown"

        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            # Convert to major.minor format (e.g., 12020 -> "12.2")
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            cuda_version_str = f"{cuda_major}.{cuda_minor}"
        except Exception:
            cuda_version_str = "Unknown"

        gpu_info_list = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get UUID
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")

            # Get model name
            try:
                model = pynvml.nvmlDeviceGetName(handle)
                if isinstance(model, bytes):
                    model = model.decode("utf-8")
            except Exception:
                model = "Unknown GPU"

            # Get memory (convert bytes to GB)
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gb = f"{memory_info.total / (1024**3):.1f}"
            except Exception:
                memory_gb = "Unknown"

            # Get compute capability
            try:
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                compute_cap = f"{major}.{minor}"
            except Exception:
                compute_cap = "Unknown"

            # Get PCI Bus ID
            try:
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                pci_bus_id = pci_info.busId
                if isinstance(pci_bus_id, bytes):
                    pci_bus_id = pci_bus_id.decode("utf-8")
            except Exception:
                pci_bus_id = "Unknown"

            # Get power limit (management/default limit, not current enforced limit)
            try:
                # Get default power limit in milliwatts
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                power_limit_w = f"{power_limit_mw / 1000:.0f}"
            except Exception:
                power_limit_w = "Unknown"

            # Get MIG mode status
            try:
                current_mode, _ = pynvml.nvmlDeviceGetMigMode(handle)
                mig_mode = "Enabled" if current_mode == 1 else "Disabled"
            except Exception:
                # MIG not supported on this GPU
                mig_mode = "NotSupported"

            gpu_info_list.append(
                {
                    "gpu_uuid": uuid,
                    "gpu_index": str(i),
                    "gpu_model": model,
                    "gpu_memory_gb": memory_gb,
                    "compute_capability": compute_cap,
                    "pci_bus_id": pci_bus_id,
                    "driver_version": driver_version,
                    "cuda_version": cuda_version_str,
                    "power_limit_w": power_limit_w,
                    "mig_mode": mig_mode,
                }
            )

        pynvml.nvmlShutdown()
        return gpu_info_list

    except ImportError:
        logger.warning(
            "pynvml not available - GPU info metric will not be populated. "
            "Install with: pip install nvidia-ml-py"
        )
        return []
    except Exception as e:
        logger.warning(f"Failed to query GPU info via NVML: {e!r}")
        return []


def initialize_gpu_info_metric(extra_labels: Optional[Dict[str, str]] = None) -> None:
    """Initialize the GPU info metric with comprehensive GPU hardware metadata.

    Creates a Prometheus Gauge metric exposing GPU UUIDs and hardware specs for each GPU
    visible to this worker. This metric uses the "info pattern" - one series per GPU
    with value=1 and labels containing static hardware metadata.

    The metric is automatically registered with prometheus_client.REGISTRY and will
    be included in /metrics endpoint output alongside other Dynamo metrics.

    Metadata includes:
        - Identifiers: UUID, index, PCI Bus ID
        - Hardware specs: Model name, memory capacity, compute capability
        - Software: Driver version, CUDA version
        - Configuration: Power limit, MIG mode status

    Args:
        extra_labels: Optional dict of additional labels to include on all series
                     (e.g., {"model": "Qwen3", "dynamo_component": "backend"})

    Example metric output:
        # Single GPU worker (comprehensive metadata)
        dynamo_worker_gpu_info{
            gpu_uuid="GPU-ef6ef310-6bb8-8bfa-1b66-9fb6e8479ee8",
            gpu_index="0",
            gpu_model="NVIDIA RTX 6000 Ada Generation",
            gpu_memory_gb="48.0",
            compute_capability="8.9",
            pci_bus_id="00000000:01:00.0",
            driver_version="535.129.03",
            cuda_version="12.2",
            power_limit_w="300",
            mig_mode="Disabled",
            model="Qwen/Qwen3-0.6B",
            dynamo_component="backend",
            dynamo_namespace="dynamo"
        } 1.0

        # Multi-GPU worker (TP=2) - one series per GPU
        dynamo_worker_gpu_info{gpu_index="0", ...} 1.0
        dynamo_worker_gpu_info{gpu_index="1", ...} 1.0

    Usage in worker initialization:
        from dynamo.common.utils.gpu import initialize_gpu_info_metric

        # After component/endpoint setup
        initialize_gpu_info_metric(
            extra_labels={
                "model": config.model_name,
                "dynamo_component": config.component,
                "dynamo_namespace": config.namespace,
            }
        )

    Prometheus query examples:
        # Get worker's GPU hardware
        dynamo_worker_gpu_info{model="Qwen3"}

        # Find GPUs by model
        dynamo_worker_gpu_info{gpu_model=~".*H100.*"}

        # Find GPUs with specific CUDA version
        dynamo_worker_gpu_info{cuda_version="12.2"}

        # Find GPUs with compute capability >= 8.0
        dynamo_worker_gpu_info{compute_capability=~"[89]\\..+"}

        # Join with DCGM metrics
        DCGM_FI_DEV_GPU_UTIL
          * on(UUID) group_right(model, gpu_index, compute_capability)
        label_replace(dynamo_worker_gpu_info, "UUID", "$1", "gpu_uuid", "(.*)")

        # Count GPUs by model
        count by (gpu_model) (dynamo_worker_gpu_info)

    Use cases without DCGM:
        - Hardware inventory: What GPUs are in the cluster?
        - Compatibility checking: Do all workers have compatible CUDA versions?
        - Capacity planning: Total memory across all GPUs
        - Debugging: Check driver versions, compute capabilities
        - MIG verification: Which workers are using MIG mode?
    """
    global _gpu_info_metric

    # Create metric if not already created
    if _gpu_info_metric is None:
        # Define all static GPU metadata labels (directly from NVML)
        gpu_labels = [
            "gpu_uuid",
            "gpu_index",
            "gpu_model",
            "gpu_memory_gb",
            "compute_capability",
            "pci_bus_id",
            "driver_version",
            "cuda_version",
            "power_limit_w",
            "mig_mode",
        ]

        # Add dynamic extra labels (e.g., model, component, namespace)
        all_labels = gpu_labels + (list(extra_labels.keys()) if extra_labels else [])

        _gpu_info_metric = Gauge(
            "dynamo_component_gpu_info",
            "GPU hardware metadata for correlation and hardware inventory (info pattern)",
            all_labels,
        )

    # Get comprehensive GPU information
    gpu_info_list = get_gpu_info()
    if not gpu_info_list:
        logger.debug(
            "No GPU info available - dynamo_worker_gpu_info metric not populated"
        )
        return

    # Set metric series for each GPU
    extra_labels = extra_labels or {}
    for gpu_info in gpu_info_list:
        # Combine GPU labels with extra labels
        all_labels = {**gpu_info, **extra_labels}

        # Set gauge to 1 (info pattern convention)
        _gpu_info_metric.labels(**all_labels).set(1)

    logger.info(
        f"Initialized GPU info metric with {len(gpu_info_list)} GPU(s): "
        f"{[(g['gpu_uuid'], g['gpu_model'], g['compute_capability']) for g in gpu_info_list]}"
    )
