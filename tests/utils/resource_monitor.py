# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Resource monitoring for ManagedDeployment.

Provides CPU, memory, and GPU metrics collection for pods in a deployment.
Supports both context manager and explicit start/stop patterns.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kr8s.objects import Pod

    from tests.utils.managed_deployment import ManagedDeployment


@dataclass
class ResourceSnapshot:
    """Single resource usage snapshot for a pod."""

    timestamp: float
    pod_name: str
    service_name: str

    # Memory (bytes)
    memory_used_bytes: int
    memory_total_bytes: int
    memory_available_bytes: int

    # CPU (percentage, calculated from /proc/stat delta)
    cpu_usage_percent: float

    # GPU (optional, per-GPU list)
    # Each dict: {index, memory_used_mb, memory_total_mb, utilization_percent}
    gpu_metrics: Optional[List[dict]] = None


@dataclass
class ResourceMonitorConfig:
    """Configuration for resource monitoring."""

    enabled: bool = False
    interval_seconds: float = 10.0
    include_gpu: bool = True
    write_to_pvc: bool = True
    pvc_output_path: str = "/tmp/service_logs/resource_metrics.jsonl"


class ResourceMonitor:
    """Resource monitoring for a ManagedDeployment.

    Supports both context manager and explicit start/stop patterns.

    Usage (context manager):
        async with ResourceMonitor(deployment, config) as monitor:
            # monitoring runs in background
            ...
        # monitor.history contains all snapshots

    Usage (explicit):
        monitor = ResourceMonitor(deployment, config)
        await monitor.start()
        ...
        history = await monitor.stop()
    """

    def __init__(
        self,
        deployment: "ManagedDeployment",
        config: Optional[ResourceMonitorConfig] = None,
    ):
        self._deployment = deployment
        self._config = config or ResourceMonitorConfig(enabled=True)
        self._history: List[ResourceSnapshot] = []
        self._task: Optional[asyncio.Task] = None
        self._prev_cpu_stats: Dict[str, dict] = {}
        self._logger = deployment._logger

    async def __aenter__(self) -> "ResourceMonitor":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @property
    def history(self) -> List[ResourceSnapshot]:
        """All collected resource snapshots."""
        return self._history

    @property
    def config(self) -> ResourceMonitorConfig:
        """Current monitoring configuration."""
        return self._config

    async def start(self):
        """Start background resource monitoring task."""
        self._history = []
        self._prev_cpu_stats = {}
        self._task = asyncio.create_task(self._monitoring_loop())
        self._logger.info(
            f"Started resource monitoring (interval={self._config.interval_seconds}s, "
            f"gpu={self._config.include_gpu}, write_to_pvc={self._config.write_to_pvc})"
        )

    async def stop(self) -> List[ResourceSnapshot]:
        """Stop background monitoring and return collected history.

        Returns:
            List of all collected ResourceSnapshots
        """
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._logger.info(
            f"Stopped resource monitoring, collected {len(self._history)} snapshots"
        )
        return self._history

    async def save_history_locally(self, log_dir: str):
        """Save resource monitoring history to local log directory.

        Args:
            log_dir: Directory to save the metrics file
        """
        if not self._history:
            return

        try:
            metrics_file = os.path.join(log_dir, "resource_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump([asdict(s) for s in self._history], f, indent=2)
            self._logger.info(
                f"Saved {len(self._history)} resource snapshots to {metrics_file}"
            )
        except Exception as e:
            self._logger.warning(f"Failed to save resource history: {e}")

    async def _monitoring_loop(self):
        """Background loop that collects resource metrics periodically."""
        while True:
            try:
                snapshots = await self.get_all_resource_usage()

                # Flatten and store snapshots
                for service_snapshots in snapshots.values():
                    self._history.extend(service_snapshots)

                    # Write to PVC if enabled
                    if self._config.write_to_pvc and service_snapshots:
                        await self._write_metrics_to_pvc(service_snapshots)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._logger.warning(f"Resource monitoring error: {e}")

            await asyncio.sleep(self._config.interval_seconds)

    async def _exec_in_pod(
        self, pod: "Pod", command: List[str], timeout: float = 10.0
    ) -> Any:
        """Execute command in pod with timeout.

        Args:
            pod: The pod to execute the command in
            command: Command as a list of strings
            timeout: Timeout in seconds

        Returns:
            The exec result with stdout, stderr, and returncode
        """
        return await asyncio.wait_for(
            asyncio.create_task(asyncio.to_thread(pod.exec, command)),
            timeout=timeout,
        )

    def _parse_meminfo(self, meminfo_output: str) -> dict:
        """Parse /proc/meminfo output.

        Args:
            meminfo_output: Raw output from cat /proc/meminfo

        Returns:
            dict with 'total', 'available', 'used' keys (all in bytes)
        """
        values = {}
        for line in meminfo_output.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                # Values are in kB, convert to bytes
                value = int(parts[1]) * 1024
                values[key] = value

        return {
            "total": values.get("MemTotal", 0),
            "available": values.get("MemAvailable", 0),
            "used": values.get("MemTotal", 0) - values.get("MemAvailable", 0),
        }

    def _parse_nvidia_smi(self, output: str) -> List[dict]:
        """Parse nvidia-smi CSV output.

        Args:
            output: Output from nvidia-smi --query-gpu=... --format=csv,noheader,nounits

        Returns:
            List of dicts with GPU metrics
        """
        gpus = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "memory_used_mb": int(parts[1]),
                            "memory_total_mb": int(parts[2]),
                            "utilization_percent": int(parts[3]),
                        }
                    )
                except ValueError:
                    # Skip lines that can't be parsed
                    continue
        return gpus

    def _calculate_cpu_usage(self, pod_name: str, proc_stat: str) -> float:
        """Calculate CPU usage from /proc/stat (requires previous sample for delta).

        Args:
            pod_name: Pod name for tracking previous stats
            proc_stat: Raw output from cat /proc/stat

        Returns:
            CPU usage percentage (0-100)
        """
        # Parse first line: cpu user nice system idle iowait irq softirq
        first_line = proc_stat.strip().split("\n")[0]
        parts = first_line.split()[1:]  # Skip "cpu" label
        values = [int(p) for p in parts[:7]]

        idle = values[3] + values[4]  # idle + iowait
        total = sum(values)

        # Calculate delta from previous sample
        prev = self._prev_cpu_stats.get(pod_name, {"total": total, "idle": idle})
        prev_total = prev.get("total", total)
        prev_idle = prev.get("idle", idle)

        # Store current values for next calculation
        self._prev_cpu_stats[pod_name] = {"total": total, "idle": idle}

        total_delta = total - prev_total
        idle_delta = idle - prev_idle

        if total_delta == 0:
            return 0.0

        return 100.0 * (1.0 - idle_delta / total_delta)

    def _get_service_for_pod(self, pod: "Pod") -> str:
        """Get service name for a pod from its labels.

        Args:
            pod: The pod object

        Returns:
            Service name or 'unknown'
        """
        # Try to get service name from labels
        labels = pod.labels or {}

        # Check for nvidia.com/selector label (format: deployment-servicename)
        selector = labels.get("nvidia.com/selector", "")
        deployment_name = self._deployment._deployment_name
        if selector and deployment_name:
            # Extract service name from selector (e.g., "my-deploy-frontend" -> "frontend")
            prefix = f"{deployment_name}-"
            if selector.startswith(prefix):
                return selector[len(prefix) :]

        return "unknown"

    async def get_pod_resource_usage(
        self, pod: "Pod", service_name: str, include_gpu: bool = True
    ) -> Optional[ResourceSnapshot]:
        """Get current resource usage for a single pod via exec.

        Args:
            pod: The pod to query
            service_name: Name of the service this pod belongs to
            include_gpu: Whether to include GPU metrics (default True)

        Returns:
            ResourceSnapshot with current metrics, or None if failed
        """
        try:
            # Memory via /proc/meminfo
            mem_result = await self._exec_in_pod(pod, ["cat", "/proc/meminfo"])
            memory = self._parse_meminfo(mem_result.stdout.decode())

            # CPU via /proc/stat
            cpu_result = await self._exec_in_pod(pod, ["cat", "/proc/stat"])
            cpu_usage = self._calculate_cpu_usage(pod.name, cpu_result.stdout.decode())

            # Determine if GPU metrics should be collected
            should_include_gpu = (
                include_gpu if self._config is None else self._config.include_gpu
            )

            # GPU via nvidia-smi (if enabled and available)
            gpu_metrics = None
            if should_include_gpu:
                try:
                    gpu_result = await self._exec_in_pod(
                        pod,
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=5.0,
                    )
                    if gpu_result.returncode == 0:
                        gpu_metrics = self._parse_nvidia_smi(gpu_result.stdout.decode())
                except Exception:
                    # nvidia-smi not available, that's okay
                    pass

            return ResourceSnapshot(
                timestamp=time.time(),
                pod_name=pod.name,
                service_name=service_name,
                memory_used_bytes=memory["used"],
                memory_total_bytes=memory["total"],
                memory_available_bytes=memory["available"],
                cpu_usage_percent=cpu_usage,
                gpu_metrics=gpu_metrics,
            )

        except Exception as e:
            self._logger.warning(f"Failed to get resource usage for {pod.name}: {e}")
            return None

    async def get_all_resource_usage(
        self, include_gpu: bool = True
    ) -> Dict[str, List[ResourceSnapshot]]:
        """Get resource usage for all pods in all services.

        Args:
            include_gpu: Whether to include GPU metrics (default True)

        Returns:
            Dict mapping service name to list of ResourceSnapshots
        """
        results: Dict[str, List[ResourceSnapshot]] = {}

        service_pods = self._deployment.get_pods()

        for service_name, pods in service_pods.items():
            results[service_name] = []
            for pod in pods:
                # Only query running pods
                if pod.status.phase != "Running":
                    continue
                try:
                    snapshot = await self.get_pod_resource_usage(
                        pod, service_name, include_gpu=include_gpu
                    )
                    if snapshot:
                        results[service_name].append(snapshot)
                except Exception as e:
                    self._logger.warning(f"Failed to get resources for {pod.name}: {e}")

        return results

    async def _write_metrics_to_pvc(self, snapshots: List[ResourceSnapshot]):
        """Write metrics to PVC via exec into a pod.

        Args:
            snapshots: List of snapshots to write as JSONL
        """
        # Find any running pod to write metrics
        service_pods = self._deployment.get_pods()
        for service_name, pods in service_pods.items():
            for pod in pods:
                if pod.status.phase != "Running":
                    continue
                try:
                    for snapshot in snapshots:
                        json_line = json.dumps(asdict(snapshot))
                        # Escape single quotes for shell safety
                        escaped_json = json_line.replace("'", "'\\''")
                        await self._exec_in_pod(
                            pod,
                            [
                                "sh",
                                "-c",
                                f"echo '{escaped_json}' >> {self._config.pvc_output_path}",
                            ],
                            timeout=5.0,
                        )
                    return  # Successfully wrote to one pod
                except Exception as e:
                    self._logger.debug(
                        f"Failed to write metrics via {pod.name}: {e}, trying next pod"
                    )
                    continue

        self._logger.warning("No available pod to write resource metrics to PVC")
