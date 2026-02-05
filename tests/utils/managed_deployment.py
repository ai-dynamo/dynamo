# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import re
import secrets
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import kr8s
import requests
import yaml
from kr8s.objects import Pod, Service
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

# LogStreamManager removed - using PVC-based log collection only


def _get_workspace_dir() -> str:
    """Get workspace directory without depending on dynamo.common package.

    This allows tests to run without requiring dynamo package to be installed.
    """
    # Start from this file's location and walk up to find workspace root
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Workspace root has pyproject.toml
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)

    # Fallback: assume workspace is 3 levels up from tests/utils/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


class ServiceSpec:
    """Wrapper around a single service in the deployment spec."""

    def __init__(self, service_name: str, service_spec: dict):
        self._name = service_name
        self._spec = service_spec

    @property
    def name(self) -> str:
        """The service name (read-only)"""
        return self._name

    # ----- Image -----
    @property
    def image(self) -> Optional[str]:
        """Container image for the service"""
        try:
            return self._spec["extraPodSpec"]["mainContainer"]["image"]
        except KeyError:
            return None

    @image.setter
    def image(self, value: str):
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in self._spec["extraPodSpec"]:
            self._spec["extraPodSpec"]["mainContainer"] = {}
        self._spec["extraPodSpec"]["mainContainer"]["image"] = value

    @property
    def component_type(self):
        return self._spec["componentType"]

    @property
    def envs(self) -> list[dict[str, str]]:
        """Environment variables for the service"""
        return self._spec.get("envs", [])

    @envs.setter
    def envs(self, value: list[dict[str, str]]):
        self._spec["envs"] = value

    # ----- Replicas -----
    @property
    def replicas(self) -> int:
        return self._spec.get("replicas", 0)

    @replicas.setter
    def replicas(self, value: int):
        self._spec["replicas"] = value

    @property
    def model(self) -> Optional[str]:
        """Model being served by this service (checks both --model and --model-path)"""
        try:
            args_list = self._spec["extraPodSpec"]["mainContainer"]["args"]
        except KeyError:
            return None
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)
        for i, part in enumerate(parts):
            if part in ["--model", "--model-path"]:
                return parts[i + 1] if i + 1 < len(parts) else None
        return None

    @model.setter
    def model(self, value: str):
        if "extraPodSpec" not in self._spec:
            return
        if "mainContainer" not in self._spec["extraPodSpec"]:
            return

        args_list = self._spec["extraPodSpec"]["mainContainer"].get("args", [])
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)

        # Try to update --model first, then --model-path
        model_index = None
        for i, part in enumerate(parts):
            if part in ["--model", "--model-path"]:
                model_index = i
                break

        if model_index is not None:
            if model_index + 1 < len(parts):
                parts[model_index + 1] = value
            else:
                return
        else:
            return

        # Store args as a list of separate strings for proper command-line parsing
        # WRONG: [" ".join(parts)] creates ["--model Qwen/Qwen3-0.6B"] (single string)
        # RIGHT: parts creates ["--model", "Qwen/Qwen3-0.6B"] (separate strings)
        self._spec["extraPodSpec"]["mainContainer"]["args"] = parts

    # ----- GPUs -----
    @property
    def gpus(self) -> int:
        try:
            return int(self._spec["resources"]["limits"]["gpu"])
        except KeyError:
            return 0

    @gpus.setter
    def gpus(self, value: int):
        if "resources" not in self._spec:
            self._spec["resources"] = {}
        if "limits" not in self._spec["resources"]:
            self._spec["resources"]["limits"] = {}
        self._spec["resources"]["limits"]["gpu"] = str(value)

    @property
    def tensor_parallel_size(self) -> int:
        """Get tensor parallel size from vLLM arguments"""
        try:
            args_list = self._spec["extraPodSpec"]["mainContainer"]["args"]
        except KeyError:
            return 1  # Default tensor parallel size

        args_str = " ".join(args_list)
        parts = shlex.split(args_str)
        for i, part in enumerate(parts):
            if part == "--tensor-parallel-size":
                return int(parts[i + 1]) if i + 1 < len(parts) else 1
        return 1

    @tensor_parallel_size.setter
    def tensor_parallel_size(self, value: int):
        if "extraPodSpec" not in self._spec:
            return
        if "mainContainer" not in self._spec["extraPodSpec"]:
            return

        args_list = self._spec["extraPodSpec"]["mainContainer"].get("args", [])
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)

        # Find existing tensor-parallel-size argument
        tp_index = None
        for i, part in enumerate(parts):
            if part == "--tensor-parallel-size":
                tp_index = i
                break

        if tp_index is not None:
            # Update existing value
            if tp_index + 1 < len(parts):
                parts[tp_index + 1] = str(value)
            else:
                parts.append(str(value))
        else:
            # Add new argument
            parts.extend(["--tensor-parallel-size", str(value)])

        # Store args as a list of separate strings for proper command-line parsing
        # When TP > 1, this setter is called and adds --tensor-parallel-size to args.
        # WRONG: [" ".join(parts)] would create ["--model Qwen/Qwen3-0.6B --tensor-parallel-size 2"]
        #        causing argparse to fail with "IndexError: list index out of range"
        # RIGHT: parts creates ["--model", "Qwen/Qwen3-0.6B", "--tensor-parallel-size", "2"]
        self._spec["extraPodSpec"]["mainContainer"]["args"] = parts

        # Auto-adjust GPU count to match tensor parallel size
        self.gpus = value

    # ----- Readiness Probe -----
    def set_readiness_probe(
        self,
        period_seconds: int = 10,
        initial_delay_seconds: int = 0,
        timeout_seconds: int = 4,
        failure_threshold: int = 3,
        path: str = "/health",
        port: int = 9090,
    ):
        """Set readiness probe configuration for this service.

        Args:
            period_seconds: How often to perform the probe (default: 10)
            initial_delay_seconds: Delay before first probe (default: 0)
            timeout_seconds: Probe timeout (default: 4)
            failure_threshold: Failures before marking unready (default: 3)
            path: HTTP path to probe (default: "/health")
            port: Port to probe (default: 9090 for workers, use 8000 for frontend)
        """
        self._spec["readinessProbe"] = {
            "httpGet": {
                "path": path,
                "port": port,
            },
            "periodSeconds": period_seconds,
            "initialDelaySeconds": initial_delay_seconds,
            "timeoutSeconds": timeout_seconds,
            "failureThreshold": failure_threshold,
        }

    # ----- Termination Grace Period -----
    def set_termination_grace_period(self, seconds: int = 60):
        """Set termination grace period for this service's pods.

        Args:
            seconds: Grace period in seconds (default: 60)
        """
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {}
        self._spec["extraPodSpec"]["terminationGracePeriodSeconds"] = seconds

    # ----- Log Collection -----
    def enable_log_collection(self, log_dir: str, pvc_name: str):
        """Wrap service command to log output to PVC.

        Args:
            log_dir: Container directory for log files (e.g., "/tmp/service_logs")
            pvc_name: Name of the PVC to mount for log storage
        """
        # Check if already wrapped to avoid double wrapping
        main_container = self._spec.get("extraPodSpec", {}).get("mainContainer", {})
        existing_command = main_container.get("command", [])
        if (
            len(existing_command) >= 3
            and existing_command[:2] == ["/bin/bash", "-c"]
            and "tee -a" in existing_command[2]
        ):
            return  # Already wrapped

        # Ensure extraPodSpec exists
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in self._spec["extraPodSpec"]:
            self._spec["extraPodSpec"]["mainContainer"] = {}

        main_container = self._spec["extraPodSpec"]["mainContainer"]

        # Get original command and args
        original_command = main_container.get("command", [])
        original_args = main_container.get("args", [])

        # Use defaults if not explicitly set
        if not original_command and not original_args:
            if self.component_type == "frontend":
                original_command = ["python3"]
                original_args = ["-m", "dynamo.frontend"]
            else:
                original_command = ["python3"]
                original_args = []

        # Build the full command string
        full_command = " ".join(original_command + original_args)

        # Create service subdirectory (lowercase service name)
        # Note: CRD doesn't support subPath in volumeMounts, so we create the
        # service_logs/ prefix in the wrapper script to match the download job's subPath
        service_subdir = self._name.lower()
        service_log_dir = f"{log_dir}/service_logs/{service_subdir}"

        # Create simplified wrapper script (no header)
        wrapper_script = f"""#!/bin/bash
set -e
mkdir -p {service_log_dir}
LOG_FILE="{service_log_dir}/${{POD_NAME}}_$(date +%s).log"
exec {full_command} > >(tee -a "$LOG_FILE") 2>&1"""

        # Set the wrapped command
        main_container["command"] = ["/bin/bash", "-c", wrapper_script]

        # Remove args since we're now using a shell script
        if "args" in main_container:
            del main_container["args"]

        # Add volume mount at service level (no subPath - CRD doesn't support it)
        # The wrapper script creates service_logs/{service}/ structure on the PVC
        if "volumeMounts" not in self._spec:
            self._spec["volumeMounts"] = []

        # Check if mount already exists
        mount_exists = any(
            mount.get("name") == pvc_name for mount in self._spec["volumeMounts"]
        )
        if not mount_exists:
            self._spec["volumeMounts"].append({"name": pvc_name, "mountPoint": log_dir})

        # Add POD_NAME env var (for log file naming via downward API)
        if "envs" not in self._spec:
            self._spec["envs"] = []

        pod_name_exists = any(
            env.get("name") == "POD_NAME" for env in self._spec["envs"]
        )
        if not pod_name_exists:
            self._spec["envs"].append(
                {
                    "name": "POD_NAME",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                }
            )

        # Add POD_NAMESPACE env var
        namespace_exists = any(
            env.get("name") == "POD_NAMESPACE" for env in self._spec["envs"]
        )
        if not namespace_exists:
            self._spec["envs"].append(
                {
                    "name": "POD_NAMESPACE",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
                }
            )

    # ----- Environment Variables -----
    def set_env_var(self, name: str, value: str):
        """Set an environment variable on this service.

        If the environment variable already exists, update its value.
        Otherwise, add a new environment variable.

        Args:
            name: Name of the environment variable
            value: Value of the environment variable
        """
        envs = self.envs or []
        for env in envs:
            if env.get("name") == name:
                env["value"] = value
                self.envs = envs
                return
        envs.append({"name": name, "value": value})
        self.envs = envs


class DeploymentSpec:
    def __init__(
        self, base: str, endpoint="/v1/chat/completions", port=8000, system_port=9090
    ):
        """Load the deployment YAML file"""
        with open(base, "r") as f:
            self._deployment_spec = yaml.safe_load(f)
        self._endpoint = endpoint
        self._port = port
        self._system_port = system_port

    @property
    def name(self) -> str:
        """Deployment name"""
        return self._deployment_spec["metadata"]["name"]

    @name.setter
    def name(self, value: str):
        self._deployment_spec["metadata"]["name"] = value

    @property
    def port(self) -> int:
        """Deployment port"""
        return self._port

    @property
    def system_port(self) -> int:
        """Deployment port"""
        return self._system_port

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def frontend_service(self) -> ServiceSpec:
        for service in self.services:
            if service.component_type == "frontend":
                return service

    @property
    def namespace(self) -> str:
        """Deployment namespace"""
        return self._deployment_spec["metadata"]["namespace"]

    @namespace.setter
    def namespace(self, value: str):
        self._deployment_spec["metadata"]["namespace"] = value

    def get_in_cluster_frontend_url(self, namespace: str) -> str:
        """Compute the in-cluster URL for the frontend service.

        Args:
            namespace: The Kubernetes namespace where the deployment is running

        Returns:
            The fully qualified in-cluster URL for the frontend service
        """
        return (
            f"http://{self.name.lower()}-"
            f"{self.frontend_service.name.lower()}."
            f"{namespace.lower()}.svc.cluster.local:{self.port}"
        )

    def disable_grove(self):
        if "annotations" not in self._deployment_spec["metadata"]:
            self._deployment_spec["metadata"]["annotations"] = {}
        self._deployment_spec["metadata"]["annotations"][
            "nvidia.com/enable-grove"
        ] = "false"

    def get_model(self, service_name: Optional[str] = None) -> Optional[str]:
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            if service.model:
                return service.model
        return None

    def set_model(self, model: str, service_name: Optional[str] = None):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.model = model

    def set_image(self, image: str, service_name: Optional[str] = None):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.image = image

    def set_tensor_parallel(self, tp_size: int, service_names: Optional[list] = None):
        """Scale deployment for different tensor parallel configurations

        Args:
            tp_size: Target tensor parallel size
            service_names: List of service names to update (defaults to worker services)
        """
        if service_names is None:
            # Auto-detect worker services (services with GPU requirements)
            service_names = [svc.name for svc in self.services if svc.gpus > 0]

        for service_name in service_names:
            service = self[service_name]
            service.tensor_parallel_size = tp_size
            service.gpus = tp_size

    def set_logging(self, enable_jsonl: bool = True, log_level: str = "debug"):
        """Configure logging for the deployment

        Args:
            enable_jsonl: Enable JSON line logging (sets DYN_LOGGING_JSONL=true)
            log_level: Set log level (sets DYN_LOG to specified level)
        """
        spec = self._deployment_spec
        if "envs" not in spec["spec"]:
            spec["spec"]["envs"] = []

        # Remove any existing logging env vars to avoid duplicates
        spec["spec"]["envs"] = [
            env
            for env in spec["spec"]["envs"]
            if env.get("name") not in ["DYN_LOGGING_JSONL", "DYN_LOG"]
        ]

        if enable_jsonl:
            spec["spec"]["envs"].append({"name": "DYN_LOGGING_JSONL", "value": "true"})

        if log_level:
            spec["spec"]["envs"].append({"name": "DYN_LOG", "value": log_level})

    def get_logging_config(self) -> dict:
        """Get current logging configuration

        Returns:
            dict with 'jsonl_enabled' and 'log_level' keys
        """
        envs = self._deployment_spec.get("spec", {}).get("envs", [])

        jsonl_enabled = False
        log_level = None

        for env in envs:
            if env.get("name") == "DYN_LOGGING_JSONL":
                jsonl_enabled = env.get("value") in ["true", "1"]
            elif env.get("name") == "DYN_LOG":
                log_level = env.get("value")

        return {"jsonl_enabled": jsonl_enabled, "log_level": log_level}

    @property
    def services(self) -> list[ServiceSpec]:
        """List of ServiceSpec objects"""
        return [
            ServiceSpec(svc, spec)
            for svc, spec in self._deployment_spec["spec"]["services"].items()
        ]

    def __getitem__(self, service_name: str) -> ServiceSpec:
        """Allow dict-like access: d['Frontend']"""
        return ServiceSpec(
            service_name, self._deployment_spec["spec"]["services"][service_name]
        )

    def spec(self):
        return self._deployment_spec

    def add_arg_to_service(self, service_name: str, arg_name: str, arg_value: str):
        """
        Add or override a command-line argument for a specific service

        Args:
            service_name: Name of the service (e.g., "VllmDecodeWorker", "TRTLLMWorker")
            arg_name: Argument name (e.g., "--max-model-len", "--max-seq-len")
            arg_value: Argument value (e.g., "1024")
        """
        service = self.get_service(service_name)
        service_spec = service._spec

        # Ensure args list exists
        if "extraPodSpec" not in service_spec:
            service_spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in service_spec["extraPodSpec"]:
            service_spec["extraPodSpec"]["mainContainer"] = {}
        if "args" not in service_spec["extraPodSpec"]["mainContainer"]:
            service_spec["extraPodSpec"]["mainContainer"]["args"] = []

        args_list = service_spec["extraPodSpec"]["mainContainer"]["args"]

        # Convert to list if needed (sometimes it's a single string)
        if isinstance(args_list, str):
            import shlex

            args_list = shlex.split(args_list)
            service_spec["extraPodSpec"]["mainContainer"]["args"] = args_list

        # Find existing argument
        arg_index = None
        for i, arg in enumerate(args_list):
            if arg == arg_name:
                arg_index = i
                break

        if arg_index is not None:
            # Argument found, check if it has a value
            if arg_index + 1 < len(args_list) and not args_list[
                arg_index + 1
            ].startswith("-"):
                # Has a value, replace it
                args_list[arg_index + 1] = arg_value
            else:
                # No value after the argument, insert the value
                args_list.insert(arg_index + 1, arg_value)
        else:
            # Add new argument
            args_list.extend([arg_name, arg_value])

    def get_service(self, service_name: str) -> ServiceSpec:
        """
        Get a specific service from the deployment spec
        """
        if service_name not in self._deployment_spec["spec"]["services"]:
            raise ValueError(f"Service '{service_name}' not found in deployment spec")

        return ServiceSpec(
            service_name, self._deployment_spec["spec"]["services"][service_name]
        )

    def set_service_replicas(self, service_name: str, replicas: int):
        """
        Set the number of replicas for a specific service
        """
        service = self.get_service(service_name)
        service.replicas = replicas

    def set_service_readiness_probe(
        self, service_name: str, period_seconds: int, **kwargs
    ):
        """Set readiness probe for a specific service.

        Args:
            service_name: Name of the service (e.g., "TRTLLMDecodeWorker")
            period_seconds: How often to perform the probe
            **kwargs: Additional probe options (initial_delay_seconds, timeout_seconds, etc.)
        """
        service = self.get_service(service_name)
        service.set_readiness_probe(period_seconds=period_seconds, **kwargs)

    def set_service_termination_grace_period(self, service_name: str, seconds: int):
        """Set termination grace period for a specific service.

        Args:
            service_name: Name of the service (e.g., "TRTLLMDecodeWorker")
            seconds: Grace period in seconds
        """
        service = self.get_service(service_name)
        service.set_termination_grace_period(seconds)

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)

    def enable_log_collection(
        self,
        pvc_name=None,
        pvc_size="1Gi",
        storage_class=None,
        container_log_dir="/tmp/service_logs",
        enable_all_services=True,
        service_names=None,
    ):
        """
        Enable log collection using a PersistentVolumeClaim with RWX (ReadWriteMany) access.

        This approach creates a PVC that can be shared across all pods in the deployment,
        allowing reliable log collection that persists through pod restarts and deletions.

        IMPORTANT: Requires a storage class that supports ReadWriteMany (RWX) access mode.
        If the cluster does not support RWX, an error will be raised at deployment time.

        Args:
            pvc_name: Name of the PVC to create (auto-generated if not provided)
            pvc_size: Size of the PVC (e.g., "1Gi", "500Mi")
            storage_class: Storage class name (must support RWX). If None, uses cluster default.
            container_log_dir: Directory inside container where logs are written
            enable_all_services: If True, wrap commands for all services
            service_names: List of specific service names to wrap (used if enable_all_services is False)

        Raises:
            RuntimeError: If the storage class does not support RWX access mode
        """
        if enable_all_services:
            target_services = self.services
        else:
            target_services = [self[name] for name in (service_names or [])]

        # Generate unique PVC name if not provided - do this once and store it
        if pvc_name is None:
            import random
            import time

            # Use timestamp + random to ensure uniqueness even if multiple deployments start simultaneously
            timestamp = int(time.time())
            rand_suffix = random.randint(1000, 9999)
            pvc_name = f"{self.name}-logs-{timestamp}-{rand_suffix}"

        # Store PVC info for later use by ManagedDeployment
        self._log_collection_pvc_name = pvc_name
        self._log_collection_pvc_size = pvc_size
        self._log_collection_storage_class = storage_class
        self._log_collection_container_dir = container_log_dir

        # Debug: Log the generated PVC name for consistency verification
        print(f"[DEBUG] Generated PVC name: {pvc_name}")

        # Add PVC at deployment level (following recipe pattern)
        if "pvcs" not in self._deployment_spec["spec"]:
            self._deployment_spec["spec"]["pvcs"] = []

        # Remove any existing log PVCs to avoid conflicts
        self._deployment_spec["spec"]["pvcs"] = [
            pvc
            for pvc in self._deployment_spec["spec"]["pvcs"]
            if not pvc.get("name", "").endswith("-logs-pvc")
            and pvc.get("name") != pvc_name
        ]

        # Add the log collection PVC (will be created by ManagedDeployment)
        self._deployment_spec["spec"]["pvcs"].append(
            {
                "name": pvc_name,
                "create": False,  # PVC will be created manually by ManagedDeployment
            }
        )

        # Enable log collection for all target services
        for service in target_services:
            service.enable_log_collection(container_log_dir, pvc_name)


class PodProcess:
    def __init__(self, pod: Pod, line: str):
        self.pid = int(re.split(r"\s+", line)[1])
        self.command = " ".join(
            re.split(r"\s+", line)[10:]
        )  # Columns 10+ are the command
        self._pod = pod

    def kill(self, signal=None):
        """Kill this process in the given pod"""

        if not signal:
            if self.pid == 1:
                signal = "SIGINT"
            else:
                signal = "SIGKILL"
        # Python processes need signal handlers for graceful shutdown
        if self.pid == 1 and signal == "SIGKILL" and "python" in self.command.lower():
            logging.info(
                f"PID 1 is a Python process ({self.command[:50]}...), "
                "changing SIGKILL to SIGINT for graceful shutdown"
            )
            signal = "SIGINT"

        logging.info("Killing PID %s with %s", self.pid, signal)

        return self._pod.exec(["kill", f"-{signal}", str(self.pid)])

    def wait(self, timeout: int = 60):
        """Wait for this process to exit in the given pod"""
        # Simple implementation; adjust as needed
        for _ in range(timeout):
            try:
                result = self._pod.exec(
                    ["kill", "-0", str(self.pid)]
                )  # Check if process exists
                if result.returncode != 0:
                    return True  # Process exited
                time.sleep(1)
            except Exception:
                return True
        return False  # Timed out


@dataclass
class ManagedDeployment:
    log_dir: str
    deployment_spec: DeploymentSpec
    namespace: str
    skip_service_restart: bool = False
    enable_volume_log_collection: bool = (
        False  # PVC-based log collection (requires RWX storage class)
    )
    container_log_dir: str = "/tmp/service_logs"  # Directory for volume logs

    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    _deployment_name: Optional[str] = None
    _apps_v1: Optional[Any] = None
    _active_port_forwards: List[Any] = field(default_factory=list)

    # PVC-based log collection
    _log_collection_pvc_created: bool = field(
        default=False, init=False
    )  # Track if we created a PVC
    _log_collection_pvc_verified: bool = field(
        default=False, init=False
    )  # Track if PVC was successfully verified/bound

    # Resource monitoring
    _resource_config: Optional[ResourceMonitorConfig] = field(default=None, init=False)
    _resource_history: List[ResourceSnapshot] = field(default_factory=list, init=False)
    _monitoring_task: Optional[asyncio.Task] = field(default=None, init=False)
    _prev_cpu_stats: Dict[str, dict] = field(
        default_factory=dict, init=False
    )  # Per-pod CPU stats for delta calculation

    @property
    def frontend_service_name(self):
        return self.deployment_spec.frontend_service.name

    def get_log_pvc_name(self) -> Optional[str]:
        """Return the PVC name if log collection is enabled.

        This allows ManagedLoad to use the same PVC for storing load test results.

        Returns:
            The PVC name if log collection is enabled, None otherwise.
        """
        return getattr(self.deployment_spec, "_log_collection_pvc_name", None)

    def _load_template(self, template_name: str) -> str:
        """Load a template file from the templates directory."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        template_path = os.path.join(template_dir, template_name)
        with open(template_path, "r") as f:
            return f.read()

    def __post_init__(self):
        self._deployment_name = self.deployment_spec.name

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
            self._in_cluster = True
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()
        k8s_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(k8s_client)
        self._core_api = client.CoreV1Api(k8s_client)
        self._apps_v1 = client.AppsV1Api()

    async def _wait_for_pods(self, label, expected, timeout=300):
        for _ in range(timeout):
            assert self._core_api is not None, "Kubernetes API not initialized"
            pods = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label
            )
            running = sum(
                1
                for pod in pods.items
                if any(
                    cond.type == "Ready" and cond.status == "True"
                    for cond in (pod.status.conditions or [])
                )
            )
            if running == expected:
                return True
            await asyncio.sleep(1)
        raise Exception(f"Didn't Reach Expected Pod Count {label}=={expected}")

    async def _scale_statfulset(self, name, label, replicas):
        body = {"spec": {"replicas": replicas}}
        assert self._apps_v1 is not None, "Kubernetes API not initialized"
        await self._apps_v1.patch_namespaced_stateful_set_scale(
            name, self.namespace, body
        )
        await self._wait_for_pods(label, replicas)

    async def _restart_stateful(self, name, label):
        self._logger.info(f"Restarting {name} {label}")

        await self._scale_statfulset(name, label, 0)
        assert self._core_api is not None, "Kubernetes API not initialized"
        nats_pvc = await self._core_api.list_namespaced_persistent_volume_claim(
            self.namespace, label_selector=label
        )
        for pvc in nats_pvc.items:
            await self._core_api.delete_namespaced_persistent_volume_claim(
                pvc.metadata.name, self.namespace
            )

        await self._scale_statfulset(name, label, 1)

        self._logger.info(f"Restarted {name} {label}")

    async def wait_for_unready(self, timeout: int = 1800, sleep=1, log_interval=60):
        """
        Wait for the custom resource to be unready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        return await self._wait_for_condition(
            timeout, sleep, log_interval, False, "pending"
        )

    async def wait_for_ready(self, timeout: int = 1800, sleep=1, log_interval=60):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        return await self._wait_for_condition(
            timeout, sleep, log_interval, True, "successful"
        )

    async def _wait_for_condition(
        self,
        timeout: int = 1800,
        sleep=1,
        log_interval=60,
        desired_ready_condition_val: bool = True,
        desired_state_val: str = "successful",
    ):
        start_time = time.time()

        self._logger.info(
            f"Waiting for Deployment {self._deployment_name} to have Ready condition {desired_ready_condition_val} and state {desired_state_val}"
        )

        attempt = 0

        while (time.time() - start_time) < timeout:
            try:
                attempt += 1
                assert self._custom_api is not None, "Kubernetes API not initialized"
                status = await self._custom_api.get_namespaced_custom_object(  # type: ignore[awaitable-is-not-coroutine]
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
                # Check both conditions:
                # 1. Ready condition is True
                # 2. State is successful
                status_obj = status.get("status", {})  # type: ignore[attr-defined]
                conditions = status_obj.get("conditions", [])  # type: ignore[attr-defined]
                current_state = status_obj.get("state", "unknown")  # type: ignore[attr-defined]

                observed_ready_condition_val = ""
                for condition in conditions:
                    if condition.get("type") == "Ready":
                        observed_ready_condition_val = condition.get("status")
                        if observed_ready_condition_val == str(
                            desired_ready_condition_val
                        ):
                            break

                observed_state_val = status_obj.get("state")  # type: ignore[attr-defined]

                if (
                    observed_ready_condition_val == str(desired_ready_condition_val)
                    and observed_state_val == desired_state_val
                ):
                    self._logger.info(f"Current deployment state: {current_state}")
                    self._logger.info(f"Current conditions: {conditions}")
                    self._logger.info(
                        f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                    )

                    self._logger.info(
                        f"Deployment {self._deployment_name} has Ready condition {desired_ready_condition_val} and state {desired_state_val}"
                    )
                    return True
                else:
                    if attempt % log_interval == 0:
                        self._logger.info(f"Current deployment state: {current_state}")
                        self._logger.info(f"Current conditions: {conditions}")
                        self._logger.info(
                            f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                        )
                        self._logger.info(
                            f"Deployment has Ready condition {observed_ready_condition_val} and state {observed_state_val}, desired condition {desired_ready_condition_val} and state {desired_state_val}"
                        )

            except exceptions.ApiException as e:
                self._logger.info(
                    f"API Exception while checking deployment status: {e}"
                )
                self._logger.info(f"Status code: {e.status}, Reason: {e.reason}")
            except Exception as e:
                self._logger.info(
                    f"Unexpected exception while checking deployment status: {e}"
                )
            await asyncio.sleep(sleep)
        raise TimeoutError("Deployment failed to become ready within timeout")

    async def get_deployment_status(self) -> dict:
        """Get current deployment status (state and conditions).

        Returns:
            dict with 'state' and 'conditions' keys
        """
        assert self._custom_api is not None, "Kubernetes API not initialized"
        status = await self._custom_api.get_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.namespace,
            plural="dynamographdeployments",
            name=self._deployment_name,
        )
        status_obj = status.get("status", {})
        return {
            "state": status_obj.get("state"),
            "conditions": status_obj.get("conditions", []),
        }

    async def _restart_nats(self):
        NATS_STS_NAME = "dynamo-platform-nats"
        NATS_LABEL = "app.kubernetes.io/component=nats"

        await self._restart_stateful(NATS_STS_NAME, NATS_LABEL)

    async def _restart_etcd(self):
        ETCD_STS_NAME = "dynamo-platform-etcd"
        ETCD_LABEL = "app.kubernetes.io/component=etcd"

        await self._restart_stateful(ETCD_STS_NAME, ETCD_LABEL)

    async def _create_deployment(self):
        """
        Create a DynamoGraphDeployment from either a dict or yaml file path.

        Args:
            deployment: Either a dict containing the deployment spec or a path to a yaml file
        """

        # Extract service names

        self._services = self.deployment_spec.services

        self._logger.info(
            f"Starting Deployment {self._deployment_name} with spec {self.deployment_spec}"
        )

        try:
            assert self._custom_api is not None, "Kubernetes API not initialized"
            await self._custom_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec.spec(),
            )
            # Save deployment spec to file instead of logging to console
            os.makedirs(self.log_dir, exist_ok=True)
            spec_file = os.path.join(self.log_dir, "deployment_spec.yaml")
            with open(spec_file, "w") as f:
                yaml.dump(self.deployment_spec.spec(), f, default_flow_style=False)
            self._logger.info(f"Deployment spec saved to {spec_file}")
            self._logger.info(f"Deployment Started {self._deployment_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                # Fallback: Replace the existing deployment with the new spec
                # This ensures the new spec (e.g., different replica counts) is applied
                self._logger.info(
                    f"Deployment {self._deployment_name} already exists, replacing with new spec..."
                )
                try:
                    await self._custom_api.replace_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural="dynamographdeployments",
                        name=self._deployment_name,
                        body=self.deployment_spec.spec(),
                    )
                    self._logger.info(
                        f"Deployment {self._deployment_name} replaced successfully"
                    )
                except exceptions.ApiException as replace_e:
                    self._logger.error(
                        f"Failed to replace deployment {self._deployment_name}: {replace_e}"
                    )
                    raise
            else:
                self._logger.info(
                    f"Failed to create deployment {self._deployment_name}: {e}"
                )
                raise

    async def apply_service_changes(self, service_names: list[str]):
        """Apply current service state to K8s by patching the CR.

        This patches the Kubernetes custom resource with the current
        environment variables from the specified services. Use this after
        modifying service state (e.g., via ServiceSpec.set_env_var) to
        apply those changes to the cluster.

        Args:
            service_names: List of service names to apply changes for
        """
        if not service_names:
            raise ValueError("service_names cannot be empty for apply_service_changes")

        patch_body: dict[str, Any] = {"spec": {"services": {}}}

        for service_name in service_names:
            service = self.deployment_spec[service_name]
            patch_body["spec"]["services"][service_name] = {"envs": service.envs}

        try:
            assert self._custom_api is not None, "Kubernetes API not initialized"
            await self._custom_api.patch_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                name=self._deployment_name,
                body=patch_body,
                _content_type="application/merge-patch+json",
            )
        except exceptions.ApiException as e:
            self._logger.info(
                f"Failed to patch deployment {self._deployment_name}: {e}"
            )
            raise

    async def get_pod_names(self, service_names: list[str] | None = None) -> list[str]:
        if not service_names:
            service_names = [service.name for service in self.deployment_spec.services]

        pod_names: list[str] = []

        for service_name in service_names:
            label_selector = (
                f"nvidia.com/selector={self._deployment_name}-{service_name.lower()}"
            )
            assert self._core_api is not None, "Kubernetes API not initialized"
            pods: client.V1PodList = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label_selector
            )
            for pod in pods.items:
                pod_names.append(pod.metadata.name)

        return pod_names

    def get_processes(self, pod: Pod) -> list[PodProcess]:
        """Get list of processes in the given pod"""
        result = pod.exec(["ps", "-aux"])
        lines = result.stdout.decode().splitlines()
        # Skip header line
        processes = [PodProcess(pod, line) for line in lines[1:]]
        return processes

    def get_service(self, service_name=None):
        if not service_name:
            service_name = ""
        full_service_name = f"{self._deployment_name}-{service_name.lower()}"

        return Service.get(full_service_name, namespace=self.namespace)

    def get_pods(self, service_names: list[str] | None = None) -> dict[str, list[Pod]]:
        result: dict[str, list[Pod]] = {}

        if not service_names:
            service_names = [service.name for service in self.deployment_spec.services]

        for service_name in service_names:
            # List pods for this service using the selector label
            # nvidia.com/selector: deployment-name-service
            label_selector = (
                f"nvidia.com/selector={self._deployment_name}-{service_name.lower()}"
            )

            pods: list[Pod] = []

            for pod in kr8s.get(
                "pods", namespace=self.namespace, label_selector=label_selector
            ):
                pods.append(pod)  # type: ignore[arg-type]

            result[service_name] = pods

        return result

    def get_pod_manifest_logs_metrics(self, service_name: str, pod: Pod, suffix=""):
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)

        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.yaml"), "w") as f:
                f.write(pod.to_yaml())
        except Exception as e:
            self._logger.error(e)
        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.log"), "w") as f:
                f.write("\n".join(pod.logs()))
        except Exception as e:
            self._logger.error(e)
        try:
            previous_logs = pod.logs(previous=True)
            with open(
                os.path.join(directory, f"{pod.name}{suffix}.previous.log"), "w"
            ) as f:
                f.write("\n".join(previous_logs))
        except Exception as e:
            self._logger.debug(e)

        self._get_pod_metrics(pod, service_name, suffix)

    def _get_service_logs(self, service_name=None, suffix=""):
        service_names = None
        if service_name:
            service_names = [service_name]

        service_pods = self.get_pods(service_names)

        for service, pods in service_pods.items():
            for pod in pods:
                self.get_pod_manifest_logs_metrics(service, pod, suffix)

    def _get_pod_metrics(
        self, pod: Pod, service_name: str, suffix="", use_services_dir: bool = False
    ):
        # When using PVC-based collection, save to services/{service_lowercase}/
        # to match the PVC log extraction path
        if use_services_dir:
            directory = os.path.join(self.log_dir, "services", service_name.lower())
        else:
            directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)
        port = None
        if service_name == self.frontend_service_name:
            port = self.deployment_spec.port
        else:
            port = self.deployment_spec.system_port

        pf = self.port_forward(pod, port)

        if not pf:
            self._logger.error(f"Unable to get metrics for {service_name}")
            return

        content = None

        try:
            url = f"http://localhost:{pf.local_port}/metrics"

            response = requests.get(url, timeout=30)
            content = None
            try:
                content = response.text
            except ValueError:
                pass

        except Exception as e:
            self._logger.error(str(e))

        if content:
            with open(
                os.path.join(directory, f"{pod.name}.metrics{suffix}.log"), "w"
            ) as f:
                f.write(content)

    def _collect_service_metrics(self, use_services_dir: bool = True):
        """Collect metrics from all services.

        Args:
            use_services_dir: If True, save to services/{service}/ directory
                            to match PVC log extraction path
        """
        self._logger.info("Collecting metrics from all services...")
        service_pods = self.get_pods()

        for service, pods in service_pods.items():
            for pod in pods:
                try:
                    self._get_pod_metrics(
                        pod, service, use_services_dir=use_services_dir
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to collect metrics from {pod.name}: {e}"
                    )

    # =========================================================================
    # Resource Monitoring Methods
    # =========================================================================

    async def _exec_in_pod(
        self, pod: Pod, command: List[str], timeout: float = 10.0
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

    def _get_service_for_pod(self, pod: Pod) -> str:
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
        if selector and self._deployment_name:
            # Extract service name from selector (e.g., "my-deploy-frontend" -> "frontend")
            prefix = f"{self._deployment_name}-"
            if selector.startswith(prefix):
                return selector[len(prefix) :]

        return "unknown"

    async def get_pod_resource_usage(
        self, pod: Pod, service_name: str, include_gpu: bool = True
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
            # Use config if monitoring is active, otherwise use the include_gpu parameter
            should_include_gpu = include_gpu
            if self._resource_config is not None:
                should_include_gpu = self._resource_config.include_gpu

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

        service_pods = self.get_pods()

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

    async def start_resource_monitoring(
        self, config: Optional[ResourceMonitorConfig] = None
    ):
        """Start background resource monitoring task.

        Args:
            config: Monitoring configuration (uses defaults if not provided)
        """
        self._resource_config = config or ResourceMonitorConfig(enabled=True)
        self._resource_history = []
        self._prev_cpu_stats = {}

        self._monitoring_task = asyncio.create_task(self._resource_monitoring_loop())
        self._logger.info(
            f"Started resource monitoring (interval={self._resource_config.interval_seconds}s, "
            f"gpu={self._resource_config.include_gpu}, write_to_pvc={self._resource_config.write_to_pvc})"
        )

    async def stop_resource_monitoring(self) -> List[ResourceSnapshot]:
        """Stop background monitoring and return collected history.

        Returns:
            List of all collected ResourceSnapshots
        """
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        self._logger.info(
            f"Stopped resource monitoring, collected {len(self._resource_history)} snapshots"
        )
        return self._resource_history

    async def _resource_monitoring_loop(self):
        """Background loop that collects resource metrics periodically."""
        while True:
            try:
                snapshots = await self.get_all_resource_usage()

                # Flatten and store snapshots
                for service_snapshots in snapshots.values():
                    self._resource_history.extend(service_snapshots)

                    # Write to PVC if enabled
                    if (
                        self._resource_config
                        and self._resource_config.write_to_pvc
                        and service_snapshots
                    ):
                        await self._write_metrics_to_pvc(service_snapshots)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._logger.warning(f"Resource monitoring error: {e}")

            await asyncio.sleep(self._resource_config.interval_seconds)

    async def _write_metrics_to_pvc(self, snapshots: List[ResourceSnapshot]):
        """Write metrics to PVC via exec into a pod.

        Args:
            snapshots: List of snapshots to write as JSONL
        """
        if not self._resource_config:
            return

        # Find any running pod to write metrics
        service_pods = self.get_pods()
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
                                f"echo '{escaped_json}' >> {self._resource_config.pvc_output_path}",
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

    async def _save_resource_history_locally(self):
        """Save resource monitoring history to local log directory."""
        if not self._resource_history:
            return

        try:
            metrics_file = os.path.join(self.log_dir, "resource_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump([asdict(s) for s in self._resource_history], f, indent=2)
            self._logger.info(
                f"Saved {len(self._resource_history)} resource snapshots to {metrics_file}"
            )
        except Exception as e:
            self._logger.warning(f"Failed to save resource history: {e}")

    async def _delete_deployment(self):
        """
        Delete the DynamoGraphDeployment CR.
        """
        try:
            if self._deployment_name and self._custom_api is not None:
                await self._custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
        except exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

    async def _wait_for_deletion(self, timeout: int = 120):
        """
        Wait for the deployment CR to be fully deleted from Kubernetes.

        This prevents a race condition where _create_deployment() is called
        before the previous deployment is fully removed, resulting in a 409
        error and the new deployment spec never being applied.

        Args:
            timeout: Maximum time to wait in seconds (default 120s)
        """
        if not self._deployment_name or self._custom_api is None:
            return

        start_time = time.time()
        self._logger.info(
            f"Waiting for deployment {self._deployment_name} to be fully deleted..."
        )

        while (time.time() - start_time) < timeout:
            try:
                await self._custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
                # CR still exists, wait
                await asyncio.sleep(2)
            except exceptions.ApiException as e:
                if e.status == 404:
                    self._logger.info(
                        f"Deployment {self._deployment_name} fully deleted"
                    )
                    return
                raise

        self._logger.warning(
            f"Deployment {self._deployment_name} deletion timed out after {timeout}s, "
            "proceeding anyway (will attempt to patch if exists)"
        )

    def port_forward(
        self, pod: Pod, remote_port: int, max_connection_attempts: int = 3
    ):
        """Attempt to connect to a pod and return the port-forward object on success.

        Note: Port forwards run in background threads. When pods are terminated,
        the async cleanup may fail, which is expected and can be safely ignored.
        """
        try:
            # Create port forward - this runs in a background thread
            # Use 127.0.0.1 (localhost) instead of 0.0.0.0 to prevent port conflicts
            port_forward = pod.portforward(
                remote_port=remote_port,
                local_port=0,  # Auto-assign an available port
                address="127.0.0.1",  # Use localhost for better isolation and conflict prevention
            )
            port_forward.start()

            # Try to connect with exponential backoff
            backoff_delay = 0.5  # Start with 500ms

            for attempt in range(max_connection_attempts):
                time.sleep(backoff_delay)
                backoff_delay = min(
                    backoff_delay * 1.5, 5.0
                )  # Double delay, max 5 seconds

                # Check if port is assigned
                if port_forward.local_port == 0:
                    self._logger.debug(
                        f"Port not yet assigned for pod {pod.name} (attempt {attempt+1}/{max_connection_attempts})"
                    )
                    continue

                # Try to connect to the port forwarded service
                test_url = f"http://localhost:{port_forward.local_port}/"
                try:
                    # Send HEAD request to test connection
                    response = requests.head(test_url, timeout=5)
                    if response.status_code in (200, 404):  # 404 is acceptable
                        self._active_port_forwards.append(port_forward)
                        return port_forward
                except (requests.ConnectionError, requests.Timeout) as e:
                    self._logger.warning(
                        f"Connection test failed for pod {pod.name} (attempt {attempt+1}/{max_connection_attempts}): {e}"
                    )

                # Restart port-forward for next attempt (except on last attempt)
                if attempt == max_connection_attempts - 1:
                    continue
                try:
                    port_forward.stop()
                    port_forward.start()
                except Exception as e:
                    self._logger.debug(
                        f"Error restarting port forward for pod {pod.name}: {e}"
                    )
                    break

            # All attempts failed
            self._logger.warning(
                f"Port forward failed after {max_connection_attempts} attempts for pod {pod.name}"
            )
            try:
                port_forward.stop()
            except Exception:
                pass  # Ignore errors during cleanup
            return None

        except Exception as e:
            self._logger.warning(
                f"Failed to create port forward for pod {pod.name}: {e}"
            )
            return None

    async def _cleanup(self):
        try:
            # Collect logs via K8s API only if PVC-based collection is not enabled
            if not self.enable_volume_log_collection:
                self._get_service_logs()
            else:
                # When using PVC-based collection, still collect metrics
                # (metrics are fetched via HTTP, not from PVC)
                self._collect_service_metrics(use_services_dir=True)

            # Stop port forwards
            self._logger.info(
                f"Cleaning up {len(self._active_port_forwards)} active port forwards"
            )
            for port_forward in self._active_port_forwards:
                try:
                    port_forward.stop()
                except RuntimeError as e:
                    if "anext()" in str(e) or "already running" in str(e):
                        self._logger.debug(f"Port forward cleanup: {e}")
                    else:
                        self._logger.warning(
                            f"Unexpected error stopping port forward: {e}"
                        )
                except Exception as e:
                    self._logger.debug(f"Error stopping port forward: {e}")
            self._active_port_forwards.clear()
        finally:
            # Clean up all resources (deployment, then extract logs from PVC, then delete PVC)
            await self._cleanup_all_resources()

    async def __aenter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self.deployment_spec.namespace = self.namespace
            self._deployment_name = self.deployment_spec.name
            logging.getLogger("httpx").setLevel(logging.WARNING)
            await self._init_kubernetes()

            # Run delete deployment and service restarts in parallel
            tasks = [self._delete_deployment()]
            if not self.skip_service_restart:
                tasks.extend([self._restart_etcd(), self._restart_nats()])
            await asyncio.gather(*tasks)

            # Wait for CR to be fully deleted to avoid race condition
            # where _create_deployment() gets 409 (Already exists) and
            # the new deployment spec is never applied
            await self._wait_for_deletion()

            # Enable PVC-based log collection before deployment creation
            if self.enable_volume_log_collection:
                # Check if PVC-based logging is already configured (by enable_log_collection)
                pvc_configured = hasattr(
                    self.deployment_spec, "_log_collection_pvc_name"
                )

                if pvc_configured:
                    self._logger.info(
                        f"PVC-based log collection configured: {self.deployment_spec._log_collection_pvc_name}"
                    )
                    # Validate RWX support and create PVC
                    await self._create_log_collection_pvc()
                else:
                    # Auto-configure PVC-based logging with defaults
                    self._logger.info("Auto-configuring PVC-based log collection")
                    self.deployment_spec.enable_log_collection(
                        container_log_dir=self.container_log_dir,
                        enable_all_services=True,
                    )
                    await self._create_log_collection_pvc()

            await self._create_deployment()
            await self.wait_for_ready()
            # Note: Download job will be created during cleanup after pods exit

        except:
            await self._cleanup()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Log if we're exiting due to an exception (Ctrl-C, etc.)
        if exc_type is not None:
            self._logger.warning(
                f"Exiting due to exception ({exc_type.__name__}), running cleanup"
            )
        # Always run cleanup, catching any cleanup errors to ensure we don't mask original exception
        try:
            await self._cleanup()
        except Exception as cleanup_error:
            self._logger.error(f"Error during cleanup: {cleanup_error}")

    async def create_log_download_job(
        self,
        local_output_dir: str,
        container_log_dir="/tmp/service_logs",
        job_name=None,
        download_timeout=300,
    ):
        """
        Create a Kubernetes job to download log files from the PVC-based service-logs volume.

        This job will:
        1. Mount the same PVC as the deployment services
        2. Create a tar archive of all log files
        3. Keep the job pod alive for extraction (similar to ManagedAIPerfDeployment pattern)

        Args:
            local_output_dir: Local directory to save the downloaded logs
            container_log_dir: Container directory where logs are stored (should match enable_log_collection)
            job_name: Optional custom job name (defaults to deployment-name-log-download)
            download_timeout: Timeout in seconds for the download job

        Returns:
            dict: Information about the created job
        """
        if not self._custom_api:
            raise RuntimeError(
                "Kubernetes API not initialized. Call _init_kubernetes() first."
            )

        # Generate job name
        if not job_name:
            job_name = (
                f"{self.deployment_spec.name}-log-download-{secrets.token_hex(4)}"
            )

        os.makedirs(local_output_dir, exist_ok=True)

        # Get PVC name
        pvc_name = self._get_download_job_volume_config()["persistentVolumeClaim"][
            "claimName"
        ]

        # Check if PVC exists and has RWX access mode before creating download job
        self._logger.info(
            f"Checking if PVC {pvc_name} exists before creating download job..."
        )
        try:
            pvc = await self._core_api.read_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            access_modes = pvc.spec.access_modes or []
            storage_class = pvc.spec.storage_class_name
            capacity = (
                pvc.status.capacity.get("storage", "unknown")
                if pvc.status.capacity
                else "unknown"
            )
            phase = pvc.status.phase

            self._logger.info(
                f"PVC {pvc_name} found: phase={phase}, access_modes={access_modes}, "
                f"storage_class={storage_class}, capacity={capacity}"
            )

            # Check for ReadWriteMany (RWX) access mode
            has_rwx = "ReadWriteMany" in access_modes
            if not has_rwx:
                self._logger.warning(
                    f"PVC {pvc_name} does not have ReadWriteMany access mode "
                    f"(has: {access_modes}). Download job may fail if other pods are using the PVC."
                )

            self._logger.info(
                f"PVC {pvc_name} exists, proceeding with download job creation"
            )
        except exceptions.ApiException as e:
            if e.status == 404:
                self._logger.warning(
                    f"PVC {pvc_name} does not exist - skipping download job creation"
                )
                return {
                    "success": False,
                    "error": f"PVC {pvc_name} does not exist",
                    "job_name": None,
                }
            else:
                self._logger.warning(f"Error checking PVC {pvc_name}: {e}")

        # Load and render template
        template = self._load_template("log_download_job.yaml")
        template = template.replace("TEMPLATE_JOB_NAME", job_name)
        template = template.replace("TEMPLATE_NAMESPACE", self.namespace)
        template = template.replace(
            "TEMPLATE_DEPLOYMENT_NAME", self.deployment_spec.name
        )
        template = template.replace("TEMPLATE_CONTAINER_LOG_DIR", container_log_dir)
        template = template.replace("TEMPLATE_PVC_NAME", pvc_name)

        # Parse the rendered template
        job_spec = yaml.safe_load(template)

        # Create the job using BatchV1Api
        try:
            batch_api = client.BatchV1Api()
            await batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._logger.info(f"Log download job created: {job_name}")

            return {
                "success": True,
                "job_name": job_name,
                "namespace": self.namespace,
                "local_output_dir": local_output_dir,
            }

        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.warning(f"Job {job_name} already exists")
                return {
                    "success": True,
                    "job_name": job_name,
                    "namespace": self.namespace,
                    "local_output_dir": local_output_dir,
                    "note": "job_already_existed",
                }
            else:
                self._logger.error(f"Failed to create job {job_name}: {e}")
                raise

    async def extract_logs_from_download_job(
        self, job_name: str, local_output_dir: str
    ):
        """
        Extract logs from a log download job pod by creating tar on-demand.

        This method creates the tar archive at extraction time (not at job start),
        ensuring all logs up to this point are captured.

        Args:
            job_name: Name of the log download job
            local_output_dir: Local directory to save the extracted logs

        Returns:
            dict: Extraction results including file count and paths
        """
        container_log_dir = self.container_log_dir

        try:
            # Find the job pod
            pods = []
            job_label = f"job-name={job_name}"
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=job_label,
            )
            for pod in pod_generator:
                pods.append(pod)

            if not pods:
                raise Exception(f"No pods found for job {job_name}")

            pod = pods[0]

            # Wait for job to be ready
            self._logger.info("Waiting for log download job to be ready...")
            for attempt in range(60):  # Wait up to 60 seconds
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(
                                pod.exec,
                                ["test", "-f", "/tmp/log_archive/job_ready.txt"],
                            )
                        ),
                        timeout=5.0,
                    )
                    if result.returncode == 0:
                        break
                except Exception:
                    pass

                self._logger.info(
                    f"Waiting for download job to be ready... (attempt {attempt + 1}/60)"
                )
                await asyncio.sleep(1)
            else:
                self._logger.warning(
                    "Download job did not become ready in expected time, proceeding anyway..."
                )

            # Create tar archive ON-DEMAND (captures all logs up to this point)
            # Logs are in service subdirectories: {container_log_dir}/{service_name}/*.log
            # Note: Download job uses subPath: service_logs, so only service logs are visible
            self._logger.info("Creating tar archive of logs on-demand...")
            create_tar_script = f"""
cd {container_log_dir} 2>/dev/null || exit 1
LOG_COUNT=$(find . -name "*.log" -type f | wc -l)
echo "LOG_COUNT:$LOG_COUNT"
if [ "$LOG_COUNT" -gt 0 ]; then
    # Archive preserving directory structure (service subdirs)
    tar -czf /tmp/log_archive/service_logs.tar.gz . 2>/dev/null
    echo "TAR_CREATED:true"
else
    echo "TAR_CREATED:false"
fi
"""
            tar_result = await asyncio.wait_for(
                asyncio.create_task(
                    asyncio.to_thread(pod.exec, ["sh", "-c", create_tar_script])
                ),
                timeout=30.0,
            )

            # Parse the output to get log count
            output = tar_result.stdout.decode() if tar_result.stdout else ""
            log_count = 0
            tar_created = False
            for line in output.split("\n"):
                if line.startswith("LOG_COUNT:"):
                    log_count = int(line.split(":")[1])
                elif line.startswith("TAR_CREATED:"):
                    tar_created = line.split(":")[1] == "true"

            self._logger.info(f"Found {log_count} log files, tar_created={tar_created}")

            extracted_files = []

            if log_count > 0 and tar_created:
                # Extract the tar archive
                self._logger.info("Extracting service logs archive...")
                cat_result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(
                            pod.exec, ["cat", "/tmp/log_archive/service_logs.tar.gz"]
                        )
                    ),
                    timeout=60.0,
                )

                if cat_result.returncode != 0:
                    raise Exception(
                        f"Archive extraction failed with return code {cat_result.returncode}"
                    )

                # Save the archive locally
                local_archive = os.path.join(local_output_dir, "service_logs.tar.gz")
                with open(local_archive, "wb") as f:
                    f.write(cat_result.stdout)

                # Extract the archive locally
                import tarfile

                print(local_archive)

                with tarfile.open(local_archive, "r:gz") as tar:
                    tar.extractall(path=local_output_dir, filter="data")
                    extracted_files = tar.getnames()

                # Remove the temporary archive file
                os.remove(local_archive)

                self._logger.info(
                    f"Extracted {len(extracted_files)} log files to {local_output_dir}"
                )

            else:
                self._logger.info("No log files were available for download")

            # Create metadata file locally
            import json

            metadata = {
                "deployment_name": self.deployment_spec.name,
                "namespace": self.namespace,
                "extraction_timestamp": asyncio.get_event_loop().time(),
                "job_name": job_name,
                "log_count": log_count,
                "container_log_dir": container_log_dir,
            }
            metadata_path = os.path.join(local_output_dir, "download_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            extracted_files.append("download_metadata.json")

            return {
                "success": True,
                "extracted_files": extracted_files,
                "log_count": log_count,
                "local_output_dir": local_output_dir,
            }

        except Exception as e:
            print(e)
            self._logger.error(
                f"Failed to extract logs from download job {job_name}: {e}"
            )
            return {
                "success": False,
                "error": str(e),
                "local_output_dir": local_output_dir,
            }

    async def cleanup_log_download_job(self, job_name: str):
        """
        Clean up a log download job and its associated pods.

        Args:
            job_name: Name of the job to clean up
        """
        try:
            # Delete the job with foreground propagation to cascade to pods
            from kubernetes_asyncio.client.models import V1DeleteOptions

            delete_options = V1DeleteOptions(propagation_policy="Foreground")

            batch_api = client.BatchV1Api()
            await batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=delete_options,
            )
            self._logger.info(f"Log download job {job_name} deleted")

        except exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                self._logger.warning(
                    f"Failed to delete log download job {job_name}: {e}"
                )

    async def download_volume_logs_now(self, local_output_dir=None):
        """
        Download logs from volume-based collection immediately.

        This creates a temporary download job, extracts logs, and cleans up the job.
        Useful for downloading logs during test execution (not just at cleanup time).

        Args:
            local_output_dir: Optional local directory to save logs (defaults to log_dir/services_manual)

        Returns:
            dict: Download results
        """
        if not self.enable_volume_log_collection:
            return {"success": False, "error": "Volume log collection is not enabled"}

        if local_output_dir is None:
            local_output_dir = os.path.join(self.log_dir, "services_manual")

        self._logger.info("Downloading volume logs on demand...")

        try:
            # Create a temporary download job
            download_job_result = await self.create_log_download_job(
                local_output_dir=local_output_dir,
                container_log_dir=self.container_log_dir,
            )

            if not download_job_result.get("success"):
                return {"success": False, "error": "Failed to create download job"}

            job_name = download_job_result["job_name"]

            # Extract logs
            result = await self.extract_logs_from_download_job(
                job_name, local_output_dir
            )

            if result["success"]:
                self._logger.info(
                    f"Successfully downloaded {result['log_count']} log files to {local_output_dir}"
                )

            # Cleanup the temporary job
            await self.cleanup_log_download_job(job_name)

            return result

        except Exception as e:
            self._logger.error(f"Failed to download volume logs: {e}")
            return {"success": False, "error": str(e)}

    async def _create_log_collection_pvc(self) -> str:
        """
        Create a PVC for log collection based on the deployment spec configuration.

        Requires a storage class that supports ReadWriteMany (RWX) access mode.

        Returns:
            The name of the created PVC

        Raises:
            RuntimeError: If the storage class does not support RWX access mode
        """
        pvc_name = getattr(
            self.deployment_spec, "_log_collection_pvc_name", "dynamo-logs-pvc"
        )
        pvc_size = getattr(self.deployment_spec, "_log_collection_pvc_size", "1Gi")
        storage_class = getattr(
            self.deployment_spec, "_log_collection_storage_class", "nebius-shared-fs"
        )

        self._logger.info(
            f"Creating PVC {pvc_name} with storage class {storage_class} (RWX)"
        )

        # Check if PVC already exists and delete it to start fresh
        try:
            await self._core_api.read_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            self._logger.info(
                f"PVC {pvc_name} already exists, deleting it to start fresh"
            )
            await self._core_api.delete_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            # Wait a moment for deletion to complete
            await asyncio.sleep(2)
            self._logger.info(f"Deleted existing PVC {pvc_name}")
        except client.ApiException as e:
            if e.status != 404:  # Not found is expected, other errors are issues
                self._logger.warning(
                    f"Error checking/deleting existing PVC {pvc_name}: {e}"
                )
            else:
                self._logger.info(f"PVC {pvc_name} does not exist, will create new one")

        # Create PVC with RWX access mode
        pvc_spec = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": pvc_name,
                "namespace": self.namespace,
                "labels": {
                    "managed-by": "managed-deployment",
                    "deployment": self.deployment_spec.name,
                    "purpose": "log-collection",
                    "app": f"{self.deployment_spec.name}-logs",
                },
            },
            "spec": {
                "accessModes": [
                    "ReadWriteMany"
                ],  # RWX required for multi-pod log collection
                "resources": {"requests": {"storage": pvc_size}},
            },
        }

        # Only set storageClassName if explicitly provided (None uses cluster default)
        if storage_class:
            pvc_spec["spec"]["storageClassName"] = storage_class

        try:
            await self._core_api.create_namespaced_persistent_volume_claim(
                namespace=self.namespace, body=pvc_spec
            )
            self._log_collection_pvc_created = True
            self._logger.info(f"Created PVC {pvc_name} for log collection ({pvc_size})")

            # Verify PVC can be bound by creating a dummy job that mounts it
            await self._verify_pvc_binding(pvc_name, timeout=60)

            return pvc_name

        except client.ApiException as e:
            self._logger.error(f"Failed to create PVC {pvc_name}: {e}")
            raise

    async def _verify_pvc_binding(self, pvc_name: str, timeout: int = 60):
        """Verify PVC can be bound by creating a dummy job that mounts it.

        This ensures the storage class supports RWX before proceeding with deployment.
        If the PVC can't be bound (e.g., storage class doesn't support RWX), this
        will fail fast rather than leaving pods in Pending state.

        Args:
            pvc_name: Name of the PVC to verify
            timeout: Maximum time to wait for binding in seconds

        Raises:
            RuntimeError: If PVC cannot be bound within timeout
        """
        job_name = f"{pvc_name}-verify-{secrets.token_hex(4)}"
        storage_class = getattr(
            self.deployment_spec, "_log_collection_storage_class", "unknown"
        )
        binding_issue_logged = False
        self._logger.info(
            f"Verifying PVC {pvc_name} can be bound (storage class: {storage_class})..."
        )

        # Create a minimal job that just mounts the PVC and exits
        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
            },
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 60,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "verify",
                                "image": "busybox:1.35",
                                "command": [
                                    "sh",
                                    "-c",
                                    "echo 'PVC mounted successfully' && ls -la /mnt",
                                ],
                                "volumeMounts": [
                                    {"name": "test-volume", "mountPath": "/mnt"}
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "test-volume",
                                "persistentVolumeClaim": {"claimName": pvc_name},
                            }
                        ],
                    }
                },
            },
        }

        try:
            # Create the verification job
            batch_api = client.BatchV1Api()
            await batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._logger.info(f"Created PVC verification job: {job_name}")

            # Wait for job to complete (pod started = PVC bound)
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                try:
                    job = await batch_api.read_namespaced_job(
                        name=job_name, namespace=self.namespace
                    )

                    # Check if job succeeded
                    if job.status.succeeded and job.status.succeeded > 0:
                        self._logger.info(
                            f"PVC {pvc_name} verified - bound successfully"
                        )
                        self._log_collection_pvc_verified = True
                        # Cleanup job
                        await batch_api.delete_namespaced_job(
                            name=job_name,
                            namespace=self.namespace,
                            body=client.V1DeleteOptions(
                                propagation_policy="Background"
                            ),
                        )
                        return

                    # Check if job failed
                    if job.status.failed and job.status.failed > 0:
                        raise RuntimeError(
                            "PVC verification job failed - storage class may not support RWX"
                        )

                    # Check pod status for more details
                    pods = await self._core_api.list_namespaced_pod(
                        namespace=self.namespace,
                        label_selector=f"job-name={job_name}",
                    )
                    if pods.items:
                        pod = pods.items[0]
                        phase = pod.status.phase
                        if phase == "Running" or phase == "Succeeded":
                            self._logger.info(
                                f"PVC {pvc_name} is binding (pod phase: {phase})"
                            )
                        elif phase == "Pending":
                            # Check for PVC binding issues (only log once)
                            if not binding_issue_logged:
                                for condition in pod.status.conditions or []:
                                    if (
                                        condition.type == "PodScheduled"
                                        and condition.status == "False"
                                    ):
                                        if (
                                            "unbound"
                                            in (condition.message or "").lower()
                                        ):
                                            self._logger.error(
                                                f"PVC BINDING FAILED: {pvc_name} cannot be bound. "
                                                f"Storage class '{storage_class}' likely does not support "
                                                f"ReadWriteMany (RWX) access mode."
                                            )
                                            binding_issue_logged = True

                except exceptions.ApiException as e:
                    if e.status != 404:
                        self._logger.warning(f"Error checking verification job: {e}")

                await asyncio.sleep(2)

            # Timeout - cleanup and fail
            try:
                await batch_api.delete_namespaced_job(
                    name=job_name,
                    namespace=self.namespace,
                    body=client.V1DeleteOptions(propagation_policy="Background"),
                )
            except Exception:
                pass

            raise RuntimeError(
                f"PVC '{pvc_name}' failed to bind within {timeout}s.\n"
                f"Storage class '{storage_class}' does not support ReadWriteMany (RWX) access mode.\n"
                f"Multi-pod log collection requires RWX. Please use a different storage class."
            )

        except exceptions.ApiException as e:
            self._logger.error(f"Failed to create PVC verification job: {e}")
            raise

    async def _cleanup_log_collection_pvc(self):
        """
        Clean up the log collection PVC if we created it.
        """
        if not self._log_collection_pvc_created:
            return

        pvc_name = getattr(
            self.deployment_spec, "_log_collection_pvc_name", "dynamo-logs-pvc"
        )

        try:
            await self._core_api.delete_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            self._logger.info(f"Cleaned up log collection PVC {pvc_name}")

        except client.ApiException as e:
            if e.status != 404:  # Not found is acceptable during cleanup
                self._logger.warning(f"Failed to cleanup PVC {pvc_name}: {e}")

    def _get_download_job_volume_config(self) -> dict:
        """
        Get the PVC volume configuration for the log download job.

        Returns:
            dict: Volume configuration for the download job using PVC

        Raises:
            RuntimeError: If PVC-based logging is not configured
        """
        if hasattr(self.deployment_spec, "_log_collection_pvc_name"):
            pvc_name = self.deployment_spec._log_collection_pvc_name
            return {"persistentVolumeClaim": {"claimName": pvc_name}}
        else:
            raise RuntimeError(
                "PVC-based log collection is not configured. "
                "Call deployment_spec.enable_log_collection() before using ManagedDeployment."
            )

    async def _cleanup_all_resources(self):
        """
        Comprehensive cleanup of all resources created by this deployment.

        Order for volume log collection:
        1. Stop resource monitoring (if running) and save history
        2. Delete deployment (pods exit gracefully, write final logs to PVC)
        3. Wait for pods to terminate
        4. Create download job to extract logs from PVC
        5. Extract logs
        6. Delete download job
        7. Delete PVC
        """
        try:
            # Stop resource monitoring and save history BEFORE deleting deployment
            if self._monitoring_task:
                self._logger.info("Stopping resource monitoring...")
                await self.stop_resource_monitoring()
                await self._save_resource_history_locally()

            # Delete the main deployment first (pods write final logs to PVC on exit)
            self._logger.info("Deleting deployment...")
            await self._delete_deployment()

            # Wait for pods to fully terminate before extracting logs
            if self.enable_volume_log_collection:
                self._logger.info("Waiting for pods to terminate...")
                await self._wait_for_pods_terminated()

                # Now create download job to extract logs from PVC (after pods are gone)
                await self._extract_logs_from_pvc()

                # Clean up PVC
                await self._cleanup_log_collection_pvc()

            # Clean up any orphaned jobs related to this deployment
            await self._cleanup_orphaned_jobs()

        except Exception as e:
            self._logger.warning(f"Error during comprehensive cleanup: {e}")

    async def _wait_for_pods_terminated(self, timeout: int = 120):
        """Wait for all deployment pods to terminate."""
        label_selector = f"dynamo-deployment={self.deployment_spec.name}"
        for attempt in range(timeout):
            try:
                pods = await self._core_api.list_namespaced_pod(
                    namespace=self.namespace, label_selector=label_selector
                )
                if not pods.items:
                    self._logger.info("All pods terminated")
                    return
                running = [
                    p.metadata.name
                    for p in pods.items
                    if p.status.phase not in ("Succeeded", "Failed")
                ]
                if not running:
                    self._logger.info("All pods completed")
                    return
                self._logger.info(
                    f"Waiting for {len(running)} pods to terminate... ({attempt + 1}/{timeout})"
                )
            except Exception as e:
                self._logger.debug(f"Error checking pods: {e}")
            await asyncio.sleep(1)
        self._logger.warning(f"Timeout waiting for pods to terminate after {timeout}s")

    async def _extract_logs_from_pvc(self):
        """Create download job, extract logs, and cleanup job."""
        # Skip if PVC was never successfully verified/bound
        if not self._log_collection_pvc_verified:
            self._logger.info(
                "Skipping log extraction - PVC was not successfully bound"
            )
            return

        try:
            services_dir = os.path.join(self.log_dir, "services")
            os.makedirs(services_dir, exist_ok=True)

            # Create download job
            self._logger.info("Creating log download job...")
            download_job_result = await self.create_log_download_job(
                local_output_dir=services_dir,
                container_log_dir=self.container_log_dir,
            )

            if not download_job_result.get("success"):
                self._logger.warning("Failed to create log download job")
                return

            job_name = download_job_result["job_name"]

            # Extract logs
            extraction_result = await self.extract_logs_from_download_job(
                job_name, services_dir
            )
            if extraction_result.get("success"):
                self._logger.info(
                    f"Successfully extracted {extraction_result.get('log_count', 0)} log files"
                )
            else:
                self._logger.warning(
                    f"Log extraction failed: {extraction_result.get('error', 'Unknown error')}"
                )

            # Cleanup download job
            await self.cleanup_log_download_job(job_name)

        except Exception as e:
            self._logger.warning(f"Error extracting logs from PVC: {e}")

    async def _cleanup_orphaned_jobs(self):
        """Clean up any jobs that might be left behind."""
        try:
            # Get all jobs in the namespace that are related to this deployment
            batch_api = client.BatchV1Api()
            jobs = await batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=f"deployment={self.deployment_spec.name}",
            )

            for job in jobs.items:
                job_name = job.metadata.name
                try:
                    # Delete the job with cascade to clean up pods
                    await batch_api.delete_namespaced_job(
                        name=job_name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(propagation_policy="Background"),
                    )
                    self._logger.info(f"Cleaned up orphaned job: {job_name}")
                except client.ApiException as e:
                    if e.status != 404:
                        self._logger.warning(f"Failed to cleanup job {job_name}: {e}")

        except Exception as e:
            self._logger.warning(f"Error cleaning up orphaned jobs: {e}")


async def main():
    LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,  # ISO 8601 UTC format
    )

    # Get workspace directory
    workspace_dir = _get_workspace_dir()

    deployment_spec = DeploymentSpec(
        os.path.join(workspace_dir, "examples/backends/vllm/deploy/agg.yaml")
    )

    deployment_spec.disable_grove()

    print(deployment_spec._deployment_spec)

    deployment_spec.name = "foo"

    deployment_spec.set_image("nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1")

    # Configure logging
    deployment_spec.set_logging(enable_jsonl=True, log_level="debug")

    print(f"Logging config: {deployment_spec.get_logging_config()}")

    async with ManagedDeployment(
        namespace="test", log_dir=".", deployment_spec=deployment_spec
    ):
        time.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
