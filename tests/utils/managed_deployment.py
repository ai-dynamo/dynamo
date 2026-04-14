# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

import kr8s
import requests
import yaml
from kr8s.objects import Pod, Service
from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions

from tests.utils.test_output import resolve_test_output_path


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


class ServiceSpec:
    """Wrapper around a single service in the deployment spec."""

    def __init__(self, service_name: str, service_spec: dict):
        self._name = service_name
        self._spec = service_spec

    def _ensure_path(self, *keys):
        """Ensure a nested dict path exists, returning the innermost dict."""
        d = self._spec
        for key in keys:
            if key not in d:
                d[key] = {}
            d = d[key]
        return d

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
        self._ensure_path("extraPodSpec", "mainContainer")["image"] = value

    @property
    def component_type(self):
        return self._spec["componentType"]

    @property
    def frontend_sidecar_image(self) -> Optional[str]:
        """Container image for the frontendSidecar (if present)."""
        try:
            return self._spec["frontendSidecar"]["image"]
        except KeyError:
            return None

    @frontend_sidecar_image.setter
    def frontend_sidecar_image(self, value: str):
        self._ensure_path("frontendSidecar")["image"] = value

    @property
    def envs(self) -> list[dict[str, str]]:
        """Environment variables for the service"""
        return self._spec.get("envs", [])

    @envs.setter
    def envs(self, value: list[dict[str, str]]):
        self._spec["envs"] = value

    def _get_args(self) -> list[str]:
        """Return the container args list, normalising scalar strings to a list in-place.

        Always returns the same list object that is stored in the spec, so
        in-place mutations (append / index assignment) are reflected immediately
        without an explicit writeback.
        """
        try:
            container = self._spec["extraPodSpec"]["mainContainer"]
        except KeyError:
            return []
        if "args" not in container:
            container["args"] = []
        args = container["args"]
        if isinstance(args, str):
            args = args.split()
            container["args"] = args
        return args

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
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg in ["--model", "--model-path"]:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    return args[i + 1]
        return None

    @model.setter
    def model(self, value: str):
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg in ["--model", "--model-path"]:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = value
                return

    # ----- GPUs -----
    @property
    def gpus(self) -> int:
        try:
            return int(self._spec["resources"]["limits"]["gpu"])
        except KeyError:
            return 0

    @gpus.setter
    def gpus(self, value: int):
        self._ensure_path("resources", "limits")["gpu"] = str(value)

    @property
    def tensor_parallel_size(self) -> int:
        """Get tensor parallel size from vLLM arguments"""
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg == "--tensor-parallel-size":
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    return int(args[i + 1])
                return 1
        return 1

    @tensor_parallel_size.setter
    def tensor_parallel_size(self, value: int):
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg == "--tensor-parallel-size":
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = str(value)
                else:
                    args.append(str(value))
                self.gpus = value
                return
        args.extend(["--tensor-parallel-size", str(value)])
        self.gpus = value

    # ----- Args -----
    def set_arg(self, arg_name: str, arg_value: str):
        """Set or override a command-line argument for this service.

        If the argument already exists, its value is updated.
        Otherwise, the argument is appended.

        Args:
            arg_name: Argument name (e.g., "--max-model-len", "--kv-cache-dtype")
            arg_value: Argument value (e.g., "1024", "fp8")

        Example:
            service.set_arg("--max-model-len", "4096")
        """
        container = self._ensure_path("extraPodSpec", "mainContainer")
        if "args" not in container:
            container["args"] = []
        args = container["args"]

        # Normalize string to list
        if isinstance(args, str):
            import shlex

            args = shlex.split(args)
            container["args"] = args

        # Find and update existing arg, or append
        for i, arg in enumerate(args):
            if arg == arg_name:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = arg_value
                else:
                    args.insert(i + 1, arg_value)
                return
        args.extend([arg_name, arg_value])

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
        self._ensure_path("extraPodSpec")["terminationGracePeriodSeconds"] = seconds

    # ----- Helpers for spec mutation -----
    def _add_volume_mount(self, name: str, mount_point: str):
        """Add a volume mount if not already present."""
        if "volumeMounts" not in self._spec:
            self._spec["volumeMounts"] = []
        if not any(m.get("name") == name for m in self._spec["volumeMounts"]):
            self._spec["volumeMounts"].append({"name": name, "mountPoint": mount_point})

    def _add_env_var(self, name: str, value=None, value_from=None):
        """Add an env var if not already present."""
        if "envs" not in self._spec:
            self._spec["envs"] = []
        if not any(e.get("name") == name for e in self._spec["envs"]):
            env = {"name": name}
            if value_from:
                env["valueFrom"] = value_from
            elif value is not None:
                env["value"] = value
            self._spec["envs"].append(env)

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

        main_container = self._ensure_path("extraPodSpec", "mainContainer")

        # Get original command and args (with defaults)
        original_command = main_container.get("command", [])
        original_args = main_container.get("args", [])
        if not original_command and not original_args:
            original_command = ["python3"]
            original_args = (
                ["-m", "dynamo.frontend"] if self.component_type == "frontend" else []
            )

        full_command = " ".join(original_command + original_args)
        service_log_dir = f"{log_dir}/service_logs/{self._name.lower()}"

        # Load wrapper script template
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates", "log_wrapper.sh"
        )
        with open(template_path) as f:
            wrapper_script = f.read()
        wrapper_script = wrapper_script.replace("{{SERVICE_LOG_DIR}}", service_log_dir)
        wrapper_script = wrapper_script.replace("{{FULL_COMMAND}}", full_command)

        # Set the wrapped command (replaces original command + args)
        main_container["command"] = ["/bin/bash", "-c", wrapper_script]
        if "args" in main_container:
            del main_container["args"]

        # Add volume mount and env vars for log file naming
        self._add_volume_mount(pvc_name, log_dir)
        self._add_env_var(
            "POD_NAME", value_from={"fieldRef": {"fieldPath": "metadata.name"}}
        )
        self._add_env_var(
            "POD_NAMESPACE",
            value_from={"fieldRef": {"fieldPath": "metadata.namespace"}},
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
        """Load the deployment YAML file.

        Args:
            base: Path to the deployment YAML file
            endpoint: API endpoint path (default: /v1/chat/completions)
            port: Frontend port (default: 8000)
            system_port: System/metrics port (default: 9090)
        """
        self._base_path = base
        with open(base, "r") as f:
            self._deployment_spec = yaml.safe_load(f)
        self._endpoint = endpoint
        self._port = port
        self._system_port = system_port

    @classmethod
    def from_backend(
        cls,
        backend: str,
        deployment_type: str = "agg",
        workspace_dir: str = "/workspace",
        **kwargs,
    ) -> "DeploymentSpec":
        """Create a DeploymentSpec from backend name and deployment type.

        Args:
            backend: Backend name ("vllm", "trtllm", "sglang", "mocker")
            deployment_type: Deployment type ("agg", "disagg", etc.)
            workspace_dir: Workspace root directory
            **kwargs: Additional arguments passed to DeploymentSpec.__init__

        Example:
            spec = DeploymentSpec.from_backend("vllm", "disagg")
            spec.set_worker_replicas(2)
        """
        yaml_path = (
            f"{workspace_dir}/examples/backends/{backend}/deploy/{deployment_type}.yaml"
        )
        return cls(yaml_path, **kwargs)

    @property
    def backend(self) -> str:
        """Auto-detect backend from YAML path or service names.

        Returns:
            Backend name ("vllm", "trtllm", "sglang", "mocker", or "unknown")
        """
        # Try to infer from YAML path
        if hasattr(self, "_base_path"):
            path = self._base_path.lower()
            for name in ("vllm", "trtllm", "sglang", "mocker"):
                if f"/backends/{name}/" in path or f"/{name}/" in path:
                    return name

        # Fall back to service name inspection
        service_names = " ".join(s.name.lower() for s in self.services)
        if "vllm" in service_names:
            return "vllm"
        if "trtllm" in service_names:
            return "trtllm"
        if "sglang" in service_names:
            return "sglang"
        if "mocker" in service_names:
            return "mocker"
        return "unknown"

    def worker_services(self) -> list[str]:
        """Return worker service names (non-frontend services).

        Example:
            spec = DeploymentSpec.from_backend("vllm", "disagg")
            spec.worker_services()  # ["VllmPrefillWorker", "VllmDecodeWorker"]
        """
        return [s.name for s in self.services if s.component_type != "frontend"]

    def set_worker_replicas(self, replicas: int):
        """Set replicas for all worker services.

        Args:
            replicas: Number of replicas for each worker service

        Example:
            spec.set_worker_replicas(2)  # Sets all workers to 2 replicas
        """
        for name in self.worker_services():
            self[name].replicas = replicas

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

    def set_frontend_sidecar_image(
        self, image: str, service_name: Optional[str] = None
    ):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.frontend_sidecar_image = image

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

    def get_service(self, service_name: str) -> ServiceSpec:
        """Get a specific service by name.

        Args:
            service_name: Name of the service (e.g., "VllmWorker", "Frontend")

        Raises:
            ValueError: If service not found in deployment spec
        """
        if service_name not in self._deployment_spec["spec"]["services"]:
            raise ValueError(
                f"Service '{service_name}' not found in deployment spec. "
                f"Available: {list(self._deployment_spec['spec']['services'].keys())}"
            )
        return ServiceSpec(
            service_name, self._deployment_spec["spec"]["services"][service_name]
        )

    def set_service_replicas(self, service_name: str, replicas: int):
        """Set the number of replicas for a specific service."""
        self.get_service(service_name).replicas = replicas

    def set_service_readiness_probe(
        self, service_name: str, period_seconds: int = 10, **kwargs
    ):
        """Set readiness probe for a specific service.

        Args:
            service_name: Name of the service (e.g., "TRTLLMDecodeWorker")
            period_seconds: How often to perform the probe (default: 10)
            **kwargs: Additional options (initial_delay_seconds, timeout_seconds,
                      failure_threshold, path, port)
        """
        self.get_service(service_name).set_readiness_probe(
            period_seconds=period_seconds, **kwargs
        )

    def set_service_termination_grace_period(self, service_name: str, seconds: int):
        """Set termination grace period for a specific service.

        Args:
            service_name: Name of the service
            seconds: Grace period in seconds
        """
        self.get_service(service_name).set_termination_grace_period(seconds)

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

        logging.getLogger(__name__).debug(f"Generated PVC name: {pvc_name}")

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
class PodStatusDetail:
    """Container-level status snapshot for a single container in a pod."""

    pod_name: str
    container_name: str
    state: Literal["Waiting", "Terminated", "Running", "Unknown"]
    reason: str = ""
    message: str = ""
    exit_code: Optional[int] = None
    restart_count: int = 0

    def format(self) -> str:
        result = f"{self.pod_name}/{self.container_name}: {self.state}"
        if self.reason:
            result += f": {self.reason}"
        if self.message:
            result += f" ({self.message})"
        if self.exit_code is not None:
            result += f" (exit_code={self.exit_code})"
        if self.restart_count > 0:
            result += f" [restarts={self.restart_count}]"
        return result


@dataclass
class ManagedDeployment:
    log_dir: str
    deployment_spec: DeploymentSpec
    namespace: str
    skip_service_restart: bool = (
        True  # Default: skip restart. Pass False to restart NATS/etcd.
    )
    container_log_dir: str = "/tmp/service_logs"  # Directory for PVC-based logs

    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    # Initialized from deployment_spec.name in __post_init__; placeholder needed for dataclass ordering
    _deployment_name: str = field(default="")
    _apps_v1: Optional[Any] = None
    _active_port_forwards: List[Any] = field(default_factory=list)

    # PVC-based log collection
    _log_collection_pvc_created: bool = field(
        default=False, init=False
    )  # Track if we created a PVC
    _log_collection_pvc_verified: bool = field(
        default=False, init=False
    )  # Track if PVC was successfully verified/bound

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
        self.log_dir = resolve_test_output_path(self.log_dir)

    async def _init_kubernetes(self):
        """Initialize kubernetes clients."""
        from tests.utils.k8s_helpers import init_kubernetes_clients

        (
            self._core_api,
            _,
            self._custom_api,
            self._apps_v1,
            self._in_cluster,
        ) = await init_kubernetes_clients(need_custom_api=True, need_apps_api=True)

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
        """Restart a StatefulSet by scaling to 0, deleting PVCs, and scaling back.

        Silently skips if the StatefulSet does not exist.
        """
        try:
            assert self._apps_v1 is not None, "Kubernetes API not initialized"
            await self._apps_v1.read_namespaced_stateful_set(name, self.namespace)
        except exceptions.ApiException as e:
            if e.status == 404:
                self._logger.info(f"StatefulSet {name} not found, skipping restart")
                return
            raise

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
                        pod_details = await self._get_pod_status_details()
                        if pod_details:
                            for d in pod_details:
                                self._logger.info(f"  Pod status: {d.format()}")
                        pod_events = await self._get_pod_events()
                        if pod_events:
                            self._logger.info("  Pod warning events:")
                            for ev in pod_events:
                                self._logger.info(f"    {ev}")

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

        # Collect pod diagnostics before raising
        pod_details = await self._get_pod_status_details()
        elapsed = time.time() - start_time
        msg = (
            f"Deployment {self._deployment_name} failed to reach "
            f"Ready={desired_ready_condition_val}, state={desired_state_val} "
            f"within {elapsed:.0f}s (timeout={timeout}s)"
        )
        if pod_details:
            detail_lines = "\n".join(f"  {d.format()}" for d in pod_details)
            msg += f"\n\nPod status at timeout:\n{detail_lines}"
        raise TimeoutError(msg)

    async def _get_pod_status_details(self) -> List[PodStatusDetail]:
        """Collect container-level status for all pods owned by this deployment.

        Returns a list of PodStatusDetail objects. Returns empty list on any
        API failure so callers never need to guard against exceptions.
        """
        try:
            assert self._core_api is not None, "Kubernetes API not initialized"
            label = f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name}"
            pods = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label
            )

            details: List[PodStatusDetail] = []
            for pod in pods.items:
                pod_name = pod.metadata.name
                pod_status = pod.status
                phase = pod_status.phase if pod_status else "Unknown"

                container_statuses = (
                    pod_status.container_statuses if pod_status else None
                )
                if not container_statuses:
                    details.append(
                        PodStatusDetail(
                            pod_name=pod_name,
                            container_name="*",
                            state="Unknown",
                            reason=f"{phase} (no container status)",
                        )
                    )
                    continue

                for cs in container_statuses:
                    state: Literal[
                        "Waiting", "Terminated", "Running", "Unknown"
                    ] = "Unknown"
                    reason = ""
                    message = ""
                    exit_code: Optional[int] = None

                    if cs.state and cs.state.waiting:
                        state = "Waiting"
                        reason = cs.state.waiting.reason or ""
                        message = cs.state.waiting.message or ""
                    elif cs.state and cs.state.terminated:
                        state = "Terminated"
                        reason = cs.state.terminated.reason or ""
                        exit_code = cs.state.terminated.exit_code
                    elif cs.state and cs.state.running:
                        state = "Running"

                    details.append(
                        PodStatusDetail(
                            pod_name=pod_name,
                            container_name=cs.name,
                            state=state,
                            reason=reason,
                            message=message,
                            exit_code=exit_code,
                            restart_count=cs.restart_count or 0,
                        )
                    )

            return details

        except exceptions.ApiException as e:
            self._logger.debug(f"Failed to collect pod status details: {e}")
            return []

    async def _get_pod_events(self) -> List[str]:
        """Fetch warning events for pods in this deployment's namespace."""
        try:
            assert self._core_api is not None, "Kubernetes API not initialized"
            events = await self._core_api.list_namespaced_event(self.namespace)
            warnings = []
            for event in events.items:
                if event.type != "Normal" and event.involved_object.kind == "Pod":
                    name = event.involved_object.name or "unknown"
                    reason = event.reason or ""
                    msg = event.message or ""
                    warnings.append(f"{name}: {reason} - {msg}")
            return warnings[-10:]
        except Exception as e:
            self._logger.debug(f"Failed to collect pod events: {e}")
            return []

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

        for original_name in service_names:
            label_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name},"
                f"nvidia.com/dynamo-component={original_name}"
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

        for original_name in service_names:
            # List pods using stable labels that are not affected by worker hash suffixes.
            label_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name},"
                f"nvidia.com/dynamo-component={original_name}"
            )

            pods: list[Pod] = []

            for pod in kr8s.get(
                "pods", namespace=self.namespace, label_selector=label_selector
            ):
                pods.append(pod)  # type: ignore[arg-type]

            result[original_name] = pods

        return result

    def _get_pod_manifest(self, pod: Pod, service_name: str, suffix=""):
        """Save pod manifest (YAML). Sync - no pod exec needed."""
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)
        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.yaml"), "w") as f:
                f.write(pod.to_yaml())
        except Exception as e:
            self._logger.error(e)

    def _get_pod_logs(self, pod: Pod, service_name: str, suffix=""):
        """Save pod logs (current + previous). Sync - uses K8s logs API."""
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)
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

    async def _get_pod_metrics(
        self, pod: Pod, service_name: str, suffix="", use_services_dir: bool = False
    ):
        """Fetch HTTP metrics. Async - needs exec for timestamp."""
        if use_services_dir:
            directory = os.path.join(self.log_dir, "services", service_name.lower())
        else:
            directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)

        port = (
            self.deployment_spec.port
            if service_name == self.frontend_service_name
            else self.deployment_spec.system_port
        )

        pf = self.port_forward(pod, port)
        if not pf:
            self._logger.error(f"Unable to get metrics for {service_name}")
            return

        content = None
        try:
            response = requests.get(
                f"http://localhost:{pf.local_port}/metrics", timeout=30
            )
            content = response.text
        except Exception as e:
            self._logger.error(str(e))

        if content:
            timestamp = await self._get_container_timestamp(pod)
            filename = f"{pod.name}_{timestamp}.metrics{suffix}.log"
            with open(os.path.join(directory, filename), "w") as f:
                f.write(content)

    async def _collect_service_metrics(self, use_services_dir: bool = True):
        """Collect only HTTP metrics from all services.

        Args:
            use_services_dir: If True, save to services/{service}/ directory
                            to match PVC log extraction path
        """
        self._logger.info("Collecting metrics from all services...")
        service_pods = self.get_pods()

        for service, pods in service_pods.items():
            for pod in pods:
                try:
                    await self._get_pod_metrics(
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

    async def _get_container_timestamp(self, pod: Pod) -> str:
        """Read container start timestamp written by wrapper script.

        The wrapper script writes the timestamp to /tmp/.{pod_name}.start_time
        This ensures exact match with service log filenames.
        """
        try:
            result = await self._exec_in_pod(
                pod, ["cat", f"/tmp/.{pod.name}.start_time"], timeout=5.0
            )
            return result.stdout.decode().strip()
        except Exception:
            # Fallback if timestamp file doesn't exist
            return str(int(time.time()))

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
            # Collect metrics via HTTP (logs are collected from PVC during resource cleanup)
            await self._collect_service_metrics(use_services_dir=True)

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

            # Set up PVC-based log collection before deployment creation
            pvc_configured = hasattr(self.deployment_spec, "_log_collection_pvc_name")
            if pvc_configured:
                self._logger.info(
                    f"PVC-based log collection configured: {self.deployment_spec._log_collection_pvc_name}"
                )
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

    async def download_volume_logs_now(self, local_output_dir=None):
        """Download service logs from PVC immediately.

        Creates a temporary extraction job, downloads logs, and cleans up.

        Args:
            local_output_dir: Local directory to save logs (defaults to log_dir/services_manual)

        Returns:
            dict: Extraction results
        """
        if local_output_dir is None:
            local_output_dir = os.path.join(self.log_dir, "services_manual")

        pvc_name = getattr(self.deployment_spec, "_log_collection_pvc_name", None)
        if not pvc_name:
            return {"success": False, "error": "No PVC configured for log collection"}

        from tests.utils.pvc_extractor import PvcExtractor

        extractor = PvcExtractor(namespace=self.namespace, logger=self._logger)
        await extractor.init()

        return await extractor.extract(
            pvc_name=pvc_name,
            sub_path="service_logs",
            container_path=self.container_log_dir,
            file_patterns=["*.log"],
            local_output_dir=local_output_dir,
        )

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

        batch_api = client.BatchV1Api()
        job_created = False
        verification_error = None

        try:
            # Create the verification job
            await batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            job_created = True
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
                        return

                    # Check if job failed
                    if job.status.failed and job.status.failed > 0:
                        verification_error = RuntimeError(
                            "PVC verification job failed - storage class may not support RWX"
                        )
                        break

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
            else:
                # Timeout reached
                verification_error = RuntimeError(
                    f"PVC '{pvc_name}' failed to bind within {timeout}s.\n"
                    f"Storage class '{storage_class}' does not support ReadWriteMany (RWX) access mode.\n"
                    f"Multi-pod log collection requires RWX. Please use a different storage class."
                )

        except exceptions.ApiException as e:
            self._logger.error(f"Failed to create PVC verification job: {e}")
            raise

        finally:
            # Always cleanup the verification job and its pods
            if job_created:
                try:
                    self._logger.info(f"Cleaning up PVC verification job: {job_name}")
                    # Use Foreground propagation to ensure pods are deleted too
                    await batch_api.delete_namespaced_job(
                        name=job_name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(propagation_policy="Foreground"),
                    )
                    # Wait briefly for cleanup to complete
                    for _ in range(10):
                        try:
                            await batch_api.read_namespaced_job(
                                name=job_name, namespace=self.namespace
                            )
                            await asyncio.sleep(1)
                        except exceptions.ApiException as e:
                            if e.status == 404:
                                break
                    self._logger.info(f"PVC verification job {job_name} cleaned up")
                except Exception as cleanup_error:
                    self._logger.warning(
                        f"Failed to cleanup verification job {job_name}: {cleanup_error}"
                    )

        # Raise any verification error after cleanup
        if verification_error:
            raise verification_error

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

    async def _cleanup_all_resources(self):
        """
        Comprehensive cleanup of all resources created by this deployment.

        Order for volume log collection:
        1. Delete deployment (pods exit gracefully, write final logs to PVC)
        2. Wait for pods to terminate
        3. Create download job to extract logs from PVC
        4. Extract logs
        5. Delete download job
        6. Delete PVC
        """
        try:
            # Delete the main deployment first (pods write final logs to PVC on exit)
            self._logger.info("Deleting deployment...")
            await self._delete_deployment()

            # Wait for pods to terminate, then extract logs from PVC
            self._logger.info("Waiting for pods to terminate...")
            await self._wait_for_pods_terminated()
            await self._extract_logs_from_pvc()
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
        """Extract service logs from PVC using PvcExtractor."""
        if not self._log_collection_pvc_verified:
            self._logger.info(
                "Skipping log extraction - PVC was not successfully bound"
            )
            return

        try:
            from tests.utils.pvc_extractor import PvcExtractor

            pvc_name = getattr(self.deployment_spec, "_log_collection_pvc_name", None)
            if not pvc_name:
                self._logger.warning("No PVC name found for log extraction")
                return

            services_dir = os.path.join(self.log_dir, "services")
            os.makedirs(services_dir, exist_ok=True)

            extractor = PvcExtractor(namespace=self.namespace, logger=self._logger)
            await extractor.init()

            result = await extractor.extract(
                pvc_name=pvc_name,
                sub_path="service_logs",
                container_path=self.container_log_dir,
                file_patterns=["*.log"],
                local_output_dir=services_dir,
            )

            if result.get("success"):
                self._logger.info(
                    f"Extracted {result.get('file_count', 0)} log files to {services_dir}"
                )
            else:
                self._logger.warning(
                    f"Log extraction failed: {result.get('error', 'Unknown error')}"
                )

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
