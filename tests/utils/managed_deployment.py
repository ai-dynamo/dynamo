# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import re
import secrets
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

import kr8s
import requests
import yaml
from kr8s.objects import Pod, Service
from kubernetes_asyncio import client, config
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
    def frontend_sidecar_image(self) -> Optional[str]:
        """Container image for the frontendSidecar (if present)."""
        try:
            return self._spec["frontendSidecar"]["image"]
        except KeyError:
            return None

    @frontend_sidecar_image.setter
    def frontend_sidecar_image(self, value: str):
        if "frontendSidecar" not in self._spec:
            self._spec["frontendSidecar"] = {}
        self._spec["frontendSidecar"]["image"] = value

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
            args = shlex.split(args)
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
        if "resources" not in self._spec:
            self._spec["resources"] = {}
        if "limits" not in self._spec["resources"]:
            self._spec["resources"]["limits"] = {}
        self._spec["resources"]["limits"]["gpu"] = str(value)

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

    # ----- Spec helpers (DGH-703) -----
    def _ensure_path(self, *keys):
        """Ensure a nested dict path exists, returning the innermost dict."""
        d = self._spec
        for key in keys:
            if key not in d:
                d[key] = {}
            d = d[key]
        return d

    @property
    def component_type(self) -> Optional[str]:
        """Service component type (e.g. ``frontend`` for the frontend service)."""
        return self._spec.get("componentType")

    # ----- Args -----
    def set_arg(self, arg_name: str, arg_value: str):
        """Set or override a command-line argument for this service.

        If the argument already exists, its value is updated. Otherwise it is appended.

        Args:
            arg_name: Argument name (e.g. ``"--max-model-len"``)
            arg_value: Argument value (e.g. ``"4096"``)
        """
        container = self._ensure_path("extraPodSpec", "mainContainer")
        if "args" not in container:
            container["args"] = []
        args = container["args"]
        if isinstance(args, str):
            args = shlex.split(args)
            container["args"] = args
        for i, arg in enumerate(args):
            if arg == arg_name:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = arg_value
                else:
                    args.insert(i + 1, arg_value)
                return
        args.extend([arg_name, arg_value])

    # ----- Volume mounts and env vars -----
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
            env: dict[str, Any] = {"name": name}
            if value_from:
                env["valueFrom"] = value_from
            elif value is not None:
                env["value"] = value
            self._spec["envs"].append(env)

    # ----- Log collection -----
    def enable_log_collection(self, log_dir: str, pvc_name: str):
        """Wrap this service's command to tee output into a PVC-mounted log dir."""
        main_container = self._spec.get("extraPodSpec", {}).get("mainContainer", {})
        existing_command = main_container.get("command", [])
        if (
            len(existing_command) >= 3
            and existing_command[:2] == ["/bin/bash", "-c"]
            and "tee -a" in existing_command[2]
        ):
            return  # already wrapped

        main_container = self._ensure_path("extraPodSpec", "mainContainer")
        original_command = main_container.get("command", [])
        original_args = main_container.get("args", [])
        if not original_command and not original_args:
            original_command = ["python3"]
            original_args = (
                ["-m", "dynamo.frontend"] if self.component_type == "frontend" else []
            )

        full_command = " ".join(original_command + original_args)
        service_log_dir = f"{log_dir}/service_logs/{self._name.lower()}"

        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates", "log_wrapper.sh"
        )
        with open(template_path) as f:
            wrapper_script = f.read()
        wrapper_script = wrapper_script.replace("{{SERVICE_LOG_DIR}}", service_log_dir)
        wrapper_script = wrapper_script.replace("{{FULL_COMMAND}}", full_command)

        main_container["command"] = ["/bin/bash", "-c", wrapper_script]
        if "args" in main_container:
            del main_container["args"]

        self._add_volume_mount(pvc_name, log_dir)
        self._add_env_var(
            "POD_NAME", value_from={"fieldRef": {"fieldPath": "metadata.name"}}
        )
        self._add_env_var(
            "POD_NAMESPACE",
            value_from={"fieldRef": {"fieldPath": "metadata.namespace"}},
        )

    # ----- ported from _2 (full surface area) -----
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

    def set_termination_grace_period(self, seconds: int = 60):
        """Set termination grace period for this service's pods.

        Args:
            seconds: Grace period in seconds (default: 60)
        """
        self._ensure_path("extraPodSpec")["terminationGracePeriodSeconds"] = seconds

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
        self._base_path = base
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
    def frontend_service(self) -> "ServiceSpec":
        """Return the first service whose componentType is ``frontend``."""
        for service in self.services:
            if service.component_type == "frontend":
                return service
        raise LookupError(
            f"Deployment {self.name} has no service with componentType 'frontend'"
        )

    def get_in_cluster_frontend_url(self, namespace: str) -> str:
        """Compute the in-cluster URL of the frontend service.

        The DNS name follows the operator's service-naming convention
        ``<deployment>-<service>.<namespace>.svc.cluster.local``.
        """
        return (
            f"http://{self.name.lower()}-"
            f"{self.frontend_service.name.lower()}."
            f"{namespace.lower()}.svc.cluster.local:{self.port}"
        )

    @property
    def system_port(self) -> int:
        """Deployment port"""
        return self._system_port

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def namespace(self) -> str:
        """Deployment namespace"""
        return self._deployment_spec["metadata"]["namespace"]

    @namespace.setter
    def namespace(self, value: str):
        self._deployment_spec["metadata"]["namespace"] = value

    def disable_grove(self):
        if "annotations" not in self._deployment_spec["metadata"]:
            self._deployment_spec["metadata"]["annotations"] = {}
        self._deployment_spec["metadata"]["annotations"][
            "nvidia.com/enable-grove"
        ] = "false"

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

    def set_service_replicas(self, service_name: str, replicas: int):
        """Convenience wrapper around ``spec[name].replicas = N``.

        Kept for legacy callers; new code should use the per-service form
        directly so different workers can carry different replica counts.
        """
        self[service_name].replicas = replicas

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)

    # ----- DGH-703 unified-framework APIs -----
    @classmethod
    def from_backend(
        cls,
        backend: str,
        deployment_type: str = "agg",
        workspace_dir: str = "/workspace",
        **kwargs,
    ) -> "DeploymentSpec":
        """Create a DeploymentSpec from a backend name and deployment type.

        Args:
            backend: Backend name (``"vllm"``, ``"trtllm"``, ``"sglang"``, ``"mocker"``)
            deployment_type: Deployment shape (``"agg"``, ``"disagg"``, ...)
            workspace_dir: Workspace root containing ``examples/backends/<backend>/deploy/``
            **kwargs: Forwarded to :class:`DeploymentSpec` (``endpoint``, ``port``, ...)

        Example::

            spec = DeploymentSpec.from_backend("vllm", "disagg")
            spec.set_worker_replicas(2)
        """
        yaml_path = (
            f"{workspace_dir}/examples/backends/{backend}/deploy/{deployment_type}.yaml"
        )
        return cls(yaml_path, **kwargs)

    @property
    def backend(self) -> str:
        """Auto-detect backend from the YAML path or service names.

        Returns one of ``"vllm"``, ``"trtllm"``, ``"sglang"``, ``"mocker"``,
        or ``"unknown"`` if neither path nor service names match.
        """
        if hasattr(self, "_base_path") and self._base_path:
            path = self._base_path.lower()
            for name in ("vllm", "trtllm", "sglang", "mocker"):
                if f"/backends/{name}/" in path or f"/{name}/" in path:
                    return name
        service_names = " ".join(s.name.lower() for s in self.services)
        for name in ("vllm", "trtllm", "sglang", "mocker"):
            if name in service_names:
                return name
        return "unknown"

    def worker_services(self) -> list[str]:
        """Return worker service names — services whose ``componentType`` is not ``frontend``."""
        return [s.name for s in self.services if s.component_type != "frontend"]

    def set_worker_replicas(self, replicas: int) -> None:
        """Set ``replicas`` for every worker service."""
        for name in self.worker_services():
            self[name].replicas = replicas

    def enable_log_collection(
        self,
        pvc_name: Optional[str] = None,
        pvc_size: str = "1Gi",
        storage_class: Optional[str] = None,
        container_log_dir: str = "/tmp/service_logs",
        enable_all_services: bool = True,
        service_names: Optional[list[str]] = None,
    ) -> None:
        """Enable PVC-backed log collection for this deployment.

        Creates an RWX PVC declaration in the deployment spec and wraps every
        target service's command to tee output into ``container_log_dir``. The
        PVC itself is materialized later by :class:`ManagedDeployment`.

        Requires a storage class that supports ReadWriteMany; if absent, the
        deployment will fail at PVC binding time.
        """
        if enable_all_services:
            target_services = self.services
        else:
            target_services = [self[name] for name in (service_names or [])]

        if pvc_name is None:
            timestamp = int(time.time())
            rand_suffix = secrets.randbelow(9000) + 1000
            pvc_name = f"{self.name}-logs-{timestamp}-{rand_suffix}"

        self._log_collection_pvc_name = pvc_name
        self._log_collection_pvc_size = pvc_size
        self._log_collection_storage_class = storage_class
        self._log_collection_container_dir = container_log_dir

        if "pvcs" not in self._deployment_spec["spec"]:
            self._deployment_spec["spec"]["pvcs"] = []

        # Drop any prior log PVCs so reruns don't accumulate stale entries.
        self._deployment_spec["spec"]["pvcs"] = [
            pvc
            for pvc in self._deployment_spec["spec"]["pvcs"]
            if not pvc.get("name", "").endswith("-logs-pvc")
            and pvc.get("name") != pvc_name
        ]
        # ``create: False`` — ManagedDeployment provisions the PVC explicitly.
        self._deployment_spec["spec"]["pvcs"].append(
            {"name": pvc_name, "create": False}
        )

        for service in target_services:
            service.enable_log_collection(container_log_dir, pvc_name)

    # ----- ported from _2 (full surface area) -----


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
    # TODO: this should be determined by the deployment_spec
    # the service containing component_type: Frontend determines what is actually the frontend service
    frontend_service_name: str = "Frontend"
    skip_service_restart: bool = False

    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    # Initialized from deployment_spec.name in __post_init__; placeholder needed for dataclass ordering
    _deployment_name: str = field(default="")
    _apps_v1: Optional[Any] = None
    _active_port_forwards: List[Any] = field(default_factory=list)

    def __post_init__(self):
        self._deployment_name = self.deployment_spec.name
        self.log_dir = resolve_test_output_path(self.log_dir)
        # Lifecycle bookkeeping for the log-collection PVC. Cleanup paths
        # check these before doing work so a failed PVC create doesn't try
        # to clean up something that was never made.
        self._log_collection_pvc_created = False
        self._log_collection_pvc_verified = False
        # Mirror DeploymentSpec.enable_log_collection's container_log_dir so
        # the extractor and per-pod log paths agree.
        self.container_log_dir = "/tmp/service_logs"
        # Initial deployment startup time (seconds). Populated by the first
        # ``_wait_for_ready`` success in ``__aenter__``; ``None`` means we
        # haven't reached Ready yet.
        self.startup_seconds: Optional[float] = None

    async def _init_kubernetes(self):
        """Initialize kubernetes client.

        Priority order:
        1. KUBECONFIG environment variable (CI scenario with proper RBAC)
        2. In-cluster config (for pods without explicit kubeconfig)
        3. Default kubeconfig (~/.kube/config)
        """
        kubeconfig_path = os.environ.get("KUBECONFIG")

        if kubeconfig_path and os.path.exists(kubeconfig_path):
            # Explicit kubeconfig provided (CI scenario) - use it first
            self._logger.info(f"Loading kubeconfig from KUBECONFIG: {kubeconfig_path}")
            await config.load_kube_config(config_file=kubeconfig_path)
            self._in_cluster = False
            self._logger.info("Successfully loaded kubeconfig from KUBECONFIG")
        else:
            try:
                # Try in-cluster config (for pods without explicit kubeconfig)
                self._logger.info("Attempting in-cluster kubernetes config")
                config.load_incluster_config()
                self._in_cluster = True
                self._logger.info("Successfully loaded in-cluster kubernetes config")
            except Exception as e:
                # Fallback to default kube config file (for local development)
                self._logger.warning(
                    f"In-cluster config failed ({type(e).__name__}: {e}), "
                    f"falling back to default kubeconfig (~/.kube/config)"
                )
                await config.load_kube_config()
                self._in_cluster = False
                self._logger.info("Successfully loaded default kubeconfig")

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

    async def _wait_for_ready(self, timeout: int = 1800, sleep=1, log_interval=60):
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
                    elapsed = time.time() - start_time
                    self._logger.info(f"Current deployment state: {current_state}")
                    self._logger.info(f"Current conditions: {conditions}")
                    self._logger.info(f"Elapsed time: {elapsed:.1f}s / {timeout}s")

                    self._logger.info(
                        f"Deployment {self._deployment_name} has Ready condition {desired_ready_condition_val} and state {desired_state_val}"
                    )
                    # Capture the FIRST observed ready elapsed-time as the
                    # deployment's startup time. Later waits (e.g. after a
                    # rolling-upgrade) leave this untouched.
                    if (
                        desired_ready_condition_val is True
                        and getattr(self, "startup_seconds", None) is None
                    ):
                        self.startup_seconds = elapsed
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
            self._logger.info(self.deployment_spec.spec())
            self._logger.info(f"Deployment Started {self._deployment_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.info(f"Deployment {self._deployment_name} already exists")
            else:
                self._logger.info(
                    f"Failed to create deployment {self._deployment_name}: {e}"
                )
                raise

    async def trigger_rolling_upgrade(self, service_names: list[str]):
        """Trigger a rolling upgrade by stamping a unique env var on each
        named service and then publishing the change via
        :py:meth:`apply_service_changes`.

        The env var is a no-op for the worker process; its purpose is to
        change the pod template so the operator rolls the deployment.
        """
        if not service_names:
            raise ValueError(
                "service_names cannot be empty for trigger_rolling_upgrade"
            )
        for service_name in service_names:
            self.deployment_spec[service_name].set_env_var(
                "TEST_ROLLING_UPDATE_TRIGGER", secrets.token_hex(8)
            )
        try:
            await self.apply_service_changes(service_names)
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

    def _get_pod_metrics(self, pod: Pod, service_name: str, suffix=""):
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

    async def _delete_deployment(self):
        """Delete the DynamoGraphDeployment CR and wait for it to fully drain.

        Two-stage wait so a subsequent ``_create_deployment`` never races a
        prior incarnation:
          1. Wait for the CR itself to disappear from the API.
          2. Wait for the pods labeled with this deployment to terminate.
        Without this, leftover pods from a previous (especially failed) run
        keep cycling alongside the new ones.
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

        await self._wait_for_cr_deleted()
        await self._wait_for_pods_terminated()

    async def _scrub_namespace(self) -> None:
        """Wipe prior-run artifacts from the test namespace.

        Deletes every ``DynamoGraphDeployment`` in the namespace, the
        log-collection PVCs left behind by past failed runs, and any
        leftover ``load-*`` / ``pvc-extract-*`` / ``*-verify-*`` jobs.
        Then waits for all pods labeled with this deployment to drain so
        the new run starts on a clean slate.

        Called from ``__aenter__`` so test authors don't have to remember
        to scrub between runs — the framework guarantees a clean baseline.
        """
        if self._custom_api is None or self._core_api is None:
            return

        # 1. Delete all DynamoGraphDeployments in the namespace.
        try:
            cr_list = await self._custom_api.list_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
            )
            for item in cr_list.get("items", []):
                name = item.get("metadata", {}).get("name")
                if not name:
                    continue
                try:
                    await self._custom_api.delete_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural="dynamographdeployments",
                        name=name,
                    )
                    self._logger.info(f"scrub: deleted prior CR {name}")
                except exceptions.ApiException as e:
                    if e.status != 404:
                        self._logger.debug(f"scrub: CR {name}: {e}")
        except exceptions.ApiException as e:
            self._logger.debug(f"scrub: list CRs failed: {e}")

        # 2. Delete log-collection PVCs (labelled by enable_log_collection).
        try:
            pvcs = await self._core_api.list_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                label_selector="purpose=log-collection",
            )
            for pvc in pvcs.items:
                try:
                    await self._core_api.delete_namespaced_persistent_volume_claim(
                        name=pvc.metadata.name, namespace=self.namespace
                    )
                    self._logger.info(
                        f"scrub: deleted prior log PVC {pvc.metadata.name}"
                    )
                except exceptions.ApiException as e:
                    if e.status != 404:
                        self._logger.debug(f"scrub: PVC {pvc.metadata.name}: {e}")
        except exceptions.ApiException as e:
            self._logger.debug(f"scrub: list PVCs failed: {e}")

        # 3. Delete leftover jobs (load runners, log extractors, PVC-verify probes).
        try:
            batch_api = client.BatchV1Api()
            jobs = await batch_api.list_namespaced_job(namespace=self.namespace)
            stale_prefixes = ("load-", "pvc-extract-")
            stale_substrs = ("-verify-",)
            for job in jobs.items:
                name = job.metadata.name
                if not (
                    any(name.startswith(p) for p in stale_prefixes)
                    or any(s in name for s in stale_substrs)
                ):
                    continue
                try:
                    await batch_api.delete_namespaced_job(
                        name=name,
                        namespace=self.namespace,
                        propagation_policy="Background",
                    )
                    self._logger.info(f"scrub: deleted prior job {name}")
                except exceptions.ApiException as e:
                    if e.status != 404:
                        self._logger.debug(f"scrub: job {name}: {e}")
        except exceptions.ApiException as e:
            self._logger.debug(f"scrub: list jobs failed: {e}")

        # 4. Wait for prior pods labeled with this deployment to drain.
        await self._wait_for_pods_terminated()

    async def _wait_for_cr_deleted(self, timeout: int = 120):
        """Poll the apiserver until the CR returns 404."""
        if not self._deployment_name or self._custom_api is None:
            return
        start = time.time()
        while (time.time() - start) < timeout:
            try:
                await self._custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
                await asyncio.sleep(2)
            except exceptions.ApiException as e:
                if e.status == 404:
                    return
                raise
        self._logger.warning(
            f"CR {self._deployment_name} deletion timed out after {timeout}s"
        )

    async def _wait_for_pods_terminated(self, timeout: int = 120):
        """Wait until no non-terminal pods remain for this deployment.

        The operator's pods are labeled with ``nvidia.com/selector=<deployment>-<service>``
        for each service; we walk every service and wait on each label set.
        """
        if self._core_api is None:
            return
        start = time.time()
        services = [s.name for s in self.deployment_spec.services]
        while (time.time() - start) < timeout:
            still_running: list[str] = []
            for service_name in services:
                selector = f"nvidia.com/selector={self._deployment_name}-{service_name.lower()}"
                try:
                    pods = await self._core_api.list_namespaced_pod(
                        namespace=self.namespace, label_selector=selector
                    )
                except exceptions.ApiException as e:
                    self._logger.debug(f"list pods failed for {selector}: {e}")
                    continue
                for p in pods.items:
                    if p.status.phase not in ("Succeeded", "Failed"):
                        still_running.append(p.metadata.name)
            if not still_running:
                return
            self._logger.info(
                f"Waiting for {len(still_running)} prior pods to terminate: "
                f"{still_running[:3]}{'…' if len(still_running) > 3 else ''}"
            )
            await asyncio.sleep(2)
        self._logger.warning(
            f"Pod termination timed out after {timeout}s — proceeding anyway"
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
            # Collect logs/metrics first; any PFs opened here will be tracked and stopped below.
            self._get_service_logs()
            self._logger.info(
                f"Cleaning up {len(self._active_port_forwards)} active port forwards"
            )
            for port_forward in self._active_port_forwards:
                try:
                    port_forward.stop()
                except RuntimeError as e:
                    # Expected error when pod is terminated:
                    # "anext(): asynchronous generator is already running"
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
            # Drain logs from PVC BEFORE deleting deployment + PVC, so we
            # never lose a run's logs to a teardown race.
            try:
                await self._extract_logs_from_pvc()
            except Exception as e:
                self._logger.warning(f"log extraction failed: {e}")
            try:
                await self._cleanup_orphaned_jobs()
            except Exception as e:
                self._logger.warning(f"orphaned job cleanup failed: {e}")
            await self._delete_deployment()
            try:
                await self._cleanup_log_collection_pvc()
            except Exception as e:
                self._logger.warning(f"log PVC cleanup failed: {e}")

    async def __aenter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self.deployment_spec.namespace = self.namespace
            self._deployment_name = self.deployment_spec.name
            logging.getLogger("httpx").setLevel(logging.WARNING)
            await self._init_kubernetes()

            # Scrub the namespace clean of any state left over from prior
            # runs (failed-mid-test deployments, orphan PVCs, stale load
            # jobs). Without this, residual pods can sit alongside the
            # current run's pods and confuse readiness checks.
            tasks = [self._scrub_namespace()]
            if not self.skip_service_restart:
                tasks.extend([self._restart_etcd(), self._restart_nats()])
            await asyncio.gather(*tasks)

            # Materialize the log-collection PVC declared by
            # DeploymentSpec.enable_log_collection. The spec carries the PVC
            # by reference (`create: false`) so the operator does not try to
            # create it; ManagedDeployment owns its lifecycle so the PVC and
            # the deployment go up and come down together.
            await self._create_log_collection_pvc()

            await self._create_deployment()
            await self._wait_for_ready()

        except:
            await self._cleanup()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def wait_for_ready(self, timeout: int = 1800, sleep=1, log_interval=60):
        """Public alias used by events.WaitForRecovery and RollingUpgrade.

        Same semantics as the private :py:meth:`_wait_for_ready` — waits for
        the DynamoGraphDeployment to reach Ready=True / state=successful.
        """
        return await self._wait_for_ready(
            timeout=timeout, sleep=sleep, log_interval=log_interval
        )

    def get_log_pvc_name(self) -> Optional[str]:
        """Return the configured log-collection PVC name, or None if not enabled.

        ManagedLoad uses this to mount the same PVC for storing aiperf
        artifacts so a single download job can collect everything.
        """
        return getattr(self.deployment_spec, "_log_collection_pvc_name", None)

    async def apply_service_changes(self, service_names: list[str]):
        """Push whatever in-memory mutations have been made on the named
        services to the cluster, in a single merge-patch.

        The caller mutates services via ``ServiceSpec`` setters (``image``,
        ``replicas``, ``set_env_var``, ``set_arg``, …); this method then
        publishes those changes. We push each service's full spec dict and
        let the apiserver's merge-patch semantics overlay them on the CR —
        unchanged fields on the server are preserved, mutated ones win.

        Used by RollingUpgrade and any other event that needs the operator
        to reconcile after spec edits.
        """
        if not service_names:
            raise ValueError("service_names cannot be empty for apply_service_changes")
        patch_body: dict = {"spec": {"services": {}}}
        for service_name in service_names:
            service = self.deployment_spec[service_name]
            patch_body["spec"]["services"][service_name] = service._spec
        assert self._custom_api is not None, "Kubernetes API not initialized"
        try:
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

    def _get_pod_manifest(self, pod: Pod, service_name: str, suffix: str = ""):
        """Save pod manifest YAML to log_dir/<service>/<pod>.yaml."""
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)
        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.yaml"), "w") as f:
                f.write(pod.to_yaml())
        except Exception as e:
            self._logger.error(e)

    async def _create_log_collection_pvc(self) -> Optional[str]:
        """Create the RWX log-collection PVC referenced by DeploymentSpec.

        No-op when the spec did not enable log collection. Recreates the PVC
        on every run so log content from a previous run does not leak into
        this one.

        Returns the PVC name on success, or ``None`` when log collection is
        not configured.
        """
        pvc_name = getattr(self.deployment_spec, "_log_collection_pvc_name", None)
        if not pvc_name:
            return None
        pvc_size = getattr(self.deployment_spec, "_log_collection_pvc_size", "1Gi")
        storage_class = getattr(
            self.deployment_spec, "_log_collection_storage_class", None
        )

        assert self._core_api is not None, "Kubernetes API not initialized"
        self._logger.info(
            f"Creating log-collection PVC {pvc_name} ({pvc_size}, "
            f"sc={storage_class or 'cluster-default'}, RWX)"
        )

        # Delete any prior incarnation so this run starts clean.
        try:
            await self._core_api.read_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            await self._core_api.delete_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            await asyncio.sleep(2)
        except exceptions.ApiException as e:
            if e.status != 404:
                raise

        pvc_spec: dict = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": pvc_name,
                "namespace": self.namespace,
                "labels": {
                    "managed-by": "managed-deployment",
                    "deployment": self.deployment_spec.name,
                    "purpose": "log-collection",
                },
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {"requests": {"storage": pvc_size}},
            },
        }
        if storage_class:
            pvc_spec["spec"]["storageClassName"] = storage_class

        await self._core_api.create_namespaced_persistent_volume_claim(
            namespace=self.namespace, body=pvc_spec
        )
        # Fast-fail if storage class doesn't actually deliver RWX — leaving
        # this implicit means pods get stuck in Pending instead of giving us
        # an actionable error here.
        await self._verify_pvc_binding(pvc_name)
        self._log_collection_pvc_created = True
        return pvc_name

    # ----- ported from _2 (full surface area) -----
    def _load_template(self, template_name: str) -> str:
        """Load a template file from the templates directory."""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        template_path = os.path.join(template_dir, template_name)
        with open(template_path, "r") as f:
            return f.read()

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


class ManagedDGDR:
    """Async helper for managing DynamoGraphDeploymentRequest custom resources.

    Provides CRUD operations and phase-polling against the DGDR CRD using the
    ``kubernetes_asyncio`` client, following the same patterns as
    ``ManagedDeployment`` (shared kubeconfig initialisation, timeout logic,
    structured error messages).

    Typical usage from a pytest fixture::

        dgdr = ManagedDGDR(namespace="default")
        await dgdr.init()
        await dgdr.create(manifest)
        phase = await dgdr.wait_for_phase(name, "Ready", timeout=600)
        await dgdr.delete(name)
        await dgdr.close()
    """

    # CRD coordinates for DGDR
    DGDR_GROUP = "nvidia.com"
    DGDR_VERSION = "v1beta1"
    DGDR_PLURAL = "dynamographdeploymentrequests"

    # CRD coordinates for DGD (for mocker cleanup)
    DGD_PLURAL = "dynamographdeployments"

    DEFAULT_POLL_INTERVAL = 10  # seconds

    def __init__(
        self,
        namespace: str = "default",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.namespace = namespace
        self._custom_api: Optional[client.CustomObjectsApi] = None
        self._api_client: Optional[client.ApiClient] = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._loop = loop

    def run(self, coro):
        """Run an async coroutine synchronously using the stored event loop.

        Convenience for callers that are not themselves async (e.g. pytest
        fixtures and synchronous test methods).
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop set on ManagedDGDR; pass loop= at construction or call init() first"
            )
        return self._loop.run_until_complete(coro)

    async def init(self) -> None:
        """Initialise the kubernetes_asyncio client.

        Priority: KUBECONFIG env → in-cluster → ~/.kube/config  (same as
        ManagedDeployment._init_kubernetes).
        """
        kubeconfig_path = os.environ.get("KUBECONFIG")

        if kubeconfig_path and os.path.exists(kubeconfig_path):
            self._logger.info("Loading kubeconfig from KUBECONFIG: %s", kubeconfig_path)
            await config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                self._logger.info("Attempting in-cluster kubernetes config")
                config.load_incluster_config()
            except Exception as e:
                self._logger.warning(
                    "In-cluster config failed (%s: %s), falling back to default kubeconfig",
                    type(e).__name__,
                    e,
                )
                await config.load_kube_config()

        self._api_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(self._api_client)

    async def close(self) -> None:
        """Close the underlying API client."""
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None

    # ----- CRUD -----

    async def create(self, manifest: dict) -> str:
        """Create a DGDR custom resource.  Returns the resource name."""
        assert self._custom_api is not None, "call init() first"
        name = manifest["metadata"]["name"]
        await self._custom_api.create_namespaced_custom_object(
            group=self.DGDR_GROUP,
            version=self.DGDR_VERSION,
            namespace=self.namespace,
            plural=self.DGDR_PLURAL,
            body=manifest,
        )
        self._logger.info("Created DGDR %s/%s", self.namespace, name)
        return name

    async def get(self, name: str) -> Optional[dict]:
        """Get a DGDR as a dict, or ``None`` if not found."""
        assert self._custom_api is not None, "call init() first"
        try:
            return await self._custom_api.get_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version=self.DGDR_VERSION,
                namespace=self.namespace,
                plural=self.DGDR_PLURAL,
                name=name,
            )
        except exceptions.ApiException as e:
            if e.status == 404:
                return None
            raise

    async def delete(self, name: str, ignore_not_found: bool = True) -> None:
        """Delete a DGDR."""
        assert self._custom_api is not None, "call init() first"
        try:
            await self._custom_api.delete_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version=self.DGDR_VERSION,
                namespace=self.namespace,
                plural=self.DGDR_PLURAL,
                name=name,
            )
            self._logger.info("Deleted DGDR %s/%s", self.namespace, name)
        except exceptions.ApiException as e:
            if e.status == 404 and ignore_not_found:
                return
            raise

    async def list(self, label_selector: str = "") -> List[dict]:
        """List DGDRs, optionally filtered by label selector.  Returns items."""
        assert self._custom_api is not None, "call init() first"
        resp = await self._custom_api.list_namespaced_custom_object(
            group=self.DGDR_GROUP,
            version=self.DGDR_VERSION,
            namespace=self.namespace,
            plural=self.DGDR_PLURAL,
            label_selector=label_selector,
        )
        return resp.get("items", [])

    async def server_dry_run(self, manifest: dict) -> dict:
        """Apply with server-side dry-run to validate admission webhooks.

        Returns the API response dict.  Raises ``ApiException`` on rejection.
        """
        assert self._custom_api is not None, "call init() first"
        return await self._custom_api.create_namespaced_custom_object(
            group=self.DGDR_GROUP,
            version=self.DGDR_VERSION,
            namespace=self.namespace,
            plural=self.DGDR_PLURAL,
            body=manifest,
            dry_run="All",
        )

    # ----- Phase helpers -----

    async def get_phase(self, name: str) -> Optional[str]:
        """Return ``status.phase`` of the named DGDR, or ``None``."""
        obj = await self.get(name)
        if obj is None:
            return None
        return obj.get("status", {}).get("phase")

    async def get_condition(self, name: str, condition_type: str) -> Optional[dict]:
        """Return the named condition dict from ``status.conditions``."""
        obj = await self.get(name)
        if obj is None:
            return None
        for c in obj.get("status", {}).get("conditions", []):
            if c.get("type") == condition_type:
                return c
        return None

    async def wait_for_phase(
        self,
        name: str,
        target_phase: str,
        timeout: int = 3600,
        fail_fast_phases: Optional[List[str]] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> str:
        """Poll until the DGDR reaches *target_phase* or times out.

        Returns the final observed phase.  Raises ``AssertionError`` on
        fail-fast and ``TimeoutError`` on timeout.
        """
        if fail_fast_phases is None:
            fail_fast_phases = ["Failed"]

        deadline = time.monotonic() + timeout
        last_phase: Optional[str] = None

        while time.monotonic() < deadline:
            current = await self.get_phase(name)
            if current != last_phase:
                self._logger.info("DGDR %s/%s phase: %s", self.namespace, name, current)
                last_phase = current

            if current == target_phase:
                return current
            if current in fail_fast_phases:
                obj = await self.get(name)
                conditions = obj.get("status", {}).get("conditions", []) if obj else []
                raise AssertionError(
                    f"DGDR {self.namespace}/{name} reached fail-fast phase {current!r} "
                    f"while waiting for {target_phase!r}. conditions={conditions}"
                )
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {timeout}s waiting for DGDR {self.namespace}/{name} "
            f"to reach phase {target_phase!r}. Last phase: {last_phase!r}"
        )

    async def wait_for_any_phase(
        self,
        name: str,
        target_phases: List[str],
        timeout: int = 3600,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> str:
        """Poll until the DGDR reaches any of *target_phases*.  Returns matched phase."""
        deadline = time.monotonic() + timeout
        last_phase: Optional[str] = None

        while time.monotonic() < deadline:
            current = await self.get_phase(name)
            if current != last_phase:
                self._logger.info("DGDR %s/%s phase: %s", self.namespace, name, current)
                last_phase = current
            if current in target_phases:
                return current
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {timeout}s waiting for DGDR {self.namespace}/{name} "
            f"to reach any of {target_phases!r}. Last phase: {last_phase!r}"
        )

    # ----- DGD helpers (for mocker cleanup) -----

    async def delete_dgd(self, name: str, ignore_not_found: bool = True) -> None:
        """Delete a DynamoGraphDeployment resource."""
        assert self._custom_api is not None, "call init() first"
        try:
            await self._custom_api.delete_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version="v1alpha1",
                namespace=self.namespace,
                plural=self.DGD_PLURAL,
                name=name,
            )
            self._logger.info("Deleted DGD %s/%s", self.namespace, name)
        except exceptions.ApiException as e:
            if e.status == 404 and ignore_not_found:
                return
            raise

    async def get_dgd(self, name: str) -> Optional[dict]:
        """Get a DynamoGraphDeployment, or ``None`` if not found."""
        assert self._custom_api is not None, "call init() first"
        try:
            return await self._custom_api.get_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version="v1alpha1",
                namespace=self.namespace,
                plural=self.DGD_PLURAL,
                name=name,
            )
        except exceptions.ApiException as e:
            if e.status == 404:
                return None
            raise


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
