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
from typing import Any, Dict, List, Optional

import kr8s
import requests
import yaml
from kr8s.objects import Pod, Service
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

# Hardware fault injection support (optional)
try:
    from tests.utils.hw_fault_helpers import HWFaultConfig, HWFaultManager

    HW_FAULTS_AVAILABLE = True
except ImportError:
    HW_FAULTS_AVAILABLE = False
    HWFaultConfig = None
    HWFaultManager = None


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

    def set_dynamo_namespace(self, dynamo_ns: str):
        """Set the dynamoNamespace for all services.

        This provides logical isolation within a Kubernetes namespace,
        ensuring services only discover and communicate with each other.

        Args:
            dynamo_ns: The dynamo namespace to set for all services
        """
        for service_name, service_spec in self._deployment_spec["spec"][
            "services"
        ].items():
            service_spec["dynamoNamespace"] = dynamo_ns

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

    def set_service_env_var(self, service_name: str, name: str, value: str):
        """
        Set an environment variable for a specific service
        """
        service = self.get_service(service_name)
        envs = service.envs if service.envs is not None else []

        # if env var already exists, update it
        for env in envs:
            if env["name"] == name:
                env["value"] = value
                service.envs = envs  # Save back to trigger the setter
                return

        # if env var does not exist, add it
        envs.append({"name": name, "value": value})
        service.envs = envs  # Save back to trigger the setter

    def get_service_env_vars(self, service_name: str) -> list[dict]:
        """
        Get all environment variables for a specific service

        Returns:
            List of environment variable dicts (e.g., [{"name": "VAR", "value": "val"}])
        """
        service = self.get_service(service_name)
        return service.envs

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

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)


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
    # TODO: this should be determined by the deployment_spec
    # the service containing component_type: Frontend determines what is actually the frontend service
    frontend_service_name: str = "Frontend"
    skip_service_restart: bool = False
    
    # Hardware fault injection support
    enable_hw_faults: bool = False
    hw_fault_config: Optional[Dict[str, Any]] = None

    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    _deployment_name: Optional[str] = None
    _apps_v1: Optional[Any] = None
    _active_port_forwards: List[Any] = field(default_factory=list)
    _hw_fault_manager: Optional[Any] = None

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

self._logger.info(f"Waiting for DGD {self._deployment_name} to be ready...")

        attempt = 0
        expected_pods = self._get_expected_pod_count()

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

# Also check if all pods are actually running (fallback for slow DGD updates)
                ready_pods, total_pods = await self._count_ready_pods()
                pods_ready = ready_pods >= expected_pods and ready_pods == total_pods

                ready_condition = observed_ready_condition_val == str(desired_ready_condition_val)
                state_successful = observed_state_val == desired_state_val

                if ready_condition and state_successful:
                    self._logger.info(f"✓ DGD ready ({time.time() - start_time:.0f}s)")
                    return True
                elif pods_ready and not (ready_condition and state_successful):
                    # Pods are ready but DGD status hasn't updated yet - use pods as source of truth
                    self._logger.info(
                        f"✓ Pods ready ({ready_pods}/{expected_pods}) ({time.time() - start_time:.0f}s)"
                    )
                    self._logger.info(
                        f"  (DGD status pending update: state={current_state}, Ready={ready_condition})"
                    )
                    return True
                else:
                    if attempt % log_interval == 0:
self._logger.info(
                            f"  [{time.time() - start_time:.0f}s] Pods: {ready_pods}/{expected_pods}, State: {current_state}"
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

    def _get_expected_pod_count(self) -> int:
        """Get the expected number of pods from the deployment spec."""
        total = 0
        for service in self.deployment_spec.services:
            total += service.replicas
        return total

    async def _count_ready_pods(self) -> tuple:
        """Count ready and total pods for this deployment."""
        try:
            assert self._core_api is not None
            label_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name}"
            )
            pods = await self._core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector,
            )
            total = len(pods.items)
            ready = 0
            for pod in pods.items:
                if pod.status and pod.status.phase == "Running":
                    # Check all containers are ready
                    if pod.status.container_statuses:
                        all_ready = all(
                            cs.ready for cs in pod.status.container_statuses
                        )
                        if all_ready:
                            ready += 1
            return ready, total
        except Exception as e:
            self._logger.debug(f"Error counting pods: {e}")
            return 0, 0

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
        """
        Triggers a rolling update for a list of services
        This is a dummy update - sets an env var on the service
        """

        if not service_names:
            raise ValueError(
                "service_names cannot be empty for trigger_rolling_upgrade"
            )

        patch_body: dict[str, Any] = {"spec": {"services": {}}}

        for service_name in service_names:
            self.deployment_spec.set_service_env_var(
                service_name, "TEST_ROLLING_UPDATE_TRIGGER", secrets.token_hex(8)
            )

            updated_envs = self.deployment_spec.get_service_env_vars(service_name)
            patch_body["spec"]["services"][service_name] = {"envs": updated_envs}

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

            # Cleanup hardware fault manager (uncordon nodes, remove patches, etc.)
            if self._hw_fault_manager:
                try:
                    self._logger.info(
                        "[HW Faults] Cleaning up hardware fault artifacts..."
                    )
                    await self._hw_fault_manager.cleanup()
                    self._logger.info("[HW Faults] ✓ Hardware fault cleanup complete")
                except Exception as e:
                    self._logger.warning(f"[HW Faults] Cleanup error (non-fatal): {e}")

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
            await self._delete_deployment()

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

            await self._create_deployment()
            await self._wait_for_ready()

            # Initialize hardware fault manager if enabled
            if self.enable_hw_faults:
                await self._init_hw_fault_manager()

        except:
            await self._cleanup()
            raise
        return self

    async def _init_hw_fault_manager(self):
        """Initialize hardware fault injection manager."""
        if not HW_FAULTS_AVAILABLE:
            self._logger.warning(
                "[HW Faults] hw_fault_helpers not available - HW fault features disabled"
            )
            return

        self._logger.info("[HW Faults] Initializing hardware fault manager...")

        # Create config from dict if provided
        config = (
            HWFaultConfig.from_dict(self.hw_fault_config)
            if self.hw_fault_config
            else HWFaultConfig()
        )

        self._hw_fault_manager = HWFaultManager(
            deployment_name=self._deployment_name,
            namespace=self.namespace,
            config=config,
            logger=self._logger,
        )

        # Setup CUDA fault injection library
        if await self._hw_fault_manager.setup():
            self._logger.info("[HW Faults] ✓ Hardware fault manager ready")
        else:
            self._logger.warning("[HW Faults] Failed to setup hardware fault manager")

    # =========================================================================
    # Hardware Fault Injection Methods
    # =========================================================================

    async def setup_cuda_passthrough(self, xid_type: int = 79) -> bool:
        """
        Setup CUDA library in passthrough mode (faults disabled).

        Patches deployment to load the CUDA intercept library with faults DISABLED.
        Pods will restart ONCE to load the library. After that, use toggle_cuda_faults()
        to enable/disable faults without additional restarts.

        Args:
            xid_type: XID error type to configure (79, 48, 94, 95, 43, 74)

        Returns:
            True if successful
        """
        if not self._hw_fault_manager:
            self._logger.error("[HW Faults] Hardware fault manager not initialized")
            return False

        return await self._hw_fault_manager.setup_cuda_passthrough(xid_type=xid_type)

    async def toggle_cuda_faults(self, enable: bool = True) -> bool:
        """
        Toggle CUDA faults ON or OFF without restarting pods.

        Writes to hostPath file that the CUDA intercept library reads at runtime.
        Requires setup_cuda_passthrough() to have been called first.

        Args:
            enable: True to enable faults, False to disable

        Returns:
            True if toggle succeeded
        """
        if not self._hw_fault_manager:
            self._logger.error("[HW Faults] Hardware fault manager not initialized")
            return False

        return await self._hw_fault_manager.toggle_cuda_faults(enable=enable)

    async def inject_hw_fault(
        self,
        fault_type: str = "xid",
        xid_type: int = 79,
        gpu_id: int = 0,
    ) -> Optional[str]:
        """
        Inject hardware fault via fault injection API.

        This is OPTIONAL - if the fault injection API is not running, XID injection
        will be skipped. CUDA faults via LD_PRELOAD can still be used for testing.

        Args:
            fault_type: Type of fault ('xid' for now)
            xid_type: XID error type (79, 48, 94, 95, 43, 74)
            gpu_id: GPU ID to target

        Returns:
            Fault ID if successful, None otherwise
        """
        if not self._hw_fault_manager:
            self._logger.error("[HW Faults] Hardware fault manager not initialized")
            return None

        if fault_type == "xid":
            return await self._hw_fault_manager.inject_xid_fault(
                xid_type=xid_type,
                gpu_id=gpu_id,
            )
        else:
            self._logger.error(f"[HW Faults] Unknown fault type: {fault_type}")
            return None

    def get_hw_fault_target_node(self) -> Optional[str]:
        """
        Get the target node for hardware fault injection.

        Returns the node where worker pods are running (for fault targeting).
        Works with both vLLM (VllmDecodeWorker, VllmPrefillWorker) and
        SGLang (decode, prefill) service names.

        Returns:
            Node name or None if not available
        """
        if self._hw_fault_manager and self._hw_fault_manager._target_node:
            return self._hw_fault_manager._target_node

        # Auto-detect from running pods
        # Check for worker services: vLLM uses "Worker" suffix, SGLang uses "decode"/"prefill"
        worker_indicators = ["worker", "decode", "prefill"]

        pods = self.get_pods()
        self._logger.info(
            f"[HW Faults] Scanning {len(pods)} services for target node: {list(pods.keys())}"
        )

        for service_name, service_pods in pods.items():
            service_lower = service_name.lower()
            is_worker = any(ind in service_lower for ind in worker_indicators)

            if is_worker:
                self._logger.info(
                    f"[HW Faults] Checking {len(service_pods)} pods in {service_name}"
                )
                for pod in service_pods:
                    try:
                        # kr8s pods use .spec.nodeName (camelCase) - same as in test log_pod_status
                        node_name = None
                        if hasattr(pod, "spec") and hasattr(pod.spec, "nodeName"):
                            node_name = pod.spec.nodeName

                        if node_name:
                            self._logger.info(
                                f"[HW Faults] Auto-detected target node from {service_name}/{pod.name}: {node_name}"
                            )
                            # Store in manager for later use
                            if self._hw_fault_manager:
                                self._hw_fault_manager.set_target_node(node_name)
                            return node_name
                        else:
                            self._logger.info(
                                f"[HW Faults] Pod {pod.name}: no nodeName assigned yet"
                            )

                    except Exception as e:
                        self._logger.warning(
                            f"[HW Faults] Failed to get node from pod {getattr(pod, 'name', 'unknown')}: {e}"
                        )
                        continue

        self._logger.warning("[HW Faults] Could not auto-detect target node from pods")
        return None

    async def remove_node_affinity(self) -> bool:
        """
        Remove node affinity from worker pods to allow rescheduling.

        This is needed during recovery because:
        1. CUDA passthrough setup may pin workers to a specific node
        2. When that node is cordoned, workers can't reschedule
        3. Removing affinity allows workers to schedule on healthy nodes

        Returns:
            True if successful
        """
        if not self._hw_fault_manager:
            self._logger.error("[HW Faults] Hardware fault manager not initialized")
            return False

        return await self._hw_fault_manager.remove_node_affinity()

    async def cleanup_cuda_spec_without_restart(self) -> bool:
        """
        Clean up CUDA fault injection from deployment spec WITHOUT restarting pods.
        
        Used when relying on NVSentinel's node-drainer to evict pods.
        The spec is cleaned so new pods (after eviction) will be clean.
        
        Returns:
            True if successful
        """
        if not self._hw_fault_manager:
            self._logger.error("[HW Faults] Hardware fault manager not initialized")
            return False

        return await self._hw_fault_manager.cleanup_cuda_spec_without_restart()

    def is_node_cordoned(self, node_name: str) -> bool:
        """
        Check if a node is cordoned (unschedulable).

        Args:
            node_name: Name of the node to check

        Returns:
            True if node is cordoned
        """
        if self._hw_fault_manager:
            return self._hw_fault_manager.is_node_cordoned(node_name)

        # Fallback: direct k8s check
        try:
            from kubernetes import client as sync_client, config as sync_config

            sync_config.load_kube_config()
            v1 = sync_client.CoreV1Api()
            node = v1.read_node(node_name)
            return node.spec.unschedulable or False
        except Exception as e:
            self._logger.error(f"[HW Faults] Failed to check node status: {e}")
            return False

    async def wait_for_all_pods_ready(
        self, timeout: int = 300, min_pods: int = None
    ) -> bool:
        """
        Wait for all pods in the deployment to be ready.

        Args:
            timeout: Maximum time to wait in seconds
            min_pods: Minimum number of pods to wait for (uses expected count if None)

        Returns:
            True if all pods ready, raises TimeoutError otherwise
        """
        label = f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name}"

        # Use expected pod count from deployment spec if not specified
        expected_pods = (
            min_pods if min_pods is not None else self._get_expected_pod_count()
        )

        start_time = time.time()
        last_log_time = 0  # Track when we last logged
        log_interval = 30  # Log every 30 seconds
        
        while (time.time() - start_time) < timeout:
            try:
                assert self._core_api is not None
                pods = await self._core_api.list_namespaced_pod(
                    self.namespace, label_selector=label
                )

                total_pods = len(pods.items)
                ready_pods = sum(
                    1
                    for pod in pods.items
                    if any(
                        cond.type == "Ready" and cond.status == "True"
                        for cond in (pod.status.conditions or [])
                    )
                )

                # Wait for expected number of pods to all be ready
                if (
                    ready_pods >= expected_pods
                    and ready_pods == total_pods
                    and total_pods >= expected_pods
                ):
                    self._logger.info(f"[HW Faults] All {total_pods} pods ready")
                    return True

                elapsed = int(time.time() - start_time)
                # Log progress every 30 seconds
                if elapsed - last_log_time >= log_interval:
                    last_log_time = elapsed
                    self._logger.info(
                        f"[HW Faults] [{elapsed}s/{timeout}s] Waiting for pods: {ready_pods}/{total_pods} ready (expecting {expected_pods})"
                    )

            except Exception as e:
                self._logger.debug(f"[HW Faults] Error checking pods: {e}")

            await asyncio.sleep(5)

        raise TimeoutError(f"Pods not ready within {timeout}s")

    async def wait_for_pods_on_healthy_nodes(
        self,
        exclude_node: str,
        timeout: int = 360,
    ) -> bool:
        """
        Wait for pods to be scheduled on healthy nodes (not the excluded node).

        Args:
            exclude_node: Node to exclude (typically the cordoned/faulty node)
            timeout: Maximum time to wait in seconds

        Returns:
            True if pods rescheduled successfully
        """
        label = f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name}"

        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                assert self._core_api is not None
                pods = await self._core_api.list_namespaced_pod(
                    self.namespace, label_selector=label
                )

                # Check if any pod is still on the excluded node
                # Note: kubernetes_asyncio uses snake_case (node_name) but kr8s uses camelCase (nodeName)
                pods_on_excluded = []
                all_pod_nodes = []
                for pod in pods.items:
                    # Try both attribute styles
                    node = getattr(pod.spec, 'node_name', None) or getattr(pod.spec, 'nodeName', None)
                    all_pod_nodes.append(f"{pod.metadata.name}={node}")
                    if node == exclude_node:
                        pods_on_excluded.append(pod.metadata.name)

                if not pods_on_excluded:
                    self._logger.info(
                        f"[HW Faults] All pods rescheduled off {exclude_node}"
                    )
                    return True

                # Debug: show where pods actually are
                elapsed = int(time.time() - start_time)
                if elapsed % 30 < 6:  # Log every ~30s
                    self._logger.info(
                        f"[HW Faults] [{elapsed}s] Waiting for pods to leave {exclude_node}: {pods_on_excluded}"
                    )
                    self._logger.debug(f"[HW Faults] Pod locations: {all_pod_nodes}")

            except Exception as e:
                self._logger.debug(f"[HW Faults] Error checking pods: {e}")

            await asyncio.sleep(5)

        self._logger.warning(
            f"[HW Faults] Timeout waiting for pods to leave {exclude_node}"
        )
        return False

    def collect_metrics(self, phase: str = ""):
        """
        Collect metrics from all pods for the given phase.

        Args:
            phase: Phase name for labeling metrics files
        """
        suffix = f".{phase}" if phase else ""
        self._get_service_logs(suffix=suffix)

    # =========================================================================
    # End Hardware Fault Injection Methods
    # =========================================================================

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()


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
