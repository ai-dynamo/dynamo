# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any, Optional

import kr8s
import kubernetes
import requests
import yaml
from kr8s.objects import Pod as kr8s_Pod
from kr8s.objects import Service as kr8s_Service
from kubernetes_asyncio import client, config


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

    # ----- Replicas -----
    @property
    def replicas(self) -> int:
        return self._spec.get("replicas", 0)

    @replicas.setter
    def replicas(self, value: int):
        self._spec["replicas"] = value

    @property
    def model(self) -> Optional[str]:
        """Model being served by this service"""
        try:
            args_list = self._spec["extraPodSpec"]["mainContainer"]["args"]
        except KeyError:
            return None
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)
        for i, part in enumerate(parts):
            if part == "--model":
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

        model_index = None
        for i, part in enumerate(parts):
            if part == "--model":
                model_index = i
                break

        if model_index is not None:
            if model_index + 1 < len(parts):
                parts[model_index + 1] = value
            else:
                return
        else:
            return

        self._spec["extraPodSpec"]["mainContainer"]["args"] = [" ".join(parts)]

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

        self._spec["extraPodSpec"]["mainContainer"]["args"] = [" ".join(parts)]

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
    def services(self) -> list:
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

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)


class PodProcess:
    def __init__(self, pod: kr8s_Pod, line: str):
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
    frontend_service_name: Optional[str] = "Frontend"

    _custom_api: Optional[Any] = None
    _core_api: Optional[Any] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    _deployment_name: Optional[str] = None
    _apps_v1: Optional[Any] = None

    def __post_init__(self):
        self._deployment_name = self.deployment_spec.name

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            await config.load_incluster_config()
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

    async def _wait_for_ready(self, timeout: int = 1800, sleep=1, log_interval=60):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        start_time = time.time()

        self._logger.info(f"Waiting for Deployment {self._deployment_name}")

        attempt = 0

        while (time.time() - start_time) < timeout:
            try:
                attempt += 1
                assert self._custom_api is not None, "Kubernetes API not initialized"
                status = await self._custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
                # Check both conditions:
                # 1. Ready condition is True
                # 2. State is successful
                status_obj = status.get("status", {})
                conditions = status_obj.get("conditions", [])
                current_state = status_obj.get("state", "unknown")

                ready_condition = False
                for condition in conditions:
                    if (
                        condition.get("type") == "Ready"
                        and condition.get("status") == "True"
                    ):
                        ready_condition = True
                        break

                state_successful = status_obj.get("state") == "successful"

                if ready_condition and state_successful:
                    self._logger.info(f"Current deployment state: {current_state}")
                    self._logger.info(f"Current conditions: {conditions}")
                    self._logger.info(
                        f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                    )

                    self._logger.info(f"Deployment {self._deployment_name} is ready")
                    return True
                else:
                    if attempt % log_interval == 0:
                        self._logger.info(f"Current deployment state: {current_state}")
                        self._logger.info(f"Current conditions: {conditions}")
                        self._logger.info(
                            f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                        )
                        self._logger.info(
                            f"Deployment not ready yet - Ready condition: {ready_condition}, State successful: {state_successful}"
                        )

            except kubernetes.client.rest.ApiException as e:
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
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.info(f"Deployment {self._deployment_name} already exists")
            else:
                self._logger.info(
                    f"Failed to create deployment {self._deployment_name}: {e}"
                )
                raise

    def get_processes(self, pod) -> list:
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

        return kr8s_Service.get(full_service_name, namespace=self.namespace)

    def get_pods(self, service_name=None):
        result = {}

        service_list = []

        if not service_name:
            service_list = [service.name for service in self.deployment_spec.services]
        else:
            service_list = [service_name]

        for service in service_list:
            # List pods for this service using the selector label
            # nvidia.com/selector: deployment-name-service
            label_selector = (
                f"nvidia.com/selector={self._deployment_name}-{service.lower()}"
            )

            pods = []

            for pod in kr8s.get(
                "pods", namespace=self.namespace, label_selector=label_selector
            ):
                pods.append(pod)

            result[service] = pods

        return result

    def get_pod_logs(self, service, pod, suffix=""):
        directory = os.path.join(self.log_dir, service)
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

        self._get_pod_metrics(pod, service, suffix)

    def _get_service_logs(self, service_name=None, suffix=""):
        service_pods = self.get_pods(service_name)

        for service, pods in service_pods.items():
            for i, pod in enumerate(pods):
                self.get_pod_logs(service, pod, suffix)

    def _get_pod_metrics(self, pod, service_name, suffix=""):
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
        except client.exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

    def port_forward(self, pod, remote_port, max_connection_attempts=3):
        """Attempt to connect to a pod and return the port-forward object on success."""
        port_forward = pod.portforward(
            remote_port=remote_port,
            local_port=0,
            address="0.0.0.0",
        )
        port_forward.start()

        for _ in range(max_connection_attempts):
            if port_forward.local_port == 0:
                time.sleep(1)
                continue

            test_url = f"http://localhost:{port_forward.local_port}/"
            try:
                # Send HEAD request to test connection
                response = requests.head(test_url, timeout=5)
                if response.status_code in (200, 404):  # 404 is acceptable
                    return port_forward
            except (requests.ConnectionError, requests.Timeout) as e:
                self._logger.warning(f"Connection test failed for pod {pod.name}: {e}")

            # Retry port-forward
            port_forward.stop()
            port_forward.start()
            time.sleep(1)

        # All attempts failed
        port_forward.stop()
        return None

    async def _cleanup(self):
        try:
            self._get_service_logs()
        finally:
            await self._delete_deployment()

    async def __aenter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self.deployment_spec.namespace = self.namespace
            self._deployment_name = self.deployment_spec.name
            logging.getLogger("httpx").setLevel(logging.WARNING)
            await self._init_kubernetes()
            await self._delete_deployment()
            await self._restart_etcd()
            await self._restart_nats()
            await self._create_deployment()
            await self._wait_for_ready()

        except:
            await self._cleanup()
            raise
        return self

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

    deployment_spec = DeploymentSpec(
        "/workspace/components/backends/vllm/deploy/agg.yaml"
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
