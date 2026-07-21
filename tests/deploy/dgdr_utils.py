# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for live-cluster DynamoGraphDeploymentRequest tests."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import yaml
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

logger = logging.getLogger(__name__)

DGDR_GROUP = "nvidia.com"
DGDR_VERSION = "v1beta1"
DGDR_PLURAL = "dynamographdeploymentrequests"
DGD_VERSION = "v1beta1"
DGD_PLURAL = "dynamographdeployments"
DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_DGD_SUFFIX = "-dgd"
_MAX_PROFILER_RESOURCE_NAME_LENGTH = 45
_LONGEST_PROFILER_SERVICE_NAME = "TRTLLMPrefillWorker"
_MAX_DGDR_NAME_LENGTH = (
    _MAX_PROFILER_RESOURCE_NAME_LENGTH
    - len(_DGD_SUFFIX)
    - len(_LONGEST_PROFILER_SERVICE_NAME)
)
_MAX_LABEL_VALUE_LENGTH = 63
_NAME_DIGEST_LENGTH = 6

PHASE_ORDER = {
    "Pending": 0,
    "Profiling": 1,
    "Ready": 2,
    "Deploying": 3,
    "Deployed": 4,
    "Failed": -1,
}

DEFAULT_MOCKER_HARDWARE = {
    "gpuSku": "h100_sxm",
    "vramMb": 81920.0,
    "numGpusPerNode": 8,
    "totalGpus": 8,
}


@dataclass(frozen=True)
class DGDRTestConfig:
    """Configuration shared by the ported DGDR tests."""

    namespace: str
    image: str
    model: str = "Qwen/Qwen3-0.6B"
    backend: str = "vllm"
    mocker: bool = True
    profiling_timeout: int = 3600
    deploy_timeout: int = 600
    name_prefix: str = ""
    pvc_name: str = ""
    pvc_model_path: str = ""
    pvc_mount_path: str = "/home/dynamo/.cache/huggingface"
    total_gpus: int = 0
    hf_token_secret: str = ""


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def unique_name(config_: DGDRTestConfig, suffix: str) -> str:
    """Return a short, collision-resistant Kubernetes resource name."""

    # Bound both profiler "<dgdr>-dgd<Service>" names and Grove
    # "<namespace>-<dgdr>-dgd" label values.
    namespace_limit = (
        _MAX_LABEL_VALUE_LENGTH - len(config_.namespace) - len("-") - len(_DGD_SUFFIX)
    )
    max_length = min(_MAX_DGDR_NAME_LENGTH, namespace_limit)
    stem_length = max_length - _NAME_DIGEST_LENGTH - len("-")
    if stem_length < 1:
        raise ValueError(
            f"namespace {config_.namespace!r} is too long for DGDR test resources"
        )

    stem = (config_.name_prefix or f"dgdr-{suffix}")[:stem_length].rstrip("-") or "d"
    digest_source = f"{config_.name_prefix}:{suffix}:{time.time_ns()}"
    digest = hashlib.sha1(digest_source.encode(), usedforsecurity=False).hexdigest()[
        :_NAME_DIGEST_LENGTH
    ]
    return f"{stem}-{digest}"


def build_dgdr(
    config_: DGDRTestConfig,
    name: str,
    spec_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v1beta1 DGDR manifest with mocker or real-GPU defaults."""

    spec: dict[str, Any] = {
        "model": config_.model,
        "backend": config_.backend,
        "image": config_.image,
        "searchStrategy": "rapid",
    }
    if spec_overrides:
        _deep_merge(spec, spec_overrides)

    if config_.mocker:
        features = spec.setdefault("features", {})
        features.setdefault("mocker", {"enabled": True})
        hardware = spec.setdefault("hardware", {})
        for key, value in DEFAULT_MOCKER_HARDWARE.items():
            hardware.setdefault(key, value)
    else:
        if config_.pvc_name and "modelCache" not in spec:
            spec["modelCache"] = {
                "pvcName": config_.pvc_name,
                "pvcModelPath": config_.pvc_model_path,
                "pvcMountPath": config_.pvc_mount_path,
            }
        if config_.total_gpus:
            spec.setdefault("hardware", {}).setdefault("totalGpus", config_.total_gpus)
        if config_.hf_token_secret:
            profiling_job = spec.setdefault("overrides", {}).setdefault(
                "profilingJob", {}
            )
            profiling_job.setdefault("template", {}).setdefault("spec", {})[
                "containers"
            ] = [
                {
                    "name": "profiler",
                    "env": [
                        {
                            "name": "HF_TOKEN",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": config_.hf_token_secret,
                                    "key": "HF_TOKEN",
                                }
                            },
                        }
                    ],
                }
            ]

    return {
        "apiVersion": f"{DGDR_GROUP}/{DGDR_VERSION}",
        "kind": "DynamoGraphDeploymentRequest",
        "metadata": {
            "name": name,
            "namespace": config_.namespace,
            "labels": {"test.dynamo/managed": "true"},
        },
        "spec": spec,
    }


def parse_final_dgd(content: str) -> dict[str, Any]:
    """Return the last YAML document after validating that it is a DGD."""

    documents = [document for document in yaml.safe_load_all(content) if document]
    if not documents:
        raise AssertionError(
            "final_config.yaml must contain at least one YAML document"
        )
    result = documents[-1]
    if result.get("kind") != "DynamoGraphDeployment":
        raise AssertionError(
            "last final_config.yaml document must be a DynamoGraphDeployment"
        )
    return result


def total_worker_gpus(dgd: dict[str, Any]) -> int:
    """Compute requested GPUs across v1alpha1 services or v1beta1 components."""

    spec = dgd.get("spec", {})
    services = spec.get("services", {})
    if services:
        return sum(
            int(service.get("replicas", 0))
            * int(service.get("resources", {}).get("limits", {}).get("gpu", 0))
            for service in services.values()
        )

    total = 0
    for component in spec.get("components", []):
        replicas = int(component.get("replicas", 0))
        containers = (
            component.get("podTemplate", {}).get("spec", {}).get("containers", [])
        )
        for container in containers:
            limits = container.get("resources", {}).get("limits", {})
            total += replicas * int(limits.get("nvidia.com/gpu", 0))
    return total


def kubectl(*args: str, input_: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run kubectl without a shell and retain stdout/stderr for assertions."""

    return subprocess.run(
        ["kubectl", *args],
        input=input_,
        text=True,
        capture_output=True,
        check=False,
    )


class DGDRCleanupError(RuntimeError):
    """Report every DGDR that could not be cleaned up."""

    def __init__(self, namespace: str, failures: list[tuple[str, Exception]]):
        self.failures = tuple(failures)
        details = "; ".join(
            f"{namespace}/{name}: {type(error).__name__}: {error}"
            for name, error in failures
        )
        super().__init__(f"Failed to clean up DGDR resources: {details}")


class ManagedDGDR:
    """Own DGDR resources for one pytest item and clean them up idempotently."""

    def __init__(self, config_: DGDRTestConfig):
        self.config = config_
        self._api_client: client.ApiClient | None = None
        self.custom: client.CustomObjectsApi | None = None
        self.core: client.CoreV1Api | None = None
        self.batch: client.BatchV1Api | None = None
        self.apiextensions: client.ApiextensionsV1Api | None = None
        self._created_names: list[str] = []

    async def init(self) -> None:
        kubeconfig = os.environ.get("KUBECONFIG")
        if kubeconfig and os.path.exists(kubeconfig):
            await config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                await config.load_kube_config()

        self._api_client = client.ApiClient()
        self.custom = client.CustomObjectsApi(self._api_client)
        self.core = client.CoreV1Api(self._api_client)
        self.batch = client.BatchV1Api(self._api_client)
        self.apiextensions = client.ApiextensionsV1Api(self._api_client)

    async def close(self) -> None:
        if self._api_client:
            await self._api_client.close()
        self._api_client = None

    def _require_clients(self) -> None:
        assert self.custom is not None, "call init() first"
        assert self.core is not None, "call init() first"
        assert self.batch is not None, "call init() first"
        assert self.apiextensions is not None, "call init() first"

    async def create(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._require_clients()
        name = manifest["metadata"]["name"]
        await self.ensure_absent(name)
        assert self.custom is not None
        result = await self.custom.create_namespaced_custom_object(
            group=DGDR_GROUP,
            version=DGDR_VERSION,
            namespace=self.config.namespace,
            plural=DGDR_PLURAL,
            body=manifest,
        )
        self._created_names.append(name)
        logger.info("Created DGDR %s/%s", self.config.namespace, name)
        return result

    async def dry_run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._require_clients()
        assert self.custom is not None
        return await self.custom.create_namespaced_custom_object(
            group=DGDR_GROUP,
            version=DGDR_VERSION,
            namespace=self.config.namespace,
            plural=DGDR_PLURAL,
            body=manifest,
            dry_run="All",
        )

    async def get(self, name: str) -> dict[str, Any] | None:
        self._require_clients()
        assert self.custom is not None
        try:
            return await self.custom.get_namespaced_custom_object(
                group=DGDR_GROUP,
                version=DGDR_VERSION,
                namespace=self.config.namespace,
                plural=DGDR_PLURAL,
                name=name,
            )
        except exceptions.ApiException as error:
            if error.status == 404:
                return None
            raise

    async def get_versioned(self, name: str, version: str) -> dict[str, Any]:
        self._require_clients()
        assert self.custom is not None
        return await self.custom.get_namespaced_custom_object(
            group=DGDR_GROUP,
            version=version,
            namespace=self.config.namespace,
            plural=DGDR_PLURAL,
            name=name,
        )

    async def get_dgd(self, name: str) -> dict[str, Any] | None:
        self._require_clients()
        assert self.custom is not None
        try:
            return await self.custom.get_namespaced_custom_object(
                group=DGDR_GROUP,
                version=DGD_VERSION,
                namespace=self.config.namespace,
                plural=DGD_PLURAL,
                name=name,
            )
        except exceptions.ApiException as error:
            if error.status == 404:
                return None
            raise

    async def wait_for_phase(
        self, name: str, target: str, timeout: int | None = None
    ) -> dict[str, Any]:
        return await self._wait_for_phase(name, target, at_least=False, timeout=timeout)

    async def wait_for_phase_at_least(
        self, name: str, target: str, timeout: int | None = None
    ) -> dict[str, Any]:
        return await self._wait_for_phase(name, target, at_least=True, timeout=timeout)

    async def _wait_for_phase(
        self, name: str, target: str, at_least: bool, timeout: int | None
    ) -> dict[str, Any]:
        timeout = timeout or self.config.profiling_timeout
        deadline = time.monotonic() + timeout
        last_phase: str | None = None
        while time.monotonic() < deadline:
            result = await self.get(name)
            phase = result.get("status", {}).get("phase") if result else None
            if phase != last_phase:
                logger.info("DGDR %s/%s phase: %s", self.config.namespace, name, phase)
                last_phase = phase
            if phase == "Failed" and target != "Failed":
                conditions = result.get("status", {}).get("conditions", [])
                raise AssertionError(
                    f"DGDR {self.config.namespace}/{name} failed while waiting for "
                    f"{target}: {conditions}"
                )
            if result and (
                phase == target
                or (
                    at_least
                    and phase in PHASE_ORDER
                    and PHASE_ORDER[phase] >= PHASE_ORDER[target]
                )
            ):
                return result
            await asyncio.sleep(5)
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for DGDR "
            f"{self.config.namespace}/{name} to reach {target}; last phase={last_phase}"
        )

    async def get_output_dgd(self, name: str) -> dict[str, Any]:
        self._require_clients()
        assert self.core is not None
        configmap = await self.core.read_namespaced_config_map(
            f"dgdr-output-{name}", self.config.namespace
        )
        data = configmap.data or {}
        content = data.get("final_config.yaml")
        if not content:
            raise AssertionError(
                f"ConfigMap dgdr-output-{name} must contain final_config.yaml"
            )
        return parse_final_dgd(content)

    async def assert_profiling_job_succeeded(self, name: str) -> None:
        self._require_clients()
        assert self.batch is not None
        job = await self.batch.read_namespaced_job(name, self.config.namespace)
        assert (job.status.succeeded or 0) >= 1, (
            f"profiling job {name} did not succeed: "
            f"succeeded={job.status.succeeded}, failed={job.status.failed}"
        )

    async def wait_for_dgd_successful(
        self, name: str, timeout: int | None = None
    ) -> dict[str, Any]:
        timeout = timeout or self.config.deploy_timeout
        deadline = time.monotonic() + timeout
        last_state = None
        while time.monotonic() < deadline:
            dgd = await self.get_dgd(name)
            last_state = dgd.get("status", {}).get("state") if dgd else None
            if last_state and last_state.lower() == "successful":
                return dgd
            await asyncio.sleep(10)
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for DGD {name} to become "
            f"successful; last state={last_state}"
        )

    async def assert_dgd_pods_ready(self, dgd_name: str) -> None:
        self._require_clients()
        assert self.core is not None
        pods = await self.core.list_namespaced_pod(
            self.config.namespace, label_selector=f"{DGD_LABEL}={dgd_name}"
        )
        assert pods.items, f"DGD {dgd_name} should own at least one pod"
        for pod in pods.items:
            if pod.status.phase == "Succeeded":
                continue
            assert (
                pod.status.phase == "Running"
            ), f"pod {pod.metadata.name} is {pod.status.phase}, expected Running"
            for status in pod.status.container_statuses or []:
                assert (
                    status.ready
                ), f"container {status.name} in pod {pod.metadata.name} is not ready"

    async def assert_services(
        self, dgd_name: str, expectations: dict[str, int]
    ) -> None:
        dgd = await self.get_dgd(dgd_name)
        assert dgd is not None
        dgd_status = dgd.get("status", {})
        statuses = dgd_status.get("components") or dgd_status.get("services", {})
        for service_name, minimum in expectations.items():
            status = statuses.get(service_name)
            assert (
                status is not None
            ), f"service {service_name!r} missing from DGD {dgd_name} status"
            replicas = int(status.get("replicas", 0))
            assert replicas >= minimum
            ready = status.get("availableReplicas", status.get("readyReplicas"))
            if ready is not None:
                assert int(ready) == replicas

    async def get_crd(self) -> client.V1CustomResourceDefinition:
        self._require_clients()
        assert self.apiextensions is not None
        return await self.apiextensions.read_custom_resource_definition(
            "dynamographdeploymentrequests.nvidia.com"
        )

    async def ensure_absent(self, name: str) -> None:
        current = await self.get(name)
        if current is None:
            return
        await self._cleanup_name(name, failed=False)

    async def cleanup(self, failed: bool) -> None:
        failures: list[tuple[str, Exception]] = []
        for name in reversed(self._created_names):
            try:
                await self._cleanup_name(name, failed=failed)
            except Exception as error:
                failures.append((name, error))
                logger.exception(
                    "Failed to clean up DGDR %s/%s", self.config.namespace, name
                )

        if failures:
            failed_names = {name for name, _ in failures}
            self._created_names = [
                name for name in self._created_names if name in failed_names
            ]
            raise DGDRCleanupError(self.config.namespace, failures)

        self._created_names.clear()

    async def _cleanup_name(self, name: str, failed: bool) -> None:
        self._require_clients()
        assert self.custom is not None
        current = await self.get(name)
        dgd_name = current.get("status", {}).get("dgdName") if current else None
        job_name = (
            current.get("status", {}).get("profilingJobName") if current else None
        )
        if failed and current:
            await self._log_diagnostics(current)

        # Stop reconciliation before deleting children that are not owned by the DGDR.
        try:
            await self.custom.delete_namespaced_custom_object(
                group=DGDR_GROUP,
                version=DGDR_VERSION,
                namespace=self.config.namespace,
                plural=DGDR_PLURAL,
                name=name,
            )
        except exceptions.ApiException as error:
            if error.status != 404:
                raise
        await self._wait_until_dgdr_absent(name)

        if dgd_name:
            try:
                await self.custom.delete_namespaced_custom_object(
                    group=DGDR_GROUP,
                    version=DGD_VERSION,
                    namespace=self.config.namespace,
                    plural=DGD_PLURAL,
                    name=dgd_name,
                )
            except exceptions.ApiException as error:
                if error.status != 404:
                    raise
            await self._wait_until_dgd_absent(dgd_name)

        if job_name:
            await self._delete_profiling_job(job_name)
        await self._delete_output_configmap(name)

    async def _wait_until_dgdr_absent(self, name: str) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if await self.get(name) is None:
                return
            await asyncio.sleep(5)
        raise TimeoutError(f"DGDR {self.config.namespace}/{name} did not terminate")

    async def _wait_until_dgd_absent(self, name: str) -> None:
        self._require_clients()
        assert self.core is not None
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            dgd = await self.get_dgd(name)
            pods = await self.core.list_namespaced_pod(
                self.config.namespace, label_selector=f"{DGD_LABEL}={name}"
            )
            if dgd is None and not pods.items:
                return
            await asyncio.sleep(5)
        raise TimeoutError(f"DGD {self.config.namespace}/{name} did not terminate")

    async def _delete_profiling_job(self, name: str) -> None:
        self._require_clients()
        assert self.batch is not None
        assert self.core is not None
        try:
            await self.batch.delete_namespaced_job(
                name,
                self.config.namespace,
                propagation_policy="Foreground",
            )
        except exceptions.ApiException as error:
            if error.status != 404:
                raise

        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                await self.batch.read_namespaced_job(name, self.config.namespace)
                job_exists = True
            except exceptions.ApiException as error:
                if error.status != 404:
                    raise
                job_exists = False
            pods = await self.core.list_namespaced_pod(
                self.config.namespace, label_selector=f"job-name={name}"
            )
            if not job_exists and not pods.items:
                return
            await asyncio.sleep(5)
        raise TimeoutError(
            f"profiling job {self.config.namespace}/{name} did not terminate"
        )

    async def _delete_output_configmap(self, name: str) -> None:
        self._require_clients()
        assert self.core is not None
        try:
            await self.core.delete_namespaced_config_map(
                f"dgdr-output-{name}", self.config.namespace
            )
        except exceptions.ApiException as error:
            if error.status != 404:
                raise

    async def _log_diagnostics(self, dgdr: dict[str, Any]) -> None:
        self._require_clients()
        logger.error("DGDR failure diagnostics:\n%s", json.dumps(dgdr, indent=2))

        status = dgdr.get("status", {})
        profiling_job_name = status.get("profilingJobName")
        if profiling_job_name:
            await self._log_profiling_job_diagnostics(profiling_job_name)

        dgd_name = status.get("dgdName")
        if dgd_name:
            try:
                dgd = await self.get_dgd(dgd_name)
                logger.error("DGD failure diagnostics:\n%s", json.dumps(dgd, indent=2))
            except exceptions.ApiException as error:
                logger.warning(
                    "Could not read DGD %s/%s: %s",
                    self.config.namespace,
                    dgd_name,
                    error,
                )
            await self._log_pod_diagnostics(f"{DGD_LABEL}={dgd_name}")

    async def _log_profiling_job_diagnostics(self, name: str) -> None:
        self._require_clients()
        assert self.batch is not None
        try:
            job = await self.batch.read_namespaced_job(name, self.config.namespace)
            logger.error("Profiling Job failure diagnostics:\n%s", job.to_str())
        except exceptions.ApiException as error:
            logger.warning(
                "Could not read profiling Job %s/%s: %s",
                self.config.namespace,
                name,
                error,
            )
        await self._log_pod_diagnostics(f"job-name={name}")

    async def _log_pod_diagnostics(self, label_selector: str) -> None:
        self._require_clients()
        assert self.core is not None
        try:
            pods = await self.core.list_namespaced_pod(
                self.config.namespace, label_selector=label_selector
            )
        except exceptions.ApiException as error:
            logger.warning(
                "Could not list pods in %s matching %s: %s",
                self.config.namespace,
                label_selector,
                error,
            )
            return

        for pod in pods.items:
            logger.error("Pod failure diagnostics:\n%s", pod.to_str())
            containers = [
                *(pod.spec.init_containers or []),
                *(pod.spec.containers or []),
            ]
            for container in containers:
                try:
                    logs = await self.core.read_namespaced_pod_log(
                        pod.metadata.name,
                        self.config.namespace,
                        container=container.name,
                        tail_lines=300,
                    )
                    logger.error(
                        "Pod %s container %s logs:\n%s",
                        pod.metadata.name,
                        container.name,
                        logs,
                    )
                except exceptions.ApiException as error:
                    logger.warning(
                        "Could not read logs for %s/%s: %s",
                        pod.metadata.name,
                        container.name,
                        error,
                    )


async def run_lifecycle(
    manager: ManagedDGDR,
    manifest: dict[str, Any],
    *,
    verify_configmap: bool = True,
    expected_services: dict[str, int] | None = None,
    verify_inference: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Exercise create, profile, optional auto-deploy, and runtime verification."""

    await manager.create(manifest)
    name = manifest["metadata"]["name"]
    result = await manager.wait_for_phase_at_least(name, "Ready")
    status = result.get("status", {})
    if not status.get("profilingJobName"):
        raise AssertionError("profilingJobName must be set")

    output = await manager.get_output_dgd(name) if verify_configmap else None
    if output and expected_services:
        services = output.get("spec", {}).get("services", {})
        components = {
            component.get("name"): component
            for component in output.get("spec", {}).get("components", [])
        }
        for service in expected_services:
            if service not in services and service not in components:
                raise AssertionError(f"service {service!r} missing from output DGD")

    if manifest["spec"].get("autoApply", True) is False:
        return result, output

    result = await manager.wait_for_phase(
        name,
        "Deployed",
        manager.config.profiling_timeout + manager.config.deploy_timeout,
    )
    dgd_name = result.get("status", {}).get("dgdName")
    if not dgd_name:
        raise AssertionError("dgdName must be set after deployment")
    if manager.config.mocker:
        return result, output

    await manager.wait_for_dgd_successful(dgd_name)
    await manager.assert_dgd_pods_ready(dgd_name)
    if expected_services:
        await manager.assert_services(dgd_name, expected_services)
    if verify_inference:
        verify_inference_endpoints(
            manager.config.namespace, dgd_name, manifest["spec"]["model"]
        )
    return result, output


def verify_inference_endpoints(namespace: str, dgd_name: str, model: str) -> None:
    """Send model-list and chat requests from a short-lived in-cluster pod."""

    url = f"http://{dgd_name}-frontend.{namespace}.svc.cluster.local:8000"
    suffix = dgd_name[:12]
    models = kubectl(
        "run",
        f"inference-models-{suffix}",
        "--rm",
        "-i",
        "--restart=Never",
        "-n",
        namespace,
        "--image=curlimages/curl:latest",
        "--",
        "curl",
        "-sf",
        "--max-time",
        "10",
        f"{url}/v1/models",
    )
    assert models.returncode == 0, models.stderr
    assert model in models.stdout

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
        }
    )
    chat = kubectl(
        "run",
        f"inference-chat-{suffix}",
        "--rm",
        "-i",
        "--restart=Never",
        "-n",
        namespace,
        "--image=curlimages/curl:latest",
        "--",
        "curl",
        "-sf",
        "--max-time",
        "60",
        "-H",
        "Content-Type: application/json",
        "-d",
        body,
        f"{url}/v1/chat/completions",
    )
    assert chat.returncode == 0, chat.stderr
    assert "choices" in chat.stdout
