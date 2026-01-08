# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ManagedLoad - YAML template-based load testing using shared PVC.

This module provides a simplified load testing framework that:
1. Uses a YAML template for the Job spec (instead of generating dynamically)
2. Modifies only the aiperf command as needed
3. Uses shared PVC with ManagedDeployment for storing results
4. Uses a dedicated download job for extracting results
"""

import asyncio
import json
import logging
import os
import secrets
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import kr8s
import yaml
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions


def _get_template_dir() -> str:
    """Get the templates directory path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


@dataclass
class LoadConfig:
    """Configuration for load test parameters."""

    endpoint_url: str
    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer: Optional[str] = None

    # Load parameters
    concurrency: int = 8
    request_count: Optional[int] = None
    duration_minutes: Optional[float] = None

    # Token parameters
    input_tokens_mean: int = 512
    input_tokens_stddev: int = 0
    output_tokens_mean: int = 64
    output_tokens_stddev: int = 0

    # Request parameters
    streaming: bool = True
    request_rate: Optional[float] = None
    request_timeout_seconds: float = 30.0
    warmup_requests: Optional[int] = None

    # Inference parameters
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = True

    # Extra inference parameters
    extra_inputs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warmup_requests is None:
            self.warmup_requests = min(self.concurrency, 10)
        if self.tokenizer is None:
            self.tokenizer = self.model_name


@dataclass
class ManagedLoad:
    """YAML template-based load testing using shared PVC."""

    namespace: str
    load_config: LoadConfig
    pvc_name: str  # Shared PVC from ManagedDeployment
    template_path: Optional[str] = None
    log_dir: Optional[str] = None
    job_name: Optional[str] = None
    container_results_dir: str = "/tmp/aiperf"

    # Internal state
    _core_api: Optional[client.CoreV1Api] = None
    _batch_api: Optional[client.BatchV1Api] = None
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _job_created: bool = False
    _terminated: bool = False
    _download_job_name: Optional[str] = None
    _unique_suffix: str = field(default_factory=lambda: secrets.token_hex(4))

    def __post_init__(self):
        # Generate unique job name if not provided
        if self.job_name is None:
            self.job_name = f"load-test-{self._unique_suffix}"

        # Use default template path if not provided
        if self.template_path is None:
            self.template_path = os.path.join(_get_template_dir(), "load_job.yaml")

        # Set up local output directory
        if self.log_dir:
            self.local_output_dir = os.path.join(self.log_dir, "load_results")
            os.makedirs(self.local_output_dir, exist_ok=True)
            self._logger.info(f"Load test results will be saved to: {self.local_output_dir}")
        else:
            self.local_output_dir = None

    async def _init_kubernetes(self):
        """Initialize kubernetes client."""
        try:
            config.load_incluster_config()
        except Exception:
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        self._core_api = client.CoreV1Api(k8s_client)
        self._batch_api = client.BatchV1Api(k8s_client)

    def _load_template(self) -> dict:
        """Load and parse YAML template."""
        with open(self.template_path, "r") as f:
            return yaml.safe_load(f)

    def _build_aiperf_command(self) -> str:
        """Build aiperf command string from LoadConfig."""
        cfg = self.load_config

        args = [
            "aiperf", "profile",
            "--artifact-dir", self.container_results_dir,
            "--model", cfg.model_name,
            "--tokenizer", cfg.tokenizer,
            "--endpoint-type", "chat",
            "--endpoint", "/v1/chat/completions",
            "--url", cfg.endpoint_url,
            "--synthetic-input-tokens-mean", str(cfg.input_tokens_mean),
            "--synthetic-input-tokens-stddev", str(cfg.input_tokens_stddev),
            "--output-tokens-mean", str(cfg.output_tokens_mean),
            "--output-tokens-stddev", str(cfg.output_tokens_stddev),
            "--concurrency", str(cfg.concurrency),
            "--warmup-request-count", str(cfg.warmup_requests),
            "--request-timeout-seconds", str(cfg.request_timeout_seconds),
            "--num-dataset-entries", "12800",
            "--random-seed", "100",
            "--workers-max", "252",
            "--record-processors", "32",
            "--ui", "simple",
            "--verbose",
        ]

        # Add duration or request count
        if cfg.duration_minutes:
            args.extend(["--benchmark-duration", str(cfg.duration_minutes * 60)])
        elif cfg.request_count:
            args.extend(["--request-count", str(cfg.request_count)])

        if cfg.streaming:
            args.append("--streaming")

        if cfg.request_rate:
            args.extend(["--request-rate", str(cfg.request_rate)])

        # Build extra inputs
        extra_inputs = {
            "max_tokens": cfg.output_tokens_mean,
            "min_tokens": cfg.output_tokens_mean,
            "temperature": cfg.temperature,
            "repetition_penalty": cfg.repetition_penalty,
        }

        if cfg.ignore_eos:
            extra_inputs["ignore_eos"] = True

        if cfg.extra_inputs:
            extra_inputs.update(cfg.extra_inputs)

        for key, value in extra_inputs.items():
            args.extend(["--extra-inputs", f"{key}:{value}"])

        return " ".join(args)

    def _apply_config_to_template(self, template: dict) -> dict:
        """Apply LoadConfig to template - set command, PVC, namespace."""
        # Set metadata
        template["metadata"]["name"] = self.job_name
        template["metadata"]["namespace"] = self.namespace

        # Get pod spec
        pod_spec = template["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]

        # Build the aiperf command
        aiperf_cmd = self._build_aiperf_command()

        # Update the container args with the aiperf command
        # The template has AIPERF_CMD placeholder
        original_script = container["args"][0]
        updated_script = original_script.replace("AIPERF_CMD", aiperf_cmd)
        container["args"] = [updated_script]

        # Update environment variable for endpoint URL
        for env in container.get("env", []):
            if env["name"] == "ENDPOINT_URL":
                env["value"] = self.load_config.endpoint_url

        # Set PVC reference
        for volume in pod_spec.get("volumes", []):
            if "persistentVolumeClaim" in volume:
                volume["persistentVolumeClaim"]["claimName"] = self.pvc_name

        # Update volume mount path if needed
        for mount in container.get("volumeMounts", []):
            if mount["name"] == "results-volume":
                mount["mountPath"] = self.container_results_dir

        return template

    async def _create_job(self):
        """Create the load test job in Kubernetes."""
        template = self._load_template()
        job_spec = self._apply_config_to_template(template)

        self._logger.info(f"Creating load test job: {self.job_name}")

        try:
            assert self._batch_api is not None, "Kubernetes API not initialized"
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._job_created = True
            self._logger.info(f"Load test job created: {self.job_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.warning(f"Job {self.job_name} already exists")
                self._job_created = True
            else:
                self._logger.error(f"Failed to create job {self.job_name}: {e}")
                raise

    async def run(self, wait_for_completion: bool = True) -> Dict[str, Any]:
        """Start the load test job."""
        await self._create_job()

        if wait_for_completion:
            success = await self.wait_for_completion()
            return {"success": success, "job_name": self.job_name}
        else:
            self._logger.info(f"Load test job {self.job_name} started (not waiting)")
            return {"success": True, "job_name": self.job_name}

    async def _wait_for_status_marker(
        self, marker_file: str, marker_description: str, timeout: int
    ) -> bool:
        """Wait for a specific status marker file to appear in the pod."""
        start_time = time.time()

        self._logger.info(
            f"Waiting for {marker_description} in job {self.job_name} (timeout: {timeout}s)"
        )

        while (time.time() - start_time) < timeout:
            if self._terminated:
                self._logger.info(f"{marker_description} wait terminated by request")
                return False

            try:
                # Find the pod for this job
                pods = []
                pod_generator = kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                for pod in pod_generator:
                    pods.append(pod)

                if pods:
                    pod = pods[0]

                    # Check if the status marker exists
                    try:
                        result = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(pod.exec, ["test", "-f", marker_file])
                            ),
                            timeout=10.0,
                        )

                        if result.returncode == 0:
                            self._logger.info(
                                f"{marker_description} marker found in job {self.job_name}"
                            )
                            return True

                    except (asyncio.TimeoutError, Exception):
                        pass

                    # Check for job failure
                    assert self._batch_api is not None
                    job = await self._batch_api.read_namespaced_job(
                        name=self.job_name, namespace=self.namespace
                    )

                    if job.status.failed:
                        self._logger.error(
                            f"Load test job {self.job_name} failed while waiting for {marker_description}"
                        )
                        return False

                    # Check if job completed (not running anymore)
                    if job.status.completion_time is not None:
                        self._logger.info(
                            f"Job {self.job_name} completed while waiting for {marker_description}"
                        )
                        return True

            except exceptions.ApiException as e:
                self._logger.warning(f"Error checking {marker_description} status: {e}")

            await asyncio.sleep(5)

        if self._terminated:
            return False
        else:
            raise TimeoutError(f"{marker_description} did not appear within {timeout}s")

    async def wait_for_started(self, timeout: int = 300) -> bool:
        """Wait for the load test to start (5 minute timeout by default)."""
        marker_file = f"{self.container_results_dir}/status/started"
        return await self._wait_for_status_marker(
            marker_file, "Load test start", timeout
        )

    async def wait_for_completion(self, timeout: Optional[int] = None) -> bool:
        """Wait for the load test to complete."""
        if timeout is None:
            if self.load_config.duration_minutes:
                timeout = int(self.load_config.duration_minutes * 60) + 300  # 5 min buffer
            elif self.load_config.request_count:
                timeout = max(self.load_config.request_count * 2 + 60, 300)
            else:
                timeout = 600  # Default 10 minutes

        marker_file = f"{self.container_results_dir}/status/completed"
        return await self._wait_for_status_marker(
            marker_file, "Load test completion", timeout
        )

    async def terminate(self):
        """Gracefully terminate the running load test."""
        self._logger.info(f"Terminating load test in job {self.job_name}")
        self._terminated = True

        if not self._job_created:
            self._logger.warning("No job created to terminate")
            return

        try:
            # Find and signal the pod
            pods = []
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=f"job-name={self.job_name}",
            )
            for pod in pod_generator:
                pods.append(pod)

            if pods:
                pod = pods[0]
                self._logger.info(f"Sending termination signal to pod {pod.name}")

                # Send SIGINT first for graceful shutdown
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGINT", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGINT sent to aiperf process")
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGINT: {e}")

                await asyncio.sleep(5)

                # If still running, send SIGTERM
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGTERM", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                except Exception:
                    pass

                self._logger.info("Load test termination completed")
            else:
                self._logger.warning("No pods found to terminate")

        except Exception as e:
            self._logger.error(f"Error during termination: {e}")

    async def is_running(self) -> bool:
        """Check if the load test is currently running."""
        if not self._job_created or self._terminated:
            return False

        try:
            assert self._batch_api is not None
            job = await self._batch_api.read_namespaced_job(
                name=self.job_name, namespace=self.namespace
            )

            # Job is running if not completed and not failed
            is_active = job.status.completion_time is None and (
                job.status.failed is None or job.status.failed == 0
            )

            if is_active:
                # Double check by looking for running pods
                pods = []
                pod_generator = kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                for pod in pod_generator:
                    if pod.status.phase in ["Running", "Pending"]:
                        pods.append(pod)
                return len(pods) > 0

            return False

        except exceptions.ApiException as e:
            if e.status == 404:
                return False
            self._logger.warning(f"Error checking job status: {e}")
            return False

    async def create_results_download_job(self) -> str:
        """Create dedicated download job for results extraction."""
        job_name = f"load-results-download-{self._unique_suffix}"

        download_script = f"""#!/bin/sh
set -e

echo "=== LOAD RESULTS DOWNLOAD JOB STARTED ==="
echo "PVC: {self.pvc_name}"
echo "Results dir: {self.container_results_dir}"

mkdir -p /tmp/download

if [ ! -d "{self.container_results_dir}" ]; then
    echo "Results directory does not exist yet"
fi

echo "ready" > /tmp/download/job_ready.txt
echo "=== DOWNLOAD JOB READY ==="

while true; do
    FILE_COUNT=$(find {self.container_results_dir} -type f 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] Download job alive - $FILE_COUNT files available"
    sleep 60
done
"""

        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "load-results-download",
                    "managed-by": "managed-load",
                },
            },
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "load-results-download",
                            "job-name": job_name,
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "download",
                                "image": "busybox:1.35",
                                "command": ["/bin/sh", "-c", download_script],
                                "volumeMounts": [
                                    {
                                        "name": "results-volume",
                                        "mountPath": self.container_results_dir,
                                        "readOnly": True,
                                    }
                                ],
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"cpu": "500m", "memory": "512Mi"},
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "results-volume",
                                "persistentVolumeClaim": {
                                    "claimName": self.pvc_name,
                                },
                            }
                        ],
                    },
                },
            },
        }

        try:
            assert self._batch_api is not None
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._download_job_name = job_name
            self._logger.info(f"Results download job created: {job_name}")
            return job_name

        except exceptions.ApiException as e:
            if e.status == 409:
                self._logger.warning(f"Download job {job_name} already exists")
                self._download_job_name = job_name
                return job_name
            raise

    async def extract_results(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract results from PVC via download job."""
        output_dir = output_dir or self.local_output_dir
        if output_dir is None:
            output_dir = "load_results"

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Create download job if not exists
            if self._download_job_name is None:
                await self.create_results_download_job()

            # Wait for download job to be ready
            self._logger.info("Waiting for download job to be ready...")
            for attempt in range(60):
                try:
                    pods = []
                    pod_generator = kr8s.get(
                        "pods",
                        namespace=self.namespace,
                        label_selector=f"job-name={self._download_job_name}",
                    )
                    for pod in pod_generator:
                        pods.append(pod)

                    if pods:
                        pod = pods[0]
                        result = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(
                                    pod.exec, ["test", "-f", "/tmp/download/job_ready.txt"]
                                )
                            ),
                            timeout=5.0,
                        )
                        if result.returncode == 0:
                            break
                except Exception:
                    pass

                self._logger.info(f"Waiting for download job... (attempt {attempt + 1}/60)")
                await asyncio.sleep(1)
            else:
                self._logger.warning("Download job did not become ready in time")

            # Find the download job pod
            pods = []
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=f"job-name={self._download_job_name}",
            )
            for pod in pod_generator:
                pods.append(pod)

            if not pods:
                raise Exception(f"No pods found for download job {self._download_job_name}")

            pod = pods[0]

            # Create tar archive on-demand
            self._logger.info("Creating tar archive of results on-demand...")
            create_tar_script = f"""
cd {self.container_results_dir} 2>/dev/null || exit 1
FILE_COUNT=$(find . -type f \\( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" \\) | wc -l)
echo "FILE_COUNT:$FILE_COUNT"
if [ "$FILE_COUNT" -gt 0 ]; then
    find . -type f \\( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" \\) | tar -czf /tmp/download/results.tar.gz -T -
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

            # Parse output
            output = tar_result.stdout.decode() if tar_result.stdout else ""
            file_count = 0
            tar_created = False
            for line in output.split("\n"):
                if line.startswith("FILE_COUNT:"):
                    file_count = int(line.split(":")[1])
                elif line.startswith("TAR_CREATED:"):
                    tar_created = line.split(":")[1] == "true"

            self._logger.info(f"Found {file_count} result files, tar_created={tar_created}")

            extracted_files = []

            if file_count > 0 and tar_created:
                # Extract the tar archive
                self._logger.info("Extracting results archive...")
                cat_result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(pod.exec, ["cat", "/tmp/download/results.tar.gz"])
                    ),
                    timeout=60.0,
                )

                if cat_result.returncode != 0:
                    raise Exception(
                        f"Archive extraction failed with return code {cat_result.returncode}"
                    )

                # Save and extract locally
                local_archive = Path(output_dir) / "results.tar.gz"
                local_archive.write_bytes(cat_result.stdout)

                with tarfile.open(local_archive, "r:gz") as tar:
                    tar.extractall(path=output_dir)
                    extracted_files = tar.getnames()

                local_archive.unlink()

                self._logger.info(f"Extracted {len(extracted_files)} files to {output_dir}")
            else:
                self._logger.info("No result files were available for download")

            # Cleanup download job
            await self._cleanup_download_job()

            return {
                "success": True,
                "output_dir": output_dir,
                "file_count": file_count,
                "extracted_files": extracted_files,
            }

        except Exception as e:
            self._logger.error(f"Failed to extract results: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_dir": output_dir,
            }

    async def get_results(self) -> Optional[Dict[str, Any]]:
        """Get parsed results JSON."""
        try:
            # Try to extract results from pod directly first
            pods = []
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=f"job-name={self.job_name}",
            )
            for pod in pod_generator:
                pods.append(pod)

            if pods:
                pod = pods[0]
                result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(
                            pod.exec,
                            ["cat", f"{self.container_results_dir}/profile_export_aiperf.json"],
                        )
                    ),
                    timeout=30.0,
                )

                if result.returncode == 0:
                    return json.loads(result.stdout.decode())

            # If direct extraction failed, try via download job
            extract_result = await self.extract_results()
            if extract_result.get("success"):
                output_dir = extract_result["output_dir"]
                results_file = Path(output_dir) / "profile_export_aiperf.json"
                if results_file.exists():
                    with open(results_file) as f:
                        return json.load(f)

            return None

        except Exception as e:
            self._logger.error(f"Failed to get results: {e}")
            return None

    async def _cleanup_download_job(self):
        """Clean up the download job."""
        if self._download_job_name is None:
            return

        try:
            from kubernetes_asyncio.client.models import V1DeleteOptions

            delete_options = V1DeleteOptions(propagation_policy="Foreground")

            assert self._batch_api is not None
            await self._batch_api.delete_namespaced_job(
                name=self._download_job_name,
                namespace=self.namespace,
                body=delete_options,
            )
            self._logger.info(f"Download job {self._download_job_name} deleted")
            self._download_job_name = None

        except exceptions.ApiException as e:
            if e.status != 404:
                self._logger.warning(f"Failed to delete download job: {e}")

    async def _cleanup(self):
        """Clean up the load test job and associated resources."""
        # Cleanup download job first
        await self._cleanup_download_job()

        # Cleanup main job
        if self._job_created and self._batch_api:
            try:
                from kubernetes_asyncio.client.models import V1DeleteOptions

                delete_options = V1DeleteOptions(propagation_policy="Foreground")

                await self._batch_api.delete_namespaced_job(
                    name=self.job_name,
                    namespace=self.namespace,
                    body=delete_options,
                )
                self._logger.info(f"Load test job {self.job_name} deleted")

            except exceptions.ApiException as e:
                if e.status != 404:
                    self._logger.warning(f"Failed to delete job {self.job_name}: {e}")

    async def __aenter__(self) -> "ManagedLoad":
        """Create the load job."""
        await self._init_kubernetes()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup job resources."""
        # Try to extract results before cleanup
        if self._job_created and self.local_output_dir:
            try:
                await self.extract_results()
            except Exception as e:
                self._logger.warning(f"Failed to extract results during cleanup: {e}")

        await self._cleanup()


async def main():
    """Example usage of ManagedLoad."""
    logging.basicConfig(level=logging.INFO)

    # This would typically come from ManagedDeployment.get_log_pvc_name()
    pvc_name = "example-pvc"

    load_config = LoadConfig(
        endpoint_url="http://trtllm-frontend:8000",
        model_name="Qwen/Qwen3-0.6B",
        concurrency=8,
        duration_minutes=5,
        streaming=True,
    )

    async with ManagedLoad(
        namespace="default",
        pvc_name=pvc_name,
        load_config=load_config,
        log_dir="./test_results",
    ) as load:
        await load.run(wait_for_completion=False)
        await load.wait_for_started()

        # Do other operations...

        await load.wait_for_completion()
        results = await load.get_results()

        if results:
            print("Load test results:")
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
