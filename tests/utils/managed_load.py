# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ManagedLoad - YAML template-based load testing using shared PVC.

This module provides a simplified load testing framework that:
1. Uses a YAML template for the Job spec (instead of generating dynamically)
2. Modifies only the aiperf command as needed
3. Uses shared PVC with ManagedDeployment for storing results
4. Uses PvcExtractor for extracting results from shared PVC
"""

import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import kr8s
import yaml
from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions


def _get_template_dir() -> str:
    """Get the templates directory path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


@dataclass
class LoadConfig:
    """Configuration for load test parameters.

    Note: endpoint_path is the API suffix (e.g., "/v1/chat/completions").
    The base URL (host:port) is passed separately to ManagedLoad.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer: Optional[str] = None
    endpoint_path: str = "/v1/chat/completions"  # API endpoint suffix

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
    """YAML template-based load testing using shared PVC.

    Args:
        namespace: Kubernetes namespace for the load test job
        load_config: Load test configuration (concurrency, tokens, etc.)
        pvc_name: Shared PVC from ManagedDeployment for storing results
        endpoint_url: Base URL of the frontend service (e.g., http://frontend:8000)
    """

    namespace: str
    load_config: LoadConfig
    pvc_name: str  # Shared PVC from ManagedDeployment
    endpoint_url: str  # Base URL (host:port) - passed by caller
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
    _load_completed: bool = False
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
            self.local_output_dir = os.path.join(self.log_dir, "load")
            os.makedirs(self.local_output_dir, exist_ok=True)
            self._logger.info(
                f"Load test results will be saved to: {self.local_output_dir}"
            )
        else:
            self.local_output_dir = None

    async def _init_kubernetes(self):
        """Initialize kubernetes clients."""
        from tests.utils.k8s_helpers import init_kubernetes_clients

        self._core_api, self._batch_api, _, _, _ = await init_kubernetes_clients()

    def _load_template(self) -> dict:
        """Load and parse YAML template."""
        with open(self.template_path, "r") as f:
            return yaml.safe_load(f)

    def _build_aiperf_command(self) -> str:
        """Build aiperf command string from LoadConfig."""
        cfg = self.load_config

        args = [
            "aiperf",
            "profile",
            "--artifact-dir",
            self.container_results_dir,
            "--model",
            cfg.model_name,
            "--tokenizer",
            cfg.tokenizer,
            "--endpoint-type",
            "chat",
            "--endpoint",
            cfg.endpoint_path,
            "--url",
            self.endpoint_url,
            "--synthetic-input-tokens-mean",
            str(cfg.input_tokens_mean),
            "--synthetic-input-tokens-stddev",
            str(cfg.input_tokens_stddev),
            "--output-tokens-mean",
            str(cfg.output_tokens_mean),
            "--output-tokens-stddev",
            str(cfg.output_tokens_stddev),
            "--concurrency",
            str(cfg.concurrency),
            "--warmup-request-count",
            str(cfg.warmup_requests),
            "--request-timeout-seconds",
            str(cfg.request_timeout_seconds),
            "--num-dataset-entries",
            "12800",
            "--random-seed",
            "100",
            "--workers-max",
            "252",
            "--record-processors",
            "32",
            "--ui",
            "simple",
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

        # Update environment variables
        for env in container.get("env", []):
            if env["name"] == "ENDPOINT_URL":
                env["value"] = self.endpoint_url
            elif env["name"] == "MODEL_NAME":
                env["value"] = self.load_config.model_name

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
                timeout = (
                    int(self.load_config.duration_minutes * 60) + 300
                )  # 5 min buffer
            elif self.load_config.request_count:
                timeout = max(self.load_config.request_count * 2 + 60, 300)
            else:
                timeout = 600  # Default 10 minutes

        marker_file = f"{self.container_results_dir}/status/completed"
        result = await self._wait_for_status_marker(
            marker_file, "Load test completion", timeout
        )
        if result:
            self._load_completed = True
        return result

    async def terminate(self):
        """Gracefully terminate the running load test.

        Sends SIGINT to aiperf, waits for graceful shutdown, then SIGTERM.
        Container exits after aiperf completes - results are on PVC.

        If load already completed naturally, skips signaling.
        """
        self._logger.info(f"Terminating load test in job {self.job_name}")
        self._terminated = True

        if not self._job_created:
            self._logger.warning("No job created to terminate")
            return

        # Skip signaling if load already completed naturally
        if self._load_completed:
            self._logger.info(
                "Load test already completed, skipping termination signals"
            )
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

                # Send SIGINT for graceful shutdown
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGINT", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGINT sent to aiperf process")
                    else:
                        self._logger.warning(
                            "SIGINT failed, process may already be stopped"
                        )
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGINT: {e}")

                # Wait for graceful shutdown
                await asyncio.sleep(5)

                # Send SIGTERM if still running
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGTERM", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGTERM sent to aiperf process")
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGTERM: {e}")

                # Wait for pod to complete (container exits after aiperf finishes)
                self._logger.info("Waiting for pod to complete...")
                for attempt in range(60):  # Wait up to 1 minute
                    try:
                        pod.refresh()
                        phase = pod.status.phase
                        if phase in ("Succeeded", "Failed"):
                            self._logger.info(f"Pod completed with phase: {phase}")
                            break
                    except Exception:
                        pass

                    if attempt < 59:
                        await asyncio.sleep(1)
                else:
                    self._logger.warning(
                        "Pod did not complete in time, proceeding anyway"
                    )

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

    async def _delete_pod(self) -> None:
        """Delete the load test pod."""
        try:
            pods = list(
                kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
            )

            if not pods:
                return

            pod = pods[0]
            self._logger.info(f"Deleting pod {pod.name}")
            pod.delete(force=True)

            # Wait for pod to be deleted
            for _ in range(30):
                try:
                    remaining = list(
                        kr8s.get(
                            "pods",
                            namespace=self.namespace,
                            label_selector=f"job-name={self.job_name}",
                        )
                    )
                    if not remaining:
                        self._logger.info("Pod deleted")
                        break
                except Exception:
                    break
                await asyncio.sleep(1)

        except Exception as e:
            self._logger.warning(f"Failed to delete pod: {e}")

    async def _debug_pod_files(self) -> None:
        """Debug helper: List files in the pod's results directory."""
        try:
            pods = list(
                kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
            )

            if not pods:
                self._logger.info("DEBUG: No pods found")
                return

            pod = pods[0]
            self._logger.info(f"DEBUG: Pod {pod.name} phase: {pod.status.phase}")

            if pod.status.phase != "Running":
                return

            # List files in results directory
            result = await asyncio.wait_for(
                asyncio.create_task(
                    asyncio.to_thread(
                        pod.exec,
                        ["ls", "-la", self.container_results_dir],
                    )
                ),
                timeout=10.0,
            )
            if result.returncode == 0:
                self._logger.info(
                    f"DEBUG: Files in {self.container_results_dir}:\n{result.stdout.decode()}"
                )

        except Exception as e:
            self._logger.warning(f"DEBUG: Failed to list files: {e}")

    async def get_results(self) -> Optional[Dict[str, Any]]:
        """Get parsed results JSON from PVC via PvcExtractor."""
        try:
            # Debug: Show what files are in the pod before deletion
            await self._debug_pod_files()

            # Delete the pod (results are already in PVC)
            await self._delete_pod()

            # Extract from PVC using PvcExtractor
            from tests.utils.pvc_extractor import PvcExtractor

            extractor = PvcExtractor(namespace=self.namespace, logger=self._logger)
            await extractor.init()

            output_dir = self.local_output_dir or "load"
            result = await extractor.extract(
                pvc_name=self.pvc_name,
                sub_path="aiperf",
                container_path=self.container_results_dir,
                file_patterns=["*.json", "*.jsonl", "*.csv", "*.log"],
                local_output_dir=output_dir,
            )

            if result.get("success"):
                output_path = Path(result["output_dir"])
                available_files = (
                    list(output_path.iterdir()) if output_path.exists() else []
                )
                self._logger.info(
                    f"Available result files: {[f.name for f in available_files]}"
                )

                json_file = output_path / "profile_export_aiperf.json"
                if json_file.exists():
                    self._logger.info(f"Using aiperf summary: {json_file}")
                    with open(json_file) as f:
                        return json.load(f)

                self._logger.warning(
                    f"No profile_export_aiperf.json found in {output_path}. "
                    f"Available files: {[f.name for f in available_files]}"
                )

            return None

        except Exception:
            self._logger.exception("Failed to get results")
            return None

    async def _cleanup(self):
        """Clean up the load test job."""
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
        # Log if we're exiting due to an exception (Ctrl-C, etc.)
        if exc_type is not None:
            self._logger.warning(
                f"Exiting due to exception ({exc_type.__name__}), running cleanup"
            )
        else:
            # Only try to extract results if we're not exiting due to an exception
            if self._job_created and self.local_output_dir:
                try:
                    await self.get_results()
                except Exception as e:
                    self._logger.warning(
                        f"Failed to extract results during cleanup: {e}"
                    )

        # Always run cleanup, catching any cleanup errors
        try:
            await self._cleanup()
        except Exception as cleanup_error:
            self._logger.error(f"Error during cleanup: {cleanup_error}")
