# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PvcExtractor - Unified PVC file extraction via temporary download jobs.

Replaces duplicate extraction logic in ManagedDeployment and ManagedLoad.
Pattern: create busybox job → mount PVC → tar files → cat to local → extract → cleanup.
"""

import asyncio
import logging
import os
import secrets
import tarfile
from pathlib import Path
from typing import Optional

import kr8s
from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions

from tests.utils.k8s_helpers import init_kubernetes_clients

logger = logging.getLogger(__name__)


class PvcExtractor:
    """Extracts files from a K8s PVC via a temporary download job.

    Usage:
        extractor = PvcExtractor(namespace="my-ns")
        await extractor.init()

        # Extract service logs
        result = await extractor.extract(
            pvc_name="my-pvc",
            sub_path="service_logs",
            container_path="/tmp/service_logs",
            file_patterns=["*.log"],
            local_output_dir="/local/path/services",
        )

        # Extract load results (same class, different sub_path)
        result = await extractor.extract(
            pvc_name="my-pvc",
            sub_path="aiperf",
            container_path="/tmp/aiperf",
            file_patterns=["*.json", "*.jsonl", "*.csv", "*.log"],
            local_output_dir="/local/path/load",
        )
    """

    def __init__(self, namespace: str, logger: Optional[logging.Logger] = None):
        self.namespace = namespace
        self._logger = logger or logging.getLogger(__name__)
        self._batch_api: Optional[client.BatchV1Api] = None
        self._core_api: Optional[client.CoreV1Api] = None
        self._active_jobs: list[str] = []

    async def init(self):
        """Initialize Kubernetes clients."""
        self._core_api, self._batch_api, _, _, _ = await init_kubernetes_clients()

    async def extract(
        self,
        pvc_name: str,
        sub_path: str,
        container_path: str,
        file_patterns: list[str],
        local_output_dir: str,
        job_prefix: str = "pvc-extract",
        ready_timeout: int = 60,
        tar_timeout: int = 30,
        cat_timeout: int = 60,
    ) -> dict:
        """Extract files from a PVC to a local directory.

        Creates a temporary busybox job that mounts the PVC, tars matching files,
        and streams the archive back for local extraction.

        Args:
            pvc_name: Name of the PVC to extract from
            sub_path: PVC sub-path to mount (e.g., "service_logs", "aiperf")
            container_path: Mount path inside the container
            file_patterns: File glob patterns to include (e.g., ["*.log", "*.json"])
            local_output_dir: Local directory to extract files into
            job_prefix: Prefix for the job name
            ready_timeout: Seconds to wait for job pod to be ready
            tar_timeout: Seconds to wait for tar creation
            cat_timeout: Seconds to wait for tar download

        Returns:
            dict with keys: success, file_count, extracted_files, output_dir, error
        """
        os.makedirs(local_output_dir, exist_ok=True)
        job_name = f"{job_prefix}-{secrets.token_hex(4)}"

        try:
            # Step 1: Check PVC exists
            if not await self._pvc_exists(pvc_name):
                return {
                    "success": False,
                    "error": f"PVC {pvc_name} does not exist",
                    "output_dir": local_output_dir,
                }

            # Step 2: Create download job
            await self._create_download_job(
                job_name, pvc_name, sub_path, container_path
            )

            # Step 3: Wait for pod to be ready
            pod = await self._wait_for_ready(job_name, ready_timeout)
            if pod is None:
                return {
                    "success": False,
                    "error": "Download job pod did not become ready",
                    "output_dir": local_output_dir,
                }

            # Step 4: Create tar archive on-demand
            find_expr = self._build_find_expr(file_patterns)
            tar_script = f"""
cd {container_path} 2>/dev/null || exit 1
FILE_LIST=$(find . -type f {find_expr} | sort)
FILE_COUNT=$(echo "$FILE_LIST" | grep -c . || echo 0)
echo "FILE_COUNT:$FILE_COUNT"
if [ "$FILE_COUNT" -gt 0 ]; then
    echo "$FILE_LIST" | tar -czf /tmp/download/archive.tar.gz -T -
    echo "TAR_CREATED:true"
else
    echo "TAR_CREATED:false"
fi
"""
            tar_result = await asyncio.wait_for(
                asyncio.to_thread(pod.exec, ["sh", "-c", tar_script]),
                timeout=tar_timeout,
            )

            output = tar_result.stdout.decode() if tar_result.stdout else ""
            file_count = 0
            tar_created = False
            for line in output.split("\n"):
                if line.startswith("FILE_COUNT:"):
                    try:
                        file_count = int(line.split(":")[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("TAR_CREATED:"):
                    tar_created = line.split(":")[1].strip() == "true"

            self._logger.info(
                f"PVC {pvc_name}/{sub_path}: {file_count} files, tar_created={tar_created}"
            )

            # Step 5: Download and extract tar
            extracted_files = []
            if file_count > 0 and tar_created:
                cat_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        pod.exec, ["cat", "/tmp/download/archive.tar.gz"]
                    ),
                    timeout=cat_timeout,
                )

                if cat_result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to download archive (exit code {cat_result.returncode})"
                    )

                local_archive = Path(local_output_dir) / "archive.tar.gz"
                local_archive.write_bytes(cat_result.stdout)

                with tarfile.open(local_archive, "r:gz") as tar:
                    tar.extractall(path=local_output_dir, filter="data")
                    extracted_files = tar.getnames()

                local_archive.unlink()

                self._logger.info(
                    f"Extracted {len(extracted_files)} files to {local_output_dir}"
                )
            else:
                self._logger.info("No files matched for extraction")

            # Step 6: Cleanup
            await self._delete_job(job_name)

            return {
                "success": True,
                "file_count": file_count,
                "extracted_files": extracted_files,
                "output_dir": local_output_dir,
            }

        except Exception as e:
            self._logger.error(f"PVC extraction failed: {e}")
            # Best-effort cleanup
            await self._delete_job(job_name)
            return {
                "success": False,
                "error": str(e),
                "output_dir": local_output_dir,
            }

    async def cleanup(self):
        """Delete any remaining download jobs."""
        for job_name in list(self._active_jobs):
            await self._delete_job(job_name)

    # --- Internal methods ---

    @staticmethod
    def _build_find_expr(patterns: list[str]) -> str:
        """Build a find -name expression from glob patterns.

        Example: ["*.log", "*.json"] -> '\\( -name "*.log" -o -name "*.json" \\)'
        """
        if not patterns:
            return ""
        parts = [f'-name "{p}"' for p in patterns]
        return "\\( " + " -o ".join(parts) + " \\)"

    async def _pvc_exists(self, pvc_name: str) -> bool:
        """Check if a PVC exists in the namespace."""
        try:
            assert self._core_api is not None
            await self._core_api.read_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            return True
        except exceptions.ApiException as e:
            if e.status == 404:
                return False
            self._logger.warning(f"Error checking PVC {pvc_name}: {e}")
            return False

    async def _create_download_job(
        self,
        job_name: str,
        pvc_name: str,
        sub_path: str,
        container_path: str,
    ):
        """Create a busybox job that mounts the PVC and waits for extraction commands."""
        download_script = f"""#!/bin/sh
set -e
mkdir -p /tmp/download
echo "ready" > /tmp/download/job_ready.txt
echo "=== DOWNLOAD JOB READY ==="
while true; do
    FILE_COUNT=$(find {container_path} -type f 2>/dev/null | wc -l)
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
                    "app": "pvc-extractor",
                    "managed-by": "pvc-extractor",
                },
            },
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "pvc-extractor",
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
                                        "name": "pvc-volume",
                                        "mountPath": container_path,
                                        "subPath": sub_path,
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
                                "name": "pvc-volume",
                                "persistentVolumeClaim": {
                                    "claimName": pvc_name,
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
            self._active_jobs.append(job_name)
            self._logger.info(f"Download job created: {job_name}")
        except exceptions.ApiException as e:
            if e.status == 409:
                self._logger.warning(f"Download job {job_name} already exists")
                self._active_jobs.append(job_name)
            else:
                raise

    async def _wait_for_ready(self, job_name: str, timeout: int = 60):
        """Wait for the download job pod to be ready and return it."""
        for attempt in range(timeout):
            try:
                pods = list(
                    kr8s.get(
                        "pods",
                        namespace=self.namespace,
                        label_selector=f"job-name={job_name}",
                    )
                )

                if pods:
                    pod = pods[0]
                    try:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                pod.exec,
                                ["test", "-f", "/tmp/download/job_ready.txt"],
                            ),
                            timeout=5.0,
                        )
                        if result.returncode == 0:
                            return pod
                    except Exception:
                        pass
            except Exception:
                pass

            if attempt % 10 == 9:
                self._logger.info(
                    f"Waiting for download job... (attempt {attempt + 1}/{timeout})"
                )
            await asyncio.sleep(1)

        self._logger.warning(
            f"Download job {job_name} did not become ready in {timeout}s"
        )
        return None

    async def _delete_job(self, job_name: str):
        """Delete a job with foreground propagation."""
        if job_name not in self._active_jobs:
            return

        try:
            from kubernetes_asyncio.client.models import V1DeleteOptions

            assert self._batch_api is not None
            await self._batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=V1DeleteOptions(propagation_policy="Foreground"),
            )
            self._logger.info(f"Download job {job_name} deleted")
        except exceptions.ApiException as e:
            if e.status != 404:
                self._logger.warning(f"Failed to delete job {job_name}: {e}")

        if job_name in self._active_jobs:
            self._active_jobs.remove(job_name)
