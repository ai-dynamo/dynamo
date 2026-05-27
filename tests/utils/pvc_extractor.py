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
import shutil
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
        tar_timeout: int = 600,
        cat_timeout: int = 120,
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
            tar_timeout: Upper bound (cap) on tar-creation timeout. The actual
                wait is computed dynamically from observed source size
                (see _GZIP_BYTES_PER_SEC); ``tar_timeout`` is the safety cap so
                a runaway gzip can't hang the test indefinitely.
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

            # Step 4: Two-phase tar creation.
            #
            # Phase A — discover (fast, fixed 30s budget): list matching
            # files + measure their total size, write the list to a temp
            # file so phase B can reuse it without re-running find. We
            # deliberately keep this phase small so we don't burn the
            # cap before we even know what we're dealing with.
            #
            # Phase B — tar+gzip with a *computed* timeout based on the
            # observed source size. The user-supplied ``tar_timeout``
            # parameter is the safety cap; the actual wait is
            #   max(60, ceil(src_bytes / _GZIP_BYTES_PER_SEC) + 60)
            # then clamped to tar_timeout. This means:
            #   - 200 MB rung → ~80s timeout (instead of 600s ceiling)
            #   - 1.1 GB steady → ~170s timeout (vs the 30s that broke us)
            #   - 10 GB cliff → would compute ~1100s but capped at 600s
            # The 60s floor covers very small rungs where overhead dominates.
            find_expr = self._build_find_expr(file_patterns)
            discover_script = f"""
cd {container_path} 2>/dev/null || exit 1
FILE_LIST=$(find . -type f {find_expr} | sort)
FILE_COUNT=$(echo "$FILE_LIST" | grep -c . || echo 0)
echo "FILE_COUNT:$FILE_COUNT"
if [ "$FILE_COUNT" -gt 0 ]; then
    BYTES=$(echo "$FILE_LIST" | xargs -r du -cb 2>/dev/null | tail -1 | awk '{{print $1}}')
    echo "SOURCE_BYTES:$BYTES"
    echo "$FILE_LIST" > /tmp/download/files.txt
fi
"""
            discover_result = await asyncio.wait_for(
                asyncio.to_thread(pod.exec, ["sh", "-c", discover_script]),
                timeout=30,
            )

            file_count = 0
            source_bytes = 0
            for line in (
                discover_result.stdout.decode() if discover_result.stdout else ""
            ).split("\n"):
                if line.startswith("FILE_COUNT:"):
                    try:
                        file_count = int(line.split(":")[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("SOURCE_BYTES:"):
                    try:
                        source_bytes = int(line.split(":")[1].strip())
                    except ValueError:
                        pass

            tar_created = False
            tar_bytes = 0

            if file_count > 0:
                # Compute dynamic tar timeout from observed size.
                computed_timeout = max(
                    60,
                    int(source_bytes / self._GZIP_BYTES_PER_SEC) + 60,
                )
                effective_timeout = min(computed_timeout, tar_timeout)

                self._logger.info(
                    f"PVC {pvc_name}/{sub_path}: {file_count} files, "
                    f"src={source_bytes / 1024 / 1024:.1f}MiB, "
                    f"tar_timeout={effective_timeout}s "
                    f"(computed={computed_timeout}s, cap={tar_timeout}s)"
                )

                # IMPORTANT: tar runs in a *fresh* shell from a separate
                # `pod.exec`, so CWD is `/` — not container_path. files.txt
                # holds relative paths like `./profile_export.json`, which
                # tar would look up from `/` and silently produce an empty
                # archive (header+footer only, ~10 KiB), then TAR_CREATED
                # would still say true. Always `cd container_path` first;
                # also make tar/stat failures observable rather than
                # swallowed by the unconditional echo.
                tar_script = f"""
set -e
cd {container_path}
tar -czf /tmp/download/archive.tar.gz -T /tmp/download/files.txt
echo "TAR_CREATED:true"
TAR_BYTES=$(stat -c %s /tmp/download/archive.tar.gz 2>/dev/null || echo 0)
echo "TAR_BYTES:$TAR_BYTES"
"""
                tar_result = await asyncio.wait_for(
                    asyncio.to_thread(pod.exec, ["sh", "-c", tar_script]),
                    timeout=effective_timeout,
                )

                for line in (
                    tar_result.stdout.decode() if tar_result.stdout else ""
                ).split("\n"):
                    if line.startswith("TAR_CREATED:"):
                        tar_created = line.split(":")[1].strip() == "true"
                    elif line.startswith("TAR_BYTES:"):
                        try:
                            tar_bytes = int(line.split(":")[1].strip())
                        except ValueError:
                            pass

                self._logger.info(
                    f"PVC {pvc_name}/{sub_path}: tar.gz={tar_bytes / 1024 / 1024:.1f}MiB, "
                    f"tar_created={tar_created}"
                )
            else:
                self._logger.info(f"PVC {pvc_name}/{sub_path}: 0 files matched")

            # Step 5: Download and extract tar
            #
            # IMPORTANT: a single `pod.exec(["cat", <tar>])` buffers the
            # entire tar in the kr8s/k8s python client's response and
            # rides on one long-lived WebSocket. Through Teleport-proxied
            # kubectl streams, that connection dies with "websocket:
            # close 1006 (abnormal closure): unexpected EOF" for tars
            # over ~150 MB — we lost 10/10 long-run rungs (~645 MB each)
            # to this on 2026-05-19. The fix is split-server-side +
            # per-chunk fetch: each chunk's transfer is short enough to
            # dodge the Teleport WebSocket close, and a failed chunk can
            # be retried without redownloading the whole archive.
            extracted_files = []
            if file_count > 0 and tar_created:
                local_archive = Path(local_output_dir) / "archive.tar.gz"
                await self._download_archive_chunked(
                    pod=pod,
                    remote_path="/tmp/download/archive.tar.gz",
                    local_path=local_archive,
                    cat_timeout=cat_timeout,
                )

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
            # Include exception type — empty str(e) is common for
            # asyncio.TimeoutError and ExceptionGroup, which is exactly
            # what we hit on the S1 3.2 GB cliff extraction and were
            # left guessing at because the original log read
            # ``PVC extraction failed: `` with no body.
            self._logger.error(f"PVC extraction failed: {type(e).__name__}: {e}")
            # Best-effort cleanup
            await self._delete_job(job_name)
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "output_dir": local_output_dir,
            }

    async def cleanup(self):
        """Delete any remaining download jobs."""
        for job_name in list(self._active_jobs):
            await self._delete_job(job_name)

    # --- Internal methods ---

    # Conservative gzip throughput estimate (bytes/sec) used to scale the
    # tar-creation timeout to observed source size. Real busybox `tar -czf`
    # on aiperf jsonl payloads measured ~25-50 MB/s in prior rungs;
    # we use 10 MB/s as a pessimistic baseline so the computed timeout
    # has headroom even on a busy node. A 1.1 GB rung → ~170s computed
    # budget; the rescue extraction completed in ~30s, so we have 5x margin.
    _GZIP_BYTES_PER_SEC = 10 * 1024 * 1024

    # Chunk size for the split-then-fetch download path. 50 MiB keeps
    # each chunk's transfer comfortably under the Teleport-proxied
    # WebSocket's working window — we empirically saw 124 MiB single
    # transfers die with "websocket: close 1006" in prior runs (this
    # was a 3.2 GB cliff-load JSONL on the S1 cascade test), and the
    # original 100 MiB ceiling was right at that failure edge. Manual
    # rescue at 50 MiB succeeded on all 5 chunks. Smaller = more
    # round-trips but each chunk fits inside both the WS window and the
    # per-chunk timeout with headroom for jitter.
    _CHUNK_BYTES = 50 * 1024 * 1024

    # Per-chunk retry budget. Each failed chunk gets re-fetched
    # independently — the archive is not rebuilt. Bumped 4 → 8 alongside
    # the A.3 streaming refactor: with the in-memory buffer eliminated,
    # the dominant failure mode shifts from "chunk too big" to "transient
    # network blip", which retries fix cleanly. Total grace at linear
    # 2*attempt backoff: 2+4+6+...+16 = 72 seconds across 8 attempts.
    _CHUNK_MAX_RETRIES = 8

    async def _download_archive_chunked(
        self,
        pod,
        remote_path: str,
        local_path: Path,
        cat_timeout: int,
    ) -> None:
        """Download a remote archive in chunks and reassemble locally.

        Step A: split the archive server-side into 100 MiB parts.
        Step B: cat each part separately via pod.exec, with per-chunk
                retry. Each chunk is short enough that the Teleport
                proxied WebSocket survives.
        Step C: concatenate parts into ``local_path``.
        Step D: clean up parts on the pod.

        Replaces a single ``cat`` of the full archive, which buffers
        the entire tar in memory and rides one long-lived WebSocket
        — that stream dies for ~150+ MB tars through Teleport.
        """

        # Step A: split server-side
        split_script = (
            "set -e\n"
            f"split -b {self._CHUNK_BYTES} {remote_path} {remote_path}.part-\n"
            f"ls {remote_path}.part-* | sort\n"
        )
        split_result = await asyncio.wait_for(
            asyncio.to_thread(pod.exec, ["sh", "-c", split_script]),
            timeout=60,
        )
        if split_result.returncode != 0:
            err = (split_result.stderr or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"split failed (rc={split_result.returncode}): {err}")
        part_names = [
            line.strip()
            for line in split_result.stdout.decode(
                "utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        self._logger.info(
            f"_download_archive_chunked: split into {len(part_names)} part(s)"
        )

        # Step B: stream each part directly to a per-chunk tempfile,
        # then append to the main archive only on success.
        #
        # kr8s `Pod.exec(stdout=BinaryIO, capture_output=False)` writes
        # the WebSocket exec stdout into the file handle as bytes arrive
        # — no full-chunk in-memory buffering. Earlier versions of this
        # method captured cat_result.stdout (50 MiB of bytes in Python
        # memory per chunk), which interacted badly with Teleport-proxied
        # WebSocket framing under load and surfaced as TimeoutError on
        # the asyncio.wait_for. Streaming to disk lets the kernel buffer
        # at TCP-receive size, decouples timing of the WS read loop from
        # the Python event loop, and keeps memory use ~constant
        # regardless of chunk size. Verified on the S1 cliff (3.2 GB
        # raw / 224 MB gz) in prior runs — see the cascade reproduction work
        # vault findings/related-investigations.md for the failure case.
        #
        # We stage each chunk in a sibling tempfile (`.part-<idx>.tmp`)
        # so a chunk that fails mid-stream can be retried without
        # corrupting the main archive. Successful chunks get appended
        # to the main archive via shutil.copyfileobj (16 KiB kernel
        # buffer), then the tempfile is unlinked.
        try:
            # Truncate the local file before appending parts
            local_path.write_bytes(b"")
            for idx, part_name in enumerate(part_names):
                chunk_tmp = local_path.parent / (f".{local_path.name}.chunk-{idx}.tmp")
                last_err: Optional[Exception] = None
                for attempt in range(1, self._CHUNK_MAX_RETRIES + 1):
                    try:
                        # Fresh tempfile each attempt — discard any
                        # partial bytes from a prior failed try.
                        chunk_tmp.unlink(missing_ok=True)
                        with chunk_tmp.open("wb") as chunk_fh:
                            cat_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    pod.exec,
                                    ["cat", part_name],
                                    stdout=chunk_fh,
                                    capture_output=False,
                                ),
                                timeout=cat_timeout,
                            )
                        if cat_result.returncode != 0:
                            raise RuntimeError(
                                f"cat returned rc={cat_result.returncode}"
                            )
                        chunk_bytes = chunk_tmp.stat().st_size
                        # Append this chunk's tempfile to the main archive.
                        with local_path.open("ab") as out:
                            with chunk_tmp.open("rb") as inp:
                                shutil.copyfileobj(inp, out)
                        chunk_tmp.unlink(missing_ok=True)
                        self._logger.info(
                            f"  chunk {idx + 1}/{len(part_names)} "
                            f"({chunk_bytes:,} bytes) try {attempt} ✓"
                        )
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        partial = chunk_tmp.stat().st_size if chunk_tmp.exists() else 0
                        chunk_tmp.unlink(missing_ok=True)
                        self._logger.warning(
                            f"  chunk {idx + 1}/{len(part_names)} attempt "
                            f"{attempt} failed after {partial:,} streamed "
                            f"bytes: {type(e).__name__}: {e}"
                        )
                        if attempt < self._CHUNK_MAX_RETRIES:
                            await asyncio.sleep(2 * attempt)
                else:
                    raise RuntimeError(
                        f"chunk {idx + 1}/{len(part_names)} ({part_name}) failed "
                        f"after {self._CHUNK_MAX_RETRIES} attempts: {last_err}"
                    )
        finally:
            # Step D: best-effort cleanup of the parts on the pod
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        pod.exec, ["sh", "-c", f"rm -f {remote_path}.part-*"]
                    ),
                    timeout=30,
                )
            except Exception as e:  # noqa: BLE001
                self._logger.debug(
                    f"_download_archive_chunked: cleanup of parts failed (non-fatal): {e}"
                )

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
