# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run rendered Parse 3.5 pages through an OpenAI-compatible Batch API.

This is a bounded tutorial adapter for the model package's single-page mode.
The caller supplies Curator's already-rendered tasks and keeps their page images
for the existing postprocessor. Batch output order has no bearing on page order.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pyarrow as pa
from PIL import Image

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

_MAX_INPUT_BYTES = 190_000_000  # Leave room below the gateway's 200 MB upload limit.
_MAX_INPUT_LINES = 50_000
_MAX_PILOT_PAGES = 1_000
_MAX_PILOT_IMAGE_BYTES = 300_000_000
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _request(model_name: str, image: bytes) -> dict[str, Any]:
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
        _server_generation_config,
    )
    from nemotron_parse.inference import DocumentParser

    class CapturedPage(Exception):
        pass

    captured: dict[str, Any] = {}

    def capture(
        page: Image.Image, *, prompt: str, stage: str, settings: Any
    ) -> dict[str, Any]:
        if stage != "fullpage":
            raise ValueError(f"Unexpected Parse stage: {stage}")
        stream = io.BytesIO()
        page.save(stream, format="PNG")
        captured.update(
            image=stream.getvalue(), prompt=prompt, settings=asdict(settings)
        )
        raise CapturedPage

    parser = DocumentParser(model=model_name, backend=capture)
    with Image.open(io.BytesIO(image)) as opened:
        try:
            parser.parse(opened, mode="single")
        except CapturedPage:
            pass
    if not captured:
        raise RuntimeError("Parse single mode did not request a full page")
    generation = _server_generation_config(
        {**captured["settings"], "skip_special_tokens": False}
    )
    wire_settings = {
        "max_tokens": generation.max_tokens,
        "n": generation.n,
        "seed": generation.seed,
        "stop": generation.stop,
        "stream": generation.stream,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        **(generation.extra_kwargs or {}).get("extra_body", {}),
    }
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": captured["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(captured["image"]).decode("ascii")
                        },
                    },
                ],
            }
        ],
        **wire_settings,
    }


def _input_chunks(
    requests: dict[str, dict[str, Any]], *, max_bytes: int, max_lines: int
) -> list[bytes]:
    if not 0 < max_bytes <= _MAX_INPUT_BYTES or not 0 < max_lines <= _MAX_INPUT_LINES:
        raise ValueError("Batch input limits exceed the gateway contract")
    chunks: list[bytes] = []
    current: list[bytes] = []
    size = 0
    for custom_id, body in requests.items():
        line = (
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        if len(line) > max_bytes:
            raise ValueError(
                f"One page exceeds the Batch input byte limit: {custom_id}"
            )
        if current and (size + len(line) > max_bytes or len(current) == max_lines):
            chunks.append(b"".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append(b"".join(current))
    return chunks


def _batch_results(content: bytes, expected: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for raw in content.splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        custom_id = row.get("custom_id")
        if custom_id not in expected or custom_id in found:
            raise ValueError(f"Unknown or duplicate Batch custom_id: {custom_id!r}")
        found[custom_id] = row
    return found


def _replay(image: bytes, model_name: str, body: dict[str, Any]) -> dict[str, Any]:
    from nemotron_parse.inference import DocumentParser

    choice = body["choices"][0]
    reply = {
        "raw_out": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage", {}),
    }
    parser = DocumentParser(model=model_name, backend=lambda *_a, **_k: reply)
    with Image.open(io.BytesIO(image)) as opened:
        return parser.parse(opened, mode="single")


def _validate_response(body: dict[str, Any], model_name: str) -> dict[str, int]:
    if not isinstance(body, dict) or body.get("model") != model_name:
        raise ValueError("Response model differs from the requested model")
    if not isinstance(body.get("id"), str) or not body["id"].strip():
        raise ValueError("Response lacks a nonempty response ID")
    choices = body.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ValueError("Expected exactly one response choice")
    if choices[0].get("finish_reason") not in ("stop", "eos"):
        raise ValueError("Response lacks an accepted final finish reason")
    message = choices[0].get("message")
    if (
        not isinstance(message, dict)
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
    ):
        raise ValueError("Response lacks nonempty text content")
    usage = body.get("usage")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0 for key in fields
    ):
        raise ValueError("Response lacks nonnegative integer token usage")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("Response token usage does not add up")
    return {key: usage[key] for key in fields}


def page_coverage(
    rows: list[dict[str, Any]], written_pages: set[tuple[str, int]]
) -> dict[str, Any]:
    """Account for pages with no parsed blocks without calling them visually blank."""
    expected = {
        (row["sample_id"], row["page_number"])
        for row in rows
        if row.get("page_number") is not None
    }
    empty = []
    for row in rows:
        if row.get("parse_document") and not row.get("materialize_error"):
            document = json.loads(row["parse_document"])
            if document["blocks"] == []:
                empty.append(
                    {
                        "sample_id": row["sample_id"],
                        "page_number": row["page_number"],
                        "image_sha256": hashlib.sha256(
                            row["binary_content"]
                        ).hexdigest(),
                    }
                )
    empty_ids = {(row["sample_id"], row["page_number"]) for row in empty}
    return {
        "structural_page_coverage_complete": len(expected) == len(rows)
        and not empty_ids
        and written_pages == expected,
        "empty_output_pages": empty,
        "empty_output_policy": "No parsed blocks; retained in this page ledger for image review, not accepted as blank",
        "image_grounding_accepted": False,
        "image_grounding_review": "not performed",
    }


class BatchClient:
    """Minimal file/job transport; one job is submitted for each bounded JSONL chunk."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        poll_interval_s: float = 2.0,
        evidence_path: Path | None = None,
        absolute_deadline_monotonic: float | None = None,
        max_upload_file_bytes: int | None = None,
        max_upload_body_bytes: int | None = None,
    ):
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if absolute_deadline_monotonic is not None and not math.isfinite(
            absolute_deadline_monotonic
        ):
            raise ValueError("absolute_deadline_monotonic must be finite")
        if (max_upload_file_bytes is None) != (max_upload_body_bytes is None):
            raise ValueError("Upload file and body caps must be supplied together")
        if max_upload_file_bytes is not None and (
            type(max_upload_file_bytes) is not int
            or type(max_upload_body_bytes) is not int
            or max_upload_file_bytes <= 0
            or max_upload_body_bytes < max_upload_file_bytes
        ):
            raise ValueError(
                "Upload caps must be positive with body cap at least file cap"
            )
        self.absolute_deadline_monotonic = absolute_deadline_monotonic
        self.max_upload_file_bytes = max_upload_file_bytes
        self.max_upload_body_bytes = max_upload_body_bytes
        self.check_deadline("Batch client construction")
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=60.0)
        self.poll_interval_s = poll_interval_s
        self.evidence_path = evidence_path
        self.profile: dict[str, Any] = {
            "clock": "client perf_counter elapsed seconds",
            "runs_started": 0,
            "runs_completed": 0,
            "runs_failed": 0,
            "phases": {},
        }
        self.jobs: dict[str, str] = {}
        self._raw_file_ids: dict[str, set[str]] = {}
        self._raw_retained: dict[tuple[str, str], tuple[int, int, str]] = {}
        self.request_submission_attempts = 0
        self.requests_confirmed_submitted = 0
        if evidence_path is not None:
            evidence_path.touch(exist_ok=False)

    def check_deadline(self, stage: str) -> float | None:
        if self.absolute_deadline_monotonic is None:
            return None
        remaining = self.absolute_deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Original reader deadline expired: {stage}")
        return remaining

    def _request_options(self) -> dict[str, Any]:
        remaining = self.check_deadline("HTTP request")
        if remaining is None:
            return {}
        configured = self.client.timeout
        return {
            "timeout": httpx.Timeout(
                **{
                    name: min(remaining, value) if value is not None else remaining
                    for name, value in (
                        (name, getattr(configured, name))
                        for name in ("connect", "read", "write", "pool")
                    )
                }
            )
        }

    def _retain_raw_download(self, job_id: str, file_id: str, content: bytes) -> None:
        """Save downloaded bytes and their job association before parsing results."""
        if self.evidence_path is None:
            return
        if job_id not in self.jobs or file_id not in self._raw_file_ids.get(
            job_id, set()
        ):
            raise ValueError("Raw Batch file is not advertised by an owned job")
        key = (job_id, file_id)
        digest = hashlib.sha256(content).hexdigest()
        previous = self._raw_retained.get(key)
        if previous is not None and previous[2] != digest:
            raise ValueError("Owned raw Batch content changed")
        directory = self.evidence_path.parent / "native-output"
        directory.mkdir(mode=0o700, exist_ok=True)
        name = hashlib.sha256((job_id + "/" + file_id).encode()).hexdigest() + ".jsonl"
        path = directory / name
        if previous is None:
            # Exclusive creation preserves evidence from an earlier run or another client.
            with os.fdopen(
                os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            for parent in (directory, directory.parent):
                fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        with os.fdopen(
            os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
        ) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Raw Batch output is not a regular file")
            identity = (info.st_dev, info.st_ino, digest)
            if previous is not None and identity != previous:
                raise ValueError("Owned raw Batch file identity changed")
            readback = hashlib.sha256()
            size = 0
            while size <= len(content):
                block = stream.read(min(1024**2, len(content) + 1 - size))
                if not block:
                    break
                size += len(block)
                readback.update(block)
            if size != len(content) or readback.hexdigest() != digest:
                raise ValueError("Raw Batch output readback differs")
        self._record(
            "raw_result_retained",
            job_id=job_id,
            file_id=file_id,
            path=str(path),
            bytes=len(content),
            sha256=digest,
            reused=previous is not None,
        )
        self._raw_retained[key] = identity

    def _record(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "utc": datetime.now(UTC).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        if self.evidence_path is not None:
            with self.evidence_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                if event == "raw_result_retained":
                    stream.flush()
                    os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)

    @contextmanager
    def _measure(self, phase: str, **fields: Any):
        """Keep cumulative caller elapsed time, including failed operations."""
        counter = self.profile["phases"].setdefault(
            phase, {"elapsed_s": 0.0, "calls": 0, "failures": 0}
        )
        counter["calls"] += 1
        started = time.perf_counter()
        failed = True
        try:
            yield
            failed = False
        finally:
            elapsed = time.perf_counter() - started
            counter["elapsed_s"] += elapsed
            counter["failures"] += int(failed)
            try:
                self._record(
                    "client_phase",
                    phase=phase,
                    elapsed_s=elapsed,
                    failed=failed,
                    **fields,
                )
            except Exception:
                # Profiling evidence must not change the transport's outcome.
                pass

    def _post_job(self, content: bytes) -> str:
        input_sha256 = hashlib.sha256(content).hexdigest()
        self._record(
            "upload_attempt",
            input_sha256=input_sha256,
            requests=len(content.splitlines()),
        )
        with self._measure("upload"):
            self.check_deadline("file upload preparation")
            if self.max_upload_file_bytes is None:
                response = self.client.post(
                    self.base_url + "/files",
                    data={"purpose": "batch"},
                    files={"file": ("parse-pages.jsonl", content, "application/jsonl")},
                    **self._request_options(),
                )
            else:
                if len(content) > self.max_upload_file_bytes:
                    raise ValueError("Actual JSONL file exceeds selected upload cap")
                with self._measure("multipart_encode"):
                    request = self.client.build_request(
                        "POST",
                        self.base_url + "/files",
                        data={"purpose": "batch"},
                        files={
                            "file": ("parse-pages.jsonl", content, "application/jsonl")
                        },
                        **self._request_options(),
                    )
                    pieces: list[bytes] = []
                    size = 0
                    try:
                        for index, piece in enumerate(request.stream):
                            self.check_deadline("actual multipart encoding")
                            if index >= 4096 or not isinstance(piece, bytes):
                                raise ValueError(
                                    "Actual multipart encoding exceeds selected chunk contract"
                                )
                            size += len(piece)
                            if size > self.max_upload_body_bytes:
                                raise ValueError(
                                    "Actual multipart body exceeds selected upload cap"
                                )
                            pieces.append(piece)
                        body = b"".join(pieces)
                        self.check_deadline("actual multipart encoding completion")
                        headers = request.headers.copy()
                        headers.pop("transfer-encoding", None)
                        headers["Content-Length"] = str(len(body))
                        outgoing = httpx.Request(
                            request.method,
                            request.url,
                            headers=headers,
                            content=body,
                            extensions=dict(request.extensions),
                        )
                    finally:
                        request.stream.close()
                self._record(
                    "upload_wire_prepared",
                    input_bytes=len(content),
                    multipart_bytes=len(body),
                    input_sha256=input_sha256,
                    multipart_sha256=hashlib.sha256(body).hexdigest(),
                    content_type=headers.get("content-type"),
                    multipart_chunks=len(pieces),
                )
                self.check_deadline("actual multipart send")
                outgoing.extensions["timeout"] = (
                    self._request_options()["timeout"].as_dict()
                    if self.absolute_deadline_monotonic is not None
                    else outgoing.extensions.get("timeout", {})
                )
                response = self.client.send(outgoing)
            self.check_deadline("file upload response")
            response.raise_for_status()
            file_id = response.json()["id"]
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("File upload returned no file ID")
        self._record("input_uploaded", input_file_id=file_id, input_sha256=input_sha256)
        self.request_submission_attempts += len(content.splitlines())
        self._record(
            "job_create_attempt",
            input_file_id=file_id,
            input_sha256=input_sha256,
            requests=len(content.splitlines()),
        )
        try:
            with self._measure("job_create", input_file_id=file_id):
                response = self.client.post(
                    self.base_url + "/batches",
                    json={
                        "input_file_id": file_id,
                        "endpoint": "/v1/chat/completions",
                        "completion_window": "24h",
                    },
                    **self._request_options(),
                )
                self.check_deadline("job create response")
                response.raise_for_status()
                job_id = response.json()["id"]
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError("Batch creation returned no job ID")
                if job_id in self.jobs:
                    raise ValueError(
                        "Batch creation returned an already submitted job ID"
                    )
        except BaseException:
            # An error response or lost acknowledgement cannot prove that no job was created.
            self._record(
                "job_create_unresolved",
                input_file_id=file_id,
                input_sha256=input_sha256,
                action="Owner must reconcile the submission; do not resubmit automatically",
            )
            raise
        self.jobs[job_id] = "submitted"
        self.requests_confirmed_submitted += len(content.splitlines())
        self._record(
            "job_created",
            job_id=job_id,
            input_file_id=file_id,
            input_sha256=input_sha256,
        )
        return job_id

    def run(self, chunks: list[bytes]) -> dict[str, dict[str, Any]]:
        self.profile["runs_started"] += 1
        pending: dict[str, set[str]] = {}
        results: dict[str, dict[str, Any]] = {}
        try:
            for chunk in chunks:
                self.check_deadline("native job submission")
                job_id = self._post_job(chunk)
                pending[job_id] = {
                    json.loads(line)["custom_id"] for line in chunk.splitlines()
                }
            while pending:
                for job_id, expected in list(pending.items()):
                    with self._measure("status_http", job_id=job_id):
                        response = self.client.get(
                            f"{self.base_url}/batches/{job_id}",
                            **self._request_options(),
                        )
                        self.check_deadline("job status response")
                        response.raise_for_status()
                        job = response.json()
                        if job.get("id") != job_id:
                            raise ValueError(
                                "Batch status response has a different job ID"
                            )
                    self.jobs[job_id] = job["status"]
                    self._record(
                        "job_status",
                        job_id=job_id,
                        status=job["status"],
                        request_counts=job.get("request_counts"),
                        output_file_id=job.get("output_file_id"),
                        error_file_id=job.get("error_file_id"),
                    )
                    if job["status"] not in _TERMINAL:
                        continue
                    if job["status"] != "completed":
                        raise RuntimeError(
                            f"Batch job {job_id} ended with {job['status']}"
                        )
                    job_results: dict[str, dict[str, Any]] = {}
                    advertised_files = tuple(
                        file_id
                        for file_id in (
                            job.get("output_file_id"),
                            job.get("error_file_id"),
                        )
                        if file_id
                    )
                    if any(
                        not isinstance(file_id, str) for file_id in advertised_files
                    ):
                        raise ValueError("Batch result file ID is not a string")
                    self._raw_file_ids.setdefault(job_id, set()).update(
                        advertised_files
                    )
                    for file_id in advertised_files:
                        with self._measure(
                            "result_download", job_id=job_id, file_id=file_id
                        ):
                            response = self.client.get(
                                f"{self.base_url}/files/{file_id}/content",
                                **self._request_options(),
                            )
                            self.check_deadline("result download response")
                            response.raise_for_status()
                        self._record(
                            "result_downloaded",
                            job_id=job_id,
                            file_id=file_id,
                            bytes=len(response.content),
                            sha256=hashlib.sha256(response.content).hexdigest(),
                        )
                        self._retain_raw_download(job_id, file_id, response.content)
                        for custom_id, row in _batch_results(
                            response.content, expected
                        ).items():
                            if custom_id in results or custom_id in job_results:
                                raise ValueError(f"Duplicate Batch result: {custom_id}")
                            job_results[custom_id] = row
                    if set(job_results) != expected:
                        raise ValueError(
                            f"Batch job {job_id} output IDs do not cover submitted IDs"
                        )
                    counts = job.get("request_counts")
                    count_keys = ("total", "completed", "failed")
                    if not isinstance(counts, dict) or any(
                        type(counts.get(key)) is not int or counts[key] < 0
                        for key in count_keys
                    ):
                        raise ValueError(
                            f"Batch job {job_id} lacks valid terminal request counts"
                        )
                    completed = sum(
                        row.get("error") is None
                        and isinstance(row.get("response"), dict)
                        and row["response"].get("status_code") == 200
                        for row in job_results.values()
                    )
                    observed = {
                        "total": len(expected),
                        "completed": completed,
                        "failed": len(expected) - completed,
                    }
                    if {key: counts[key] for key in count_keys} != observed:
                        raise ValueError(
                            f"Batch job {job_id} terminal counts disagree with downloaded results"
                        )
                    self._record(
                        "job_reconciled", job_id=job_id, request_counts=observed
                    )
                    results.update(job_results)
                    del pending[job_id]
                if pending:
                    remaining = self.check_deadline("poll delay")
                    with self._measure("poll_sleep"):
                        time.sleep(
                            self.poll_interval_s
                            if remaining is None
                            else min(self.poll_interval_s, remaining)
                        )
                    self.check_deadline("poll delay completion")
        except BaseException as exc:
            self.profile["runs_failed"] += 1
            cancellations = []
            evidence_errors = []

            def record_cleanup(event: str, **fields: Any) -> None:
                try:
                    self._record(event, **fields)
                except Exception as record_exc:
                    evidence_errors.append(
                        f"{event} {fields.get('job_id', '')}: {record_exc}"
                    )

            for job_id, status in list(self.jobs.items()):
                if status in _TERMINAL:
                    continue
                record_cleanup("cancel_attempt", job_id=job_id)
                try:
                    response = self.client.post(
                        f"{self.base_url}/batches/{job_id}/cancel",
                        **self._request_options(),
                    )
                    response.raise_for_status()
                    cancellation = response.json()
                    if not isinstance(cancellation, dict):
                        raise ValueError("Cancellation returned no job object")
                    terminal = (
                        cancellation.get("id") == job_id
                        and cancellation.get("status") in _TERMINAL
                    )
                    if terminal:
                        self.jobs[job_id] = cancellation["status"]
                    record_cleanup(
                        "cancel_response",
                        job_id=job_id,
                        status=cancellation.get("status"),
                        terminal_observed=terminal,
                        gpu_release_verified=False,
                    )
                    cancellations.append(
                        f"{job_id}: requested; terminal observed={terminal}"
                    )
                except Exception as cancel_exc:
                    record_cleanup(
                        "cancel_unresolved", job_id=job_id, error=str(cancel_exc)
                    )
                    cancellations.append(
                        f"{job_id}: cancellation failed ({cancel_exc})"
                    )
            record_cleanup(
                "run_failed", error=str(exc), jobs=self.jobs, gpu_release_verified=False
            )
            detail = f"submitted jobs: {', '.join(cancellations)}; cleanup evidence errors: {evidence_errors}"
            if not isinstance(exc, Exception):
                exc.add_note(detail)
                raise
            raise RuntimeError(f"Batch pilot failed: {exc}; {detail}") from exc
        self.profile["runs_completed"] += 1
        return results


def infer_tasks(
    tasks: list[InterleavedBatch],
    *,
    model_name: str,
    gateway: BatchClient,
    model_path: str | None = None,
    max_bytes: int = _MAX_INPUT_BYTES,
    max_lines: int = _MAX_INPUT_LINES,
    timings: dict[str, Any] | None = None,
    accounting: dict[str, Any] | None = None,
) -> list[InterleavedBatch]:
    """Aggregate page requests across tasks, then restore each task for Curator postprocessing."""
    from nemo_curator.tasks import InterleavedBatch

    gateway.check_deadline("task materialization")
    rows_by_task = [task.to_pyarrow().to_pylist() for task in tasks]
    gateway.check_deadline("task materialization completion")
    images = [
        row["binary_content"]
        for rows in rows_by_task
        for row in rows
        if not row.get("materialize_error") and row.get("binary_content")
    ]
    if len(images) > _MAX_PILOT_PAGES or sum(map(len, images)) > _MAX_PILOT_IMAGE_BYTES:
        raise ValueError(
            f"Batch pilot exceeds {_MAX_PILOT_PAGES} pages or "
            f"{_MAX_PILOT_IMAGE_BYTES} rendered-image bytes ({len(images)} pages); "
            "split the manifest into smaller runs"
        )
    request_started = time.perf_counter()
    requests = {}
    for task_index, rows in enumerate(rows_by_task):
        for row_index, row in enumerate(rows):
            gateway.check_deadline("page request construction")
            if not row.get("materialize_error") and row.get("binary_content"):
                requests[f"page-{task_index}-{row_index}"] = _request(
                    model_name, row["binary_content"]
                )
            gateway.check_deadline("page request construction completion")
    request_build_s = time.perf_counter() - request_started
    if accounting is not None:
        accounting.update(
            requests_prepared=len(requests),
            client_retries=0,
            responses={},
            rejected_responses={},
            request_sha256={
                custom_id: hashlib.sha256(
                    json.dumps(body, sort_keys=True).encode()
                ).hexdigest()
                for custom_id, body in requests.items()
            },
        )
    gateway_started = time.perf_counter()
    gateway.check_deadline("JSONL chunk construction")
    chunks = _input_chunks(requests, max_bytes=max_bytes, max_lines=max_lines)
    gateway.check_deadline("JSONL chunk construction completion")
    try:
        results = gateway.run(chunks) if requests else {}
    finally:
        if timings is not None:
            timings.update(
                request_build_s=request_build_s,
                gateway_s=time.perf_counter() - gateway_started,
                client_profile=gateway.profile,
            )
    gateway_s = time.perf_counter() - gateway_started
    replay_started = time.perf_counter()
    output: list[InterleavedBatch] = []
    for task_index, (task, rows) in enumerate(zip(tasks, rows_by_task, strict=True)):
        for row_index, row in enumerate(rows):
            gateway.check_deadline("page response replay")
            if row.get("materialize_error") or not row.get("binary_content"):
                row["parse_document"] = None
                continue
            custom_id = f"page-{task_index}-{row_index}"
            result = results.get(custom_id)
            try:
                if result is None:
                    raise ValueError("Batch result missing")
                response = result.get("response") or {}
                if (
                    result.get("error") is not None
                    or response.get("status_code") != 200
                ):
                    raise ValueError(
                        f"Batch request failed: {result.get('error') or response.get('status_code')}"
                    )
                usage = _validate_response(response["body"], model_name)
                document = _replay(row["binary_content"], model_name, response["body"])
                if (
                    not isinstance(document.get("blocks"), list)
                    or not document["blocks"]
                ):
                    raise ValueError("Parse returned no decoded page blocks")
                gateway.check_deadline("page response replay completion")
                if accounting is not None:
                    accounting["responses"][custom_id] = {
                        "response_id": response["body"].get("id"),
                        "model": model_name,
                        "finish_reason": response["body"]["choices"][0][
                            "finish_reason"
                        ],
                        "usage": usage,
                        "output_sha256": hashlib.sha256(
                            response["body"]["choices"][0]["message"][
                                "content"
                            ].encode()
                        ).hexdigest(),
                        "blocks": len(document["blocks"]),
                    }
                row["parse_document"] = json.dumps(document)
                row["materialize_error"] = None
            except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                if accounting is not None:
                    accounting["rejected_responses"][custom_id] = str(exc)
                row["parse_document"] = None
                row["materialize_error"] = json.dumps(
                    {
                        "stage": "parse_request",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        output.append(
            InterleavedBatch(
                dataset_name=task.dataset_name,
                data=pa.Table.from_pylist(rows),
                _metadata={
                    **task._metadata,
                    "parse_mode": "single",
                    "model_path": model_path or model_name,
                },
                _stage_perf=task._stage_perf,
            )
        )
    if timings is not None:
        replay_s = time.perf_counter() - replay_started
        timings.update(
            request_build_s=request_build_s,
            gateway_s=gateway_s,
            replay_s=replay_s,
        )
    gateway.check_deadline("inferred task completion")
    return output
