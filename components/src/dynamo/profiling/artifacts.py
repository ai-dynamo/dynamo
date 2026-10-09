# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-filesystem Kineto validation; other stores implement the same protocol."""

import asyncio
import gzip
import hashlib
import re
import time
from pathlib import Path
from typing import Protocol

from .models import Config


class ArtifactValidator(Protocol):
    async def validate(self, evidence: dict, config: Config) -> dict:
        ...


class BoundedReader:
    def __init__(self, stream, limit: int, deadline: float):
        self.stream = stream
        self.limit = limit
        self.deadline = deadline
        self.bytes = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Artifact validation deadline expired")
        data = self.stream.read(min(size if size >= 0 else 65536, 65536))
        self.bytes += len(data)
        if self.bytes > self.limit:
            raise ValueError("Trace exceeds uncompressed byte limit")
        self.digest.update(data)
        return data


def inspect_trace(path: Path, config: Config, deadline: float) -> dict:
    import ijson

    before = path.stat()
    events = kernels = 0
    has_events = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        reader = BoundedReader(stream, config.max_trace_bytes, deadline)
        for prefix, event, value in ijson.parse(reader):
            if (prefix, event) == ("traceEvents", "start_array"):
                has_events = True
            elif (prefix, event) == ("traceEvents.item", "start_map"):
                events += 1
            elif (
                prefix == "traceEvents.item.cat"
                and event == "string"
                and value == "kernel"
            ):
                kernels += 1
        # Consuming the whole stream also verifies the gzip trailer and CRC.
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Trace changed during validation")
        if not has_events or not events:
            raise ValueError("Trace has no traceEvents objects")
        return {
            "location": str(path),
            "bytes": after.st_size,
            "uncompressed_bytes": reader.bytes,
            "sha256_uncompressed": reader.digest.hexdigest(),
            "events": events,
            "cuda_kernels": kernels,
            "validation": "valid",
        }


class SharedFilesystemValidator:
    async def validate(self, evidence: dict, config: Config) -> dict:
        # Parsing can take seconds for native traces; never run it on the
        # frontend's serving loop. The reader enforces a cooperative deadline.
        return await asyncio.to_thread(self._validate, evidence.copy(), config)

    def _validate(self, evidence: dict, config: Config) -> dict:
        location = evidence.get("artifact_location")
        prefix = evidence.get("artifact_prefix")
        if evidence.get("collector") != "torch_profiler" or not location or not prefix:
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "no_supported_accessible_store",
            }
        directory = Path(location)
        if not directory.is_dir():
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "artifact_store_not_mounted",
                "location": location,
            }
        import ijson

        pattern = re.compile(
            rf"^{re.escape(prefix)}_(.+)\.\d+\.pt\.trace\.json(?:\.gz)?$"
        )
        deadline = time.monotonic() + config.artifact_validation_timeout_s
        artifacts = []
        ranks = []
        try:
            for path in directory.iterdir():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Artifact enumeration deadline expired")
                match = pattern.fullmatch(path.name)
                if not match or not path.is_file():
                    continue
                rank = match.group(1)
                ranks.append(rank)
                try:
                    artifact = inspect_trace(path, config, deadline)
                except (
                    OSError,
                    EOFError,
                    ijson.JSONError,
                    ValueError,
                    TimeoutError,
                ) as exc:
                    artifact = {
                        "location": str(path),
                        "validation": "invalid",
                        "error": str(exc),
                    }
                artifacts.append({**artifact, "rank_id": rank})
        except (OSError, TimeoutError) as exc:
            return {
                "status": "invalid",
                "coverage": "partial",
                "artifacts": artifacts,
                "error": str(exc),
            }
        artifacts.sort(key=lambda artifact: artifact["location"])
        ranks.sort()
        expected = evidence.get("expected_rank_ids", [])
        missing = sorted(set(expected) - set(ranks))
        unexpected = sorted(set(ranks) - set(expected)) if expected else []
        duplicate = len(ranks) != len(set(ranks))
        invalid = any(artifact["validation"] != "valid" for artifact in artifacts)
        status = (
            "invalid"
            if invalid or duplicate
            else "partial"
            if missing or unexpected or not artifacts
            else "verified"
            if expected
            else "unverified"
        )
        return {
            "status": status,
            "coverage": "complete"
            if status == "verified"
            else "partial"
            if status in ("invalid", "partial")
            else "unknown",
            "artifacts": artifacts,
            "expected_rank_ids": expected,
            "observed_rank_ids": ranks,
            "missing_rank_ids": missing,
            "unexpected_rank_ids": unexpected,
            "duplicate_rank_files": duplicate,
            "association": "unique_session_prefix",
            "scope": "engine_rank_files",
            "excluded": "AsyncLLM CPU traces lack the native profile prefix",
        }
