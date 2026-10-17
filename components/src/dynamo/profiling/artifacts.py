# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap artifact presence checks on the capture path; the bundle job inspects."""

import asyncio
import re
from pathlib import Path

from .models import Config
from .nsys import validate_reports


def rank_files(location: str, prefix: str) -> list[tuple[str, Path]]:
    """Kineto rank traces written under this session's unique prefix."""
    pattern = re.compile(rf"{re.escape(prefix)}_(.+)\.\d+\.pt\.trace\.json(?:\.gz)?")
    return sorted(
        (match.group(1), path)
        for path in Path(location).iterdir()
        if (match := pattern.fullmatch(path.name))
    )


def present(path: Path) -> tuple[int, str | None]:
    """Size and why a native trace is unusable (None if usable), without parsing."""
    size = path.stat().st_size
    if size <= 0:
        return size, "empty"
    if path.suffix == ".gz":
        with path.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                return size, "not a gzip file"
    return size, None


class SharedFilesystemValidator:
    async def validate(self, evidence: dict, config: Config) -> dict:
        if evidence.get("collector") == "nsys":
            return await validate_reports(evidence, config)
        return await asyncio.to_thread(self._validate, evidence, config)

    def _validate(self, evidence: dict, config: Config) -> dict:
        location = evidence.get("artifact_location")
        prefix = evidence.get("artifact_prefix")
        if evidence.get("collector") != "torch_profiler" or not location or not prefix:
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "no_supported_accessible_store",
            }
        if not Path(location).is_dir():
            return {
                "status": "unverified",
                "coverage": "unknown",
                "reason": "artifact_store_not_mounted",
                "location": location,
            }
        try:
            found = rank_files(location, prefix)
            artifacts = []
            for rank, path in found:
                size, problem = present(path)
                artifacts.append(
                    {
                        "location": str(path),
                        "rank_id": rank,
                        "bytes": size,
                        "validation": "present" if problem is None else "invalid",
                        **({"error": problem} if problem else {}),
                    }
                )
        except OSError as exc:
            return {"status": "invalid", "coverage": "partial", "error": str(exc)}
        ranks = [rank for rank, _ in found]
        expected = evidence.get("expected_rank_ids", [])
        missing = sorted(set(expected) - set(ranks))
        unexpected = sorted(set(ranks) - set(expected)) if expected else []
        duplicate = len(ranks) != len(set(ranks))
        invalid = any(artifact["validation"] != "present" for artifact in artifacts)
        status = (
            "invalid"
            if invalid or duplicate
            else "partial"
            if missing or unexpected or not artifacts
            else "present"
            if expected
            else "unverified"
        )
        return {
            "status": status,
            "coverage": {"present": "present", "unverified": "unknown"}.get(
                status, "partial"
            ),
            "artifacts": artifacts,
            "expected_rank_ids": expected,
            "observed_rank_ids": sorted(ranks),
            "missing_rank_ids": missing,
            "unexpected_rank_ids": unexpected,
            "duplicate_rank_files": duplicate,
            "association": "unique_session_prefix",
            "scope": "engine_rank_files",
            "inspection": "deferred_to_bundle",
            "excluded": "AsyncLLM CPU traces lack the native profile prefix",
        }
