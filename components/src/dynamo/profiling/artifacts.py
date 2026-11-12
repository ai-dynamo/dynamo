# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap artifact presence checks on the capture path; the bundle job inspects."""

import asyncio
import re
import time
from pathlib import Path

from .models import Config
from .nsys import validate_reports

RANK_PATTERNS = {
    # <prefix>_<rank>.<stamp>.pt.trace.json[.gz]
    "vllm": r"_(.+)\.\d+\.pt\.trace\.json(?:\.gz)?",
    # <prefix>-<profile id>-TP-<n>[-DP-<n>][-PP-<n>][-EP-<n>][-<stage>].trace.json.gz
    "sglang": r"-[^-]+-(TP-\d+(?:-DP-\d+)?(?:-PP-\d+)?(?:-EP-\d+)?)(?:-[a-z]+)?"
    r"\.trace\.json(?:\.gz)?",
}


def rank_files(
    location: str, prefix: str, layout: str = "vllm"
) -> list[tuple[str, Path]]:
    """Kineto rank traces written under this session's unique prefix."""
    pattern = re.compile(re.escape(prefix) + RANK_PATTERNS[layout])
    return sorted(
        (match.group(1), path)
        for path in Path(location).iterdir()
        if (match := pattern.fullmatch(path.name))
    )


# vLLM stamps a trace while handling stop, before the stop response returns; the
# allowance covers clock skew between the worker and this host.
STAMP_ALLOWANCE_NS = 1_000_000_000


def adopt_session_traces(
    location: str, prefix: str, ranks: list[str], window: tuple[int, int]
) -> list[dict]:
    """Rename this session's rank traces that vLLM wrote under an earlier prefix.

    vLLM fixes a worker's torch trace name when it builds the profiler on the
    first profile start, so later sessions on that worker write under the first
    session's prefix. Only one session holds the lease, so a missing rank's
    single trace stamped between this session's start request and stop belongs
    to it; earlier and later sessions' files never qualify. Ambiguous windows are
    left for validation to reject.
    """
    adopted = []
    for rank in ranks:
        pattern = re.compile(
            rf"(.+)_{re.escape(rank)}\.(\d+)(\.pt\.trace\.json(?:\.gz)?)"
        )
        candidates = [
            (path, match)
            for path in Path(location).iterdir()
            if (match := pattern.fullmatch(path.name))
            and match.group(1) != prefix
            and window[0] <= int(match.group(2)) <= window[1]
        ]
        if len(candidates) != 1:
            continue
        path, match = candidates[0]
        target = path.with_name(f"{prefix}_{rank}.{match.group(2)}{match.group(3)}")
        path.rename(target)
        adopted.append(
            {"rank_id": rank, "renamed_from": path.name, "location": str(target)}
        )
    return adopted


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


def perf_artifact(evidence: dict) -> dict:
    """The frontend perf recording, kept for the archive though nothing parses it."""
    if not evidence.get("artifact"):
        return {"status": "unverified", "coverage": "unknown", "reason": "no_artifact"}
    path = Path(evidence["artifact"])
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError as exc:
        return {"status": "invalid", "coverage": "partial", "error": str(exc)}
    if size <= 0:
        return {"status": "invalid", "coverage": "partial", "error": "empty or missing"}
    return {
        "status": "present",
        "coverage": "present",
        "artifacts": [{"location": str(path), "bytes": size, "validation": "present"}],
        "inspection": "not_supported",
    }


class SharedFilesystemValidator:
    async def validate(self, evidence: dict, config: Config) -> dict:
        if evidence.get("collector") == "nsys":
            return await validate_reports(evidence, config)
        return await asyncio.to_thread(self._validate, evidence, config)

    def _validate(self, evidence: dict, config: Config) -> dict:
        if evidence.get("collector") == "linux_perf":
            return perf_artifact(evidence)
        location = evidence.get("artifact_location")
        prefix = evidence.get("artifact_prefix")
        layout = evidence.get("artifact_layout", "vllm")
        if (
            evidence.get("collector") != "torch_profiler"
            or not location
            or not prefix
            or layout not in RANK_PATTERNS
        ):
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
        expected = evidence.get("expected_rank_ids", [])
        try:
            adopted = []
            # Only vLLM keeps a worker's first trace name; SGLang names each start.
            if layout == "vllm" and expected and "start_requested_at_ns" in evidence:
                prefixed = {rank for rank, _ in rank_files(location, prefix)}
                adopted = adopt_session_traces(
                    location,
                    prefix,
                    sorted(set(expected) - prefixed),
                    (
                        evidence["start_requested_at_ns"],
                        evidence.get("stopped_at_ns", time.time_ns())
                        + STAMP_ALLOWANCE_NS,
                    ),
                )
            found = rank_files(location, prefix, layout)
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
            "association": "unique_session_prefix"
            + ("+session_window" if adopted else ""),
            **({"adopted": adopted} if adopted else {}),
            "scope": "engine_rank_files",
            "inspection": "deferred_to_bundle",
            "excluded": "AsyncLLM CPU traces lack the native profile prefix",
        }
