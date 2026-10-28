# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online backend — benchmark live endpoints with AIPerf.

The real-deployment counterpart to ``runners/sims.py``. Where the sim drives an
autoscaler *adapter* against the mocker, the online backend points **AIPerf** at
a live OpenAI-compatible endpoint (each endpoint is some autoscaler running "as
itself" behind a real serving stack — Planner, KEDA, a static deployment, …),
replays the SAME Mooncake workload traces, and collects AIPerf's client-side
metrics + goodput. We shell out to the ``aiperf`` CLI rather than import it, so
this package stays thin and decoupled from AIPerf's environment.

SLO profiles are shared with the offline scorecard: each profile becomes an
AIPerf ``--goodput`` constraint string, so offline-sim and online goodput are the
same metric by construction.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from autoscaling_arena.scorecard import DEFAULT_PROFILES, SLOProfile


@dataclass
class Endpoint:
    """A live endpoint under test: a name, a URL, the model to request, and a
    human description (e.g. which autoscaler/config sits behind it)."""

    name: str
    url: str
    model: str
    description: str = ""
    endpoint_type: str = "chat"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Endpoint":
        return Endpoint(
            name=d["name"],
            url=d["url"],
            model=d["model"],
            description=d.get("description", ""),
            endpoint_type=d.get("endpoint_type", "chat"),
        )


def load_endpoints(path: str | Path) -> list[Endpoint]:
    """Load a JSON or YAML endpoints mapping (or a bare endpoint list)."""
    source = Path(path)
    text = source.read_text()
    data = (
        yaml.safe_load(text)
        if source.suffix.lower() in {".yaml", ".yml"}
        else json.loads(text)
    )
    items = data.get("endpoints") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"{source}: endpoints must be a non-empty list")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"{source}: each endpoint must be a mapping")
    return [Endpoint.from_dict(item) for item in items]


# Map our SLO metric names onto AIPerf's goodput metric tags.
_AIPERF_GOODPUT_KEYS = {
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}


def slo_to_aiperf_goodput(profile: SLOProfile) -> Optional[str]:
    """Render an SLO profile as an AIPerf ``--goodput`` constraint string.

    e.g. ``interactive`` → ``"time_to_first_token:300 inter_token_latency:50"``.
    Values are in milliseconds (AIPerf's display unit for these metrics), so they
    match our profile thresholds 1:1. Returns ``None`` for a constraint-free profile.
    """
    parts = []
    for field_name, tag in _AIPERF_GOODPUT_KEYS.items():
        v = getattr(profile, field_name)
        if v is not None:
            parts.append(f"{tag}:{v:g}")
    return " ".join(parts) if parts else None


def build_aiperf_command(
    endpoint: Endpoint,
    trace_file: str | Path,
    artifact_dir: str | Path,
    *,
    profile: Optional[SLOProfile] = None,
    block_size: int = 512,
    streaming: bool = True,
    tokenizer: Optional[str] = None,
    aiperf_bin: str = "aiperf",
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """Build an AIPerf replay with the trace's exact token-block layout."""
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("trace block_size must be a positive integer")
    block_size_flags = {
        "--isl-block-size",
        "--prompt-input-tokens-block-size",
        "--synthetic-input-tokens-block-size",
    }
    for argument in extra_args:
        if argument.split("=", 1)[0] in block_size_flags:
            raise ValueError("extra_args cannot override the trace block_size")
    cmd = [
        aiperf_bin,
        "profile",
        "--model",
        endpoint.model,
        "--url",
        endpoint.url,
        "--endpoint-type",
        endpoint.endpoint_type,
        "--input-file",
        str(trace_file),
        "--custom-dataset-type",
        "mooncake_trace",
        "--isl-block-size",
        str(block_size),
        "--artifact-dir",
        str(artifact_dir),
    ]
    if streaming:
        cmd.append("--streaming")
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    goodput = slo_to_aiperf_goodput(profile) if profile else None
    if goodput:
        cmd += ["--goodput", goodput]
    cmd += list(extra_args)
    return cmd


def _find_summary_json(artifact_dir: Path) -> Optional[Path]:
    """Locate AIPerf's summary export (``profile_export_aiperf.json``) under the
    artifact dir (AIPerf nests it under a run subdir)."""
    matches = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    return matches[-1] if matches else None


def parse_aiperf_summary(summary_path: Path) -> dict[str, Any]:
    """Extract goodput + latency stats from AIPerf's summary JSON.

    AIPerf's ``profile_export_aiperf.json`` keys each metric at the top level by
    tag, with a stats object ``{"unit": ..., "avg": ..., "p50": ..., "p99": ...}``
    (count-style metrics like ``goodput`` / ``good_request_count`` carry the value
    in ``avg``). Verified against a real run; missing fields degrade to ``None``.
    """
    data = json.loads(summary_path.read_text())

    def stat(tag: str, key: str = "avg") -> Optional[float]:
        m = data.get(tag)
        if isinstance(m, dict):
            v = m.get(key)
            return float(v) if isinstance(v, (int, float)) else None
        if isinstance(m, (int, float)):
            return float(m)
        return None

    return {
        # AIPerf's own goodput (requests/sec meeting the --goodput SLO) — the
        # authoritative figure; the same definition our offline scorecard uses.
        "goodput_rps": stat("goodput"),
        "good_request_count": stat("good_request_count"),
        "good_request_fraction": stat("good_request_fraction"),
        "request_throughput_rps": stat("request_throughput"),
        "request_count": stat("request_count"),
        "mean_ttft_ms": stat("time_to_first_token"),
        "p99_ttft_ms": stat("time_to_first_token", "p99"),
        "mean_itl_ms": stat("inter_token_latency"),
        "p99_itl_ms": stat("inter_token_latency", "p99"),
        "mean_e2e_ms": stat("request_latency"),
        "p99_e2e_ms": stat("request_latency", "p99"),
        "benchmark_duration_s": stat("benchmark_duration"),
        "_raw_summary_path": str(summary_path),
    }


@dataclass
class EndpointMatchResult:
    endpoint: str
    workload: str
    profile: str
    metrics: dict[str, Any]
    returncode: int
    artifact_dir: str
    stderr_tail: str = ""


def run_endpoint_match(
    endpoint: Endpoint,
    trace_file: str | Path,
    profile: SLOProfile,
    *,
    artifact_root: str | Path,
    block_size: int = 512,
    streaming: bool = True,
    tokenizer: Optional[str] = None,
    aiperf_bin: str = "aiperf",
    timeout_s: Optional[float] = None,
    extra_args: tuple[str, ...] = (),
    workload_name: Optional[str] = None,
) -> EndpointMatchResult:
    """Run one AIPerf benchmark of ``endpoint`` on ``trace_file`` scored against ``profile``."""
    workload = workload_name or Path(trace_file).stem
    artifact_dir = Path(artifact_root) / endpoint.name / f"{workload}_{profile.name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_aiperf_command(
        endpoint,
        trace_file,
        artifact_dir,
        profile=profile,
        block_size=block_size,
        streaming=streaming,
        tokenizer=tokenizer,
        aiperf_bin=aiperf_bin,
        extra_args=extra_args,
    )
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return EndpointMatchResult(
            endpoint=endpoint.name,
            workload=workload,
            profile=profile.name,
            metrics={"_timeout_s": timeout_s},
            returncode=124,
            artifact_dir=str(artifact_dir),
            stderr_tail="\n".join(
                [
                    f"AIPerf timed out after {timeout_s} seconds",
                    *stderr.splitlines()[-8:],
                ]
            ),
        )
    except FileNotFoundError:
        return EndpointMatchResult(
            endpoint=endpoint.name,
            workload=workload,
            profile=profile.name,
            metrics={},
            returncode=127,
            artifact_dir=str(artifact_dir),
            stderr_tail=f"AIPerf executable not found: {aiperf_bin}",
        )
    metrics: dict[str, Any] = {}
    summary = _find_summary_json(artifact_dir)
    if summary is not None:
        try:
            metrics = parse_aiperf_summary(summary)
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            metrics = {"_parse_error": repr(e)}
    return EndpointMatchResult(
        endpoint=endpoint.name,
        workload=workload,
        profile=profile.name,
        metrics=metrics,
        returncode=proc.returncode,
        artifact_dir=str(artifact_dir),
        stderr_tail="\n".join(proc.stderr.splitlines()[-8:]),
    )


def run_endpoint_leaderboard(
    *,
    endpoints: list[Endpoint],
    workload_traces: dict[str, str],
    workload_block_sizes: Optional[dict[str, int]] = None,
    profiles: tuple[SLOProfile, ...] = DEFAULT_PROFILES,
    artifact_root: str | Path,
    aiperf_bin: str = "aiperf",
    tokenizer: Optional[str] = None,
    timeout_s: Optional[float] = None,
    on_progress=None,
) -> list[EndpointMatchResult]:
    """Sweep endpoints × workloads × SLO profiles; return one result per match.

    ``workload_traces`` maps a workload name to a materialized Mooncake trace.
    ``workload_block_sizes`` supplies the token span per hash for every workload;
    omission preserves the Mooncake default of 512 tokens.
    """
    if workload_block_sizes is not None:
        missing = workload_traces.keys() - workload_block_sizes.keys()
        if missing:
            raise ValueError(
                "missing workload block sizes: " + ", ".join(sorted(missing))
            )
    results: list[EndpointMatchResult] = []
    for ep in endpoints:
        for wl_name, trace in workload_traces.items():
            for profile in profiles:
                if on_progress:
                    on_progress(ep.name, wl_name, profile.name)
                results.append(
                    run_endpoint_match(
                        ep,
                        trace,
                        profile,
                        artifact_root=artifact_root,
                        block_size=(
                            workload_block_sizes[wl_name]
                            if workload_block_sizes is not None
                            else 512
                        ),
                        aiperf_bin=aiperf_bin,
                        tokenizer=tokenizer,
                        timeout_s=timeout_s,
                        workload_name=wl_name,
                    )
                )
    return results


def format_endpoint_leaderboard(
    results: list[EndpointMatchResult], profile: str = "interactive"
) -> str:
    """Render endpoints ranked by goodput for one SLO profile."""
    rows = [r for r in results if r.profile == profile]
    workloads = sorted({r.workload for r in rows})
    lines = [
        f"\nOnline leaderboard — SLO profile: {profile}  (goodput req/s via AIPerf)\n"
    ]
    for wl in workloads:
        lines.append(f"── {wl} " + "─" * max(0, 40 - len(wl)))
        wl_rows = sorted(
            [r for r in rows if r.workload == wl],
            key=lambda r: (
                r.metrics.get("goodput_rps")
                if r.metrics.get("goodput_rps") is not None
                else -1.0
            ),
            reverse=True,
        )
        for r in wl_rows:
            m = r.metrics
            gp = m.get("goodput_rps")
            ttft = m.get("mean_ttft_ms")
            ok = "" if r.returncode == 0 else "  [FAILED rc=%d]" % r.returncode
            goodput_label = "n/a" if gp is None else f"{gp:.2f}"
            ttft_label = "n/a" if ttft is None else f"{ttft:.0f}"
            lines.append(
                f"  {r.endpoint:<16} goodput={goodput_label:>7}  "
                f"mean_ttft={ttft_label}{ok}"
            )
        lines.append("")
    return "\n".join(lines)
