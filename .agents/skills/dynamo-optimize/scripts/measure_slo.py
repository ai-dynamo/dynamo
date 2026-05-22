#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf 0.8.0 wrapper that emits PASS/FAIL per declared --goodput metric.

Used by dynamo-optimize/SKILL.md Phases 3.4 (baseline), 3.2 (workstation
pre-val), and 4.4 (post-deploy validation).

Contract:
  - Always installs aiperf==0.8.0 (locked decision 2026-05-22). DYN-2878-
    style transformers conflicts are surfaced, not silently downgraded.
  - Reads profile_export_aiperf.json from --artifact-dir after the run.
  - Direction (lower-vs-higher-is-better) inferred from a built-in table
    that mirrors AIPerf's MetricFlags.LARGER_IS_BETTER.
  - Stdout schema (matches dynamo-skill-author/references/body-shape.md
    §Output Schema):
        PASS|<metric>|<detail>
        FAIL|<metric>|<detail>
        DELTA|<metric>|<detail>
        INVALID|<metric>|<reason>
  - Exit non-zero on any FAIL or INVALID; zero only when all dimensions PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AIPERF_VERSION_DEFAULT = "0.8.0"

# Direction table for the metric tags `dynamo-optimize` supports. Mirrors
# AIPerf MetricFlags.LARGER_IS_BETTER. Adding a tag here also requires
# verifying the metric is registered in AIPerf's registry.
LOWER_IS_BETTER = {
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
    "inter_chunk_latency",
    "time_to_second_token",
    "time_to_first_output_token",
}
LARGER_IS_BETTER = {
    "output_token_throughput",
    "output_token_throughput_per_user",
    "total_token_throughput",
    "e2e_output_token_throughput",
    "request_throughput",
    "goodput",
}


def direction_for(tag: str) -> str:
    if tag in LARGER_IS_BETTER:
        return "larger_is_better"
    if tag in LOWER_IS_BETTER:
        return "lower_is_better"
    return "unknown"


def parse_slo(slo: str) -> List[Tuple[str, float]]:
    """Parse the AIPerf --goodput grammar: space-separated KEY:VALUE pairs."""
    pairs: List[Tuple[str, float]] = []
    for token in slo.split():
        if ":" not in token:
            raise ValueError(f"Invalid --goodput pair {token!r}: expected KEY:VALUE")
        key, value = token.split(":", 1)
        try:
            pairs.append((key, float(value)))
        except ValueError as exc:
            raise ValueError(f"Invalid --goodput value for {key!r}: {value!r}") from exc
    return pairs


def install_aiperf(version: str) -> bool:
    """Install aiperf at the locked version. Return True on success."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", f"aiperf=={version}"]
    print(f"[measure_slo] installing aiperf=={version}...", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def resolve_model(url: str) -> Optional[str]:
    """Fetch /v1/models and return the first model id, or None."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 — surfacing network errors as None
        print(f"[measure_slo] /v1/models fetch failed: {exc}", file=sys.stderr)
        return None
    data = payload.get("data") or []
    if not data:
        return None
    return data[0].get("id")


def run_aiperf(
    *,
    url: str,
    model: str,
    slo: str,
    artifact_dir: Path,
    duration: int,
    extra_args: List[str],
) -> bool:
    """Invoke `aiperf profile` with the declared SLO. Return True on success."""
    cmd = [
        sys.executable,
        "-m",
        "aiperf",
        "profile",
        "-m",
        model,
        "--url",
        url,
        "--streaming",
        "--benchmark-duration",
        str(duration),
        "--artifact-dir",
        str(artifact_dir),
        "--goodput",
        slo,
        *extra_args,
    ]
    print(
        f"[measure_slo] running: {' '.join(shlex.quote(c) for c in cmd)}",
        file=sys.stderr,
    )
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def load_profile_export(artifact_dir: Path) -> Optional[dict]:
    path = artifact_dir / "profile_export_aiperf.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def metric_avg_and_unit(export: dict, tag: str) -> Tuple[Optional[float], str]:
    """Return (avg, unit) for a metric tag from a JsonExportData dict."""
    entry = export.get(tag)
    if not isinstance(entry, dict):
        return None, ""
    avg = entry.get("avg")
    unit = entry.get("unit", "")
    return (None if avg is None else float(avg)), str(unit)


def evaluate_slo(measured: float, threshold: float, direction: str) -> bool:
    if direction == "larger_is_better":
        return measured >= threshold
    return measured <= threshold


def compute_delta_pct(post: float, base: float) -> str:
    if base == 0:
        return "NaN"
    return f"{((post - base) / base * 100):+.2f}"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_slo.py",
        description=(
            "AIPerf 0.8.0 wrapper for dynamo-optimize. Emits PASS/FAIL "
            "per declared --goodput metric and an optional delta vs baseline."
        ),
    )
    parser.add_argument(
        "--url", required=True, help="Frontend URL, e.g. http://localhost:8000"
    )
    parser.add_argument(
        "--slo",
        required=True,
        help='AIPerf --goodput string, e.g. "time_to_first_token:2000 inter_token_latency:25"',
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Where AIPerf writes profile_export_aiperf.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to a previously captured baseline (same AIPerf version)",
    )
    parser.add_argument(
        "--mode",
        choices=["preval", "baseline", "postdeploy"],
        default="postdeploy",
        help="Tags the run; default postdeploy",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="AIPerf benchmark-duration seconds; default 60",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id; defaults to first model from /v1/models",
    )
    parser.add_argument(
        "--aiperf-version",
        default=os.environ.get("AIPERF_VERSION", AIPERF_VERSION_DEFAULT),
        help=f"AIPerf version pin (default {AIPERF_VERSION_DEFAULT}; override NOT recommended)",
    )

    args = parser.parse_args(argv)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    # Parse the SLO eagerly so we fail fast on bad grammar.
    try:
        slo_pairs = parse_slo(args.slo)
    except ValueError as exc:
        print(f"FAIL|slo-parse|{exc}")
        return 1

    if not slo_pairs:
        print("FAIL|slo-parse|--slo produced no KEY:VALUE pairs")
        return 1

    # Pin AIPerf.
    if not install_aiperf(args.aiperf_version):
        print(f"FAIL|aiperf-install|version={args.aiperf_version} pip-install failed")
        return 1

    # Resolve model.
    model = args.model or resolve_model(args.url)
    if not model:
        print(f"FAIL|model-resolution|/v1/models returned no entries at {args.url}")
        return 1

    # Run AIPerf.
    extra_args = shlex.split(os.environ.get("AIPERF_EXTRA_ARGS", ""))
    if not run_aiperf(
        url=args.url,
        model=model,
        slo=args.slo,
        artifact_dir=args.artifact_dir,
        duration=args.duration,
        extra_args=extra_args,
    ):
        print("FAIL|aiperf-run|aiperf profile exited non-zero")
        return 1

    export = load_profile_export(args.artifact_dir)
    if export is None:
        print(
            f"FAIL|profile-export|missing {args.artifact_dir / 'profile_export_aiperf.json'}"
        )
        return 1

    # Per-dimension PASS/FAIL.
    results: List[str] = []
    measured_values: Dict[str, float] = {}
    pass_count = 0
    fail_count = 0

    for tag, threshold in slo_pairs:
        direction = direction_for(tag)
        measured, unit = metric_avg_and_unit(export, tag)

        if measured is None:
            results.append(
                f"FAIL|{tag}|measured=missing threshold={threshold} direction={direction}"
            )
            fail_count += 1
            continue

        measured_values[tag] = measured

        if direction == "unknown":
            results.append(
                f"FAIL|{tag}|direction=unknown (not in measure_slo.py registry); "
                "add to LARGER_IS_BETTER or LOWER_IS_BETTER before using"
            )
            fail_count += 1
            continue

        ok = evaluate_slo(measured, threshold, direction)
        verdict = "PASS" if ok else "FAIL"
        results.append(
            f"{verdict}|{tag}|measured={measured}{unit} "
            f"SLO={threshold}{unit} direction={direction}"
        )
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    # Optional delta vs baseline.
    delta_invalid = 0
    if args.baseline and args.baseline.exists():
        with args.baseline.open() as f:
            baseline = json.load(f)
        baseline_version = baseline.get("aiperf_version", "")
        if baseline_version and baseline_version != args.aiperf_version:
            for tag in measured_values:
                results.append(
                    f"INVALID|{tag}|reason=different_aiperf_versions "
                    f"baseline={baseline_version} current={args.aiperf_version}"
                )
                delta_invalid = 1
        else:
            for tag, measured in measured_values.items():
                base_entry = baseline.get(tag) or {}
                base_avg = base_entry.get("avg")
                if base_avg is None:
                    results.append(
                        f"DELTA|{tag}|baseline_value=missing post_value={measured}"
                    )
                    continue
                delta_pct = compute_delta_pct(measured, float(base_avg))
                results.append(
                    f"DELTA|{tag}|post_value={measured} "
                    f"baseline_value={base_avg} delta={delta_pct}%"
                )

    # Emit summary.
    print()
    print(f"===== measure_slo results (mode={args.mode}) =====")
    for row in results:
        print(row)
    print(f"Passed: {pass_count}  Failed: {fail_count}")

    if fail_count or delta_invalid:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
