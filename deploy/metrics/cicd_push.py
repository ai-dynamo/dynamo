#!/usr/bin/env python3
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
"""
CI/CD Metrics Push Client

Lightweight CLI that runs as a GitHub Actions step to push test results and
build metrics directly to Prometheus Pushgateway and Loki.  Replaces the
cron-based artifact-download pipeline for data that the CI job already has.

Usage:
    python cicd_push.py \
        --pushgateway-url http://pushgateway:9091 \
        --loki-url http://loki:3100 \
        --test-results ./test-results/ \
        --build-metrics ./build_metrics.json \
        --framework sglang \
        --test-type pre_merge \
        --cuda-version 12.8 \
        --job-status success \
        --job-started-at 2025-04-01T12:00:00Z
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from loki_exporter import LokiExporter
from opensearch_schema import (
    FIELD_ARCH,
    FIELD_BUILD_DURATION_SEC,
    FIELD_BUILD_FRAMEWORK,
    FIELD_BUILD_PLATFORM,
    FIELD_BUILD_SIZE_BYTES,
    FIELD_BUILD_TARGET,
    FIELD_BUILT_STEPS,
    FIELD_CACHE_HIT_RATE,
    FIELD_CACHED_STEPS,
    FIELD_ERROR_MESSAGE,
    FIELD_FRAMEWORK,
    FIELD_ID,
    FIELD_LAYER_CACHED,
    FIELD_LAYER_COMMAND,
    FIELD_LAYER_DURATION_SEC,
    FIELD_LAYER_SIZE_TRANSFERRED,
    FIELD_LAYER_STAGE,
    FIELD_LAYER_STATUS,
    FIELD_LAYER_STEP_NAME,
    FIELD_LAYER_STEP_NUMBER,
    FIELD_STAGE_CACHE_HIT_RATE,
    FIELD_STAGE_DURATION_SEC,
    FIELD_STAGE_NAME,
    FIELD_STATUS,
    FIELD_TEST_CLASSNAME,
    FIELD_TEST_DURATION,
    FIELD_TEST_NAME,
    FIELD_TEST_STATUS,
    FIELD_TEST_TYPE,
    FIELD_TOTAL_STEPS,
)
from parsers.build_metrics import parse_build_metrics_json
from parsers.github_context import GitHubActionsContext
from parsers.junit_xml import parse_junit_xml_directory
from prometheus_exporter import PrometheusExporter

# ── Document builders ─────────────────────────────────────────────────────
# These construct the same s_*/l_* prefixed dicts the exporters expect.


def build_job_document(
    ctx: GitHubActionsContext,
    args: argparse.Namespace,
    timestamp: str,
) -> Dict[str, Any]:
    """Build a job-level metrics document."""
    common = ctx.to_common_fields()
    status = args.job_status or "unknown"

    doc = {**common}
    doc[FIELD_ID] = f"ci-push-job-{ctx.run_id}-{ctx.job_name}"
    doc[FIELD_STATUS] = status
    doc["l_status_number"] = (
        1 if status == "success" else (0 if status == "failure" else None)
    )
    doc["s_runner_prefix"] = _extract_runner_prefix(ctx.runner_name)
    doc["@timestamp"] = timestamp

    # Duration from --job-started-at
    if args.job_started_at:
        try:
            started = datetime.fromisoformat(args.job_started_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            doc["l_duration_sec"] = max(0, int((now - started).total_seconds()))
        except Exception:
            doc["l_duration_sec"] = 0
    else:
        doc["l_duration_sec"] = 0

    doc["l_queue_time_sec"] = 0  # Not available without API call
    return doc


def build_test_document(
    test: Dict[str, Any],
    common: Dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> Dict[str, Any]:
    """Build an individual test result document."""
    test_name = test["name"]
    test_classname = test.get("classname", "")
    full_name = f"{test_classname}::{test_name}" if test_classname else test_name

    doc = {**common}
    doc[FIELD_ID] = f"ci-push-test-{common['s_job_id']}-{hash(full_name) & 0x7FFFFFFF}"
    doc[FIELD_TEST_NAME] = test_name
    doc[FIELD_TEST_CLASSNAME] = test_classname
    doc[FIELD_TEST_DURATION] = int(test["time"] * 1000)  # ms
    doc[FIELD_TEST_STATUS] = test["status"]
    doc[FIELD_STATUS] = test["status"]
    doc["l_test_status_number"] = (
        1
        if test["status"] == "passed"
        else (0 if test["status"] in ("failed", "error") else None)
    )
    doc[FIELD_FRAMEWORK] = args.framework or "unknown"
    doc[FIELD_TEST_TYPE] = args.test_type or "unknown"
    doc[FIELD_ARCH] = args.arch or common.get("s_arch", "unknown")
    doc["s_cuda_version"] = args.cuda_version or ""
    doc["@timestamp"] = timestamp

    error_msg = test.get("error_message", "")
    if error_msg:
        doc[FIELD_ERROR_MESSAGE] = error_msg[:1000]

    return doc


def build_container_document(
    metrics: Dict[str, Any],
    common: Dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> Dict[str, Any]:
    """Build container-level build metrics document."""
    container = metrics.get("container", {})
    framework = container.get("framework", args.framework or "unknown")

    doc = {**common}
    doc[FIELD_ID] = f"ci-push-container-{common['s_job_id']}-{framework}"
    doc[FIELD_STATUS] = args.job_status or "unknown"
    doc[FIELD_BUILD_FRAMEWORK] = framework
    doc[FIELD_BUILD_TARGET] = container.get("target", "unknown")
    doc[FIELD_BUILD_PLATFORM] = container.get("platform", "unknown")
    doc[FIELD_BUILD_SIZE_BYTES] = container.get("image_size_bytes", 0)
    doc[FIELD_TOTAL_STEPS] = container.get("total_steps", 0)
    doc[FIELD_CACHED_STEPS] = container.get("cached_steps", 0)
    doc[FIELD_BUILT_STEPS] = container.get("built_steps", 0)
    doc[FIELD_CACHE_HIT_RATE] = container.get("overall_cache_hit_rate", 0.0)
    doc[FIELD_BUILD_DURATION_SEC] = container.get("build_duration_sec", 0)
    doc["s_cuda_version"] = args.cuda_version or ""

    if "build_end_time" in container:
        doc["@timestamp"] = container["build_end_time"]
    else:
        doc["@timestamp"] = timestamp

    return doc


def build_stage_document(
    stage: Dict[str, Any],
    metrics: Dict[str, Any],
    common: Dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> Dict[str, Any]:
    """Build stage-level build metrics document."""
    container = metrics.get("container", {})
    framework = container.get("framework", args.framework or "unknown")
    stage_name = stage.get("stage_name", "unknown")

    doc = {**common}
    doc[FIELD_ID] = f"ci-push-stage-{common['s_job_id']}-{framework}-{stage_name}"
    doc[FIELD_BUILD_FRAMEWORK] = framework
    doc[FIELD_STAGE_NAME] = stage_name
    doc[FIELD_STAGE_DURATION_SEC] = stage.get("build_duration_sec", 0.0)
    doc[FIELD_STAGE_CACHE_HIT_RATE] = stage.get("cache_hit_rate", 0.0)
    doc["s_cuda_version"] = args.cuda_version or ""
    doc["@timestamp"] = container.get("build_end_time", timestamp)
    return doc


def build_layer_document(
    layer: Dict[str, Any],
    metrics: Dict[str, Any],
    common: Dict[str, Any],
    args: argparse.Namespace,
    timestamp: str,
) -> Dict[str, Any]:
    """Build layer-level build event document (for Loki)."""
    container = metrics.get("container", {})
    framework = container.get("framework", args.framework or "unknown")

    doc = {**common}
    doc[FIELD_BUILD_FRAMEWORK] = framework
    doc[FIELD_LAYER_STAGE] = layer.get("stage", "unknown")
    doc[FIELD_LAYER_STEP_NUMBER] = layer.get("step_number", 0)
    doc[FIELD_LAYER_STEP_NAME] = layer.get("step_name", "unknown")
    doc[FIELD_LAYER_COMMAND] = layer.get("command", "")
    doc[FIELD_LAYER_STATUS] = layer.get("status", "unknown")
    doc[FIELD_LAYER_CACHED] = layer.get("cached", False)
    doc[FIELD_LAYER_DURATION_SEC] = layer.get("duration_sec", 0.0)
    doc[FIELD_LAYER_SIZE_TRANSFERRED] = layer.get("size_transferred", 0)
    doc["s_cuda_version"] = args.cuda_version or ""
    doc["@timestamp"] = container.get("build_end_time", timestamp)
    return doc


def _extract_runner_prefix(runner_name: str) -> str:
    """Extract runner prefix (same logic as WorkflowMetricsProcessor)."""
    if not runner_name:
        return "unknown"
    import re

    if " " in runner_name:
        parts = runner_name.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]

    version_match = re.match(r"^(.*-v\d+)", runner_name)
    if version_match:
        return version_match.group(1)

    runner_suffix_match = re.match(
        r"^(.*?)-[a-z0-9]{4,8}-runner-[a-z0-9]+$", runner_name
    )
    if runner_suffix_match:
        return runner_suffix_match.group(1)

    return runner_name


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push CI/CD metrics directly from a GitHub Actions job",
    )

    # Targets
    parser.add_argument(
        "--otlp-endpoint", help="OTLP HTTP endpoint (e.g. https://otlp-http.nvidia.com)"
    )
    parser.add_argument(
        "--otlp-token",
        help="Bearer token for OTLP auth (or set OTLP_TOKEN env / use --otlp-token-file)",
    )
    parser.add_argument(
        "--otlp-token-file", help="File containing the OTLP bearer token"
    )

    # Data sources
    parser.add_argument(
        "--test-results", help="Path to directory containing JUnit XML files"
    )
    parser.add_argument("--build-metrics", help="Path to build_metrics.json file")

    # Metadata
    parser.add_argument("--framework", help="Framework name (e.g. sglang, vllm)")
    parser.add_argument("--test-type", help="Test type (e.g. pre_merge, nightly)")
    parser.add_argument("--arch", help="Architecture (default: from RUNNER_ARCH)")
    parser.add_argument("--cuda-version", help="CUDA version")

    # Job timing
    parser.add_argument(
        "--job-started-at", help="Job start time ISO 8601 (from github.run_started_at)"
    )
    parser.add_argument(
        "--job-status", help="Job conclusion: success/failure/cancelled"
    )

    # Behavior
    parser.add_argument(
        "--service-name", default="dynamo-cicd-metrics", help="OTLP service name"
    )
    parser.add_argument("--dry-run", action="store_true", help="Log without pushing")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve OTLP token
    otlp_token = (
        args.otlp_token
        or os.environ.get("OTLP_TOKEN")
        or os.environ.get("OTEL_AUTH_TOKEN")
        or ""
    )
    if not otlp_token and args.otlp_token_file:
        try:
            with open(args.otlp_token_file) as f:
                otlp_token = f.read().strip()
        except Exception as e:
            print(f"Warning: could not read token file {args.otlp_token_file}: {e}")

    otlp_endpoint = args.otlp_endpoint or os.environ.get("OTLP_ENDPOINT") or ""

    if not otlp_endpoint:
        print("No OTLP endpoint configured, skipping metrics push.")
        sys.exit(0)
    if not args.test_results and not args.build_metrics:
        print("No test-results or build-metrics provided, skipping.")
        sys.exit(0)

    # Read GitHub context from environment
    ctx = GitHubActionsContext.from_env()
    if not ctx.repo:
        print("Warning: GITHUB_REPOSITORY not set — running outside CI?")

    if args.arch:
        pass  # explicit override
    elif ctx.runner_arch:
        args.arch = ctx.runner_arch.lower()

    common = ctx.to_common_fields()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Initialize exporters — both metrics and logs go through OTLP
    prometheus = PrometheusExporter(
        otlp_endpoint=otlp_endpoint,
        otlp_token=otlp_token,
        service_name=args.service_name,
        dry_run=args.dry_run,
    )

    loki = LokiExporter(
        otlp_endpoint=otlp_endpoint,
        otlp_token=otlp_token,
        service_name=args.service_name,
        dry_run=args.dry_run,
    )

    # ── 1. Job-level metrics ──────────────────────────────────────────
    job_doc = build_job_document(ctx, args, timestamp)
    prometheus.record_job(job_doc)
    print(f"Recorded job metrics: {ctx.job_name} status={args.job_status}")

    # ── 2. Test results ───────────────────────────────────────────────
    if args.test_results:
        test_dir = Path(args.test_results)
        if not test_dir.exists():
            print(f"Warning: test-results path does not exist: {test_dir}")
        else:
            tests = parse_junit_xml_directory(test_dir)
            print(f"Parsed {len(tests)} test results from {test_dir}")

            for test in tests:
                doc = build_test_document(test, common, args, timestamp)
                prometheus.record_test(doc)
                loki.record_test_result(doc)

    # ── 3. Build metrics ──────────────────────────────────────────────
    if args.build_metrics:
        metrics_path = Path(args.build_metrics)
        if not metrics_path.exists():
            print(f"Warning: build-metrics path does not exist: {metrics_path}")
        else:
            metrics = parse_build_metrics_json(metrics_path)
            if metrics and "container" in metrics:
                container_doc = build_container_document(
                    metrics, common, args, timestamp
                )
                prometheus.record_container(container_doc)
                print(
                    f"Recorded container metrics: {container_doc.get(FIELD_BUILD_FRAMEWORK)}"
                )

                for stage in metrics.get("stages", []):
                    stage_doc = build_stage_document(
                        stage, metrics, common, args, timestamp
                    )
                    prometheus.record_stage(stage_doc)

                for layer in metrics.get("layers", []):
                    layer_doc = build_layer_document(
                        layer, metrics, common, args, timestamp
                    )
                    loki.record_build_layer(layer_doc)

                print(
                    f"Recorded {len(metrics.get('stages', []))} stages, {len(metrics.get('layers', []))} layers"
                )
            else:
                print("Warning: build metrics missing 'container' field")

    # ── 4. Push ───────────────────────────────────────────────────────
    prometheus.push()
    prometheus.shutdown()
    loki.flush()
    loki.shutdown()

    # Summary
    prom_summary = prometheus.get_summary()
    loki_summary = loki.get_summary()
    parts = [f"{k}={v}" for k, v in sorted(prom_summary.items()) if v > 0]
    print(f"OTLP metrics: {', '.join(parts) if parts else 'none'}")
    parts = [f"{k}={v}" for k, v in sorted(loki_summary.items()) if v > 0]
    print(f"OTLP logs: {', '.join(parts) if parts else 'none'}")

    print("Done.")


if __name__ == "__main__":
    main()
