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
OpenTelemetry OTLP metrics exporter for CI/CD metrics.

Pushes metrics to an OTLP HTTP endpoint (e.g., https://otlp-http.nvidia.com/v1/metrics)
using the OpenTelemetry SDK. The public API (record_workflow, record_job, record_test, etc.)
accepts the same OpenSearch-shaped dicts as the previous Pushgateway implementation.
"""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

log = logging.getLogger(__name__)


class PrometheusExporter:
    """Exports CI/CD metrics via OpenTelemetry OTLP HTTP.

    Despite the class name (kept for backward compatibility), this now uses
    the OpenTelemetry SDK to push metrics to any OTLP-compatible endpoint.
    """

    def __init__(
        self,
        otlp_endpoint: str = "",
        otlp_token: str = "",
        service_name: str = "dynamo-cicd-metrics",
        dry_run: bool = False,
    ):
        self.dry_run = dry_run
        self.otlp_endpoint = otlp_endpoint
        self.otlp_token = otlp_token
        self.service_name = service_name
        self._counters: Dict[str, int] = defaultdict(int)
        self._provider: Optional[MeterProvider] = None

        if not self.dry_run and self.otlp_endpoint:
            self._setup_provider()

        # Create meter (works in dry-run too — just won't export)
        self._meter = metrics.get_meter("cicd-metrics", "1.0.0")
        self._instruments: Dict[str, Any] = {}

    # ── Provider setup ────────────────────────────────────────────────────

    def _setup_provider(self) -> None:
        """Configure the OpenTelemetry MeterProvider with OTLP exporter."""
        resource_attrs = {
            "service.name": self.service_name,
            "service.version": "1.0.0",
        }
        if self.otlp_token:
            resource_attrs["authorization"] = self.otlp_token

        resource = Resource.create(resource_attrs)

        headers = {}
        if self.otlp_token:
            headers["Authorization"] = f"Bearer {self.otlp_token}"

        endpoint = self.otlp_endpoint
        if not endpoint.endswith("/v1/metrics"):
            endpoint = endpoint.rstrip("/") + "/v1/metrics"

        exporter = OTLPMetricExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=30,
        )

        reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=60000,
        )

        self._provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(self._provider)

    # ── Instrument helpers ────────────────────────────────────────────────

    def _counter(self, name: str, description: str = "", unit: str = "1"):
        key = f"counter_{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_counter(
                name=name, description=description, unit=unit,
            )
        return self._instruments[key]

    def _histogram(self, name: str, description: str = "", unit: str = "1"):
        key = f"histogram_{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_histogram(
                name=name, description=description, unit=unit,
            )
        return self._instruments[key]

    def _gauge(self, name: str, description: str = "", unit: str = "1"):
        """Up-down counter used as a settable gauge."""
        key = f"gauge_{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_gauge(
                name=name, description=description, unit=unit,
            )
        return self._instruments[key]

    # ── Public API (same signatures as before) ────────────────────────────

    def record_workflow(self, doc: Dict[str, Any]) -> None:
        """Record a workflow run from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        workflow_name = doc.get("s_workflow_name", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")
        status = doc.get("s_status", "unknown")
        duration = doc.get("l_duration_sec", 0)
        queue_time = doc.get("l_queue_time_sec", 0)

        attrs = dict(repo=repo, workflow_name=workflow_name, branch=branch,
                     commit_sha=commit_sha, status=status)

        self._counter("cicd_workflow_total", "Total workflow runs").add(1, attrs)
        if duration > 0:
            self._histogram("cicd_workflow_duration_seconds", "Workflow duration", "s").record(duration, attrs)
        if queue_time > 0:
            queue_attrs = dict(repo=repo, workflow_name=workflow_name, branch=branch, commit_sha=commit_sha)
            self._histogram("cicd_workflow_queue_time_seconds", "Workflow queue time", "s").record(queue_time, queue_attrs)

        self._counters["workflows"] += 1

    def record_job(self, doc: Dict[str, Any]) -> None:
        """Record a job run from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        workflow_name = doc.get("s_workflow_name", "")
        job_name = doc.get("s_job_name", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")
        status = doc.get("s_status", "unknown")
        runner_prefix = doc.get("s_runner_prefix", "unknown")
        duration = doc.get("l_duration_sec", 0)
        queue_time = doc.get("l_queue_time_sec", 0)

        attrs = dict(repo=repo, workflow_name=workflow_name, job_name=job_name,
                     branch=branch, commit_sha=commit_sha, status=status,
                     runner_prefix=runner_prefix)

        self._counter("cicd_job_total", "Total job runs").add(1, attrs)
        if duration > 0:
            dur_attrs = {k: v for k, v in attrs.items() if k != "status"}
            self._histogram("cicd_job_duration_seconds", "Job duration", "s").record(duration, dur_attrs)
        if queue_time > 0:
            q_attrs = dict(repo=repo, job_name=job_name, branch=branch,
                           commit_sha=commit_sha, runner_prefix=runner_prefix)
            self._histogram("cicd_job_queue_time_seconds", "Job queue time", "s").record(queue_time, q_attrs)

        for severity, field in [
            ("failure", "l_annotation_failure_count"),
            ("warning", "l_annotation_warning_count"),
            ("notice", "l_annotation_notice_count"),
        ]:
            count = doc.get(field, 0)
            if count > 0:
                self._gauge("cicd_job_annotation_total", "Job annotations").set(
                    count, dict(repo=repo, job_name=job_name, commit_sha=commit_sha, severity=severity),
                )

        self._counters["jobs"] += 1

    def record_test(self, doc: Dict[str, Any]) -> None:
        """Record a test result from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        framework = doc.get("s_framework", "unknown")
        test_type = doc.get("s_test_type", "unknown")
        arch = doc.get("s_arch", "unknown")
        cuda_version = doc.get("s_cuda_version", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")
        status = doc.get("s_test_status", doc.get("s_status", "unknown"))
        test_name = doc.get("s_test_name", "")
        job_name = doc.get("s_job_name", "")
        duration_ms = doc.get("l_test_duration_ms", 0)

        base_attrs = dict(repo=repo, framework=framework, test_type=test_type, arch=arch,
                          cuda_version=cuda_version, branch=branch, commit_sha=commit_sha)

        self._counter("cicd_test_total", "Total test results").add(1, {**base_attrs, "status": status})

        if status in ("passed", "failed", "error"):
            self._gauge("cicd_test_status", "Per-test pass/fail (1=pass, 0=fail)").set(
                1 if status == "passed" else 0,
                {**base_attrs, "test_name": test_name, "job_name": job_name},
            )

        if duration_ms > 0:
            self._histogram("cicd_test_duration_seconds", "Test duration", "s").record(
                duration_ms / 1000.0, base_attrs,
            )

        self._counters["tests"] += 1

    def record_container(self, doc: Dict[str, Any]) -> None:
        """Record container build metrics from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        framework = doc.get("s_build_framework", "unknown")
        target = doc.get("s_build_target", "unknown")
        platform = doc.get("s_build_platform", "unknown")
        cuda_version = doc.get("s_cuda_version", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")

        full_attrs = dict(repo=repo, framework=framework, target=target, platform=platform,
                          cuda_version=cuda_version, branch=branch, commit_sha=commit_sha)
        cache_attrs = dict(repo=repo, framework=framework, cuda_version=cuda_version,
                           branch=branch, commit_sha=commit_sha)

        size_bytes = doc.get("l_build_size_bytes", 0)
        if size_bytes > 0:
            self._gauge("cicd_container_image_size_bytes", "Container image size", "By").set(size_bytes, full_attrs)

        duration = doc.get("l_build_duration_sec", 0)
        if duration > 0:
            self._histogram("cicd_container_build_duration_seconds", "Build duration", "s").record(duration, full_attrs)

        cache_hit_rate = doc.get("f_cache_hit_rate", -1)
        if cache_hit_rate >= 0:
            self._gauge("cicd_container_cache_hit_rate", "Docker cache hit rate").set(cache_hit_rate, cache_attrs)

        for type_name, field in [("total", "l_total_steps"), ("cached", "l_cached_steps"), ("built", "l_built_steps")]:
            val = doc.get(field, 0)
            if val > 0:
                self._gauge("cicd_container_steps_total", "Build step counts").set(
                    val, {**cache_attrs, "type": type_name},
                )

        self._counters["containers"] += 1

    def record_stage(self, doc: Dict[str, Any]) -> None:
        """Record container stage metrics from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        framework = doc.get("s_build_framework", "unknown")
        stage_name = doc.get("s_stage_name", "unknown")
        cuda_version = doc.get("s_cuda_version", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")

        attrs = dict(repo=repo, framework=framework, stage_name=stage_name,
                     cuda_version=cuda_version, branch=branch, commit_sha=commit_sha)

        duration = doc.get("f_stage_duration_sec", 0)
        if duration > 0:
            self._gauge("cicd_container_stage_duration_seconds", "Stage duration", "s").set(duration, attrs)

        cache_rate = doc.get("f_stage_cache_hit_rate", -1)
        if cache_rate >= 0:
            self._gauge("cicd_container_stage_cache_hit_rate", "Stage cache rate").set(cache_rate, attrs)

        self._counters["stages"] += 1

    def record_pr(self, doc: Dict[str, Any]) -> None:
        """Record PR metrics from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        state = doc.get("s_state", "unknown")
        is_external = str(doc.get("b_is_external", False)).lower()
        base_branch = doc.get("s_base_branch", "")

        self._counter("cicd_pr_total", "Total pull requests").add(
            1, dict(repo=repo, state=state, is_external=is_external, base_branch=base_branch),
        )

        ttm = doc.get("l_time_to_merge_hours")
        if ttm is not None and ttm > 0:
            self._histogram("cicd_pr_merge_duration_hours", "PR time-to-merge", "h").record(
                ttm, dict(repo=repo, base_branch=base_branch, is_external=is_external),
            )

        self._counters["prs"] += 1

    # ── Flush / Push ──────────────────────────────────────────────────────

    def push(self, grouping_key: Optional[Dict[str, str]] = None) -> None:
        """Flush all pending metrics to the OTLP endpoint.

        The grouping_key parameter is accepted for backward compatibility
        but is not used with OTLP (each metric carries its own attributes).
        """
        if self.dry_run:
            log.info("[DRY RUN] Would push to OTLP endpoint %s", self.otlp_endpoint)
            self._log_summary()
            return

        if not self._provider:
            log.warning("OTLP endpoint not configured, skipping push")
            return

        try:
            success = self._provider.force_flush(timeout_millis=30000)
            if success:
                log.info("Flushed metrics to OTLP endpoint %s", self.otlp_endpoint)
            else:
                log.warning("OTLP flush returned False — some metrics may not have been exported")
            self._log_summary()
        except Exception:
            log.exception("Failed to flush metrics to OTLP endpoint")

    def shutdown(self) -> None:
        """Shutdown the meter provider and release resources."""
        if self._provider:
            try:
                self._provider.shutdown()
            except Exception:
                log.exception("Error shutting down meter provider")

    def _log_summary(self) -> None:
        parts = [f"{k}={v}" for k, v in sorted(self._counters.items()) if v > 0]
        if parts:
            log.info("Metrics recorded: %s", ", ".join(parts))

    def get_summary(self) -> Dict[str, int]:
        return dict(self._counters)
