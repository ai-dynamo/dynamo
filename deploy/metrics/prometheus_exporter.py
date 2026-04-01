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
Prometheus Pushgateway exporter for CI/CD metrics.

Pushes metrics to Prometheus via Pushgateway, matching the schema defined in the
migration plan. Designed to be called alongside the existing OpenSearch uploader
during the dual-write phase.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)

log = logging.getLogger(__name__)

# Histogram bucket definitions from the plan
WORKFLOW_DURATION_BUCKETS = (60, 120, 300, 600, 900, 1200, 1800, 3600, 7200)
WORKFLOW_QUEUE_BUCKETS = (10, 30, 60, 120, 300, 600, 900)
JOB_DURATION_BUCKETS = (30, 60, 120, 300, 600, 900, 1800, 3600)
JOB_QUEUE_BUCKETS = (5, 10, 30, 60, 120, 300, 600)
TEST_DURATION_BUCKETS = (0.1, 0.5, 1, 5, 10, 30, 60, 120, 300)
CONTAINER_BUILD_DURATION_BUCKETS = (60, 120, 300, 600, 900, 1200, 1800, 3600)
PR_MERGE_DURATION_BUCKETS = (1, 4, 8, 24, 48, 72, 168, 336)


class PrometheusExporter:
    """Exports CI/CD metrics to Prometheus via Pushgateway."""

    def __init__(self, pushgateway_url: str, job_name: str = "cicd_metrics", dry_run: bool = False):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.dry_run = dry_run
        self.registry = CollectorRegistry()

        self._counters: Dict[str, int] = defaultdict(int)

        # ── Workflow metrics ──
        self.workflow_total = Counter(
            "cicd_workflow_total",
            "Total workflow runs",
            ["repo", "workflow_name", "branch", "commit_sha", "status"],
            registry=self.registry,
        )
        self.workflow_duration = Histogram(
            "cicd_workflow_duration_seconds",
            "Workflow duration in seconds",
            ["repo", "workflow_name", "branch", "commit_sha", "status"],
            buckets=WORKFLOW_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.workflow_queue_time = Histogram(
            "cicd_workflow_queue_time_seconds",
            "Workflow queue time in seconds",
            ["repo", "workflow_name", "branch", "commit_sha"],
            buckets=WORKFLOW_QUEUE_BUCKETS,
            registry=self.registry,
        )

        # ── Job metrics ──
        self.job_total = Counter(
            "cicd_job_total",
            "Total job runs",
            ["repo", "workflow_name", "job_name", "branch", "commit_sha", "status", "runner_prefix"],
            registry=self.registry,
        )
        self.job_duration = Histogram(
            "cicd_job_duration_seconds",
            "Job duration in seconds",
            ["repo", "workflow_name", "job_name", "branch", "commit_sha", "runner_prefix"],
            buckets=JOB_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.job_queue_time = Histogram(
            "cicd_job_queue_time_seconds",
            "Job queue time in seconds",
            ["repo", "job_name", "branch", "commit_sha", "runner_prefix"],
            buckets=JOB_QUEUE_BUCKETS,
            registry=self.registry,
        )
        self.job_annotation_total = Gauge(
            "cicd_job_annotation_total",
            "Job annotation counts by severity",
            ["repo", "job_name", "commit_sha", "severity"],
            registry=self.registry,
        )

        # ── Test metrics ──
        self.test_total = Counter(
            "cicd_test_total",
            "Total test results",
            ["repo", "framework", "test_type", "arch", "cuda_version", "branch", "commit_sha", "status"],
            registry=self.registry,
        )
        self.test_status = Gauge(
            "cicd_test_status",
            "Per-test pass/fail status (1=pass, 0=fail)",
            ["repo", "framework", "test_type", "arch", "cuda_version", "branch", "commit_sha", "test_name", "job_name"],
            registry=self.registry,
        )
        self.test_duration = Histogram(
            "cicd_test_duration_seconds",
            "Test duration in seconds",
            ["repo", "framework", "test_type", "arch", "cuda_version", "branch", "commit_sha"],
            buckets=TEST_DURATION_BUCKETS,
            registry=self.registry,
        )

        # ── Container build metrics ──
        self.container_image_size = Gauge(
            "cicd_container_image_size_bytes",
            "Container image size in bytes",
            ["repo", "framework", "target", "platform", "cuda_version", "branch", "commit_sha"],
            registry=self.registry,
        )
        self.container_build_duration = Histogram(
            "cicd_container_build_duration_seconds",
            "Container build duration in seconds",
            ["repo", "framework", "target", "platform", "cuda_version", "branch", "commit_sha"],
            buckets=CONTAINER_BUILD_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.container_cache_hit_rate = Gauge(
            "cicd_container_cache_hit_rate",
            "Docker layer cache hit rate (0.0-1.0)",
            ["repo", "framework", "cuda_version", "branch", "commit_sha"],
            registry=self.registry,
        )
        self.container_sccache_hit_rate = Gauge(
            "cicd_container_sccache_hit_rate",
            "sccache compilation cache hit rate (0.0-1.0)",
            ["repo", "framework", "cuda_version", "branch", "commit_sha"],
            registry=self.registry,
        )
        self.container_steps_total = Gauge(
            "cicd_container_steps_total",
            "Container build step counts by type",
            ["repo", "framework", "cuda_version", "branch", "commit_sha", "type"],
            registry=self.registry,
        )
        self.container_stage_duration = Gauge(
            "cicd_container_stage_duration_seconds",
            "Container stage duration in seconds",
            ["repo", "framework", "stage_name", "cuda_version", "branch", "commit_sha"],
            registry=self.registry,
        )
        self.container_stage_cache_hit_rate = Gauge(
            "cicd_container_stage_cache_hit_rate",
            "Container stage cache hit rate",
            ["repo", "framework", "stage_name", "cuda_version", "branch", "commit_sha"],
            registry=self.registry,
        )

        # ── PR metrics ──
        self.pr_total = Counter(
            "cicd_pr_total",
            "Total pull requests",
            ["repo", "state", "is_external", "base_branch"],
            registry=self.registry,
        )
        self.pr_merge_duration = Histogram(
            "cicd_pr_merge_duration_hours",
            "PR time-to-merge in hours",
            ["repo", "base_branch", "is_external"],
            buckets=PR_MERGE_DURATION_BUCKETS,
            registry=self.registry,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def record_workflow(self, doc: Dict[str, Any]) -> None:
        """Record a workflow run from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        workflow_name = doc.get("s_workflow_name", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")
        status = doc.get("s_status", "unknown")
        duration = doc.get("l_duration_sec", 0)
        queue_time = doc.get("l_queue_time_sec", 0)

        labels = dict(repo=repo, workflow_name=workflow_name, branch=branch,
                       commit_sha=commit_sha, status=status)
        self.workflow_total.labels(**labels).inc()
        if duration > 0:
            self.workflow_duration.labels(**labels).observe(duration)
        if queue_time > 0:
            queue_labels = dict(repo=repo, workflow_name=workflow_name,
                                branch=branch, commit_sha=commit_sha)
            self.workflow_queue_time.labels(**queue_labels).observe(queue_time)

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

        self.job_total.labels(
            repo=repo, workflow_name=workflow_name, job_name=job_name,
            branch=branch, commit_sha=commit_sha, status=status,
            runner_prefix=runner_prefix,
        ).inc()
        if duration > 0:
            self.job_duration.labels(
                repo=repo, workflow_name=workflow_name, job_name=job_name,
                branch=branch, commit_sha=commit_sha, runner_prefix=runner_prefix,
            ).observe(duration)
        if queue_time > 0:
            self.job_queue_time.labels(
                repo=repo, job_name=job_name, branch=branch,
                commit_sha=commit_sha, runner_prefix=runner_prefix,
            ).observe(queue_time)

        # Annotations
        for severity, field in [
            ("failure", "l_annotation_failure_count"),
            ("warning", "l_annotation_warning_count"),
            ("notice", "l_annotation_notice_count"),
        ]:
            count = doc.get(field, 0)
            if count > 0:
                self.job_annotation_total.labels(
                    repo=repo, job_name=job_name, commit_sha=commit_sha,
                    severity=severity,
                ).set(count)

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

        # Aggregate counter
        self.test_total.labels(
            repo=repo, framework=framework, test_type=test_type, arch=arch,
            cuda_version=cuda_version, branch=branch, commit_sha=commit_sha,
            status=status,
        ).inc()

        # Per-test status gauge (only for pass/fail, skip skipped)
        if status in ("passed", "failed", "error"):
            self.test_status.labels(
                repo=repo, framework=framework, test_type=test_type, arch=arch,
                cuda_version=cuda_version, branch=branch, commit_sha=commit_sha,
                test_name=test_name, job_name=job_name,
            ).set(1 if status == "passed" else 0)

        # Duration histogram (convert ms to seconds)
        if duration_ms > 0:
            self.test_duration.labels(
                repo=repo, framework=framework, test_type=test_type, arch=arch,
                cuda_version=cuda_version, branch=branch, commit_sha=commit_sha,
            ).observe(duration_ms / 1000.0)

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

        # Image size
        size_bytes = doc.get("l_build_size_bytes", 0)
        if size_bytes > 0:
            self.container_image_size.labels(
                repo=repo, framework=framework, target=target, platform=platform,
                cuda_version=cuda_version, branch=branch, commit_sha=commit_sha,
            ).set(size_bytes)

        # Build duration
        duration = doc.get("l_build_duration_sec", 0)
        if duration > 0:
            self.container_build_duration.labels(
                repo=repo, framework=framework, target=target, platform=platform,
                cuda_version=cuda_version, branch=branch, commit_sha=commit_sha,
            ).observe(duration)

        # Cache hit rate
        cache_hit_rate = doc.get("f_cache_hit_rate", -1)
        if cache_hit_rate >= 0:
            self.container_cache_hit_rate.labels(
                repo=repo, framework=framework, cuda_version=cuda_version,
                branch=branch, commit_sha=commit_sha,
            ).set(cache_hit_rate)

        # Step counts
        for type_name, field in [
            ("total", "l_total_steps"),
            ("cached", "l_cached_steps"),
            ("built", "l_built_steps"),
        ]:
            val = doc.get(field, 0)
            if val > 0:
                self.container_steps_total.labels(
                    repo=repo, framework=framework, cuda_version=cuda_version,
                    branch=branch, commit_sha=commit_sha, type=type_name,
                ).set(val)

        self._counters["containers"] += 1

    def record_stage(self, doc: Dict[str, Any]) -> None:
        """Record container stage metrics from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        framework = doc.get("s_build_framework", "unknown")
        stage_name = doc.get("s_stage_name", "unknown")
        cuda_version = doc.get("s_cuda_version", "")
        branch = doc.get("s_branch", "")
        commit_sha = doc.get("s_commit_sha", "")

        labels = dict(repo=repo, framework=framework, stage_name=stage_name,
                       cuda_version=cuda_version, branch=branch, commit_sha=commit_sha)

        duration = doc.get("f_stage_duration_sec", 0)
        if duration > 0:
            self.container_stage_duration.labels(**labels).set(duration)

        cache_rate = doc.get("f_stage_cache_hit_rate", -1)
        if cache_rate >= 0:
            self.container_stage_cache_hit_rate.labels(**labels).set(cache_rate)

        self._counters["stages"] += 1

    def record_pr(self, doc: Dict[str, Any]) -> None:
        """Record PR metrics from an OpenSearch-shaped document."""
        repo = doc.get("s_repo", "")
        state = doc.get("s_state", "unknown")
        is_external = str(doc.get("b_is_external", False)).lower()
        base_branch = doc.get("s_base_branch", "")

        self.pr_total.labels(
            repo=repo, state=state, is_external=is_external, base_branch=base_branch,
        ).inc()

        # Time-to-merge histogram
        ttm = doc.get("l_time_to_merge_hours")
        if ttm is not None and ttm > 0:
            self.pr_merge_duration.labels(
                repo=repo, base_branch=base_branch, is_external=is_external,
            ).observe(ttm)

        self._counters["prs"] += 1

    # ── Push to gateway ───────────────────────────────────────────────────────

    def push(self, grouping_key: Optional[Dict[str, str]] = None) -> None:
        """Push all collected metrics to the Pushgateway.

        Args:
            grouping_key: Optional dict of additional grouping labels. When pushing
                from CI, pass ``{"run_id": ..., "job_name": ...}`` to avoid
                collisions between concurrent jobs.
        """
        if self.dry_run:
            log.info("[DRY RUN] Would push to Pushgateway at %s", self.pushgateway_url)
            self._log_summary()
            return

        if not self.pushgateway_url:
            log.warning("PUSHGATEWAY_URL not configured, skipping Prometheus push")
            return

        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                grouping_key=grouping_key or {},
                registry=self.registry,
            )
            log.info("Pushed metrics to Pushgateway at %s", self.pushgateway_url)
            self._log_summary()
        except Exception:
            log.exception("Failed to push metrics to Pushgateway")

    def _log_summary(self) -> None:
        """Log a summary of recorded metrics."""
        parts = [f"{k}={v}" for k, v in sorted(self._counters.items()) if v > 0]
        if parts:
            log.info("Prometheus metrics recorded: %s", ", ".join(parts))

    def get_summary(self) -> Dict[str, int]:
        """Return counts of recorded metrics."""
        return dict(self._counters)
