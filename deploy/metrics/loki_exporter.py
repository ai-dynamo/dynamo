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
OpenTelemetry OTLP log exporter for CI/CD event logs.

Pushes structured log records to an OTLP HTTP endpoint (e.g.,
https://otlp-http.nvidia.com/v1/logs) routed to Loki.  Handles
high-cardinality data: steps, layers, reviews, commits, issues,
and full test/PR event details.

The public record_*() API accepts the same OpenSearch-shaped dicts
as the previous native-Loki implementation.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, Optional

from opentelemetry import _logs as logs_api
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

log = logging.getLogger(__name__)


class LokiExporter:
    """Exports CI/CD event logs via OpenTelemetry OTLP HTTP.

    Despite the class name (kept for backward compatibility), this now
    pushes log records through the OpenTelemetry SDK to any OTLP endpoint
    that routes to Loki (or any log backend).
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
        self._counters: Dict[str, int] = defaultdict(int)
        self._provider: Optional[LoggerProvider] = None
        self._otel_logger = None

        if not self.dry_run and self.otlp_endpoint:
            self._setup_provider(service_name)

    def _setup_provider(self, service_name: str) -> None:
        resource_attrs: Dict[str, str] = {"service.name": service_name}
        if self.otlp_token:
            resource_attrs["authorization"] = self.otlp_token

        resource = Resource.create(resource_attrs)

        headers: Dict[str, str] = {}
        if self.otlp_token:
            headers["Authorization"] = f"Bearer {self.otlp_token}"

        endpoint = self.otlp_endpoint
        if not endpoint.endswith("/v1/logs"):
            endpoint = endpoint.rstrip("/") + "/v1/logs"

        exporter = OTLPLogExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=30,
        )

        self._provider = LoggerProvider(resource=resource)
        self._provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        self._otel_logger = self._provider.get_logger("cicd-logs", "1.0.0")

    # ── Internal helper ───────────────────────────────────────────────────

    def _emit(self, body: str, attributes: Dict[str, Any]) -> None:
        """Emit a single structured log record."""
        if self.dry_run or not self._otel_logger:
            return
        # Clean attributes: OTel only accepts str/int/float/bool values
        clean_attrs = {k: v for k, v in attributes.items() if v is not None}
        self._otel_logger.emit(
            logs_api.LogRecord(
                body=body,
                severity_text="INFO",
                severity_number=logs_api.SeverityNumber.INFO,
                attributes=clean_attrs,
            )
        )

    # ── Public API (same signatures as before) ────────────────────────────

    def record_test_result(self, doc: Dict[str, Any]) -> None:
        """Record a test result event."""
        attrs = {
            "source": "cicd",
            "type": "test_result",
            "repo": doc.get("s_repo", ""),
            "branch": doc.get("s_branch", ""),
            "framework": doc.get("s_framework", "unknown"),
            "commit_sha": doc.get("s_commit_sha", ""),
            "test_name": doc.get("s_test_name", ""),
            "test_classname": doc.get("s_test_classname", ""),
            "status": doc.get("s_test_status", doc.get("s_status", "")),
            "duration_ms": doc.get("l_test_duration_ms", 0),
            "job_id": doc.get("s_job_id", ""),
            "job_name": doc.get("s_job_name", ""),
            "run_id": doc.get("s_run_id", ""),
            "workflow_name": doc.get("s_workflow_name", ""),
            "pr_id": doc.get("s_pr_id", ""),
            "arch": doc.get("s_arch", ""),
            "cuda_version": doc.get("s_cuda_version", ""),
            "test_type": doc.get("s_test_type", ""),
        }
        error_msg = doc.get("s_error_message", "")
        if error_msg:
            attrs["error_message"] = error_msg
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["test_results"] += 1

    def record_build_layer(self, doc: Dict[str, Any]) -> None:
        """Record a build layer event."""
        attrs = {
            "source": "cicd",
            "type": "build_layer",
            "repo": doc.get("s_repo", ""),
            "framework": doc.get("s_build_framework", "unknown"),
            "commit_sha": doc.get("s_commit_sha", ""),
            "stage_name": doc.get("s_stage", ""),
            "step_number": doc.get("l_step_number", 0),
            "step_name": doc.get("s_step_name", ""),
            "command": doc.get("s_command", ""),
            "cached": doc.get("b_cached", False),
            "duration_sec": doc.get("f_duration_sec", 0),
            "size_transferred": doc.get("l_size_transferred", 0),
            "job_id": doc.get("s_job_id", ""),
            "branch": doc.get("s_branch", ""),
            "cuda_version": doc.get("s_cuda_version", ""),
        }
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["build_layers"] += 1

    def record_workflow_step(self, doc: Dict[str, Any]) -> None:
        """Record a workflow step event."""
        attrs = {
            "source": "cicd",
            "type": "workflow_step",
            "repo": doc.get("s_repo", ""),
            "commit_sha": doc.get("s_commit_sha", ""),
            "step_name": doc.get("s_step_name", ""),
            "step_number": doc.get("l_step_number", 0),
            "command": doc.get("s_command", ""),
            "status": doc.get("s_status", ""),
            "duration_sec": doc.get("l_duration_sec", 0),
            "job_id": doc.get("s_job_id", ""),
            "job_name": doc.get("s_job_name", ""),
            "workflow_name": doc.get("s_workflow_name", ""),
            "branch": doc.get("s_branch", ""),
        }
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["workflow_steps"] += 1

    def record_pr_event(self, doc: Dict[str, Any]) -> None:
        """Record a PR event."""
        attrs = {
            "source": "cicd",
            "type": "pr_event",
            "repo": doc.get("s_repo", ""),
            "commit_sha": doc.get("s_commit_sha", ""),
            "pr_number": doc.get("l_pr_number", 0),
            "author": doc.get("s_author", ""),
            "author_association": doc.get("s_author_association", ""),
            "is_external": doc.get("b_is_external", False),
            "state": doc.get("s_state", ""),
            "merged_by": doc.get("s_merged_by", ""),
            "time_to_merge_hours": doc.get("l_time_to_merge_hours"),
            "base_branch": doc.get("s_base_branch", ""),
            "created_at": doc.get("ts_created_at", ""),
            "merged_at": doc.get("ts_merged_at", ""),
        }
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["pr_events"] += 1

    def record_review(self, doc: Dict[str, Any]) -> None:
        """Record a review event."""
        attrs = {
            "source": "cicd",
            "type": "review",
            "repo": doc.get("s_repo", ""),
            "reviewer": doc.get("s_reviewer", ""),
            "state": doc.get("s_state", ""),
        }
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["reviews"] += 1

    def record_commit(self, doc: Dict[str, Any]) -> None:
        """Record a commit event."""
        attrs = {
            "source": "cicd",
            "type": "commit",
            "repo": doc.get("s_repo", ""),
            "sha": doc.get("s_sha", ""),
            "author": doc.get("s_author", ""),
            "committer": doc.get("s_committer", ""),
            "branch": doc.get("s_branch", ""),
            "committed_at": doc.get("ts_committed_at", ""),
        }
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["commits"] += 1

    def record_issue(self, doc: Dict[str, Any]) -> None:
        """Record an issue event."""
        attrs = {
            "source": "cicd",
            "type": "issue",
            "repo": doc.get("s_repo", ""),
            "issue_number": doc.get("l_issue_number", 0),
            "author": doc.get("s_author", ""),
            "author_association": doc.get("s_author_association", ""),
            "is_external": doc.get("b_is_external", False),
            "state": doc.get("s_state", ""),
            "comments_count": doc.get("l_comments_count", 0),
            "labels": doc.get("s_labels", ""),
            "assignee": doc.get("s_assignee", ""),
            "closed_by": doc.get("s_closed_by", ""),
            "created_at": doc.get("ts_created_at", ""),
            "closed_at": doc.get("ts_closed_at", ""),
            "time_to_close_hours": doc.get("l_time_to_close_hours"),
        }
        body = json.dumps({k: v for k, v in attrs.items() if v}, default=str)
        self._emit(body, attrs)
        self._counters["issues"] += 1

    # ── Flush ─────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all pending log records to the OTLP endpoint."""
        if self.dry_run:
            log.info("[DRY RUN] Would flush logs to OTLP endpoint")
            return

        if not self._provider:
            log.warning("OTLP endpoint not configured, skipping log flush")
            return

        try:
            result = self._provider.force_flush(timeout_millis=30000)
            if result:
                log.info("Flushed logs to OTLP endpoint %s", self.otlp_endpoint)
            else:
                log.warning("OTLP log flush returned False")
        except Exception:
            log.exception("Failed to flush logs to OTLP endpoint")

    def shutdown(self) -> None:
        """Shutdown the logger provider."""
        if self._provider:
            try:
                self._provider.shutdown()
            except Exception:
                log.exception("Error shutting down log provider")

    def get_summary(self) -> Dict[str, int]:
        return dict(self._counters)
