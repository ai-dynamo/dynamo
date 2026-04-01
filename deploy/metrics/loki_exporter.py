# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Loki HTTP push exporter for CI/CD event logs.

Pushes structured JSON logs to Loki via the /loki/api/v1/push endpoint.
Handles high-cardinality data that doesn't belong in Prometheus:
steps, layers, reviews, commits, issues, and full test/PR event details.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# Maximum batch size before auto-flush
_MAX_BATCH_SIZE = 100


class LokiExporter:
    """Exports CI/CD event logs to Loki via HTTP push."""

    def __init__(self, loki_url: str, dry_run: bool = False):
        """
        Args:
            loki_url: Loki base URL (e.g., "http://loki:3100")
            dry_run: If True, log what would be sent without actually pushing
        """
        self.loki_url = loki_url.rstrip("/") if loki_url else ""
        self.push_url = f"{self.loki_url}/loki/api/v1/push" if self.loki_url else ""
        self.dry_run = dry_run

        # Batched entries keyed by stream label tuple
        self._streams: Dict[tuple, List[tuple]] = defaultdict(list)

        # Counters
        self._counters: Dict[str, int] = defaultdict(int)

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Public API ────────────────────────────────────────────────────────────

    def record_test_result(self, doc: Dict[str, Any]) -> None:
        """Record a test result event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "test_result",
            "repo": doc.get("s_repo", ""),
            "branch": doc.get("s_branch", ""),
            "framework": doc.get("s_framework", "unknown"),
        }
        body = {
            "commit_sha": doc.get("s_commit_sha", ""),
            "test_name": doc.get("s_test_name", ""),
            "test_classname": doc.get("s_test_classname", ""),
            "status": doc.get("s_test_status", doc.get("s_status", "")),
            "duration_ms": doc.get("l_test_duration_ms", 0),
            "error_message": doc.get("s_error_message", ""),
            "job_id": doc.get("s_job_id", ""),
            "job_name": doc.get("s_job_name", ""),
            "run_id": doc.get("s_run_id", ""),
            "workflow_name": doc.get("s_workflow_name", ""),
            "pr_id": doc.get("s_pr_id", ""),
            "arch": doc.get("s_arch", ""),
            "cuda_version": doc.get("s_cuda_version", ""),
            "test_type": doc.get("s_test_type", ""),
        }
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["test_results"] += 1

    def record_build_layer(self, doc: Dict[str, Any]) -> None:
        """Record a build layer event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "build_layer",
            "repo": doc.get("s_repo", ""),
            "framework": doc.get("s_build_framework", "unknown"),
        }
        body = {
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
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["build_layers"] += 1

    def record_workflow_step(self, doc: Dict[str, Any]) -> None:
        """Record a workflow step event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "workflow_step",
            "repo": doc.get("s_repo", ""),
        }
        body = {
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
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["workflow_steps"] += 1

    def record_pr_event(self, doc: Dict[str, Any]) -> None:
        """Record a PR event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "pr_event",
            "repo": doc.get("s_repo", ""),
        }
        body = {
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
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["pr_events"] += 1

    def record_review(self, doc: Dict[str, Any]) -> None:
        """Record a review event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "review",
            "repo": doc.get("s_repo", ""),
        }
        body = {
            "reviewer": doc.get("s_reviewer", ""),
            "state": doc.get("s_state", ""),
        }
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["reviews"] += 1

    def record_commit(self, doc: Dict[str, Any]) -> None:
        """Record a commit event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "commit",
            "repo": doc.get("s_repo", ""),
        }
        body = {
            "sha": doc.get("s_sha", ""),
            "author": doc.get("s_author", ""),
            "committer": doc.get("s_committer", ""),
            "branch": doc.get("s_branch", ""),
            "committed_at": doc.get("ts_committed_at", ""),
        }
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["commits"] += 1

    def record_issue(self, doc: Dict[str, Any]) -> None:
        """Record an issue event for Loki."""
        stream_labels = {
            "source": "cicd",
            "type": "issue",
            "repo": doc.get("s_repo", ""),
        }
        body = {
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
        self._add_entry(stream_labels, body, doc.get("@timestamp"))
        self._counters["issues"] += 1

    # ── Flush / Push ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush all buffered entries to Loki."""
        if not self._streams:
            return

        if self.dry_run:
            total = sum(len(entries) for entries in self._streams.values())
            log.info("[DRY RUN] Would push %d entries to Loki across %d streams",
                     total, len(self._streams))
            self._streams.clear()
            return

        if not self.push_url:
            log.warning("LOKI_URL not configured, skipping Loki push")
            self._streams.clear()
            return

        # Build Loki push payload
        streams = []
        for label_key, entries in self._streams.items():
            label_dict = dict(label_key)
            streams.append({
                "stream": label_dict,
                "values": entries,
            })

        payload = {"streams": streams}

        try:
            resp = self._session.post(self.push_url, json=payload, timeout=30)
            if resp.status_code in (200, 204):
                total = sum(len(s["values"]) for s in streams)
                log.info("Pushed %d entries to Loki across %d streams", total, len(streams))
            else:
                log.error("Loki push failed: HTTP %d — %s", resp.status_code, resp.text[:500])
        except Exception:
            log.exception("Failed to push entries to Loki")
        finally:
            self._streams.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add_entry(
        self,
        stream_labels: Dict[str, str],
        body: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> None:
        """Add an entry to the buffer, auto-flushing if the batch is large."""
        # Convert stream labels to a hashable key
        key = tuple(sorted(stream_labels.items()))

        # Loki expects [timestamp_ns, json_line]
        ts_ns = self._to_nanoseconds(timestamp)
        line = json.dumps(body, default=str)

        self._streams[key].append([str(ts_ns), line])

        # Auto-flush if any stream gets too large
        if len(self._streams[key]) >= _MAX_BATCH_SIZE:
            self.flush()

    @staticmethod
    def _to_nanoseconds(timestamp: Optional[str]) -> int:
        """Convert an ISO timestamp to nanoseconds since epoch, or use current time."""
        if not timestamp:
            return int(time.time() * 1e9)
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1e9)
        except Exception:
            return int(time.time() * 1e9)

    def get_summary(self) -> Dict[str, int]:
        """Return counts of recorded entries."""
        return dict(self._counters)
