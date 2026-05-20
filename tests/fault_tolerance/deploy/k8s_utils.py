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

"""Kubernetes utility functions for fault tolerance testing.

This module provides utilities for interacting with Kubernetes:
- Fetching container restart counts for a pod
- Fetching Kubernetes events for a pod
- Detecting container restart/crash events

All k8s access goes through kr8s so the harness has no kubectl system-binary
dependency.
"""

import logging

import kr8s
from kr8s.objects import Pod

logger = logging.getLogger(__name__)


def get_pod_restart_count(deployment, pod_name: str, namespace: str) -> dict:
    """Get container restart counts for a pod.

    Args:
        deployment: ManagedDeployment instance (unused; kept for API parity)
        pod_name: Name of the pod
        namespace: Kubernetes namespace

    Returns:
        Dict with container names as keys and restart counts as values.
        Example: {"main": 2, "sidecar": 0}. Empty dict on any failure.
    """
    try:
        pod = Pod.get(pod_name, namespace=namespace)
        status = pod.raw.get("status", {}) or {}
        container_statuses = status.get("containerStatuses", []) or []

        restart_counts = {}
        for container in container_statuses:
            name = container.get("name", "unknown")
            count = container.get("restartCount", 0)
            restart_counts[name] = count

            state = container.get("state", {}) or {}
            if "running" in state and count > 0:
                started_at = state["running"].get("startedAt", "unknown")
                logger.info(
                    f"Container {name} restarted {count} times, "
                    f"last started at {started_at}"
                )

        return restart_counts
    except Exception as e:
        logger.debug(f"Could not get pod restart count for {pod_name}: {e}")
        return {}


def get_k8s_events_for_pod(deployment, pod_name: str, namespace: str) -> list:
    """Get Kubernetes events for a specific pod.

    Args:
        deployment: ManagedDeployment instance (unused; kept for API parity)
        pod_name: Name of the pod (must be an exact match — the API filters
            on ``involvedObject.name``)
        namespace: Kubernetes namespace

    Returns:
        List of event dictionaries with keys: type, reason, message,
        timestamp, count. Empty list on any failure.
    """
    try:
        api = kr8s.api()
        events = list(
            api.get(
                "events",
                namespace=namespace,
                field_selector=f"involvedObject.name={pod_name}",
            )
        )
        out = []
        for event in events:
            raw = event.raw
            out.append(
                {
                    "type": raw.get("type", ""),
                    "reason": raw.get("reason", ""),
                    "message": raw.get("message", ""),
                    "timestamp": raw.get("lastTimestamp", raw.get("eventTime", "")),
                    "count": raw.get("count", 1),
                }
            )
        return out
    except Exception as e:
        logger.debug(f"Could not get K8s events for {pod_name}: {e}")
        return []


def check_container_restart_events(deployment, pod_name: str, namespace: str) -> bool:
    """Check if there are container restart/crash events for a pod.

    This looks for events like:
    - BackOff, CrashLoopBackOff: Container keeps crashing
    - Killing: Container was terminated
    - Started: Container was restarted

    Args:
        deployment: ManagedDeployment instance (unused; kept for API parity)
        pod_name: Name of the pod
        namespace: Kubernetes namespace

    Returns:
        True if restart/crash events found, False otherwise.
    """
    events = get_k8s_events_for_pod(deployment, pod_name, namespace)

    restart_related_reasons = {
        "BackOff",
        "CrashLoopBackOff",
        "Killing",
        "Started",
        "Unhealthy",
        "FailedMount",
    }

    found_restart = False
    for event in events:
        if event["reason"] in restart_related_reasons:
            logger.info(
                f"Container event detected: [{event['type']}] {event['reason']} - "
                f"{event['message']} (count: {event.get('count', 1)})"
            )
            found_restart = True

    return found_restart
