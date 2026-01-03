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
- Fetching pod events
- Listing pods in namespaces
- Logging K8s event summaries
"""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def get_pod_restart_count(deployment, pod_name: str, namespace: str) -> dict:
    """Get container restart counts for a pod.

    Args:
        deployment: ManagedDeployment instance
        pod_name: Name of the pod
        namespace: Kubernetes namespace

    Returns:
        Dict with container names as keys and restart counts as values
        Example: {"main": 2, "sidecar": 0}
    """
    try:
        cmd = [
            "kubectl",
            "get",
            "pod",
            pod_name,
            "-n",
            namespace,
            "-o",
            "json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            pod_data = json.loads(result.stdout)
            restart_counts = {}

            # Get restart counts from container statuses
            container_statuses = pod_data.get("status", {}).get("containerStatuses", [])
            for container in container_statuses:
                container_name = container.get("name", "unknown")
                restart_count = container.get("restartCount", 0)
                restart_counts[container_name] = restart_count

                # Also log if container recently restarted
                state = container.get("state", {})
                if "running" in state and restart_count > 0:
                    started_at = state["running"].get("startedAt", "unknown")
                    logger.info(
                        f"Container {container_name} restarted {restart_count} times, "
                        f"last started at {started_at}"
                    )

            return restart_counts
    except Exception as e:
        logger.debug(f"Could not get pod restart count: {e}")

    return {}


def get_k8s_events_for_pod(deployment, pod_name: str, namespace: str) -> list:
    """Get Kubernetes events for a specific pod using kubectl.

    Args:
        deployment: ManagedDeployment instance
        pod_name: Name of the pod (can be partial match)
        namespace: Kubernetes namespace

    Returns:
        List of event dictionaries with keys: type, reason, message, timestamp
    """
    try:
        # Get events for the pod using kubectl
        cmd = [
            "kubectl",
            "get",
            "events",
            "-n",
            namespace,
            "--field-selector",
            f"involvedObject.name={pod_name}",
            "-o",
            "json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            events_data = json.loads(result.stdout)
            events = []
            for item in events_data.get("items", []):
                events.append(
                    {
                        "type": item.get("type", ""),
                        "reason": item.get("reason", ""),
                        "message": item.get("message", ""),
                        "timestamp": item.get(
                            "lastTimestamp", item.get("eventTime", "")
                        ),
                        "count": item.get("count", 1),
                    }
                )
            return events
    except Exception as e:
        logger.debug(f"Could not get K8s events: {e}")

    return []


def check_container_restart_events(deployment, pod_name: str, namespace: str) -> bool:
    """Check if there are container restart/crash events for a pod.

    This looks for events like:
    - BackOff, CrashLoopBackOff: Container keeps crashing
    - Killing: Container was terminated
    - Started: Container was restarted

    Args:
        deployment: ManagedDeployment instance
        pod_name: Name of the pod
        namespace: Kubernetes namespace

    Returns:
        True if restart/crash events found, False otherwise
    """
    events = get_k8s_events_for_pod(deployment, pod_name, namespace)

    restart_related_reasons = [
        "BackOff",
        "CrashLoopBackOff",
        "Killing",
        "Started",
        "Unhealthy",
        "FailedMount",
    ]

    found_restart = False
    for event in events:
        if event["reason"] in restart_related_reasons:
            logger.info(
                f"Container event detected: [{event['type']}] {event['reason']} - "
                f"{event['message']} (count: {event.get('count', 1)})"
            )
            found_restart = True

    return found_restart
