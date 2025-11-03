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
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            events_data = json.loads(result.stdout)
            events = []
            for item in events_data.get("items", []):
                events.append({
                    "type": item.get("type", ""),
                    "reason": item.get("reason", ""),
                    "message": item.get("message", ""),
                    "timestamp": item.get("lastTimestamp", item.get("eventTime", "")),
                    "count": item.get("count", 1),
                })
            return events
    except Exception as e:
        logger.debug(f"Could not get K8s events: {e}")
    
    return []

