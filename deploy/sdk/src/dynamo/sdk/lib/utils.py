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

import io
import os
import tarfile
from datetime import datetime
from typing import Optional

import requests

from dynamo.sdk.core.protocol.deployment import Service

REQUEST_TIMEOUT = 20


def get_host_port():
    """Gets host and port from environment variables. Defaults to 0.0.0.0:8000."""
    port = int(os.environ.get("DYNAMO_PORT", 8000))
    host = os.environ.get("DYNAMO_HOST", "0.0.0.0")
    return host, port


def get_system_app_host_port():
    """Gets host and port for system app from environment variables. Defaults to choosing a random port."""
    port = int(os.environ.get("DYNAMO_SYSTEM_APP_PORT", 0))
    host = os.environ.get("DYNAMO_SYSTEM_APP_HOST", "0.0.0.0")
    return host, port


def upload_graph(
    endpoint: str,
    graph: str,
    entry_service: Service,
    session: Optional[requests.Session] = None,
    **kwargs,
) -> None:
    """Upload the entire graph as a single component/version, with a manifest of all services."""
    session = session or requests.Session()
    parts = graph.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"`graph` must be in '<name>:<version>' format, got '{graph}'."
        )
    graph_name, graph_version = parts

    # Check if component exists before POST
    comp_url = f"{endpoint}/api/v1/dynamo_components"
    comp_get_url = f"{endpoint}/api/v1/dynamo_components/{graph_name}"
    comp_exists = False
    comp_resp = session.get(comp_get_url, timeout=REQUEST_TIMEOUT)
    if comp_resp.status_code == 200:
        comp_exists = True
    elif comp_resp.status_code == 404:
        comp_exists = False
    else:
        raise RuntimeError(
            f"Failed to verify component '{graph_name}': "
            f"{comp_resp.status_code}: {comp_resp.text}"
        )
    if not comp_exists:
        comp_payload = {
            "name": graph_name,
            "description": "Registered by Dynamo's KubernetesDeploymentManager",
        }
        resp = session.post(comp_url, json=comp_payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code not in (200, 201, 409):
            raise RuntimeError(f"Failed to create component: {resp.text}")

    # Check if version exists before POST
    ver_url = f"{endpoint}/api/v1/dynamo_components/{graph_name}/versions"
    ver_get_url = (
        f"{endpoint}/api/v1/dynamo_components/{graph_name}/versions/{graph_version}"
    )
    ver_exists = False
    ver_resp = session.get(ver_get_url, timeout=REQUEST_TIMEOUT)
    if ver_resp.status_code == 200:
        ver_exists = True
    if not ver_exists:
        build_at = kwargs.get("build_at")
        if not build_at:
            build_at = datetime.utcnow()
        if isinstance(build_at, str):
            try:
                build_at = datetime.fromisoformat(build_at)
            except Exception:
                build_at = datetime.utcnow()
        manifest = {
            "service": entry_service.service_name,
            "apis": entry_service.apis,
            "size_bytes": entry_service.size_bytes,
        }
        ver_payload = {
            "name": entry_service.name,
            "description": f"Auto-registered version for {graph}",
            "resource_type": "dynamo_component_version",
            "version": graph_version,
            "manifest": manifest,
            "build_at": build_at.isoformat(),
        }
        resp = session.post(ver_url, json=ver_payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code not in (200, 201, 409):
            raise RuntimeError(f"Failed to create component version: {resp.text}")

    # Upload the graph
    build_dir = entry_service.path
    if not build_dir or not os.path.isdir(build_dir):
        raise FileNotFoundError(f"Built graph directory not found: {build_dir}")
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(build_dir, arcname=".")
    tar_stream.seek(0)
    upload_url = f"{endpoint}/api/v1/dynamo_components/{graph_name}/versions/{graph_version}/upload"
    upload_headers = {"Content-Type": "application/x-tar"}
    resp = session.put(
        upload_url,
        data=tar_stream,
        headers=upload_headers,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Failed to upload graph artifact: {resp.text}")
