# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
import typing as t
from datetime import datetime

import requests

from dynamo.sdk.core.protocol.deployment import (
    Deployment,
    DeploymentManager,
    DeploymentStatus,
    Service,
)


class KubernetesDeploymentManager(DeploymentManager):
    """
    Implementation of DeploymentManager that talks to the dynamo_store deployment API.
    Accepts **kwargs for backend-specific options.
    Handles error reporting and payload construction according to the API schema.
    Raises exceptions for errors; CLI handles user interaction.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self.session = requests.Session()
        self.namespace = "default"

    def _upload_service(self, service: Service, **kwargs) -> None:
        """Upload a service to the API store."""
        session = self.session
        endpoint = self.endpoint

        if ":" not in service.name or len((parts := service.name.split(":"))) != 2:
            raise ValueError("Invalid service name format, expected 'name:version'")
        name, version = parts
        description = f"Auto-registered by Dynamo's KubernetesDeploymentManager for {name}:{version}"

        comp_url = f"{endpoint}/api/v1/dynamo_components"
        comp_payload = {
            "name": name,
            "description": description,
        }
        labels = getattr(service, "environment", None)
        if labels and isinstance(labels, dict):
            comp_payload["labels"] = labels
        resp = session.post(comp_url, json=comp_payload)
        if resp.status_code not in (200, 201, 409):
            raise RuntimeError(f"Failed to create component: {resp.text}")

        # Try to extract manifest info from service if available, else use defaults
        manifest = {
            "service": name,
            "apis": getattr(service, "apis", {}),
            "size_bytes": getattr(service, "size_bytes", 0),
        }

        ver_url = f"{endpoint}/api/v1/dynamo_components/{name}/versions"
        build_at = kwargs.get("build_at")
        if not build_at:
            build_at = datetime.utcnow()
        if isinstance(build_at, str):
            try:
                build_at = datetime.fromisoformat(build_at)
            except Exception:
                build_at = datetime.utcnow()
        ver_payload = {
            "description": description,
            "version": version,
            "manifest": manifest,
            "build_at": build_at.isoformat(),
        }
        if labels and isinstance(labels, dict):
            ver_payload["labels"] = [{"key": k, "value": v} for k, v in labels.items()]
        resp = session.post(ver_url, json=ver_payload)
        if resp.status_code not in (200, 201, 409):
            raise RuntimeError(f"Failed to create component version: {resp.text}")

    def create_deployment(self, deployment: Deployment, **kwargs) -> str:
        """Create a new deployment. Ensures all components and versions are registered/uploaded before creating the deployment."""
        # For each service/component in the deployment, upload it to the API store
        for service in deployment.services:
            self._upload_service(service)

        # Now create the deployment
        component = kwargs.get("pipeline") or deployment.namespace
        name = deployment.name
        dev = kwargs.get("dev", False)
        envs = kwargs.get("envs")
        labels = kwargs.get("labels")
        secrets = kwargs.get("secrets")
        services = kwargs.get("services", {})
        access_authorization = kwargs.get("access_authorization", False)
        payload = {
            "name": name,
            "component": component,
            "dev": dev,
            "envs": envs,
            "labels": labels,
            "secrets": secrets,
            "services": services,
            "access_authorization": access_authorization,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        url = f"{self.endpoint}/api/v2/deployments"
        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("name", name)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else None
            msg = e.response.text if e.response else str(e)
            raise RuntimeError((status, msg, url))

    def update_deployment(
        self, deployment_id: str, deployment: Deployment, **kwargs
    ) -> None:
        """Update an existing deployment."""
        component = kwargs.get("pipeline") or deployment.namespace
        dev = kwargs.get("dev", False)
        envs = kwargs.get("envs")
        labels = kwargs.get("labels")
        secrets = kwargs.get("secrets")
        services = kwargs.get("services", {})
        access_authorization = kwargs.get("access_authorization", False)
        payload = {
            "name": deployment.name,
            "component": component,
            "dev": dev,
            "envs": envs,
            "labels": labels,
            "secrets": secrets,
            "services": services,
            "access_authorization": access_authorization,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        url = f"{self.endpoint}/api/v2/deployments/{deployment_id}"
        try:
            resp = self.session.put(url, json=payload)
            resp.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else None
            msg = e.response.text if e.response else str(e)
            raise RuntimeError((status, msg, url))

    def get_deployment(self, deployment_id: str, **kwargs) -> dict[str, t.Any]:
        """Get deployment details."""
        url = f"{self.endpoint}/api/v2/deployments/{deployment_id}"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else None
            msg = e.response.text if e.response else str(e)
            raise RuntimeError((status, msg, url))

    def list_deployments(self, **kwargs) -> list[dict[str, t.Any]]:
        """List all deployments."""
        url = f"{self.endpoint}/api/v2/deployments"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data.get("items", [])
        except requests.HTTPError as e:
            msg = e.response.text if e.response else str(e)
            raise RuntimeError(
                (e.response.status_code if e.response else None, msg, url)
            )

    def delete_deployment(self, deployment_id: str, **kwargs) -> None:
        """Delete a deployment."""
        url = f"{self.endpoint}/api/v2/deployments/{deployment_id}"
        try:
            resp = self.session.delete(url)
            resp.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else None
            msg = e.response.text if e.response else str(e)
            raise RuntimeError((status, msg, url))

    def get_status(self, deployment_id: str, **kwargs) -> DeploymentStatus:
        dep = self.get_deployment(deployment_id)
        status = dep.get("status", "unknown")
        if status == "running":
            return DeploymentStatus.RUNNING
        elif status == "failed":
            return DeploymentStatus.FAILED
        elif status == "deploying":
            return DeploymentStatus.IN_PROGRESS
        elif status == "terminated":
            return DeploymentStatus.TERMINATED
        else:
            return DeploymentStatus.PENDING

    def wait_until_ready(
        self, deployment_id: str, timeout: int = 3600, **kwargs
    ) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(deployment_id)
            if status == DeploymentStatus.RUNNING:
                return True
            elif status == DeploymentStatus.FAILED:
                return False
            time.sleep(5)
        return False

    def get_endpoint_urls(self, deployment_id: str, **kwargs) -> list[str]:
        dep = self.get_deployment(deployment_id)
        return dep.get("urls", [])
