#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from datetime import datetime
from typing import Any, Dict, List


def get_deployment_status(resource: Dict[str, Any]) -> str:
    """
    Get the current status of a deployment.
    Maps operator status to BentoML status values.
    Returns lowercase status values matching BentoML's DeploymentStatus enum.
    """
    status = resource.get("status", {})
    conditions = status.get("conditions", [])
    state = status.get("state", "")

    # First check Ready condition
    for condition in conditions:
        if condition.get("type") == "Ready":
            if condition.get("status") == "True":
                # If state is "successful", map to "running"
                if state == "successful":
                    return "running"
                return condition.get("message", "running").lower()
            elif condition.get("message"):
                return condition.get("message").lower()

    # If no Ready condition or not True, check state
    if state == "failed":
        return "failed"
    elif state == "pending":
        return "deploying"  # map pending to deploying to match BentoML states

    # Default fallback
    return "unknown"


def get_urls(resource: Dict[str, Any]) -> List[str]:
    """
    Get the URLs for a deployment.
    Returns URLs as soon as they are available from EndpointExposed condition.
    """
    urls = []
    conditions = resource.get("status", {}).get("conditions", [])

    # Check for EndpointExposed condition
    for condition in conditions:
        if (
            condition.get("type") == "EndpointExposed"
            and condition.get("status") == "True"
        ):
            if message := condition.get("message"):
                urls.append(message)
    return urls


def build_latest_revision_from_cr(cr: dict) -> dict:
    spec = cr.get("spec", {})
    meta = cr.get("metadata", {})
    now = datetime.utcnow().isoformat() + "Z"
    bento_str = spec.get("dynamoGraph", "unknown:unknown")
    if ":" in bento_str:
        bento_name, bento_version = bento_str.split(":", 1)
    else:
        bento_name, bento_version = "unknown", "unknown"
    # Dummy creator
    creator = {"name": "system", "email": "", "first_name": "", "last_name": ""}
    # Dummy repository
    repository = {
        "uid": "dummy-repo-uid",
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "name": bento_name,
        "resource_type": "bento_repository",
        "labels": [],
        "description": "",
        "latest_bento": None,
    }
    # Dummy bento
    bento = {
        "uid": "dummy-bento-uid",
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "name": bento_version,
        "resource_type": "bento",
        "labels": [],
        "description": "",
        "repository": repository,
        "version": bento_version,
        "image_build_status": "",
        "upload_status": "",
        "upload_finished_reason": "",
        "presigned_upload_url": "",
        "presigned_download_url": "",
    }
    # Target
    target = {
        "uid": "dummy-target-uid",
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "name": "default-target",
        "resource_type": "deployment_target",
        "labels": [],
        "creator": creator,
        "status": "running",
        "config": {
            "services": spec.get("services", {}),
            "access_authorization": True,
            "envs": spec.get("envs", []),
        },
        "bento": bento,
    }
    # Revision
    return {
        "uid": meta.get("uid", "dummy-uid"),
        "created_at": meta.get("creationTimestamp", now),
        "updated_at": meta.get("creationTimestamp", now),
        "deleted_at": None,
        "name": meta.get("name", "dummy-revision"),
        "resource_type": "deployment_revision",
        "labels": [],
        "creator": creator,
        "status": "running",
        "targets": [target],
    }
