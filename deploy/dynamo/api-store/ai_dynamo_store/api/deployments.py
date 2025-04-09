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

import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    CreateDeploymentSchema,
    DeploymentFullSchema,
    ResourceSchema,
    create_default_cluster,
    create_default_user,
)
from .k8s import create_dynamo_deployment

router = APIRouter(prefix="/api/v2/deployments", tags=["deployments"])


def sanitize_deployment_name(name: Optional[str], dynamo_nim: str) -> str:
    """
    Resolve a name for the DynamoDeployment that will work safely in k8s

    Args:
        name: Optional custom name
        dynamo_nim: Bento name and version (format: name:version)

    Returns:
        A unique deployment name that is at most 63 characters
    """
    if name:
        # If name is provided, truncate it to 55 chars to leave room for UUID
        base_name = name[:55]
    else:
        # Generate base name from dynamoNim
        dynamo_nim_parts = dynamo_nim.split(":")
        if len(dynamo_nim_parts) != 2:
            raise ValueError("Invalid dynamoNim format, expected 'name:version'")
        base_name = f"dep-{dynamo_nim_parts[0]}-{dynamo_nim_parts[1]}"
        # Truncate to 55 chars to leave room for UUID
        base_name = base_name[:55]

    # Add UUID and ensure total length is <= 63
    return f"{base_name}-{uuid.uuid4().hex[:7]}"


@router.post("", response_model=DeploymentFullSchema)
async def create_deployment(deployment: CreateDeploymentSchema):
    """
    Create a new deployment.

    Args:
        deployment: The deployment configuration following CreateDeploymentSchema

    Returns:
        DeploymentFullSchema: The created deployment details
    """
    try:
        # Get ownership info for labels
        ownership = {"organization_id": "default-org", "user_id": "default-user"}

        # Get the k8s namespace from environment variable
        kube_namespace = os.getenv("DEFAULT_KUBE_NAMESPACE", "dynamo")

        # Generate deployment name
        deployment_name = sanitize_deployment_name(deployment.name, deployment.bento)

        # Create the deployment using helper function
        created_crd = create_dynamo_deployment(
            name=deployment_name,
            namespace=kube_namespace,
            dynamo_nim=deployment.bento,
            labels={
                "ngc-organization": ownership["organization_id"],
                "ngc-user": ownership["user_id"],
            },
        )

        # Create response schema
        resource = ResourceSchema(
            uid=created_crd["metadata"]["uid"],
            name=created_crd["metadata"]["name"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resource_type="deployment",
            labels=[],
        )

        # Use helper functions for default resources
        creator = create_default_user()
        cluster = create_default_cluster(creator)

        deployment_schema = DeploymentFullSchema(
            **resource.dict(),
            status="running",
            kube_namespace=kube_namespace,
            creator=creator,
            cluster=cluster,
            latest_revision=None,
            manifest=None,
            urls=[f"https://{created_crd['metadata']['name']}.dynamo.example.com"],
        )

        return deployment_schema

    except Exception as e:
        print("Error creating deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
