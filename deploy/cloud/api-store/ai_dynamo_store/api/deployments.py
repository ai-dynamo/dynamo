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
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ..models.schemas import (
    CreateDeploymentSchema,
    DeploymentFullSchema,
    DeploymentListResponse,
    ResourceSchema,
    UpdateDeploymentSchema,
    create_default_cluster,
    create_default_user,
)
from .k8s import (
    create_dynamo_deployment,
    delete_dynamo_deployment,
    get_dynamo_deployment,
    get_namespace,
    list_dynamo_deployments,
    update_dynamo_deployment,
)
from .utils import build_latest_revision_from_cr, get_deployment_status, get_urls

router = APIRouter(prefix="/api/v2/deployments", tags=["deployments"])


def sanitize_deployment_name(name: Optional[str], dynamo_nim: str) -> str:
    """
    Resolve a name for the DynamoGraphDeployment that will work safely in k8s

    Args:
        name: Optional custom name
        dynamo_nim: Bento name and version (format: name:version)

    Returns:
        A unique deployment name that is at most 63 characters
    """
    if name:
        # If name is provided, truncate it to 63
        base_name = name[:63]
    else:
        # Generate base name from dynamoNim
        dynamo_nim_parts = dynamo_nim.split(":")
        if len(dynamo_nim_parts) != 2:
            raise ValueError("Invalid dynamoNim format, expected 'name:version'")
        base_name = f"dep-{dynamo_nim_parts[0]}-{dynamo_nim_parts[1]}"
        # Truncate to 63 chars
        base_name = base_name[:63]

    return base_name


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
        kube_namespace = get_namespace()

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
            envs=deployment.envs,
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
            status="deploying",
            kube_namespace=kube_namespace,
            creator=creator,
            cluster=cluster,
            latest_revision=build_latest_revision_from_cr(created_crd),
            manifest=None,
        )

        return deployment_schema

    except Exception as e:
        print("Error creating deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}", response_model=DeploymentFullSchema)
def get_deployment(name: str) -> DeploymentFullSchema:
    """
    Retrieve a deployment by name.

    Args:
        name: The name of the deployment to retrieve

    Returns:
        DeploymentFullSchema: The deployment details
    """
    try:
        kube_namespace = get_namespace()
        cr = get_dynamo_deployment(
            name=name,
            namespace=kube_namespace,
        )
        deployment_schema = DeploymentFullSchema(
            name=name,
            created_at=cr["metadata"]["creationTimestamp"],
            uid=cr["metadata"]["uid"],
            resource_type="deployment",
            labels=[],
            kube_namespace=kube_namespace,
            status=get_deployment_status(cr),
            urls=get_urls(cr),
            creator=create_default_user(),
            cluster=create_default_cluster(create_default_user()),
            latest_revision=build_latest_revision_from_cr(cr),
            manifest=None,
        )
        return deployment_schema
    except HTTPException as e:
        raise e
    except Exception as e:
        print("Error retrieving deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}", response_model=DeploymentFullSchema)
def delete_deployment(name: str) -> DeploymentFullSchema:
    """
    Delete a deployment by name.

    Args:
        name: The name of the deployment to delete

    Returns:
        DeploymentFullSchema: The deleted deployment details
    """
    try:
        kube_namespace = get_namespace()
        # Get deployment details before deletion
        cr = get_dynamo_deployment(name, kube_namespace)
        deployment_schema = DeploymentFullSchema(
            name=name,
            created_at=cr["metadata"]["creationTimestamp"],
            uid=cr["metadata"]["uid"],
            resource_type="deployment",
            labels=[],
            kube_namespace=kube_namespace,
            status=get_deployment_status(cr),
            urls=get_urls(cr),
            creator=create_default_user(),
            cluster=create_default_cluster(create_default_user()),
            latest_revision=build_latest_revision_from_cr(cr),
            manifest=None,
        )
        # Delete the deployment
        delete_dynamo_deployment(name, kube_namespace)
        return deployment_schema
    except HTTPException as e:
        raise e
    except Exception as e:
        print("Error deleting deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=DeploymentListResponse)
@router.get("", response_model=DeploymentListResponse)
def list_deployments(
    search: str = Query(default="", description="Search query"),
    dev: bool = Query(default=False, description="Filter development deployments"),
    q: str = Query(default="", description="Advanced query string"),
    all: bool = Query(default=False, description="Return all deployments"),
    count: str = Query(default="", description="Number of items to return"),
    start: str = Query(default="", description="Starting index"),
    cluster: str = Query(default="", description="Filter by cluster name"),
) -> Dict[str, Any]:
    """
    List all deployments with optional filtering.

    Args:
        search: Simple text search
        dev: Filter development deployments
        q: Advanced query string
        all: Whether to return all deployments
        count: Number of deployments to return
        start: Starting index for pagination
        cluster: Filter by cluster name

    Returns:
        Dict containing paginated deployment list
    """
    try:
        # Convert count and start to integers if they're not empty
        count_val = int(count) if count else None
        start_val = int(start) if start else None

        if count_val is not None and count_val <= 0:
            raise HTTPException(status_code=400, detail="Count must be greater than 0")
        if start_val is not None and start_val < 0:
            raise HTTPException(status_code=400, detail="Start must be non-negative")

        kube_namespace = get_namespace()
        crs = list_dynamo_deployments(
            namespace=kube_namespace,
            label_selector=q,
        )

        deployments = []
        for cr in crs:
            deployment_schema = DeploymentFullSchema(
                name=cr["metadata"]["name"],
                created_at=cr["metadata"]["creationTimestamp"],
                uid=cr["metadata"]["uid"],
                resource_type="deployment",
                labels=[],
                kube_namespace=kube_namespace,
                status=get_deployment_status(cr),
                urls=get_urls(cr),
                creator=create_default_user(),
                cluster=create_default_cluster(create_default_user()),
                latest_revision=build_latest_revision_from_cr(cr),
                manifest=None,
            )

            # Apply cluster filter if provided
            if cluster and cluster != deployment_schema.cluster.name:
                continue

            # Apply search filter if provided
            if search and search.lower() not in deployment_schema.name.lower():
                continue

            # Apply dev filter if enabled and all is not True
            if not all and dev and not deployment_schema.name.startswith("dev-"):
                continue

            deployments.append(deployment_schema)

        # Handle pagination
        total = len(deployments)
        start_idx = start_val if start_val is not None else 0
        if count_val is not None:
            deployments = deployments[start_idx : start_idx + count_val]
        else:
            deployments = deployments[start_idx:]

        return {
            "start": start_idx,
            "count": len(deployments),
            "total": total,
            "items": deployments,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("Error listing deployments:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{name}", response_model=DeploymentFullSchema)
def update_deployment(name: str, deployment: UpdateDeploymentSchema):
    """
    Update an existing deployment.

    Args:
        name: The name of the deployment to update (path param)
        deployment: The new deployment configuration (body)

    Returns:
        updated deployment details
    """
    try:
        ownership = {"organization_id": "default-org", "user_id": "default-user"}
        kube_namespace = get_namespace()
        existing_deployment = get_deployment(name)
        if existing_deployment.bento != deployment.bento:
            raise HTTPException(
                status_code=422,
                detail="Cannot update the Dynamo components of a deployment.",
            )

        deployment_name = sanitize_deployment_name(name, deployment.bento)
        updated_crd = update_dynamo_deployment(
            name=deployment_name,
            namespace=kube_namespace,
            dynamo_nim=deployment.bento,
            labels={
                "ngc-organization": ownership["organization_id"],
                "ngc-user": ownership["user_id"],
            },
            envs=deployment.envs,
        )
        resource = ResourceSchema(
            uid=updated_crd["metadata"]["uid"],
            name=updated_crd["metadata"]["name"],
            created_at=updated_crd["metadata"].get(
                "creationTimestamp", datetime.utcnow()
            ),
            updated_at=datetime.utcnow(),
            resource_type="deployment",
            labels=[],
        )
        creator = create_default_user()
        cluster = create_default_cluster(creator)
        deployment_schema = DeploymentFullSchema(
            **resource.dict(),
            status=get_deployment_status(updated_crd),
            kube_namespace=kube_namespace,
            creator=creator,
            cluster=cluster,
            latest_revision=build_latest_revision_from_cr(updated_crd),
            manifest=None,
            urls=get_urls(updated_crd),
        )
        return deployment_schema
    except Exception as e:
        print("Error updating deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
