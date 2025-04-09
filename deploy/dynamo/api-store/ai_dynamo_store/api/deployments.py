from fastapi import APIRouter, HTTPException
from datetime import datetime
import os
import uuid
from typing import Optional, Dict, List

from ..models.schemas import (
    CreateDeploymentSchema,
    DeploymentFullSchema,
    ResourceSchema,
    create_default_user,
    create_default_cluster
)
from .k8s import create_dynamo_deployment

router = APIRouter(prefix="/api/v2/deployments", tags=["deployments"])

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
        # Parse dynamoNim into name and version
        dynamo_nim_parts = deployment.bento.split(":")
        if len(dynamo_nim_parts) != 2:
            raise HTTPException(
                status_code=400,
                detail="Invalid dynamoNim format, expected 'name:version'"
            )
        
        dynamo_nim_name, dynamo_nim_version = dynamo_nim_parts
        
        # Generate deployment name if not provided
        deployment_name = deployment.name or f"dep-{dynamo_nim_name}-{dynamo_nim_version}--{uuid.uuid4().hex}"
        deployment_name = deployment_name[:63]  # Max label length for k8s
        
        # Get ownership info for labels
        ownership = {
            "organization_id": "default-org",
            "user_id": "default-user"
        }

        # Get the k8s namespace from environment variable
        kube_namespace = os.getenv("DEFAULT_KUBE_NAMESPACE", "dynamo")

        # Create the deployment using helper function
        created_crd = create_dynamo_deployment(
            name=deployment_name,
            namespace=kube_namespace,
            dynamo_nim=deployment.bento,
            labels={
                "ngc-organization": ownership["organization_id"],
                "ngc-user": ownership["user_id"]
            }
        )
        
        # Create response schema
        resource = ResourceSchema(
            uid=created_crd["metadata"]["uid"],
            name=deployment_name,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resource_type="deployment",
            labels=[]
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
            urls=[f"https://{deployment_name}.dynamo.example.com"]
        )
        
        return deployment_schema
        
    except Exception as e:
        print("Error creating deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
