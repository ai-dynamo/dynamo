from fastapi import APIRouter, HTTPException
from kubernetes import client, config
from kubernetes.client import V1ObjectMeta, V1TypeMeta
from datetime import datetime
import os
import uuid
from typing import Optional, Dict, List

from bentoml._internal.cloud.schemas.schemasv2 import (
    CreateDeploymentSchema,
    DeploymentFullSchema,
    ResourceSchema,
    UserSchema,
    ClusterSchema,
    DeploymentSchema
)

from ..models.dynamo_deployment import (
    TypeMeta,
    ObjectMeta,
    DynamoDeploymentSpec,
    DynamoDeployment
)

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
        # TODO: Implement proper ownership info retrieval
        ownership = {
            "organization_id": "default-org",  # Replace with actual org ID
            "user_id": "default-user"          # Replace with actual user ID
        }

        # Get the k8s namespace from environment variable
        kube_namespace = os.getenv("DEFAULT_KUBE_NAMESPACE", "dynamo")

        # Create DynamoDeployment using Pydantic models
        type_meta = TypeMeta(
            api_version="nvidia.com/v1alpha1",
            kind="DynamoDeployment"
        )
        
        object_meta = ObjectMeta(
            name=deployment_name,
            namespace=kube_namespace,
            labels={
                "ngc-organization": ownership["organization_id"],
                "ngc-user": ownership["user_id"]
            }
        )
        
        spec = DynamoDeploymentSpec(
            dynamo_nim=deployment.bento,
            services={}  # Empty services map as per requirement
        )
        
        dynamo_deployment = DynamoDeployment(
            type_meta=type_meta,
            object_meta=object_meta,
            spec=spec
        )
        
        # Initialize Kubernetes client
        config.load_kube_config()
        api = client.CustomObjectsApi()
        
        # Create the CRD in Kubernetes
        created_crd = api.create_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=kube_namespace,
            plural="dynamodeployments",
            body=dynamo_deployment.dict(by_alias=True)  # Use by_alias to maintain camelCase
        )
        
        # Create response schema
        resource = ResourceSchema(
            uid=created_crd["metadata"]["uid"],
            name=deployment_name,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # TODO: Replace with actual user info
        creator = UserSchema(
            uid=ownership["user_id"],
            name="default-user",
            email="default@example.com"
        )
        
        # TODO: Replace with actual cluster info
        cluster = ClusterSchema(
            uid="default-cluster",
            name="default",
            description="Default cluster"
        )
        
        deployment_schema = DeploymentSchema(
            **resource.dict(),
            status="running",
            kube_namespace=kube_namespace,
            creator=creator,
            cluster=cluster,
            latest_revision=None,
            manifest=None
        )
        
        full_schema = DeploymentFullSchema(
            **deployment_schema.dict(),
            urls=[f"https://{deployment_name}.dynamo.example.com"]
        )
        
        return full_schema
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 