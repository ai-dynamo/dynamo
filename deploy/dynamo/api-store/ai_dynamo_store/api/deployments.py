from fastapi import APIRouter, HTTPException
from kubernetes import client, config
from datetime import datetime
import os
import uuid
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

# ===== PYDANTIC MODELS FOR API SCHEMAS =====
class BaseSchema(BaseModel):
    uid: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

class ResourceSchema(BaseSchema):
    name: str
    resource_type: str
    labels: List[Dict[str, str]]

class UserSchema(BaseModel):
    name: str
    email: str
    first_name: str
    last_name: str


class ClusterSchema(ResourceSchema):
    description: str
    organization_name: str
    creator: UserSchema
    is_first: bool = False

class DeploymentConfigSchema(BaseModel):
    access_authorization: bool = False
    envs: Optional[List[Dict[str, str]]] = None
    labels: Optional[List[Dict[str, str]]] = None
    secrets: Optional[List[str]] = None
    services: Dict[str, Dict] = Field(default_factory=dict)

class UpdateDeploymentSchema(DeploymentConfigSchema):
    bento: str

class CreateDeploymentSchema(UpdateDeploymentSchema):
    name: Optional[str] = None
    dev: bool = False

class DeploymentSchema(ResourceSchema):
    status: str
    kube_namespace: str
    creator: UserSchema
    cluster: ClusterSchema
    latest_revision: Optional[Dict] = None
    manifest: Optional[Dict] = None

class DeploymentFullSchema(DeploymentSchema):
    urls: List[str] = Field(default_factory=list)
# ===== END PYDANTIC MODELS =====

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
        ownership = {
            "organization_id": "default-org",
            "user_id": "default-user"
        }

        # Get the k8s namespace from environment variable
        kube_namespace = os.getenv("DEFAULT_KUBE_NAMESPACE", "dynamo")

        # Create the custom resource directly as a dictionary
        custom_resource = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoDeployment",
            "metadata": {
                "name": deployment_name,
                "namespace": kube_namespace,
                "labels": {
                    "ngc-organization": ownership["organization_id"],
                    "ngc-user": ownership["user_id"]
                }
            },
            "spec": {
                "dynamoNim": deployment.bento,
                "services": {} # TODO: create a function to parse body into CRD compatible service definition
            }
        }
        
        # Initialize Kubernetes client
        try:
            # Try in-cluster config first (for running in k8s)
            config.load_incluster_config()
        except config.config_exception.ConfigException:
            # Fall back to kube config (for local development)
            print("Trying kube config")
            config.load_kube_config()
            
        api = client.CustomObjectsApi()

        print("Creating CRD in Kubernetes")
        print(custom_resource)
        
        # Create the CRD in Kubernetes
        created_crd = api.create_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=kube_namespace,
            plural="dynamodeployments",
            body=custom_resource
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
        
        # TODO: Replace with actual user info
        creator = UserSchema(
            name="default-user",
            email="default@example.com",
            first_name="Default",
            last_name="User"
        )
        
        # TODO: Replace with actual cluster info
        cluster = ClusterSchema(
            uid="default-cluster",
            name="default",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resource_type="cluster",
            labels=[],
            description="Default cluster",
            organization_name="default-org",
            creator=creator,
            is_first=True
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
        print("Error creating deployment:")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
