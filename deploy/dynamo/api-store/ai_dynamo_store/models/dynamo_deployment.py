from typing import Dict, Optional

from pydantic import BaseModel, Field


class TypeMeta(BaseModel):
    api_version: str = Field(..., alias="apiVersion")
    kind: str = Field(...)


class ObjectMeta(BaseModel):
    name: str = Field(...)
    namespace: str = Field(...)
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)


class Resources(BaseModel):
    cpu: Optional[str] = Field(None)
    memory: Optional[str] = Field(None)
    gpu: Optional[str] = Field(None)
    custom: Optional[Dict[str, str]] = Field(None)


class ResourceRequirements(BaseModel):
    requests: Optional[Resources] = Field(None)
    limits: Optional[Resources] = Field(None)


class Autoscaling(BaseModel):
    min_replicas: Optional[int] = Field(None, alias="minReplicas")
    max_replicas: Optional[int] = Field(None, alias="maxReplicas")


class IngressSpec(BaseModel):
    enabled: Optional[bool] = Field(None)
    use_virtual_service: Optional[bool] = Field(None, alias="useVirtualService")
    host_prefix: Optional[str] = Field(None, alias="hostPrefix")
    tls: Optional[Dict[str, str]] = Field(None)


class DynamoNimDeploymentSpec(BaseModel):
    dynamo_nim: str = Field(..., alias="dynamoNim")
    dynamo_tag: str = Field(..., alias="dynamoTag")
    service_name: Optional[str] = Field(None, alias="serviceName")
    resources: Optional[ResourceRequirements] = Field(None)
    autoscaling: Optional[Autoscaling] = Field(None)
    ingress: Optional[IngressSpec] = Field(None)
    replicas: Optional[int] = Field(None)


class DynamoNimDeployment(BaseModel):
    type_meta: TypeMeta = Field(..., alias="typeMeta")
    object_meta: ObjectMeta = Field(..., alias="objectMeta")
    spec: DynamoNimDeploymentSpec = Field(...)


class DynamoDeploymentSpec(BaseModel):
    dynamo_nim: str = Field(..., alias="dynamoNim")
    services: Dict[str, DynamoNimDeployment] = Field(default_factory=dict)


class DynamoDeployment(BaseModel):
    type_meta: TypeMeta = Field(..., alias="typeMeta")
    object_meta: ObjectMeta = Field(..., alias="objectMeta")
    spec: DynamoDeploymentSpec = Field(...)
