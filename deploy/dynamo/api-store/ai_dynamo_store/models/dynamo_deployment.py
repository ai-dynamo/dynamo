from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TypeMeta(BaseModel):
    api_version: str = Field(..., alias="apiVersion")
    kind: str = Field(...)

class ObjectMeta(BaseModel):
    name: str = Field(...)
    namespace: str = Field(...)
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)

class DynamoDeploymentSpec(BaseModel):
    dynamo_nim: str = Field(None)
    services: Dict = Field(None)

class DynamoDeployment(BaseModel):
    type_meta: TypeMeta = Field(..., alias="typeMeta")
    object_meta: ObjectMeta = Field(..., alias="objectMeta")
    spec: DynamoDeploymentSpec = Field(...)
