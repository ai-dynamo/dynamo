from __future__ import annotations

import datetime
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Query
from pydantic import BaseModel, ValidationError, field_validator
from sqlalchemy import JSON, Column
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

class TimeCreatedUpdated(SQLModel):
    created_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)


class CompoundNimUploadStatus(str, Enum):
    Pending = "pending"
    Uploading = "uploading"
    Success = "success"
    Failed = "failed"


class ImageBuildStatus(str, Enum):
    Pending = "pending"
    Building = "building"
    Success = "success"
    Failed = "failed"


class TransmissionStrategy(str, Enum):
    Proxy = "proxy"


"""
    API Request Objects
"""


class CreateCompoundNimRequest(BaseModel):
    name: str
    description: str
    labels: Optional[Dict[str, str]] = None


class CreateCompoundNimVersionRequest(BaseModel):
    description: str
    version: str
    manifest: CompoundNimVersionManifestSchema
    build_at: datetime.datetime
    labels: Optional[list[Dict[str, str]]] = None


class UpdateCompoundNimVersionRequest(BaseModel):
    manifest: CompoundNimVersionManifestSchema
    labels: Optional[list[Dict[str, str]]] = None


class ListQuerySchema(BaseModel):
    start: int = Query(0, alias="start")
    count: int = Query(0, alias="count")
    search: Optional[str] = Query(None, alias="search")
    q: Optional[str] = Query(None, alias="q")

    def get_query_map(self) -> Dict[str, Any]:
        if not self.q:
            return {}

        query = defaultdict(list)
        for piece in self.q.split():
            if ":" in piece:
                k, v = piece.split(":")
                query[k].append(v)

            else:
                # Todo: add search keywords
                continue

        return query


"""
    API Schemas
"""


class ResourceType(str, Enum):
    Organization = "organization"
    Cluster = "cluster"
    CompoundNim = "compound_nim"
    CompoundNimVersion = "compound_nim_version"
    Deployment = "deployment"
    DeploymentRevision = "deployment_revision"
    TerminalRecord = "terminal_record"
    Label = "label"


class BaseSchema(BaseModel):
    uid: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    deleted_at: Optional[datetime.datetime] = None


class BaseListSchema(BaseModel):
    total: int
    start: int
    count: int


class ResourceSchema(BaseSchema):
    name: str
    resource_type: ResourceType
    labels: List[LabelItemSchema]


class LabelItemSchema(BaseModel):
    key: str
    value: str


class OrganizationSchema(ResourceSchema):
    description: str


class UserSchema(BaseModel):
    name: str
    email: str
    first_name: str
    last_name: str


class CompoundNimVersionApiSchema(BaseModel):
    route: str
    doc: str
    input: str
    output: str


class CompoundNimVersionManifestSchema(BaseModel):
    service: str
    bentoml_version: str
    apis: Dict[str, CompoundNimVersionApiSchema]
    size_bytes: int


def _validate_manifest(v):
    try:
        # Validate that the 'manifest' matches the CompoundAIManifestSchema
        return CompoundNimVersionManifestSchema.model_validate(v).model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid manifest schema: {e}")


class CompoundNimVersionSchema(ResourceSchema):
    bento_repository_uid: str
    version: str
    description: str
    image_build_status: ImageBuildStatus
    upload_status: str
    upload_started_at: Optional[datetime.datetime]
    upload_finished_at: Optional[datetime.datetime]
    upload_finished_reason: str
    presigned_upload_url: str = ""
    presigned_download_url: str = ""
    presigned_urls_deprecated: bool = False
    transmission_strategy: TransmissionStrategy
    upload_id: str = ""
    manifest: Optional[CompoundNimVersionManifestSchema | dict[str, Any]]
    build_at: datetime.datetime

    @field_validator("manifest")
    def validate_manifest(cls, v):
        return _validate_manifest(v)


class CompoundNimVersionFullSchema(CompoundNimVersionSchema):
    repository: CompoundNimSchema


class CompoundNimSchema(ResourceSchema):
    latest_bento: Optional[CompoundNimVersionSchema]
    latest_bentos: Optional[List[CompoundNimVersionSchema]]
    n_bentos: int
    description: str


class CompoundNimSchemaWithDeploymentsSchema(CompoundNimSchema):
    deployments: List[str] = []  # mocked for now


class CompoundNimSchemaWithDeploymentsListSchema(BaseListSchema):
    items: List[CompoundNimSchemaWithDeploymentsSchema]


class CompoundNimVersionsWithNimListSchema(BaseListSchema):
    items: List[CompoundNimVersionWithNimSchema]


class CompoundNimVersionWithNimSchema(CompoundNimVersionSchema):
    repository: CompoundNimSchema


"""
    DB Models
"""


class BaseCompoundNimModel(TimeCreatedUpdated, AsyncAttrs):
    deleted_at: Optional[datetime.datetime] = SQLField(nullable=True, default=None)


class CompoundNimVersionBase(BaseCompoundNimModel):
    version: str = SQLField(default=None)
    description: str = SQLField(default="")
    file_path: Optional[str] = SQLField(default=None)
    file_oid: Optional[str] = SQLField(default=None)  # Used for GIT Lfs access
    upload_status: CompoundNimUploadStatus = SQLField()
    image_build_status: ImageBuildStatus = SQLField()
    image_build_status_syncing_at: Optional[datetime.datetime] = SQLField(default=None)
    image_build_status_updated_at: Optional[datetime.datetime] = SQLField(default=None)
    upload_started_at: Optional[datetime.datetime] = SQLField(default=None)
    upload_finished_at: Optional[datetime.datetime] = SQLField(default=None)
    upload_finished_reason: str = SQLField(default="")
    manifest: Optional[CompoundNimVersionManifestSchema | dict[str, Any]] = SQLField(
        default=None, sa_column=Column(JSON)
    )  # JSON-like field for the manifest
    build_at: datetime.datetime = SQLField()

    @field_validator("manifest")
    def validate_manifest(cls, v):
        return _validate_manifest(v)


class CompoundNimBase(BaseCompoundNimModel):
    name: str = SQLField(default="", unique=True)
    description: str = SQLField(default="")
