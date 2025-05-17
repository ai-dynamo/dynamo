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

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import Query
from pydantic import BaseModel, ValidationError, field_validator
from sqlalchemy import JSON, Column
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


class TimeCreatedUpdated(SQLModel):
    created_at: datetime = SQLField(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )
    updated_at: datetime = SQLField(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )


class DynamoComponentUploadStatus(str, Enum):
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


class CreateDynamoComponentRequest(BaseModel):
    name: str
    description: str
    labels: Optional[Dict[str, str]] = None


class CreateDynamoComponentVersionRequest(BaseModel):
    description: str
    version: str
    manifest: DynamoComponentVersionManifestSchema
    build_at: datetime
    labels: Optional[list[Dict[str, str]]] = None


class UpdateDynamoComponentVersionRequest(BaseModel):
    manifest: DynamoComponentVersionManifestSchema
    labels: Optional[list[Dict[str, str]]] = None


class ListQuerySchema(BaseModel):
    start: int = Query(default=0, ge=0, alias="start")
    count: int = Query(default=20, ge=0, alias="count")
    search: Optional[str] = Query(None, alias="search")
    q: Optional[str] = Query(default="", alias="q")
    sort_asc: bool = Query(default=False)

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
    DynamoComponent = "dynamo_component"
    DynamoComponentVersion = "dynamo_component_version"
    Deployment = "deployment"
    DeploymentRevision = "deployment_revision"
    TerminalRecord = "terminal_record"
    Label = "label"


class BaseSchema(BaseModel):
    uid: str
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


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


class DynamoComponentVersionApiSchema(BaseModel):
    route: str
    doc: str
    input: str
    output: str


class DynamoComponentVersionManifestSchema(BaseModel):
    service: str
    bentoml_version: Optional[str] = None
    apis: Dict[str, DynamoComponentVersionApiSchema]
    size_bytes: int


def _validate_manifest(v):
    try:
        # Validate that the 'manifest' matches the DynamoManifestSchema
        return DynamoComponentVersionManifestSchema.model_validate(v).model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid manifest schema: {e}")


class DynamoComponentVersionSchema(ResourceSchema):
    bento_repository_uid: str
    version: str
    description: str
    image_build_status: ImageBuildStatus
    upload_status: str
    # upload_started_at: Optional[datetime]
    # upload_finished_at: Optional[datetime]
    upload_finished_reason: str
    presigned_upload_url: str = ""
    presigned_download_url: str = ""
    presigned_urls_deprecated: bool = False
    transmission_strategy: TransmissionStrategy
    upload_id: str = ""
    manifest: Optional[Union[DynamoComponentVersionManifestSchema, Dict[str, Any]]]
    build_at: datetime

    @field_validator("manifest")
    def validate_manifest(cls, v):
        return _validate_manifest(v)


class DynamoComponentVersionFullSchema(DynamoComponentVersionSchema):
    repository: DynamoComponentSchema


class DynamoComponentSchema(ResourceSchema):
    latest_bento: Optional[DynamoComponentVersionSchema]
    latest_bentos: Optional[List[DynamoComponentVersionSchema]]
    n_bentos: int
    description: str


class DynamoComponentSchemaWithDeploymentsSchema(DynamoComponentSchema):
    deployments: List[str] = []  # mocked for now


class DynamoComponentSchemaWithDeploymentsListSchema(BaseListSchema):
    items: List[DynamoComponentSchemaWithDeploymentsSchema]


class DynamoComponentVersionsWithNimListSchema(BaseListSchema):
    items: List[DynamoComponentVersionWithNimSchema]


class DynamoComponentVersionWithNimSchema(DynamoComponentVersionSchema):
    repository: DynamoComponentSchema


"""
    DB Models
"""


class BaseDynamoComponentModel(TimeCreatedUpdated, AsyncAttrs):
    deleted_at: Optional[datetime] = SQLField(nullable=True, default=None)


class DynamoComponentVersionBase(BaseDynamoComponentModel):
    version: str = SQLField(default=None)
    description: str = SQLField(default="")
    file_path: Optional[str] = SQLField(default=None)
    file_oid: Optional[str] = SQLField(default=None)  # Used for GIT Lfs access
    upload_status: DynamoComponentUploadStatus = SQLField()
    image_build_status: ImageBuildStatus = SQLField()
    image_build_status_syncing_at: Optional[datetime] = SQLField(default=None)
    image_build_status_updated_at: Optional[datetime] = SQLField(default=None)
    upload_started_at: Optional[datetime] = SQLField(default=None)
    upload_finished_at: Optional[datetime] = SQLField(default=None)
    upload_finished_reason: str = SQLField(default="")
    manifest: Optional[
        Union[DynamoComponentVersionManifestSchema, Dict[str, Any]]
    ] = SQLField(
        default=None, sa_column=Column(JSON)
    )  # JSON-like field for the manifest
    build_at: datetime = SQLField()

    @field_validator("manifest")
    def validate_manifest(cls, v):
        return _validate_manifest(v)


class DynamoComponentBase(BaseDynamoComponentModel):
    name: str = SQLField(default="", unique=True)
    description: str = SQLField(default="")
