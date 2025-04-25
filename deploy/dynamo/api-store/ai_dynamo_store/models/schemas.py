# SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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


class DeploymentListResponse(BaseModel):
    start: int
    count: int
    total: int
    items: List[DeploymentFullSchema]


def create_default_user() -> UserSchema:
    """Create a default user schema for testing/demo purposes."""
    return UserSchema(
        name="default-user",
        email="default@example.com",
        first_name="Default",
        last_name="User",
    )


def create_default_cluster(creator: UserSchema) -> ClusterSchema:
    """Create a default cluster schema for testing/demo purposes."""
    return ClusterSchema(
        uid="default-cluster",
        name="default",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        resource_type="cluster",
        labels=[],
        description="Default cluster",
        organization_name="default-org",
        creator=creator,
        is_first=True,
    )
