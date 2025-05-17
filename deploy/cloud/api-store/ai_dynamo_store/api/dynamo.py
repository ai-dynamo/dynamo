# type: ignore  # Ignore all mypy errors in this file
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

import json
import logging
from datetime import datetime
from typing import Annotated, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, responses
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlmodel import col, desc, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from .components import (
    CreateDynamoComponentRequest,
    CreateDynamoComponentVersionRequest,
    DynamoComponentSchema,
    DynamoComponentSchemaWithDeploymentsListSchema,
    DynamoComponentSchemaWithDeploymentsSchema,
    DynamoComponentUploadStatus,
    DynamoComponentVersionFullSchema,
    DynamoComponentVersionSchema,
    DynamoComponentVersionsWithNimListSchema,
    DynamoComponentVersionWithNimSchema,
    ImageBuildStatus,
    ListQuerySchema,
    OrganizationSchema,
    ResourceType,
    TransmissionStrategy,
    UpdateDynamoComponentVersionRequest,
    UserSchema,
)
from .model import DynamoComponent, DynamoComponentVersion, make_aware, utc_now_naive
from .storage import S3Storage, get_s3_storage, get_session

API_TAG_MODELS = "dynamo"

DEFAULT_LIMIT = 3
SORTABLE_COLUMNS = {
    "created_at": col(DynamoComponent.created_at),
    "update_at": col(DynamoComponent.updated_at),
}

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


@router.get(
    "/auth/current",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def login(
    request: Request,
):
    return UserSchema(
        name="dynamo",
        email="dynamo@nvidia.com",
        first_name="dynamo",
        last_name="ai",
    )


@router.get(
    "/current_org",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def current_org(
    request: Request,
):
    return OrganizationSchema(
        uid="uid",
        created_at=datetime(2024, 9, 18, 12, 0, 0),
        updated_at=datetime(2024, 9, 18, 12, 0, 0),
        deleted_at=None,
        name="nvidia",
        resource_type=ResourceType.Organization,
        labels=[],
        description="Dynamo default organization.",
    )


# GetDynamoComponent is a FastAPI dependency that will perform stored model lookup.
async def dynamo_component_handler(
    *,
    session: AsyncSession = Depends(get_session),
    dynamo_component_name: str,
) -> DynamoComponent:
    statement = select(DynamoComponent).where(
        DynamoComponent.name == dynamo_component_name
    )
    stored_dynamo_component_result = await session.exec(statement)
    stored_dynamo_component = stored_dynamo_component_result.first()
    if not stored_dynamo_component:
        raise HTTPException(status_code=404, detail="Record not found")

    return stored_dynamo_component


GetDynamoComponent = Depends(dynamo_component_handler)


@router.get(
    "/bento_repositories/{dynamo_component_name}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_components/{dynamo_component_name}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_component(
    *,
    dynamo_component: DynamoComponent = GetDynamoComponent,
    session: AsyncSession = Depends(get_session),
):
    dynamo_component_id = dynamo_component.id
    statement = (
        select(DynamoComponentVersion)
        .where(
            DynamoComponentVersion.dynamo_component_id == dynamo_component_id,
        )
        .order_by(desc(DynamoComponentVersion.created_at))
    )

    result = await session.exec(statement)
    dynamo_components = result.all()

    latest_dynamo_component_versions = (
        await convert_dynamo_component_version_model_to_schema(
            session, list(dynamo_components), dynamo_component
        )
    )

    return DynamoComponentSchema(
        uid=dynamo_component.id,
        created_at=dynamo_component.created_at,
        updated_at=dynamo_component.updated_at,
        deleted_at=dynamo_component.deleted_at,
        name=dynamo_component.name,
        resource_type=ResourceType.DynamoComponent,
        labels=[],
        description=dynamo_component.description,
        latest_bento=None
        if not latest_dynamo_component_versions
        else latest_dynamo_component_versions[0],
        latest_bentos=latest_dynamo_component_versions,
        n_bentos=len(dynamo_components),
    )


@router.post(
    "/bento_repositories",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.post(
    "/dynamo_components",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def create_dynamo_component(
    *,
    session: AsyncSession = Depends(get_session),
    request: CreateDynamoComponentRequest,
):
    """
    Create a new respository
    """
    try:
        db_dynamo_component = DynamoComponent.model_validate(request)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))  # type: ignore

    logger.debug("Creating repository...")

    try:
        session.add(db_dynamo_component)
        await session.flush()
        await session.refresh(db_dynamo_component)
    except IntegrityError as e:
        logger.error(f"Details: {str(e)}")
        await session.rollback()
        logger.error(
            f"The requested Dynamo Component {db_dynamo_component.name} already exists in the database"
        )
        raise HTTPException(
            status_code=422,
            detail=f"The Dynamo Component {db_dynamo_component.name} already exists in the database",
        )  # type: ignore
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the repository")
        raise HTTPException(status_code=500, detail=str(e))

    await session.commit()
    logger.debug(
        f"Dynamo Component {db_dynamo_component.id} with name {db_dynamo_component.name} saved to database"
    )

    return DynamoComponentSchema(
        uid=db_dynamo_component.id,
        created_at=db_dynamo_component.created_at,
        updated_at=db_dynamo_component.updated_at,
        deleted_at=db_dynamo_component.deleted_at,
        name=db_dynamo_component.name,
        resource_type=ResourceType.DynamoComponent,
        labels=[],
        description=db_dynamo_component.description,
        latest_bentos=None,
        latest_bento=None,
        n_bentos=0,
    )


@router.get(
    "/bento_repositories",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_components",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_component_list(
    *,
    session: AsyncSession = Depends(get_session),
    query_params: ListQuerySchema = Depends(),
):
    try:
        # Base query using SQLModel's select
        statement = select(DynamoComponent)

        # Handle search query 'q'
        if query_params.q:
            statement = statement.where(
                DynamoComponent.name.ilike(f"%{query_params.q}%")
            )

        # Get total count using SQLModel
        total_statement = select(func.count(DynamoComponent.id)).select_from(statement)

        # Execute count query
        result = await session.exec(total_statement)
        total = result.scalar() or 0

        # Apply pagination and sorting
        if query_params.sort_asc is not None:
            statement = statement.order_by(
                DynamoComponent.created_at.asc()
                if query_params.sort_asc
                else DynamoComponent.created_at.desc()
            )

        statement = statement.offset(query_params.start).limit(query_params.count)

        # Execute main query
        result = await session.exec(statement)
        dynamo_components = result.scalars().all()

        # Rest of your code remains the same
        dynamo_component_schemas = await convert_dynamo_component_model_to_schema(
            session, dynamo_components
        )

        dynamo_components_with_deployments = [
            DynamoComponentSchemaWithDeploymentsSchema(
                **dynamo_component_schema.model_dump(), deployments=[]
            )
            for dynamo_component_schema in dynamo_component_schemas
        ]

        return DynamoComponentSchemaWithDeploymentsListSchema(
            total=total,
            start=query_params.start,
            count=query_params.count,
            items=dynamo_components_with_deployments,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))


async def dynamo_component_version_handler(
    *,
    session: AsyncSession = Depends(get_session),
    dynamo_component_name: str,
    version: str,
) -> tuple[DynamoComponentVersion, DynamoComponent]:
    statement = select(DynamoComponentVersion, DynamoComponent).where(
        DynamoComponentVersion.dynamo_component_id == DynamoComponent.id,
        DynamoComponentVersion.version == version,
        DynamoComponent.name == dynamo_component_name,
    )

    result = await session.exec(statement)
    records = result.all()

    if not records:
        logger.error("No Dynamo Component version record found")
        raise HTTPException(status_code=404, detail="Record not found")

    if len(records) >= 2:
        logger.error("Found multiple relations for Dynamo Component version")
        raise HTTPException(
            status_code=422,
            detail="Found multiple relations for Dynamo Component version",
        )

    return records[0]


GetDynamoComponentVersion = Depends(dynamo_component_version_handler)


@router.get(
    "/bento_repositories/{dynamo_component_name}/bentos/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_components/{dynamo_component_name}/versions/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_component_version(
    *,
    dynamo_component_entities: tuple[
        DynamoComponentVersion, DynamoComponent
    ] = GetDynamoComponentVersion,
    session: AsyncSession = Depends(get_session),
):
    dynamo_component_version, dynamo_component = dynamo_component_entities
    dynamo_component_version_schemas = (
        await convert_dynamo_component_version_model_to_schema(
            session, [dynamo_component_version], dynamo_component
        )
    )
    dynamo_component_schemas = await convert_dynamo_component_model_to_schema(
        session, [dynamo_component]
    )

    full_schema = DynamoComponentVersionFullSchema(
        **dynamo_component_version_schemas[0].model_dump(),
        repository=dynamo_component_schemas[0],
    )
    return full_schema


@router.post(
    "/bento_repositories/{dynamo_component_name}/bentos",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.post(
    "/dynamo_components/{dynamo_component_name}/versions",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def create_dynamo_component_version(
    request: CreateDynamoComponentVersionRequest,
    dynamo_component: DynamoComponent = GetDynamoComponent,
    session: AsyncSession = Depends(get_session),
):
    """
    Create a new nim
    """
    print("[DEBUG]request", request)
    try:
        # Create without validation
        db_dynamo_component_version = DynamoComponentVersion(
            **request.model_dump(),
            dynamo_component_id=dynamo_component.id,
            upload_status=DynamoComponentUploadStatus.Pending,
            image_build_status=ImageBuildStatus.Pending,
        )
        DynamoComponentVersion.model_validate(db_dynamo_component_version)
        tag = f"{dynamo_component.name}:{db_dynamo_component_version.version}"
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))  # type: ignore
    except BaseException as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))  # type: ignore

    try:
        session.add(db_dynamo_component_version)
        await session.flush()
        await session.refresh(db_dynamo_component_version)
    except IntegrityError as e:
        logger.error(f"Details: {str(e)}")
        await session.rollback()

        logger.error(f"The Dynamo Component {tag} already exists")
        raise HTTPException(
            status_code=422,
            detail=f"The Dynamo Component version {tag} already exists",
        )  # type: ignore
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the Dynamo Component")
        raise HTTPException(status_code=500, detail=str(e))

    logger.debug(
        f"Commiting {dynamo_component.name}:{db_dynamo_component_version.version} to database"
    )
    await session.commit()

    schema = await convert_dynamo_component_version_model_to_schema(
        session, [db_dynamo_component_version]
    )
    return schema[0]


@router.get(
    "/bento_repositories/{dynamo_component_name}/bentos",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_components/{dynamo_component_name}/versions",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_component_versions(
    *,
    dynamo_component: DynamoComponent = GetDynamoComponent,
    session: AsyncSession = Depends(get_session),
    query_params: ListQuerySchema = Depends(),
):
    dynamo_component_schemas = await convert_dynamo_component_model_to_schema(
        session, [dynamo_component]
    )
    dynamo_component_schema = dynamo_component_schemas[0]

    total_statement = (
        select(DynamoComponentVersion)
        .where(
            DynamoComponentVersion.dynamo_component_id == dynamo_component.id,
        )
        .order_by(desc(DynamoComponentVersion.created_at))
    )

    result = await session.exec(total_statement)
    dynamo_component_versions = result.all()
    total = len(dynamo_component_versions)

    statement = total_statement.limit(query_params.count)
    result = await session.exec(statement)
    dynamo_component_versions = list(result.all())

    dynamo_component_version_schemas = (
        await convert_dynamo_component_version_model_to_schema(
            session, dynamo_component_versions, dynamo_component
        )
    )

    items = [
        DynamoComponentVersionWithNimSchema(
            **version.model_dump(), repository=dynamo_component_schema
        )
        for version in dynamo_component_version_schemas
    ]

    return DynamoComponentVersionsWithNimListSchema(
        total=total, count=query_params.count, start=query_params.start, items=items
    )


@router.patch(
    "/bento_repositories/{dynamo_component_name}/bentos/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.patch(
    "/dynamo_components/{dynamo_component_name}/versions/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def update_dynamo_component_version(
    *,
    dynamo_component_entities: tuple[
        DynamoComponentVersion, DynamoComponent
    ] = GetDynamoComponentVersion,
    request: UpdateDynamoComponentVersionRequest,
    session: AsyncSession = Depends(get_session),
):
    dynamo_component_version, _ = dynamo_component_entities
    dynamo_component_version.manifest = request.manifest.model_dump()

    try:
        session.add(dynamo_component_version)
        await session.flush()
        await session.refresh(dynamo_component_version)
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the Dynamo Component")
        raise HTTPException(status_code=500, detail=str(e))

    logger.debug("Updating Dynamo Component")
    await session.commit()

    schema = await convert_dynamo_component_version_model_to_schema(
        session, [dynamo_component_version]
    )
    return schema[0]


@router.put(
    "/bento_repositories/{dynamo_component_name}/bentos/{version}/upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.put(
    "/dynamo_components/{dynamo_component_name}/versions/{version}/upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def upload_dynamo_component_version(
    *,
    dynamo_component_entities: tuple[
        DynamoComponentVersion, DynamoComponent
    ] = GetDynamoComponentVersion,
    file: Annotated[bytes, Body()],
    session: AsyncSession = Depends(get_session),
    s3_storage: S3Storage = Depends(get_s3_storage),
):
    dynamo_component_version, dynamo_component = dynamo_component_entities
    object_name = f"{dynamo_component.name}/{dynamo_component_version.version}"

    try:
        s3_storage.upload_file(file, object_name)

        dynamo_component_version.upload_status = DynamoComponentUploadStatus.Success
        dynamo_component_version.upload_finished_at = (
            utc_now_naive()
        )  # datetime.now(timezone.utc)
        session.add(dynamo_component_version)
        await session.commit()

        return {"message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")


def generate_file_path(version) -> str:
    return f"dynamo-{version}"


@router.get(
    "/bento_repositories/{dynamo_component_name}/bentos/{version}/download",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_components/{dynamo_component_name}/versions/{version}/download",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def download_dynamo_component_version(
    *,
    dynamo_component_entities: tuple[
        DynamoComponentVersion, DynamoComponent
    ] = GetDynamoComponentVersion,
    s3_storage: S3Storage = Depends(get_s3_storage),
):
    dynamo_component_version, dynamo_component = dynamo_component_entities
    object_name = f"{dynamo_component.name}/{dynamo_component_version.version}"

    try:
        file_data = s3_storage.download_file(object_name)
        return responses.StreamingResponse(
            iter([file_data]), media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@router.patch(
    "/bento_repositories/{dynamo_component_name}/bentos/{version}/start_upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.patch(
    "/dynamo_components/{dynamo_component_name}/versions/{version}/start_upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def start_dynamo_component_version_upload(
    *,
    dynamo_component_entities: tuple[
        DynamoComponentVersion, DynamoComponent
    ] = GetDynamoComponentVersion,
    session: AsyncSession = Depends(get_session),
):
    dynamo_component_version, _ = dynamo_component_entities
    dynamo_component_version.upload_status = DynamoComponentUploadStatus.Uploading

    try:
        session.add(dynamo_component_version)
        await session.flush()
        await session.refresh(dynamo_component_version)
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the Dynamo Component")
        raise HTTPException(status_code=500, detail=str(e))

    logger.debug("Setting Dynamo Component upload status to Uploading.")
    await session.commit()

    schema = await convert_dynamo_component_version_model_to_schema(
        session, [dynamo_component_version]
    )
    return schema[0]


@router.get("/api/v1/healthz")
async def health_check():
    return {"status": "ok"}


"""
    DB to Schema Converters
"""


async def convert_dynamo_component_model_to_schema(
    session: AsyncSession, entities: List[DynamoComponent]
) -> List[DynamoComponentSchema]:
    dynamo_component_schemas = []
    for entity in entities:
        try:
            statement = (
                select(DynamoComponentVersion)
                .where(
                    DynamoComponentVersion.dynamo_component_id == entity.id,
                )
                .order_by(desc(DynamoComponentVersion.created_at))
                .limit(DEFAULT_LIMIT)
            )

            total_statement = select(func.count(col(DynamoComponentVersion.id))).where(
                DynamoComponentVersion.dynamo_component_id == entity.id
            )
            result = await session.exec(total_statement)
            total = result.first()
            if not total:
                total = 0

            result = await session.exec(statement)
            dynamo_component_versions = list(result.all())
            dynamo_component_version_schemas = (
                await convert_dynamo_component_version_model_to_schema(
                    session, dynamo_component_versions, entity
                )
            )

            # Add timezone info for API responses
            created_at = make_aware(entity.created_at)
            updated_at = make_aware(entity.updated_at)
            deleted_at = make_aware(entity.deleted_at) if entity.deleted_at else None

            dynamo_component_schemas.append(
                DynamoComponentSchema(
                    uid=entity.id,
                    created_at=created_at,
                    updated_at=updated_at,
                    deleted_at=deleted_at,
                    name=entity.name,
                    resource_type=ResourceType.DynamoComponent,
                    labels=[],
                    latest_bento=(
                        None
                        if not dynamo_component_version_schemas
                        else dynamo_component_version_schemas[0]
                    ),
                    latest_bentos=dynamo_component_version_schemas,
                    n_bentos=total,
                    description=entity.description,
                )
            )
        except SQLAlchemyError as e:
            logger.error(
                "Something went wrong with getting associated Dynamo Component versions"
            )
            raise HTTPException(status_code=500, detail=str(e))

    return dynamo_component_schemas


async def convert_dynamo_component_version_model_to_schema(
    session: AsyncSession,
    entities: List[DynamoComponentVersion],
    dynamo_component: Optional[DynamoComponent] = None,
) -> List[DynamoComponentVersionSchema]:
    dynamo_component_version_schemas = []
    for entity in entities:
        if not dynamo_component:
            statement = select(DynamoComponent).where(
                DynamoComponent.id == entity.dynamo_component_id
            )
            results = await session.exec(statement)
            dynamo_component = results.first()

        if dynamo_component:
            # Add timezone info for API responses
            created_at = make_aware(utc_now_naive())  # make_aware(entity.created_at)
            updated_at = make_aware(utc_now_naive())  # make_aware(entity.updated_at)
            # upload_started_at = (
            #     make_aware(entity.upload_started_at)
            #     if entity.upload_started_at
            #     else None
            # )
            # upload_finished_at = (
            #     make_aware(entity.upload_finished_at)
            #     if entity.upload_finished_at
            #     else None
            # )
            build_at = make_aware(utc_now_naive())  # make_aware(entity.build_at)
            # description = entity.description or ""

            dynamo_component_version_schema = DynamoComponentVersionSchema(
                description="",
                version=entity.version,
                image_build_status=entity.image_build_status,
                upload_status=str(entity.upload_status.value),
                upload_finished_reason=entity.upload_finished_reason,
                uid=entity.id,
                name=dynamo_component.name,
                created_at=created_at,
                resource_type=ResourceType.DynamoComponentVersion,
                labels=[],
                manifest=entity.manifest,
                updated_at=updated_at,
                bento_repository_uid=dynamo_component.id,
                # upload_started_at=upload_started_at,
                # upload_finished_at=upload_finished_at,
                transmission_strategy=TransmissionStrategy.Proxy,
                build_at=build_at,
            )

            dynamo_component_version_schemas.append(dynamo_component_version_schema)
        else:
            raise HTTPException(
                status_code=500, detail="Failed to find related Dynamo Component"
            )  # Should never happen

    return dynamo_component_version_schemas
