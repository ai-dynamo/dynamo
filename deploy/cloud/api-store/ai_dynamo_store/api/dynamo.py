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
    CreateDynamoNimRequest,
    CreateDynamoNimVersionRequest,
    DynamoNimSchema,
    DynamoNimSchemaWithDeploymentsListSchema,
    DynamoNimSchemaWithDeploymentsSchema,
    DynamoNimUploadStatus,
    DynamoNimVersionFullSchema,
    DynamoNimVersionSchema,
    DynamoNimVersionsWithNimListSchema,
    DynamoNimVersionWithNimSchema,
    ImageBuildStatus,
    ListQuerySchema,
    OrganizationSchema,
    ResourceType,
    TransmissionStrategy,
    UpdateDynamoNimVersionRequest,
    UserSchema,
)
from .model import DynamoNim, DynamoNimVersion, make_aware, utc_now_naive
from .storage import S3Storage, get_s3_storage, get_session

API_TAG_MODELS = "dynamo"

DEFAULT_LIMIT = 3
SORTABLE_COLUMNS = {
    "created_at": col(DynamoNim.created_at),
    "update_at": col(DynamoNim.updated_at),
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


# GetDynamoNim is a FastAPI dependency that will perform stored model lookup.
async def dynamo_nim_handler(
    *,
    session: AsyncSession = Depends(get_session),
    dynamo_nim_name: str,
) -> DynamoNim:
    statement = select(DynamoNim).where(DynamoNim.name == dynamo_nim_name)
    stored_dynamo_nim_result = await session.exec(statement)
    stored_dynamo_nim = stored_dynamo_nim_result.first()
    if not stored_dynamo_nim:
        raise HTTPException(status_code=404, detail="Record not found")

    return stored_dynamo_nim


GetDynamoNim = Depends(dynamo_nim_handler)


@router.get(
    "/bento_repositories/{dynamo_nim_name}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_nims/{dynamo_nim_name}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_nim(
    *,
    dynamo_nim: DynamoNim = GetDynamoNim,
    session: AsyncSession = Depends(get_session),
):
    dynamo_nim_id = dynamo_nim.id
    statement = (
        select(DynamoNimVersion)
        .where(
            DynamoNimVersion.dynamo_nim_id == dynamo_nim_id,
        )
        .order_by(desc(DynamoNimVersion.created_at))
    )

    result = await session.exec(statement)
    dynamo_nims = result.all()

    latest_dynamo_nim_versions = await convert_dynamo_nim_version_model_to_schema(
        session, list(dynamo_nims), dynamo_nim
    )

    return DynamoNimSchema(
        uid=dynamo_nim.id,
        created_at=dynamo_nim.created_at,
        updated_at=dynamo_nim.updated_at,
        deleted_at=dynamo_nim.deleted_at,
        name=dynamo_nim.name,
        resource_type=ResourceType.DynamoNim,
        labels=[],
        description=dynamo_nim.description,
        latest_bento=None
        if not latest_dynamo_nim_versions
        else latest_dynamo_nim_versions[0],
        latest_bentos=latest_dynamo_nim_versions,
        n_bentos=len(dynamo_nims),
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
    "/dynamo_nims",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def create_dynamo_nim(
    *,
    session: AsyncSession = Depends(get_session),
    request: CreateDynamoNimRequest,
):
    """
    Create a new respository
    """
    try:
        db_dynamo_nim = DynamoNim.model_validate(request)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))  # type: ignore

    logger.debug("Creating repository...")

    try:
        session.add(db_dynamo_nim)
        await session.flush()
        await session.refresh(db_dynamo_nim)
    except IntegrityError as e:
        logger.error(f"Details: {str(e)}")
        await session.rollback()
        logger.error(
            f"The requested Dynamo NIM {db_dynamo_nim.name} already exists in the database"
        )
        raise HTTPException(
            status_code=422,
            detail=f"The Dynamo NIM {db_dynamo_nim.name} already exists in the database",
        )  # type: ignore
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the repository")
        raise HTTPException(status_code=500, detail=str(e))

    await session.commit()
    logger.debug(
        f"Dynamo NIM {db_dynamo_nim.id} with name {db_dynamo_nim.name} saved to database"
    )

    return DynamoNimSchema(
        uid=db_dynamo_nim.id,
        created_at=db_dynamo_nim.created_at,
        updated_at=db_dynamo_nim.updated_at,
        deleted_at=db_dynamo_nim.deleted_at,
        name=db_dynamo_nim.name,
        resource_type=ResourceType.DynamoNim,
        labels=[],
        description=db_dynamo_nim.description,
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
    "/dynamo_nims",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_nim_list(
    *,
    session: AsyncSession = Depends(get_session),
    query_params: ListQuerySchema = Depends(),
):
    try:
        # Base query using SQLModel's select
        statement = select(DynamoNim)

        # Handle search query 'q'
        if query_params.q:
            statement = statement.where(DynamoNim.name.ilike(f"%{query_params.q}%"))

        # Get total count using SQLModel
        total_statement = select(func.count(DynamoNim.id)).select_from(statement)

        # Execute count query
        result = await session.exec(total_statement)
        total = result.scalar() or 0

        # Apply pagination and sorting
        if query_params.sort_asc is not None:
            statement = statement.order_by(
                DynamoNim.created_at.asc()
                if query_params.sort_asc
                else DynamoNim.created_at.desc()
            )

        statement = statement.offset(query_params.start).limit(query_params.count)

        # Execute main query
        result = await session.exec(statement)
        dynamo_nims = result.scalars().all()

        # Rest of your code remains the same
        dynamo_nim_schemas = await convert_dynamo_nim_model_to_schema(
            session, dynamo_nims
        )

        dynamo_nims_with_deployments = [
            DynamoNimSchemaWithDeploymentsSchema(
                **dynamo_nim_schema.model_dump(), deployments=[]
            )
            for dynamo_nim_schema in dynamo_nim_schemas
        ]

        return DynamoNimSchemaWithDeploymentsListSchema(
            total=total,
            start=query_params.start,
            count=query_params.count,
            items=dynamo_nims_with_deployments,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))


async def dynamo_nim_version_handler(
    *,
    session: AsyncSession = Depends(get_session),
    dynamo_nim_name: str,
    version: str,
) -> tuple[DynamoNimVersion, DynamoNim]:
    statement = select(DynamoNimVersion, DynamoNim).where(
        DynamoNimVersion.dynamo_nim_id == DynamoNim.id,
        DynamoNimVersion.version == version,
        DynamoNim.name == dynamo_nim_name,
    )

    result = await session.exec(statement)
    records = result.all()

    if not records:
        logger.error("No Dynamo NIM version record found")
        raise HTTPException(status_code=404, detail="Record not found")

    if len(records) >= 2:
        logger.error("Found multiple relations for Dynamo NIM version")
        raise HTTPException(
            status_code=422, detail="Found multiple relations for Dynamo NIM version"
        )

    return records[0]


GetDynamoNimVersion = Depends(dynamo_nim_version_handler)


@router.get(
    "/bento_repositories/{dynamo_nim_name}/bentos/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_nims/{dynamo_nim_name}/versions/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_nim_version(
    *,
    dynamo_nim_entities: tuple[DynamoNimVersion, DynamoNim] = GetDynamoNimVersion,
    session: AsyncSession = Depends(get_session),
):
    dynamo_nim_version, dynamo_nim = dynamo_nim_entities
    dynamo_nim_version_schemas = await convert_dynamo_nim_version_model_to_schema(
        session, [dynamo_nim_version], dynamo_nim
    )
    dynamo_nim_schemas = await convert_dynamo_nim_model_to_schema(session, [dynamo_nim])

    full_schema = DynamoNimVersionFullSchema(
        **dynamo_nim_version_schemas[0].model_dump(),
        repository=dynamo_nim_schemas[0],
    )
    return full_schema


@router.post(
    "/bento_repositories/{dynamo_nim_name}/bentos",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.post(
    "/dynamo_nims/{dynamo_nim_name}/versions",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def create_dynamo_nim_version(
    request: CreateDynamoNimVersionRequest,
    dynamo_nim: DynamoNim = GetDynamoNim,
    session: AsyncSession = Depends(get_session),
):
    """
    Create a new nim
    """
    print("[DEBUG]request", request)
    try:
        # Create without validation
        db_dynamo_nim_version = DynamoNimVersion(
            **request.model_dump(),
            dynamo_nim_id=dynamo_nim.id,
            upload_status=DynamoNimUploadStatus.Pending,
            image_build_status=ImageBuildStatus.Pending,
        )
        DynamoNimVersion.model_validate(db_dynamo_nim_version)
        tag = f"{dynamo_nim.name}:{db_dynamo_nim_version.version}"
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))  # type: ignore
    except BaseException as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))  # type: ignore

    try:
        session.add(db_dynamo_nim_version)
        await session.flush()
        await session.refresh(db_dynamo_nim_version)
    except IntegrityError as e:
        logger.error(f"Details: {str(e)}")
        await session.rollback()

        logger.error(f"The Dynamo NIM {tag} already exists")
        raise HTTPException(
            status_code=422,
            detail=f"The Dynamo NIM version {tag} already exists",
        )  # type: ignore
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the Dynamo NIM")
        raise HTTPException(status_code=500, detail=str(e))

    logger.debug(
        f"Commiting {dynamo_nim.name}:{db_dynamo_nim_version.version} to database"
    )
    await session.commit()

    schema = await convert_dynamo_nim_version_model_to_schema(
        session, [db_dynamo_nim_version]
    )
    return schema[0]


@router.get(
    "/bento_repositories/{dynamo_nim_name}/bentos",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_nims/{dynamo_nim_name}/versions",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def get_dynamo_nim_versions(
    *,
    dynamo_nim: DynamoNim = GetDynamoNim,
    session: AsyncSession = Depends(get_session),
    query_params: ListQuerySchema = Depends(),
):
    dynamo_nim_schemas = await convert_dynamo_nim_model_to_schema(session, [dynamo_nim])
    dynamo_nim_schema = dynamo_nim_schemas[0]

    total_statement = (
        select(DynamoNimVersion)
        .where(
            DynamoNimVersion.dynamo_nim_id == dynamo_nim.id,
        )
        .order_by(desc(DynamoNimVersion.created_at))
    )

    result = await session.exec(total_statement)
    dynamo_nim_versions = result.all()
    total = len(dynamo_nim_versions)

    statement = total_statement.limit(query_params.count)
    result = await session.exec(statement)
    dynamo_nim_versions = list(result.all())

    dynamo_nim_version_schemas = await convert_dynamo_nim_version_model_to_schema(
        session, dynamo_nim_versions, dynamo_nim
    )

    items = [
        DynamoNimVersionWithNimSchema(
            **version.model_dump(), repository=dynamo_nim_schema
        )
        for version in dynamo_nim_version_schemas
    ]

    return DynamoNimVersionsWithNimListSchema(
        total=total, count=query_params.count, start=query_params.start, items=items
    )


@router.patch(
    "/bento_repositories/{dynamo_nim_name}/bentos/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.patch(
    "/dynamo_nims/{dynamo_nim_name}/versions/{version}",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def update_dynamo_nim_version(
    *,
    dynamo_nim_entities: tuple[DynamoNimVersion, DynamoNim] = GetDynamoNimVersion,
    request: UpdateDynamoNimVersionRequest,
    session: AsyncSession = Depends(get_session),
):
    dynamo_nim_version, _ = dynamo_nim_entities
    dynamo_nim_version.manifest = request.manifest.model_dump()

    try:
        session.add(dynamo_nim_version)
        await session.flush()
        await session.refresh(dynamo_nim_version)
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the Dynamo NIM")
        raise HTTPException(status_code=500, detail=str(e))

    logger.debug("Updating Dynamo NIM")
    await session.commit()

    schema = await convert_dynamo_nim_version_model_to_schema(
        session, [dynamo_nim_version]
    )
    return schema[0]


@router.put(
    "/bento_repositories/{dynamo_nim_name}/bentos/{version}/upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.put(
    "/dynamo_nims/{dynamo_nim_name}/versions/{version}/upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def upload_dynamo_nim_version(
    *,
    dynamo_nim_entities: tuple[DynamoNimVersion, DynamoNim] = GetDynamoNimVersion,
    file: Annotated[bytes, Body()],
    session: AsyncSession = Depends(get_session),
    s3_storage: S3Storage = Depends(get_s3_storage),
):
    dynamo_nim_version, dynamo_nim = dynamo_nim_entities
    object_name = f"{dynamo_nim.name}/{dynamo_nim_version.version}"

    try:
        s3_storage.upload_file(file, object_name)

        dynamo_nim_version.upload_status = DynamoNimUploadStatus.Success
        dynamo_nim_version.upload_finished_at = (
            utc_now_naive()
        )  # datetime.now(timezone.utc)
        session.add(dynamo_nim_version)
        await session.commit()

        return {"message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")


def generate_file_path(version) -> str:
    return f"dynamo-{version}"


@router.get(
    "/bento_repositories/{dynamo_nim_name}/bentos/{version}/download",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.get(
    "/dynamo_nims/{dynamo_nim_name}/versions/{version}/download",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def download_dynamo_nim_version(
    *,
    dynamo_nim_entities: tuple[DynamoNimVersion, DynamoNim] = GetDynamoNimVersion,
    s3_storage: S3Storage = Depends(get_s3_storage),
):
    dynamo_nim_version, dynamo_nim = dynamo_nim_entities
    object_name = f"{dynamo_nim.name}/{dynamo_nim_version.version}"

    try:
        file_data = s3_storage.download_file(object_name)
        return responses.StreamingResponse(
            iter([file_data]), media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@router.patch(
    "/bento_repositories/{dynamo_nim_name}/bentos/{version}/start_upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
@router.patch(
    "/dynamo_nims/{dynamo_nim_name}/versions/{version}/start_upload",
    responses={
        200: {"description": "Successful Response"},
        422: {"description": "Validation Error"},
    },
    tags=[API_TAG_MODELS],
)
async def start_dynamo_nim_version_upload(
    *,
    dynamo_nim_entities: tuple[DynamoNimVersion, DynamoNim] = GetDynamoNimVersion,
    session: AsyncSession = Depends(get_session),
):
    dynamo_nim_version, _ = dynamo_nim_entities
    dynamo_nim_version.upload_status = DynamoNimUploadStatus.Uploading

    try:
        session.add(dynamo_nim_version)
        await session.flush()
        await session.refresh(dynamo_nim_version)
    except SQLAlchemyError as e:
        logger.error("Something went wrong with adding the Dynamo NIM")
        raise HTTPException(status_code=500, detail=str(e))

    logger.debug("Setting Dynamo NIM upload status to Uploading.")
    await session.commit()

    schema = await convert_dynamo_nim_version_model_to_schema(
        session, [dynamo_nim_version]
    )
    return schema[0]


@router.get("/api/v1/healthz")
async def health_check():
    return {"status": "ok"}


"""
    DB to Schema Converters
"""


async def convert_dynamo_nim_model_to_schema(
    session: AsyncSession, entities: List[DynamoNim]
) -> List[DynamoNimSchema]:
    dynamo_nim_schemas = []
    for entity in entities:
        try:
            statement = (
                select(DynamoNimVersion)
                .where(
                    DynamoNimVersion.dynamo_nim_id == entity.id,
                )
                .order_by(desc(DynamoNimVersion.created_at))
                .limit(DEFAULT_LIMIT)
            )

            total_statement = select(func.count(col(DynamoNimVersion.id))).where(
                DynamoNimVersion.dynamo_nim_id == entity.id
            )
            result = await session.exec(total_statement)
            total = result.first()
            if not total:
                total = 0

            result = await session.exec(statement)
            dynamo_nim_versions = list(result.all())
            dynamo_nim_version_schemas = (
                await convert_dynamo_nim_version_model_to_schema(
                    session, dynamo_nim_versions, entity
                )
            )

            # Add timezone info for API responses
            created_at = make_aware(entity.created_at)
            updated_at = make_aware(entity.updated_at)
            deleted_at = make_aware(entity.deleted_at) if entity.deleted_at else None

            dynamo_nim_schemas.append(
                DynamoNimSchema(
                    uid=entity.id,
                    created_at=created_at,
                    updated_at=updated_at,
                    deleted_at=deleted_at,
                    name=entity.name,
                    resource_type=ResourceType.DynamoNim,
                    labels=[],
                    latest_bento=(
                        None
                        if not dynamo_nim_version_schemas
                        else dynamo_nim_version_schemas[0]
                    ),
                    latest_bentos=dynamo_nim_version_schemas,
                    n_bentos=total,
                    description=entity.description,
                )
            )
        except SQLAlchemyError as e:
            logger.error(
                "Something went wrong with getting associated Dynamo NIM versions"
            )
            raise HTTPException(status_code=500, detail=str(e))

    return dynamo_nim_schemas


async def convert_dynamo_nim_version_model_to_schema(
    session: AsyncSession,
    entities: List[DynamoNimVersion],
    dynamo_nim: Optional[DynamoNim] = None,
) -> List[DynamoNimVersionSchema]:
    dynamo_nim_version_schemas = []
    for entity in entities:
        if not dynamo_nim:
            statement = select(DynamoNim).where(DynamoNim.id == entity.dynamo_nim_id)
            results = await session.exec(statement)
            dynamo_nim = results.first()

        if dynamo_nim:
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

            dynamo_nim_version_schema = DynamoNimVersionSchema(
                description="",
                version=entity.version,
                image_build_status=entity.image_build_status,
                upload_status=str(entity.upload_status.value),
                upload_finished_reason=entity.upload_finished_reason,
                uid=entity.id,
                name=dynamo_nim.name,
                created_at=created_at,
                resource_type=ResourceType.DynamoNimVersion,
                labels=[],
                manifest=entity.manifest,
                updated_at=updated_at,
                bento_repository_uid=dynamo_nim.id,
                # upload_started_at=upload_started_at,
                # upload_finished_at=upload_finished_at,
                transmission_strategy=TransmissionStrategy.Proxy,
                build_at=build_at,
            )

            dynamo_nim_version_schemas.append(dynamo_nim_version_schema)
        else:
            raise HTTPException(
                status_code=500, detail="Failed to find related Dynamo NIM"
            )  # Should never happen

    return dynamo_nim_version_schemas
