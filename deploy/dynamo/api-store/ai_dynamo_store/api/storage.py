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

import logging
import os
from typing import Any, AsyncGenerator

import boto3
from botocore.exceptions import ClientError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)

### SQL database

DB_URL_PARTS = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"]
POSTGRES_DB_URL_FORMAT = (
    "postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def get_db_url_from_env():
    database_url = os.getenv("DATABASE_URL", None)
    if database_url:
        return database_url
    db_creds = {key: os.getenv(key) for key in DB_URL_PARTS}
    db_creds["DB_PORT"] = os.getenv("DB_PORT", "5432")
    if all(list(db_creds.values())):
        # we can construct db url from parts
        return POSTGRES_DB_URL_FORMAT.format(**db_creds)
    return None


database_url = get_db_url_from_env()
connect_args = {}
if not database_url:  # default to sqlite in-memory
    sqlite_file_name = "database.db"
    database_url = f"sqlite+aiosqlite:///{sqlite_file_name}"
    connect_args = {"check_same_thread": False}
    logger.warning(
        "WARNING: Using SQLite in-memory database, no data persistence"
    )  # noqa: T201

os.environ["API_DATABASE_URL"] = database_url
engine = create_async_engine(
    url=database_url, echo=True, pool_pre_ping=True, connect_args=connect_args
)


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    async_session = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


async def create_db_and_tables_async():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


### S3 storage

DYN_OBJECT_STORE_BUCKET = os.getenv("DYN_OBJECT_STORE_BUCKET", "dynamo-storage").lower()


def get_s3_client():
    s3_key = os.getenv("DYN_OBJECT_STORE_ID")
    s3_secret = os.getenv("DYN_OBJECT_STORE_KEY")
    s3_url = os.getenv("DYN_OBJECT_STORE_ENDPOINT")
    if not s3_url:
        raise ValueError("DYN_OBJECT_STORE_ENDPOINT is required for S3 connection")
    if not s3_key:
        raise ValueError("DYN_OBJECT_STORE_ID is required for S3 authentication")
    if not s3_secret:
        raise ValueError("DYN_OBJECT_STORE_KEY is required for S3 authentication")
    return boto3.client(
        "s3",
        aws_access_key_id=s3_key,
        aws_secret_access_key=s3_secret,
        endpoint_url=s3_url,
    )


class S3Storage:
    def __init__(self):
        self.s3_client = get_s3_client()
        self.bucket_name = DYN_OBJECT_STORE_BUCKET.replace("_", "-").lower()
        self.ensure_bucket_exists()

    def ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Bucket doesn't exist, create it
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                except ClientError as create_error:
                    logger.error(
                        f"Failed to create bucket {self.bucket_name}: {create_error}"
                    )
                    raise
            else:
                logger.error(f"Error checking bucket {self.bucket_name}: {e}")
                raise

    def upload_file(self, file_data, object_name):
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=object_name, Body=file_data
            )
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise

    def download_file(self, object_name):
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=object_name
            )
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise


S3_STORAGE_INSTANCE: S3Storage | None = None


def get_s3_storage() -> S3Storage:
    global S3_STORAGE_INSTANCE
    if S3_STORAGE_INSTANCE is None:
        S3_STORAGE_INSTANCE = S3Storage()
    assert isinstance(S3_STORAGE_INSTANCE, S3Storage)
    return S3_STORAGE_INSTANCE
