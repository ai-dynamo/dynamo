# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in S3-compatible integration coverage for artifact delivery."""

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import uuid4

import boto3
import pytest
from botocore.client import Config

from dynamo.artifacts.storage import (
    ArtifactStorageError,
    ManagedFsspecTarget,
    PresignedHttpPutTarget,
    put_artifact,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
]


@pytest.fixture
def minio_client(monkeypatch):
    endpoint = os.environ.get("DYN_ARTIFACT_MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("set DYN_ARTIFACT_MINIO_ENDPOINT to run MinIO artifact E2E")
    access_key = os.environ.get("DYN_ARTIFACT_MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("DYN_ARTIFACT_MINIO_SECRET_KEY", "minioadmin")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )
    bucket = f"dynamo-generation-artifact-{uuid4().hex}"
    client.create_bucket(Bucket=bucket)
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", "true")
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "true")
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS",
        urlsplit(endpoint).netloc,
    )
    try:
        yield client, bucket, endpoint, access_key, secret_key
    finally:
        response = client.list_objects_v2(Bucket=bucket)
        objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        client.delete_bucket(Bucket=bucket)


@pytest.mark.asyncio
async def test_minio_managed_fsspec_create_only(minio_client, monkeypatch) -> None:
    client, bucket, endpoint, access_key, secret_key = minio_client
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        json.dumps(
            {
                "minio": {
                    "url": f"s3://{bucket}",
                    "allowed_prefixes": ["artifacts"],
                    "create_only": True,
                    "storage_options": {
                        "key": access_key,
                        "secret": secret_key,
                        "client_kwargs": {
                            "endpoint_url": endpoint,
                            "region_name": "us-east-1",
                        },
                        "config_kwargs": {"s3": {"addressing_style": "path"}},
                    },
                }
            }
        ),
    )
    target = ManagedFsspecTarget(profile="minio", object_key="artifacts/managed.bin")
    payload = b"managed-minio-artifact"

    receipt = await put_artifact(payload, target)

    assert receipt.actual_bytes == len(payload)
    assert (
        client.get_object(Bucket=bucket, Key="artifacts/managed.bin")["Body"].read()
        == payload
    )
    with pytest.raises(ArtifactStorageError, match="managed artifact write failed"):
        await put_artifact(b"must-not-overwrite", target)
    assert (
        client.get_object(Bucket=bucket, Key="artifacts/managed.bin")["Body"].read()
        == payload
    )


@pytest.mark.asyncio
async def test_minio_presigned_put_create_only(minio_client) -> None:
    client, bucket, _endpoint, _, _ = minio_client
    key = "artifacts/presigned.bin"
    url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ContentType": "application/octet-stream",
            "IfNoneMatch": "*",
        },
        ExpiresIn=600,
        HttpMethod="PUT",
    )
    target = PresignedHttpPutTarget(
        url=url,
        max_bytes=1024,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        required_headers={
            "content-type": "application/octet-stream",
            "if-none-match": "*",
        },
        object_id="opaque-minio-presigned-artifact",
    )
    payload = b"presigned-minio-artifact"

    receipt = await put_artifact(payload, target)

    assert receipt.object_id == "opaque-minio-presigned-artifact"
    assert client.get_object(Bucket=bucket, Key=key)["Body"].read() == payload
    with pytest.raises(
        ArtifactStorageError, match="presigned artifact PUT was not accepted"
    ):
        await put_artifact(b"must-not-overwrite", target)
    assert client.get_object(Bucket=bucket, Key=key)["Body"].read() == payload
