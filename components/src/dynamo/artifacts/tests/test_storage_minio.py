# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real S3-compatible transfer tests; no inference service is involved."""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest

boto3 = pytest.importorskip("boto3")
np = pytest.importorskip("numpy")
opendal = pytest.importorskip("opendal")
Config = pytest.importorskip("botocore.config").Config
ClientError = pytest.importorskip("botocore.exceptions").ClientError
pytest.importorskip("msgspec")
pytest.importorskip("zstandard")

from dynamo.artifacts.storage import (  # noqa: E402 -- optional dependency guards
    ArtifactStorageError,
    ManagedProfile,
    ManagedStoreTarget,
    PresignedHttpPutTarget,
    StoragePolicy,
    put_artifact,
)
from dynamo.artifacts.vllm.format import (  # noqa: E402 -- optional dependency guards
    GenerationArtifactChoice,
    GenerationArtifactView,
    MoeRoutes,
    RouterRoutes,
    Tensor,
    encode_generation_artifact,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.timeout(60),
]


@pytest.fixture
def frozen_artifact():
    routes = MoeRoutes(
        1,
        (
            RouterRoutes(3, 256, Tensor.from_array(np.asarray([[4], [5]]))),
            RouterRoutes(9, 257, Tensor.from_array(np.asarray([[1, 256], [256, 1]]))),
        ),
    )
    choice = GenerationArtifactChoice(
        0, 2, Tensor.from_array(np.asarray([11, 12, 13, 14])), (routes,)
    )
    return encode_generation_artifact(GenerationArtifactView((choice,)))


@pytest.fixture
def store():
    endpoint = os.environ.get("DYN_ARTIFACT_MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("set DYN_ARTIFACT_MINIO_ENDPOINT for S3-compatible integration")
    access = os.environ["DYN_ARTIFACT_MINIO_ACCESS_KEY"]
    secret = os.environ["DYN_ARTIFACT_MINIO_SECRET_KEY"]
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=10,
            retries={"max_attempts": 1},
        ),
    )
    bucket = f"dynamo-artifact-{uuid4().hex}"
    client.create_bucket(Bucket=bucket)
    operator = opendal.AsyncOperator(
        "s3",
        bucket=bucket,
        endpoint=endpoint,
        region="us-east-1",
        access_key_id=access,
        secret_access_key=secret,
    )
    authority = urlsplit(endpoint).netloc
    policy = StoragePolicy(
        allowed_hosts=frozenset({authority}),
        allow_http_hosts=frozenset({authority}),
        allowed_networks=("127.0.0.0/8", "::1/128"),
    )
    try:
        yield client, bucket, operator, policy
    finally:
        listed = client.list_objects_v2(Bucket=bucket)
        keys = [{"Key": item["Key"]} for item in listed.get("Contents", ())]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        client.delete_bucket(Bucket=bucket)
        client.close()


def _grant(client, bucket, key, artifact):
    url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ContentType": "application/octet-stream",
            "ContentLength": artifact.byte_count,
            "IfNoneMatch": "*",
            "Metadata": {"artifact-sha256": artifact.sha256},
        },
        ExpiresIn=300,
        HttpMethod="PUT",
    )
    headers = {
        "content-type": "application/octet-stream",
        "content-length": str(artifact.byte_count),
        "if-none-match": "*",
        "x-amz-meta-artifact-sha256": artifact.sha256,
    }
    signed_headers = set(
        parse_qs(urlsplit(url).query)["X-Amz-SignedHeaders"][0].split(";")
    )
    assert signed_headers == set(headers) | {"host"}
    return PresignedHttpPutTarget(
        url=url,
        required_headers=headers,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        max_bytes=artifact.byte_count,
        object_id="frozen-fixture",
    )


def _read(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return body.read()


@pytest.mark.asyncio
async def test_minio_same_v1_bytes_managed_and_presigned(
    store, frozen_artifact, monkeypatch
):
    client, bucket, operator, policy = store
    managed_key = "artifacts/managed space/%25/Δ.bin"
    presigned_key = "artifacts/presigned space/%25/Δ.bin"
    managed = ManagedStoreTarget("minio", managed_key)
    profiles = {"minio": ManagedProfile(operator, ("artifacts",))}
    delegated = _grant(client, bucket, presigned_key, frozen_artifact)
    managed_receipt = await put_artifact(
        frozen_artifact.data, managed, policy=policy, profiles=profiles
    )
    for variable in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(variable, raising=False)
    presigned_receipt = await put_artifact(
        frozen_artifact.data, delegated, policy=policy
    )
    received = tuple(_read(client, bucket, key) for key in (managed_key, presigned_key))
    assert received[0] == received[1] == frozen_artifact.data
    assert managed_receipt.sha256 == presigned_receipt.sha256 == frozen_artifact.sha256
    for target in (managed, delegated):
        with pytest.raises(ArtifactStorageError) as error:
            await put_artifact(
                b"\xff" * frozen_artifact.byte_count,
                target,
                policy=policy,
                profiles=profiles,
            )
        assert error.value.outcome == "ambiguous"
    assert (
        tuple(_read(client, bucket, key) for key in (managed_key, presigned_key))
        == received
    )
    print(
        json.dumps(
            {
                "test": "same_v1_bytes_managed_and_presigned",
                "bytes": frozen_artifact.byte_count,
                "managed_sha256": hashlib.sha256(received[0]).hexdigest(),
                "presigned_sha256": hashlib.sha256(received[1]).hexdigest(),
                "equal_bytes": True,
                "encoded_space_percent_unicode_keys": True,
                "conditional_collisions_preserve_original": True,
                "delegated_cloud_credentials": False,
                "live_generation": False,
            },
            sort_keys=True,
        )
    )


@pytest.mark.asyncio
async def test_minio_byte_limits_prevent_object_creation(store, frozen_artifact):
    client, bucket, operator, policy = store
    key = "artifacts/managed-over-limit"
    target = ManagedStoreTarget("minio", key)
    profiles = {
        "minio": ManagedProfile(
            operator, ("artifacts",), max_bytes=frozen_artifact.byte_count - 1
        )
    }
    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(
            frozen_artifact.data, target, policy=policy, profiles=profiles
        )
    assert error.value.code == "too_large"
    with pytest.raises(ClientError) as missing:
        client.head_object(Bucket=bucket, Key=key)
    assert missing.value.response["ResponseMetadata"]["HTTPStatusCode"] == 404
