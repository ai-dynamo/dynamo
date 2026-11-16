# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Optional dependency checks precede runtime imports.

"""Configured authority, capture, and verified completion against a real store."""

import hashlib
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

np = pytest.importorskip("numpy")
Fernet = pytest.importorskip("cryptography.fernet").Fernet
pytest.importorskip("opendal")
pytest.importorskip("boto3")
pytest.importorskip("msgspec")
pytest.importorskip("zstandard")

from dynamo.artifacts.runtime import ArtifactRuntime
from dynamo.artifacts.storage import ArtifactStorageError
from dynamo.artifacts.tests import test_storage_minio as test_artifact_storage_minio
from dynamo.artifacts.vllm.capture import CaptureBudget, VllmArtifactSession

frozen_artifact = test_artifact_storage_minio.frozen_artifact
store = test_artifact_storage_minio.store

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.post_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]


def authority(tmp_path, store, verifiers=None):
    _, bucket, _, policy = store
    return ArtifactRuntime(
        {
            "owner": "isolated-worker",
            "operation_db": str(tmp_path / "operations.db"),
            "storage_policy": {
                "allowed_hosts": list(policy.allowed_hosts),
                "allow_http_hosts": list(policy.allow_http_hosts),
                "allowed_networks": list(policy.allowed_networks),
            },
            "profiles": {
                "minio": {
                    "service": "s3",
                    "immutable": True,
                    "allowed_prefixes": ["artifacts"],
                    "options": {
                        "bucket": bucket,
                        "endpoint": os.environ["DYN_ARTIFACT_MINIO_ENDPOINT"],
                        "region": "us-east-1",
                        "access_key_id": os.environ["DYN_ARTIFACT_MINIO_ACCESS_KEY"],
                        "secret_access_key": os.environ[
                            "DYN_ARTIFACT_MINIO_SECRET_KEY"
                        ],
                    },
                }
            },
            "verifiers": verifiers or {},
        },
        Fernet.generate_key(),
    )


@pytest_asyncio.fixture
async def authority_factory():
    instances = []

    def create(tmp_path, store, verifiers=None):
        instance = authority(tmp_path, store, verifiers)
        instances.append(instance)
        return instance

    yield create
    for instance in instances:
        await instance.close()


@pytest.mark.asyncio
async def test_capture_handoff_and_verified_managed_result(
    tmp_path, store, authority_factory
):
    runtime = authority_factory(tmp_path, store)
    destination = {
        "kind": "managed_store",
        "profile": "minio",
        "object_key": "artifacts/capture.dynexp",
    }
    request = {
        "nvext": {
            "metadata_upload": {
                "kind": "generation_artifact",
                "mode": "object_store",
                "contents": ["moe_routes"],
                "destination": destination,
            }
        }
    }
    budget = CaptureBudget(1 << 20)
    session = VllmArtifactSession.from_request(
        request,
        prompt_token_count=2,
        max_tokens=2,
        router_profile=((3, 8, 2),),
        route_capture_enabled=True,
        runtime=runtime,
        budget=budget,
    )
    rows = np.asarray([[[1, 2]], [[2, 3]], [[3, 4]]], dtype=np.int64)
    session.observe(
        SimpleNamespace(index=0, token_ids=[12, 13], routed_experts=rows), [10, 11]
    )
    pending = await session.finalize()
    assert pending["state"] == "pending"
    assert budget.used == 0
    await runtime.service.drain()
    ready = runtime.status(pending["operation_id"])
    assert ready["state"] == "ready"
    assert ready["kind"] == "generation_artifact"
    assert ready["mode"] == "object_store"
    client, bucket, _, _ = store
    stored = client.get_object(Bucket=bucket, Key=destination["object_key"])[
        "Body"
    ].read()
    assert hashlib.sha256(stored).hexdigest() == ready["sha256"]


@pytest.mark.asyncio
async def test_presigned_acknowledgement_loss_reconciles_without_retry(
    tmp_path, store, frozen_artifact, authority_factory
):
    client, bucket, _, _ = store
    key = "artifacts/delegated.dynexp"
    runtime = authority_factory(
        tmp_path,
        store,
        {
            "registration": {
                "profile": "minio",
                "url_prefix": f"{os.environ['DYN_ARTIFACT_MINIO_ENDPOINT'].rstrip('/')}/{bucket}/artifacts/",
                "object_key_prefix": "artifacts",
            }
        },
    )
    url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "IfNoneMatch": "*"},
        ExpiresIn=300,
    )
    destination = {
        "kind": "presigned_http_put",
        "url": url,
        "required_headers": {"if-none-match": "*"},
        "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat(),
        "max_bytes": frozen_artifact.byte_count,
        "object_id": key,
        "verification_ref": "registration",
    }
    transfer = runtime._transfer
    calls = []

    async def lose_ack(data, target):
        calls.append(1)
        await transfer(data, target)
        raise ArtifactStorageError(
            "commit_unknown", "response lost", outcome="ambiguous"
        )

    runtime.service.transfer = lose_ack
    pending = await runtime.submit(frozen_artifact.data, ["moe_routes"], destination)
    await runtime.service.drain()
    ready = runtime.status(pending["operation_id"])
    assert ready["state"] == "ready"
    assert ready["sha256"] == frozen_artifact.sha256
    assert len(calls) == 1
    with sqlite3.connect(runtime.store.path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(operations)")
        }
    assert "grant" not in columns
