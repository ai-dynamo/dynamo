# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for MinIO uploads on CRT-optimized hosts.

On network-optimized instances, boto3's automatic managed-transfer selection
can choose the AWS CRT client, which ignores the configured MinIO endpoint.
This test reproduces that host-dependent branch on CPU and verifies that the
adapter still reaches the configured endpoint.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest
from pytest_httpserver import HTTPServer

from tests.serve.lora_utils import MinioLoraConfig, MinioService

# The tests run in well under a second against a loopback server, but a
# regression puts the upload back on the CRT path, which addresses real S3 and
# lets botocore's retry loop stall on an unreachable endpoint. The marker caps
# that at a fast failure rather than a hung pre-merge job.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.timeout(60),
]

BUCKET = "my-loras"
LORA_NAME = "test-org/test-lora"
ADAPTER_NAME = "adapter.bin"
ADAPTER_BYTES = b"adapter payload"


@dataclass
class S3Stub:
    """A local stand-in for MinIO that records what it was actually handed."""

    endpoint: str
    objects: dict[str, bytes] = field(default_factory=dict)


@pytest.fixture
def s3_stub() -> Iterator[S3Stub]:
    # werkzeug is imported here rather than at module scope, as conftest.py's
    # image_server does, so collection stays independent of it.
    from werkzeug.wrappers import Request, Response

    server = HTTPServer(host="127.0.0.1", port=0)
    server.start()
    stub = S3Stub(endpoint=f"http://127.0.0.1:{server.port}")

    def record_object(request: Request) -> Response:
        key = request.path[len(f"/{BUCKET}/") :]
        stub.objects[key] = request.get_data()
        return Response("", status=200)

    server.expect_request(
        re.compile(rf"^/{re.escape(BUCKET)}/.+"), method="PUT"
    ).respond_with_handler(record_object)

    yield stub

    server.clear()
    server.stop()


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    """A minimal stand-in for a downloaded LoRA snapshot."""
    (tmp_path / ADAPTER_NAME).write_bytes(ADAPTER_BYTES)
    return tmp_path


def _service(stub: S3Stub) -> MinioService:
    return MinioService(
        MinioLoraConfig(endpoint=stub.endpoint, bucket=BUCKET, lora_name=LORA_NAME)
    )


def _force_crt_optimized_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this host look like the CRT-optimized instances used for H100 CI."""
    awscrt_s3 = pytest.importorskip(
        "awscrt.s3",
        reason="awscrt is not installed, so boto3 never offers the CRT path",
    )
    monkeypatch.setattr(awscrt_s3, "is_optimized_for_system", lambda: True)

    # Confirm boto3 really does resolve to the CRT client under that patch,
    # rather than letting the test pass vacuously on a boto3/awscrt combination
    # that would never have taken the failing branch.
    # _should_use_crt is boto3-private, so a rename skips rather than raising
    # ImportError, which would take the two negative controls down with it.
    try:
        from boto3.s3.transfer import TransferConfig, _should_use_crt
    except ImportError:
        pytest.skip("boto3 no longer exposes _should_use_crt under that name")

    if not _should_use_crt(TransferConfig()):
        pytest.skip("boto3 does not select the CRT transfer client in this environment")


def test_upload_lora_targets_configured_endpoint_on_crt_optimized_host(
    s3_stub: S3Stub, adapter_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter reaches the configured endpoint even where boto3 prefers CRT.

    Without an explicit transfer client the CRT path takes over here, addresses
    ``my-loras.s3.amazonaws.com`` instead of the configured endpoint, and the
    stub below receives nothing.
    """
    _force_crt_optimized_host(monkeypatch)

    _service(s3_stub).upload_lora(str(adapter_dir))

    assert s3_stub.objects == {
        f"{LORA_NAME}/{ADAPTER_NAME}": ADAPTER_BYTES,
    }
