# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for LoRA adapter staging in the MinIO test fixture.

`MinioService.upload_lora` used to fail only on the H100 CI runners, with
`InvalidAccessKeyId` on `CreateMultipartUpload`, while MinIO was healthy and
bucket operations succeeded. The cause was boto3's managed-transfer client
selection: on network-optimized instances boto3 resolves `upload_file` to the
AWS CRT client, and that path ignores the client's `endpoint_url`, so the
adapter was addressed at real S3 rather than at MinIO.

The instance shape is the only host-dependent input, so these tests reproduce
the runner difference anywhere by patching
`awscrt.s3.is_optimized_for_system`, and assert where the uploaded bytes
actually land: a local HTTP stand-in for MinIO either receives the adapter or
it does not. No GPU, no Docker and no network access are required.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List

import pytest
from pytest_httpserver import HTTPServer

from tests.serve.lora_utils import MinioLoraConfig, MinioService

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

BUCKET = "my-loras"
LORA_NAME = "test-org/test-lora"
# Deliberately more than one file, one of them nested, because upload_lora walks
# the snapshot with rglob and builds a key per file.
ADAPTER_FILES = {
    "adapter_config.json": b'{"r": 8, "lora_alpha": 16}',
    "adapter_model.safetensors": b"\x00" * 4096,
    "nested/extra.bin": b"nested payload",
}


@dataclass
class S3Stub:
    """A local stand-in for MinIO that records what it was actually handed."""

    endpoint: str
    objects: Dict[str, bytes] = field(default_factory=dict)
    bucket_methods: List[str] = field(default_factory=list)


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

    def record_bucket(request: Request) -> Response:
        stub.bucket_methods.append(request.method)
        # 404 on HEAD so create_bucket has to follow up with its PUT, which puts
        # both plain-client calls on the record.
        status = 404 if request.method == "HEAD" else 200
        return Response("", status=status)

    for method in ("HEAD", "PUT"):
        server.expect_request(f"/{BUCKET}", method=method).respond_with_handler(
            record_bucket
        )
    server.expect_request(
        re.compile(rf"^/{re.escape(BUCKET)}/.+"), method="PUT"
    ).respond_with_handler(record_object)

    yield stub

    server.clear()
    server.stop()


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Path:
    """A minimal stand-in for a downloaded LoRA snapshot."""
    for relative_path, payload in ADAPTER_FILES.items():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return tmp_path


def _service(stub: S3Stub) -> MinioService:
    return MinioService(
        MinioLoraConfig(endpoint=stub.endpoint, bucket=BUCKET, lora_name=LORA_NAME)
    )


def _expected_objects() -> Dict[str, bytes]:
    return {f"{LORA_NAME}/{name}": payload for name, payload in ADAPTER_FILES.items()}


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
    from boto3.s3.transfer import TransferConfig, _should_use_crt

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

    assert s3_stub.objects == _expected_objects()


def test_upload_lora_unchanged_on_hosts_without_crt_preference(
    s3_stub: S3Stub, adapter_dir: Path
) -> None:
    """Negative control: the same upload with the instance check left alone.

    This is the ordinary-runner case these tests already passed on, and it must
    keep delivering every file to the configured endpoint.
    """
    _service(s3_stub).upload_lora(str(adapter_dir))

    assert s3_stub.objects == _expected_objects()


def test_bucket_operations_reach_endpoint_on_crt_optimized_host(
    s3_stub: S3Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: plain client calls were never affected by any of this.

    They do not go through a transfer manager, which is why bucket setup
    succeeded on the same runners where the adapter upload failed.
    """
    _force_crt_optimized_host(monkeypatch)

    _service(s3_stub).create_bucket()

    assert s3_stub.bucket_methods == ["HEAD", "PUT"]
