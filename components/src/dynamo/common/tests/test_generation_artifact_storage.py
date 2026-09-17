# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from dynamo.common.generation_artifact_storage import (
    ArtifactStorageError,
    PresignedHttpPutTarget,
    put_artifact,
    target_from_settings,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def _allow_test_presigned_hosts(monkeypatch) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_PRESIGNED_HOSTS",
        "storage.example,example.test",
    )


@pytest.mark.asyncio
async def test_presigned_put_preserves_exact_url_and_hides_capability() -> None:
    url = "https://storage.example/object%2Fname?X-Signature=secret&part=1&part=2"
    fs = SimpleNamespace(
        _pipe_file=AsyncMock(),
        aclose=AsyncMock(),
    )
    payload = b"artifact"
    target = PresignedHttpPutTarget(
        url=url,
        max_bytes=1024,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        required_headers={
            "content-type": "application/octet-stream",
            "if-none-match": "*",
        },
        object_id="opaque-1",
    )

    with patch(
        "dynamo.common.generation_artifact_storage._ExactHttpPutFileSystem",
        return_value=fs,
    ):
        receipt = await put_artifact(payload, target)

    fs._pipe_file.assert_awaited_once_with(
        url,
        payload,
        headers={"content-type": "application/octet-stream", "if-none-match": "*"},
        allow_redirects=False,
    )
    assert "secret" not in repr(target)
    assert receipt.object_id == "opaque-1"


@pytest.mark.asyncio
async def test_presigned_put_closes_each_request_filesystem() -> None:
    target = PresignedHttpPutTarget(
        url="https://storage.example/object?signature=secret",
        max_bytes=1024,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        required_headers={"if-none-match": "*"},
        object_id="opaque-1",
    )
    filesystems = []

    def filesystem_factory(**kwargs):
        filesystem = SimpleNamespace(
            _pipe_file=AsyncMock(),
            aclose=AsyncMock(),
        )
        filesystems.append(filesystem)
        return filesystem

    with patch(
        "dynamo.common.generation_artifact_storage._ExactHttpPutFileSystem",
        side_effect=filesystem_factory,
    ):
        await put_artifact(b"first", target)
        await put_artifact(b"second", target)

    assert len(filesystems) == 2
    for filesystem in filesystems:
        filesystem.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_presigned_put_rejects_oversize_before_network() -> None:
    fs = MagicMock()
    target = PresignedHttpPutTarget(
        url="https://storage.example/object?signature=secret",
        max_bytes=3,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        required_headers={"if-none-match": "*"},
        object_id="opaque-1",
    )
    with (
        patch(
            "dynamo.common.generation_artifact_storage._ExactHttpPutFileSystem",
            return_value=fs,
        ),
        pytest.raises(ArtifactStorageError, match="max_bytes"),
    ):
        await put_artifact(b"four", target)
    fs._pipe_file.assert_not_called()


@pytest.mark.asyncio
async def test_presigned_put_rechecks_expiry_immediately_before_network() -> None:
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    target = PresignedHttpPutTarget(
        url="https://storage.example/object?signature=secret",
        max_bytes=1024,
        expires_at=expires.isoformat(),
        required_headers={"if-none-match": "*"},
        object_id="opaque-1",
    )
    fs = MagicMock()
    clock = MagicMock()
    clock.fromisoformat.side_effect = datetime.fromisoformat
    clock.now.return_value = expires + timedelta(seconds=1)
    with (
        patch("dynamo.common.generation_artifact_storage.datetime", clock),
        patch(
            "dynamo.common.generation_artifact_storage._ExactHttpPutFileSystem",
            return_value=fs,
        ),
        pytest.raises(ArtifactStorageError, match="expired"),
    ):
        await put_artifact(b"data", target)
    fs._pipe_file.assert_not_called()


def test_presigned_put_rejects_insecure_url_and_unapproved_headers() -> None:
    with pytest.raises(ArtifactStorageError, match="HTTPS"):
        PresignedHttpPutTarget(
            url="http://storage.example/object", max_bytes=1024, object_id="opaque"
        )
    with pytest.raises(ArtifactStorageError, match="header"):
        PresignedHttpPutTarget(
            url="https://storage.example/object",
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            required_headers={"authorization": "secret", "if-none-match": "*"},
            object_id="opaque",
        )


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    [
        ("x-ms-blob-type", "BlockBlob"),
        ("x-ms-version", "2025-05-05"),
        ("x-goog-content-sha256", "UNSIGNED-PAYLOAD"),
    ],
)
def test_presigned_target_accepts_provider_signed_headers(
    header_name: str, header_value: str
) -> None:
    target = PresignedHttpPutTarget(
        url="https://storage.example/object",
        max_bytes=1024,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        required_headers={header_name: header_value, "if-none-match": "*"},
        object_id="opaque",
    )

    assert target.required_headers[header_name] == header_value


@pytest.mark.parametrize(
    "header_name",
    [
        "destination",
        "x-amz-copy-source",
        "x-goog-copy-source-generation",
        "x-http-method-override",
        "x-ms-copy-source",
        "x-original-url",
        "x-rewrite-url",
    ],
)
def test_presigned_target_rejects_semantic_override_headers(
    header_name: str,
) -> None:
    with pytest.raises(ArtifactStorageError, match="header is not allowed"):
        PresignedHttpPutTarget(
            url="https://storage.example/object",
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            required_headers={header_name: "untrusted", "if-none-match": "*"},
            object_id="opaque",
        )


def test_presigned_put_rejects_unapproved_header_without_reflecting_name() -> None:
    untrusted_name = "x" * 65536

    with pytest.raises(ArtifactStorageError) as exc_info:
        PresignedHttpPutTarget(
            url="https://storage.example/object",
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            required_headers={untrusted_name: "value", "if-none-match": "*"},
            object_id="opaque",
        )

    assert str(exc_info.value) == "presigned target header is not allowed"
    assert untrusted_name not in str(exc_info.value)


def test_insecure_http_test_target_requires_exact_authority(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "true")
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", "127.0.0.1:9000")
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    target = PresignedHttpPutTarget(
        url="http://127.0.0.1:9000/bucket/key",
        max_bytes=1024,
        expires_at=expires_at,
        required_headers={"if-none-match": "*"},
        object_id="local-test",
    )
    assert target.object_id == "local-test"

    with pytest.raises(ArtifactStorageError, match="HTTPS"):
        PresignedHttpPutTarget(
            url="http://127.0.0.1:9001/bucket/key",
            max_bytes=1024,
            expires_at=expires_at,
            required_headers={"if-none-match": "*"},
            object_id="wrong-port",
        )
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", "example.test:9000"
    )
    with pytest.raises(ArtifactStorageError, match="HTTPS"):
        PresignedHttpPutTarget(
            url="http://example.test:9000/bucket/key",
            max_bytes=1024,
            expires_at=expires_at,
            required_headers={"if-none-match": "*"},
            object_id="non-loopback",
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/object",
        "https://169.254.169.254/latest/meta-data",
        "https://not-allowed.example/object",
        "https://storage.example/object#fragment",
        "https://storage.example/object?bad=%GG",
    ],
)
def test_presigned_put_rejects_unapproved_or_ambiguous_destinations(url) -> None:
    with pytest.raises(ArtifactStorageError):
        PresignedHttpPutTarget(
            url=url,
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            required_headers={"if-none-match": "*"},
            object_id="opaque",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "https://user:password@example.test/x"},
        {"max_bytes": 0},
        {"max_bytes": "1024"},
        {"required_headers": []},
        {"required_headers": {"content-type": "bad\nvalue"}},
        {"expires_at": "not-a-date"},
        {"expires_at": "2020-01-01T00:00:00Z"},
    ],
)
def test_presigned_target_rejects_invalid_capability_fields(kwargs) -> None:
    values = {
        "url": "https://example.test/object",
        "max_bytes": 1024,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "required_headers": {"if-none-match": "*"},
        "object_id": "opaque",
        **kwargs,
    }
    with pytest.raises(ArtifactStorageError):
        PresignedHttpPutTarget(**values)


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "presigned_http_put", "max_bytes": 1024, "object_id": "id"},
        {
            "kind": "presigned_http_put",
            "url": "https://example.test/x",
            "max_bytes": "1024",
            "object_id": "id",
        },
    ],
)
def test_target_from_settings_rejects_missing_or_mistyped_required_fields(
    target,
) -> None:
    with pytest.raises(ArtifactStorageError):
        target_from_settings({"delivery": {"mode": "object_store", "target": target}})


@pytest.mark.asyncio
async def test_provider_errors_are_sanitized(monkeypatch) -> None:
    capability_sentinel = "url-capability-sentinel"
    target = PresignedHttpPutTarget(
        url=f"https://example.test/object?signature={capability_sentinel}",
        max_bytes=1024,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        required_headers={"if-none-match": "*"},
        object_id="opaque",
    )
    fs = SimpleNamespace(
        _pipe_file=AsyncMock(side_effect=RuntimeError(capability_sentinel)),
        aclose=AsyncMock(),
    )
    with (
        patch(
            "dynamo.common.generation_artifact_storage._ExactHttpPutFileSystem",
            return_value=fs,
        ),
        pytest.raises(
            ArtifactStorageError, match="presigned artifact PUT failed"
        ) as error,
    ):
        await put_artifact(b"data", target)
    assert capability_sentinel not in str(error.value)
    assert error.value.__cause__ is None


async def _start_http_server(handler):
    application = web.Application()
    application.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return SimpleNamespace(runner=runner, port=port)


@pytest.mark.asyncio
async def test_real_http_put_preserves_exact_target_and_body(monkeypatch) -> None:
    observed = {}

    async def receive(request):
        observed["method"] = request.method
        observed["raw_path"] = request.raw_path
        observed["content_type"] = request.headers["content-type"]
        observed["body"] = await request.read()
        return web.Response(status=200)

    server = await _start_http_server(receive)
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "true")
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", f"127.0.0.1:{server.port}"
    )
    try:
        url = (
            f"http://127.0.0.1:{server.port}/object%2Fpart"
            "?signature=secret&part=1&part=2"
        )
        receipt = await put_artifact(
            b"exact-bytes",
            PresignedHttpPutTarget(
                url=url,
                max_bytes=1024,
                expires_at=(
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                ).isoformat(),
                required_headers={
                    "content-type": "application/octet-stream",
                    "if-none-match": "*",
                },
                object_id="exact-http",
            ),
        )
    finally:
        await server.runner.cleanup()

    assert observed == {
        "method": "PUT",
        "raw_path": "/object%2Fpart?signature=secret&part=1&part=2",
        "content_type": "application/octet-stream",
        "body": b"exact-bytes",
    }
    assert receipt.object_id == "exact-http"


@pytest.mark.asyncio
async def test_real_http_put_rejects_redirect_without_following(monkeypatch) -> None:
    redirected = False

    async def redirect(request):
        return web.Response(status=307, headers={"location": "/redirected"})

    async def destination(request):
        nonlocal redirected
        redirected = True
        return web.Response(status=200)

    application = web.Application()
    application.router.add_put("/source", redirect)
    application.router.add_put("/redirected", destination)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "true")
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", f"127.0.0.1:{port}"
    )
    try:
        target = PresignedHttpPutTarget(
            url=f"http://127.0.0.1:{port}/source",
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            required_headers={"if-none-match": "*"},
            object_id="redirect-test",
        )
        with pytest.raises(ArtifactStorageError, match="was not accepted"):
            await put_artifact(b"data", target)
    finally:
        await runner.cleanup()

    assert redirected is False


def test_presigned_target_requires_bounded_expiry_and_exact_nondefault_port(
    monkeypatch,
) -> None:
    with pytest.raises(ArtifactStorageError, match="expires_at"):
        PresignedHttpPutTarget(
            url="https://storage.example/object",
            max_bytes=1024,
            required_headers={"if-none-match": "*"},
            object_id="opaque",
        )
    with pytest.raises(ArtifactStorageError, match="lifetime"):
        PresignedHttpPutTarget(
            url="https://storage.example/object",
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            required_headers={"if-none-match": "*"},
            object_id="opaque",
        )
    with pytest.raises(ArtifactStorageError, match="allowlisted"):
        PresignedHttpPutTarget(
            url="https://storage.example:8443/object",
            max_bytes=1024,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            required_headers={"if-none-match": "*"},
            object_id="opaque",
        )
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_PRESIGNED_HOSTS", "storage.example:8443"
    )
    target = PresignedHttpPutTarget(
        url="https://storage.example:8443/object",
        max_bytes=1024,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        required_headers={"if-none-match": "*"},
        object_id="opaque",
    )
    assert target.object_id == "opaque"
