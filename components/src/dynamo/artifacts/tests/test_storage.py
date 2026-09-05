# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from dynamo.artifacts.storage import (
    ArtifactStorageError,
    ManagedFsspecTarget,
    PresignedHttpPutTarget,
    put_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def _allow_test_presigned_hosts(monkeypatch) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_PRESIGNED_HOSTS",
        "storage.example,example.test",
    )
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", "true")


@pytest.mark.asyncio
async def test_managed_fsspec_writes_exact_profile_object(monkeypatch) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"training":{"url":"s3://artifacts/run","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    payload = b"artifact-bytes"
    stored = {}
    session = MagicMock()
    session.__aexit__ = AsyncMock()

    async def pipe_file(path, data, mode):
        assert mode == "create"
        if path in stored:
            raise FileExistsError(path)
        stored[path] = data

    filesystem = SimpleNamespace(
        protocol="s3",
        async_impl=True,
        set_session=AsyncMock(return_value=session),
        _pipe_file=pipe_file,
    )

    with patch(
        "dynamo.artifacts.storage.url_to_fs",
        return_value=(filesystem, "artifacts/run"),
    ):
        receipt = await put_artifact(
            payload,
            ManagedFsspecTarget(
                profile="training", object_key="request-1/output.dynexp"
            ),
        )
        with pytest.raises(ArtifactStorageError, match="managed artifact write failed"):
            await put_artifact(
                payload,
                ManagedFsspecTarget(
                    profile="training", object_key="request-1/output.dynexp"
                ),
            )

    assert stored["artifacts/run/request-1/output.dynexp"] == payload
    assert receipt.actual_bytes == len(payload)
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    assert receipt.object_id == "training:request-1/output.dynexp"


@pytest.mark.asyncio
@pytest.mark.parametrize("object_key", ["", "/absolute", "../escape", "a/../b", "a\\b"])
async def test_managed_fsspec_rejects_unsafe_object_keys(
    monkeypatch, object_key: str
) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"training":{"url":"s3://artifacts/run","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    with pytest.raises(ArtifactStorageError, match="object_key"):
        await put_artifact(
            b"data", ManagedFsspecTarget(profile="training", object_key=object_key)
        )


@pytest.mark.asyncio
async def test_presigned_put_preserves_exact_url_and_hides_capability() -> None:
    url = "https://storage.example/object%2Fname?X-Signature=secret&part=1&part=2"
    fs = MagicMock()
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

    with patch("dynamo.artifacts.storage._ExactHttpPutFileSystem", return_value=fs):
        receipt = await put_artifact(payload, target)

    fs.pipe_file.assert_called_once_with(
        url,
        payload,
        headers={"content-type": "application/octet-stream", "if-none-match": "*"},
        allow_redirects=False,
    )
    assert "secret" not in repr(target)
    assert receipt.object_id == "opaque-1"


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
        patch("dynamo.artifacts.storage._ExactHttpPutFileSystem", return_value=fs),
        pytest.raises(ArtifactStorageError, match="max_bytes"),
    ):
        await put_artifact(b"four", target)
    fs.pipe_file.assert_not_called()


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
        patch("dynamo.artifacts.storage.datetime", clock),
        patch("dynamo.artifacts.storage._ExactHttpPutFileSystem", return_value=fs),
        pytest.raises(ArtifactStorageError, match="expired"),
    ):
        await put_artifact(b"data", target)
    fs.pipe_file.assert_not_called()


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


def test_presigned_put_allows_explicit_local_test_host(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "true")
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", "127.0.0.1")
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    target = PresignedHttpPutTarget(
        url="http://127.0.0.1:9000/bucket/key",
        max_bytes=1024,
        expires_at=expires_at,
        required_headers={"if-none-match": "*"},
        object_id="local-test",
    )
    assert target.object_id == "local-test"


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
    fs = MagicMock()
    fs.pipe_file.side_effect = RuntimeError(capability_sentinel)
    with (
        patch("dynamo.artifacts.storage._ExactHttpPutFileSystem", return_value=fs),
        pytest.raises(
            ArtifactStorageError, match="presigned artifact PUT failed"
        ) as error,
    ):
        await put_artifact(b"data", target)
    assert capability_sentinel not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.asyncio
async def test_managed_profile_and_limit_failures_are_explicit(monkeypatch) -> None:
    target = ManagedFsspecTarget(profile="missing", object_key="output.dynexp")
    with pytest.raises(ArtifactStorageError, match="profile is unknown"):
        await put_artifact(b"data", target)

    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_BYTES", "invalid")
    with pytest.raises(ArtifactStorageError, match="byte limit"):
        await put_artifact(b"data", target)

    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_BYTES", "3")
    with pytest.raises(ArtifactStorageError, match="byte limit"):
        await put_artifact(b"four", target)


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
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", "127.0.0.1")
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
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", "127.0.0.1")
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


@pytest.mark.asyncio
async def test_managed_target_is_operator_gated_and_prefix_scoped(monkeypatch) -> None:
    monkeypatch.delenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", raising=False)
    target = ManagedFsspecTarget(profile="training", object_key="run/output.dynexp")
    with pytest.raises(ArtifactStorageError, match="not enabled"):
        await put_artifact(b"data", target)

    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", "true")
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"training":{"url":"memory://artifacts/root","allowed_prefixes":["authorized"],"create_only":true}}',
    )
    with pytest.raises(ArtifactStorageError, match="prefix"):
        await put_artifact(b"data", target)
