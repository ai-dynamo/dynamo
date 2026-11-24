# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import re
import socket
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from yarl import URL

opendal = pytest.importorskip("opendal")

from dynamo.artifacts.storage import (  # noqa: E402 -- optional dependency guard
    ArtifactStorageError,
    ManagedProfile,
    ManagedStoreTarget,
    PresignedHttpPutTarget,
    StoragePolicy,
    _PolicyResolver,
    put_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def target():
    return PresignedHttpPutTarget(
        url="https://objects.example/a%2Fb?x=1&x=2&signature=secret",
        required_headers={"If-None-Match": "*"},
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        max_bytes=1024,
        object_id="object-1",
    )


@pytest.fixture
def policy():
    return StoragePolicy(allowed_hosts=frozenset({"objects.example"}))


def test_secret_target_repr_is_redacted(target):
    assert "secret" not in repr(target)
    assert "objects.example" not in repr(target)


@pytest.mark.parametrize(
    "headers",
    [
        {"If-None-Match": "*", "if-none-match": "*"},
        {"If-None-Match": "*", "Host": "elsewhere"},
        {"If-None-Match": "*", "x-custom": "secret\r\ninjected: yes"},
        {"If-None-Match": "*", "Authorization": "secret"},
        {"If-None-Match": "*", "x-amz-copy-source": "secret"},
        {},
    ],
)
def test_unsafe_headers_are_rejected(target, headers):
    with pytest.raises(ArtifactStorageError) as error:
        replace(target, required_headers=headers)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("key", ["../a", "/a", "a//b", "a/./b", "a\\b", "", "a\x00b"])
def test_managed_keys_are_confined(key):
    with pytest.raises(ArtifactStorageError):
        ManagedStoreTarget(profile="approved", object_key=key)


@pytest.mark.asyncio
async def test_managed_conditional_write_and_local_receipt():
    operator = SimpleNamespace(
        capability=lambda: SimpleNamespace(write_with_if_not_exists=True),
        write=AsyncMock(),
    )
    profile = ManagedProfile(operator=operator, allowed_prefixes=("runs",))
    data = b"\x00opaque\xff"
    receipt = await put_artifact(
        data,
        ManagedStoreTarget(profile="approved", object_key="runs/a"),
        policy=StoragePolicy(),
        profiles={"approved": profile},
    )
    operator.write.assert_awaited_once_with("runs/a", data, if_not_exists=True)
    assert receipt.actual_bytes == len(data)
    assert receipt.sha256 == hashlib.sha256(data).hexdigest()
    assert receipt.outcome == "acknowledged"
    assert receipt.provider_identity is None
    assert re.fullmatch(r"obj_[0-9a-f]{32}", receipt.object_id)
    assert "approved" not in receipt.object_id
    assert "runs/a" not in receipt.object_id


@pytest.mark.asyncio
async def test_managed_prefix_is_checked_before_io():
    operator = SimpleNamespace(write=AsyncMock())
    profile = ManagedProfile(operator=operator, allowed_prefixes=("runs",))
    with pytest.raises(ArtifactStorageError, match="prefix"):
        await put_artifact(
            b"opaque",
            ManagedStoreTarget(profile="approved", object_key="runs-evil/a"),
            policy=StoragePolicy(),
            profiles={"approved": profile},
        )
    operator.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_provider_without_atomic_create_is_rejected():
    operator = SimpleNamespace(
        capability=lambda: SimpleNamespace(write_with_if_not_exists=False),
        write=AsyncMock(),
    )
    with pytest.raises(ArtifactStorageError, match="conditional"):
        await put_artifact(
            b"opaque",
            ManagedStoreTarget(profile="approved", object_key="runs/a"),
            policy=StoragePolicy(),
            profiles={"approved": ManagedProfile(operator, ("runs",))},
        )
    operator.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_presigned_signed_length_mismatch_rejected_before_connection(
    target, policy
):
    signed = replace(
        target, required_headers={"if-none-match": "*", "content-length": "9"}
    )
    with pytest.raises(ArtifactStorageError, match="length"):
        await put_artifact(b"a", signed, policy=policy)


@pytest.mark.asyncio
async def test_presigned_expiry_rechecked_before_connection(target, policy):
    expired = replace(target, expires_at="2000-01-01T00:00:00Z")
    with pytest.raises(ArtifactStorageError, match="expired"):
        await put_artifact(b"a", expired, policy=policy)


@pytest.mark.asyncio
async def test_resolver_rejects_private_dns_answer(policy):
    resolver = _PolicyResolver(policy)
    resolver.delegate.resolve = AsyncMock(
        return_value=[{"host": "169.254.169.254", "family": socket.AF_INET}]
    )
    with pytest.raises(ArtifactStorageError, match="address"):
        await resolver.resolve("objects.example")
    await resolver.close()


@pytest.mark.asyncio
async def test_resolver_explicit_private_network_does_not_allow_metadata(policy):
    resolver = _PolicyResolver(replace(policy, allowed_networks=("0.0.0.0/0",)))
    resolver.delegate.resolve = AsyncMock(
        return_value=[{"host": "169.254.169.254", "family": socket.AF_INET}]
    )
    with pytest.raises(ArtifactStorageError, match="address"):
        await resolver.resolve("objects.example")
    await resolver.close()


@pytest.mark.asyncio
async def test_resolver_allows_explicit_private_store_network(policy):
    resolver = _PolicyResolver(replace(policy, allowed_networks=("10.42.0.0/16",)))
    answer = {"host": "10.42.1.2", "family": socket.AF_INET}
    resolver.delegate.resolve = AsyncMock(return_value=[answer])
    assert await resolver.resolve("objects.example") == [answer]
    await resolver.close()


@pytest.mark.asyncio
async def test_real_http_transport_rejects_private_dns_answer_before_connect(
    target, policy, monkeypatch
):
    resolve = AsyncMock(
        return_value=[
            {
                "hostname": "objects.example",
                "host": "127.0.0.1",
                "port": 443,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
        ]
    )
    connect = AsyncMock(side_effect=AssertionError("private address was dialed"))
    monkeypatch.setattr(aiohttp.ThreadedResolver, "resolve", resolve)
    monkeypatch.setattr(aiohttp.TCPConnector, "_wrap_create_connection", connect)

    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(b"abc", target, policy=policy)

    assert error.value.code == "destination_denied"
    resolve.assert_awaited_once()
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_http_transport_rejects_private_ip_literal_before_connect(
    target, monkeypatch
):
    connect = AsyncMock(side_effect=AssertionError("private address was dialed"))
    monkeypatch.setattr(aiohttp.TCPConnector, "_wrap_create_connection", connect)
    literal_target = replace(target, url="https://127.0.0.1/object")

    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(
            b"abc",
            literal_target,
            policy=StoragePolicy(allowed_hosts=frozenset({"127.0.0.1"})),
        )

    assert error.value.code == "destination_denied"
    connect.assert_not_awaited()


@pytest.fixture
def http_transport(monkeypatch):
    response = SimpleNamespace(status=200, headers={"x-amz-version-id": "version-1"})
    response_context = AsyncMock()
    response_context.__aenter__.return_value = response
    session = SimpleNamespace(put=Mock(return_value=response_context))
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    factory = Mock(return_value=session_context)
    connector = Mock()
    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    monkeypatch.setattr(aiohttp, "TCPConnector", connector)
    return response, response_context, session, factory, connector


@pytest.mark.asyncio
async def test_http_preserves_exact_url_and_signed_headers(
    target, policy, http_transport
):
    _, _, session, factory, connector = http_transport
    signed = replace(
        target,
        required_headers={
            "if-none-match": "*",
            "content-length": "3",
            "x-custom": "signed",
        },
    )
    receipt = await put_artifact(b"abc", signed, policy=policy)
    session.put.assert_called_once_with(
        URL(target.url, encoded=True),
        data=b"abc",
        headers={"if-none-match": "*", "content-length": "3", "x-custom": "signed"},
        allow_redirects=False,
    )
    assert str(session.put.call_args.args[0]) == target.url
    assert factory.call_args.kwargs["trust_env"] is False
    assert factory.call_args.kwargs["auto_decompress"] is False
    assert connector.call_args.kwargs["use_dns_cache"] is False
    assert receipt.provider_identity == "version-1"
    assert receipt.outcome == "acknowledged"


@pytest.mark.asyncio
async def test_signed_length_preserves_its_header_spelling(
    target, policy, http_transport
):
    signed = replace(
        target, required_headers={"if-none-match": "*", "content-length": "003"}
    )
    await put_artifact(b"abc", signed, policy=policy)
    assert http_transport[2].put.call_args.kwargs["headers"]["content-length"] == "003"


@pytest.mark.parametrize(
    "status,outcome",
    [
        (302, "rejected"),
        (403, "rejected"),
        (409, "ambiguous"),
        (412, "ambiguous"),
        (503, "ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_http_non_success_is_not_ready(
    target, policy, http_transport, status, outcome
):
    response, _, _, _, _ = http_transport
    response.status = status
    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(b"abc", target, policy=policy)
    assert error.value.outcome == outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception", [aiohttp.ClientConnectionError, asyncio.TimeoutError]
)
async def test_http_lost_ack_is_safe_ambiguous(
    target, policy, http_transport, exception
):
    _, context, _, _, _ = http_transport
    context.__aenter__.side_effect = exception("bearer-secret")
    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(b"abc", target, policy=policy)
    assert error.value.outcome == "ambiguous"
    assert error.value.code == "commit_unknown"
    assert "bearer-secret" not in str(error.value)


@pytest.mark.parametrize(
    "exception,outcome",
    [
        (opendal.exceptions.PermissionDenied, "rejected"),
        (opendal.exceptions.ConditionNotMatch, "ambiguous"),
        (opendal.exceptions.Unexpected, "ambiguous"),
        (TimeoutError, "ambiguous"),
        (asyncio.TimeoutError, "ambiguous"),
    ],
)
@pytest.mark.asyncio
async def test_managed_failures_are_sanitized(exception, outcome):
    operator = SimpleNamespace(
        capability=lambda: SimpleNamespace(write_with_if_not_exists=True),
        write=AsyncMock(side_effect=exception("bearer-secret")),
    )
    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(
            b"opaque",
            ManagedStoreTarget("approved", "runs/a"),
            policy=StoragePolicy(),
            profiles={"approved": ManagedProfile(operator, ("runs",))},
        )
    assert error.value.outcome == outcome
    assert "bearer-secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_managed_timeout_settles_write_and_requires_reconciliation():
    cleanup = asyncio.Event()

    async def write(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()

    operator = SimpleNamespace(
        capability=lambda: SimpleNamespace(write_with_if_not_exists=True), write=write
    )
    with pytest.raises(ArtifactStorageError) as error:
        await put_artifact(
            b"opaque",
            ManagedStoreTarget("approved", "runs/a"),
            policy=StoragePolicy(timeout_seconds=0.01),
            profiles={"approved": ManagedProfile(operator, ("runs",))},
        )
    assert error.value.outcome == "ambiguous"
    assert error.value.code == "commit_unknown"
    assert cleanup.is_set()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@objects.example/a",
        "https://objects.example/a#secret",
        "https://objects.example/%zz",
        "https://objects.example:invalid/a",
        "https://objects.example/\r\nb",
        "https://objects.example/raw space",
        "https://objects.example/raw\\backslash",
        "https://objects.example/raw{brace}",
        'https://objects.example/raw"quote',
        "ftp://objects.example/a",
        " https://objects.example/a",
    ],
)
def test_invalid_urls_rejected_without_echo(target, url):
    with pytest.raises(ArtifactStorageError) as error:
        replace(target, url=url)
    assert url not in str(error.value)


@pytest.mark.parametrize("max_bytes", [True, 0, -1, 64 * 1024 * 1024 + 1, "1"])
def test_invalid_limits(target, max_bytes):
    with pytest.raises(ArtifactStorageError):
        replace(target, max_bytes=max_bytes)


@pytest.mark.parametrize(
    "expires_at",
    ["2026-10-11", "2026-10-11 10:00:00+00:00", "2026-10-11T10:00:00", "invalid"],
)
def test_expiry_requires_rfc3339(target, expires_at):
    with pytest.raises(ArtifactStorageError):
        replace(target, expires_at=expires_at)


@pytest.mark.parametrize(
    ("expires_at", "microsecond"),
    [
        ("2026-10-11T10:00:00.1Z", 100000),
        ("2026-10-11T10:00:00.1234+00:00", 123400),
        ("2026-10-11T10:00:00.123456789Z", 123456),
    ],
)
def test_expiry_accepts_rfc3339_fractional_seconds(expires_at, microsecond):
    from dynamo.artifacts.storage import _expiry

    assert _expiry(expires_at).microsecond == microsecond


@pytest.mark.asyncio
async def test_presigned_requires_explicit_authority(target):
    with pytest.raises(ArtifactStorageError, match="host"):
        await put_artifact(b"a", target, policy=StoragePolicy())


@pytest.mark.asyncio
async def test_http_is_opt_in_even_with_private_network(target, policy):
    target = replace(target, url="http://objects.example/a")
    policy = replace(policy, allowed_networks=("127.0.0.0/8",))
    with pytest.raises(ArtifactStorageError, match="HTTPS"):
        await put_artifact(b"a", target, policy=policy)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.42.1.2",
        "169.254.169.254",
        "::ffff:169.254.169.254",
        "64:ff9b::a9fe:a9fe",
        "::a9fe:a9fe",
        "64:ff9b::a00:1",
        "::a00:1",
        "::127.0.0.1",
        "0.0.0.0",
        "224.0.0.1",
    ],
)
def test_addresses_denied_by_default(address):
    with pytest.raises(ArtifactStorageError, match="address"):
        StoragePolicy().check_address(address)


@pytest.mark.parametrize("address", ["64:ff9b::808:808", "::808:808"])
def test_public_embedded_ipv4_addresses_allowed(address):
    StoragePolicy().check_address(address)


def test_native_ipv6_loopback_can_be_explicitly_allowed():
    StoragePolicy(allowed_networks=("::1/128",)).check_address("::1")


@pytest.mark.asyncio
async def test_operator_limit_rejected_before_io(target, policy, http_transport):
    with pytest.raises(ArtifactStorageError, match="operator byte"):
        await put_artifact(b"ab", target, policy=replace(policy, max_bytes=1))
    http_transport[3].assert_not_called()


@pytest.mark.asyncio
async def test_target_limit_rejected_before_io(target, policy, http_transport):
    with pytest.raises(ArtifactStorageError, match="target byte"):
        await put_artifact(b"ab", replace(target, max_bytes=1), policy=policy)
    http_transport[3].assert_not_called()
