# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Optional storage checks precede runtime imports.

import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

opendal = pytest.importorskip("opendal")
fernet = pytest.importorskip("cryptography.fernet")
Fernet = fernet.Fernet

from dynamo.artifacts.runtime import (
    ArtifactRuntime,
    close_configured_artifact_runtime,
    configured_artifact_runtime,
    initialize_configured_artifact_runtime,
)
from dynamo.artifacts.storage import ArtifactStorageError, ManagedProfile

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(15),
]


@pytest.fixture
def runtime_config(tmp_path):
    return {
        "owner": "test-domain",
        "operation_db": str(tmp_path / "operations.db"),
        "profiles": {
            "approved": {
                "immutable": True,
                "service": "memory",
                "options": {},
                "allowed_prefixes": ["artifacts"],
            }
        },
        "storage_policy": {"allowed_hosts": ["objects.example"]},
    }


@pytest_asyncio.fixture
async def runtime(runtime_config):
    instance = ArtifactRuntime(runtime_config, Fernet.generate_key())
    yield instance
    await instance.close()


@pytest.fixture
def managed():
    return {
        "kind": "managed_store",
        "profile": "approved",
        "object_key": "artifacts/result",
    }


@pytest.fixture
def delegated():
    return {
        "kind": "presigned_http_put",
        "url": "https://objects.example/artifact?signature=SECRET_SENTINEL",
        "required_headers": {"if-none-match": "*"},
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "object_id": "object-42",
        "max_bytes": 1024,
        "verification_ref": "verifier-42",
    }


@pytest.fixture(autouse=True)
def configured_cache():
    configured_artifact_runtime.cache_clear()
    yield
    configured_artifact_runtime.cache_clear()


@pytest.mark.asyncio
async def test_shutdown_does_not_initialize_unused_authority(monkeypatch):
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", "/does-not-exist/authority.json")
    await close_configured_artifact_runtime()
    assert configured_artifact_runtime.cache_info().currsize == 0


@pytest.mark.asyncio
async def test_shutdown_handles_cached_disabled_authority(monkeypatch):
    monkeypatch.delenv("DYNAMO_ARTIFACT_CONFIG", raising=False)
    assert configured_artifact_runtime() is None
    await close_configured_artifact_runtime()


def test_managed_preflight_and_target_are_nonmutating(runtime, managed):
    original = dict(managed)
    runtime.validate(managed, "object_store")
    assert runtime._target(managed).profile == "approved"
    assert managed == original
    assert runtime._binding(managed) == ("approved", "artifacts/result")


@pytest.mark.asyncio
async def test_shutdown_rejects_new_submit_without_waiting_for_drain(
    runtime, managed, monkeypatch
):
    draining = asyncio.Event()
    finish_drain = asyncio.Event()

    async def slow_drain():
        draining.set()
        await finish_drain.wait()

    monkeypatch.setattr(runtime.service, "drain", slow_drain)
    closing = asyncio.create_task(runtime.close())
    try:
        await asyncio.wait_for(draining.wait(), timeout=1)
        with pytest.raises(ValueError, match="closed"):
            await asyncio.wait_for(runtime.submit(b"data", [], managed), timeout=0.1)
    finally:
        finish_drain.set()
        await closing


@pytest.mark.parametrize(
    "destination,mode,match",
    [
        ({"kind": "unknown"}, "object_store", "unsupported"),
        (
            {
                "kind": "managed_store",
                "profile": "unknown",
                "object_key": "artifacts/a",
            },
            "object_store",
            "authorized",
        ),
        (
            {
                "kind": "managed_store",
                "profile": "approved",
                "object_key": "elsewhere/a",
            },
            "object_store",
            "prefix",
        ),
        (
            {
                "kind": "managed_store",
                "profile": "approved",
                "object_key": "artifacts/a",
            },
            "relay",
            "relay",
        ),
    ],
)
def test_preflight_rejects_bad_destinations(runtime, destination, mode, match):
    with pytest.raises((ValueError, ArtifactStorageError), match=match):
        runtime.validate(destination, mode)


def test_delegated_verifier_must_bind_exact_put_url(runtime, delegated):
    with pytest.raises(ValueError, match="registration"):
        runtime.validate(delegated, "object_store")
    runtime.verifiers = {
        "verifier-42": {
            "put_url_sha256": hashlib.sha256(delegated["url"].encode()).hexdigest(),
            "profile": "approved",
            "object_key": "artifacts/result",
        }
    }
    runtime.validate(delegated, "object_store")
    assert runtime._binding(delegated) == ("approved", "artifacts/result")
    with pytest.raises(ValueError, match="bind"):
        runtime.validate(
            {**delegated, "url": delegated["url"] + "&other=1"}, "object_store"
        )


def test_delegated_verifier_cannot_escape_profile_prefix(runtime, delegated):
    runtime.verifiers = {
        "verifier-42": {
            "put_url_sha256": hashlib.sha256(delegated["url"].encode()).hexdigest(),
            "profile": "approved",
            "object_key": "private/result",
        }
    }
    with pytest.raises(ArtifactStorageError, match="prefix"):
        runtime.validate(delegated, "object_store")


def test_dynamic_verifier_accepts_fresh_grants_for_confined_keys(
    runtime_config, delegated
):
    runtime_config["verifiers"] = {
        "verifier-42": {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket/artifacts/",
            "object_key_prefix": "artifacts",
        }
    }
    runtime = ArtifactRuntime(runtime_config, Fernet.generate_key())
    try:
        first = {
            **delegated,
            "url": "https://objects.example/bucket/artifacts/one?signature=first",
        }
        second = {
            **delegated,
            "url": "https://objects.example/bucket/artifacts/two?signature=second",
        }
        runtime.validate(first, "object_store")
        runtime.validate(second, "object_store")
        assert runtime._binding(first) == ("approved", "artifacts/one")
        assert runtime._binding(second) == ("approved", "artifacts/two")
    finally:
        asyncio.run(runtime.close())


@pytest.mark.parametrize(
    "url",
    [
        "https://objects.example/bucket/elsewhere/one?signature=x",
        "https://other.example/bucket/artifacts/one?signature=x",
        "https://objects.example/bucket/artifacts/%2e%2e/secret?signature=x",
        "https://objects.example/bucket/artifacts/../secret?signature=x",
        "https://objects.example/bucket/artifacts//secret?signature=x",
    ],
)
def test_dynamic_verifier_rejects_unbound_or_ambiguous_paths(
    runtime_config, delegated, url
):
    runtime_config["storage_policy"]["allowed_hosts"].append("other.example")
    runtime_config["verifiers"] = {
        "verifier-42": {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket/artifacts/",
            "object_key_prefix": "artifacts",
        }
    }
    runtime = ArtifactRuntime(runtime_config, Fernet.generate_key())
    try:
        with pytest.raises((ValueError, ArtifactStorageError)):
            runtime.validate({**delegated, "url": url}, "object_store")
    finally:
        asyncio.run(runtime.close())


@pytest.mark.parametrize(
    "verifier",
    [
        {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket/",
            "object_key_prefix": "private",
        },
        {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket",
            "object_key_prefix": "artifacts",
        },
        {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket/?query=x",
            "object_key_prefix": "artifacts",
        },
        {
            "profile": "approved",
            "url_prefix": "https://unapproved.example/bucket/",
            "object_key_prefix": "artifacts",
        },
        {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket/../artifacts/",
            "object_key_prefix": "artifacts",
        },
    ],
)
def test_dynamic_verifier_rejects_unsafe_configuration(runtime_config, verifier):
    runtime_config["verifiers"] = {"verifier-42": verifier}
    with pytest.raises(ValueError, match="configuration is invalid"):
        ArtifactRuntime(runtime_config, Fernet.generate_key())


@pytest.mark.asyncio
async def test_dynamic_verifier_reads_the_url_bound_key(runtime_config, delegated):
    runtime_config["verifiers"] = {
        "verifier-42": {
            "profile": "approved",
            "url_prefix": "https://objects.example/bucket/artifacts/",
            "object_key_prefix": "artifacts",
        }
    }
    runtime = ArtifactRuntime(runtime_config, Fernet.generate_key())
    target = {
        **delegated,
        "url": "https://objects.example/bucket/artifacts/new-object?signature=fresh",
    }
    try:
        await runtime.profiles["approved"].operator.write(
            "artifacts/new-object", b"opaque"
        )
        verified = await runtime._verify(b"opaque", target, {})
        assert verified["sha256"] == hashlib.sha256(b"opaque").hexdigest()
        assert verified["object_id"] == delegated["object_id"]
    finally:
        await runtime.close()


def test_delegated_egress_policy_applies_before_generation(runtime, delegated):
    with pytest.raises(ArtifactStorageError, match="host"):
        runtime.validate(
            {**delegated, "url": "https://unapproved.example/a"}, "object_store"
        )


def test_config_requires_certified_immutable_profile(runtime_config):
    config = {**runtime_config, "profiles": {"bad": {"immutable": False}}}
    with pytest.raises(ValueError, match="configuration is invalid"):
        ArtifactRuntime(config, Fernet.generate_key())


def test_missing_opendal_dependency_is_actionable(
    runtime_config, monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "opendal", None)
    with pytest.raises(RuntimeError, match=r"ai-dynamo\[artifact-delivery\]"):
        ArtifactRuntime(runtime_config, Fernet.generate_key())
    path = tmp_path / "artifact-config.json"
    path.write_text(json.dumps(runtime_config))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(path))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CAPABILITY_KEY", Fernet.generate_key().decode())
    with pytest.raises(RuntimeError, match=r"ai-dynamo\[artifact-delivery\]"):
        configured_artifact_runtime()


@pytest.mark.asyncio
async def test_runtime_configures_terminal_receipt_retention(runtime_config):
    configured = {**runtime_config, "terminal_ttl_seconds": 120}
    runtime = ArtifactRuntime(configured, Fernet.generate_key())
    try:
        assert runtime.store.terminal_ttl_seconds == 120
    finally:
        await runtime.close()
    with pytest.raises(ValueError, match="configuration is invalid"):
        ArtifactRuntime(
            {**configured, "terminal_ttl_seconds": 0}, Fernet.generate_key()
        )


@pytest.mark.parametrize(
    "prefixes", ["artifacts", {"artifacts": True}, ["artifacts", 7]]
)
def test_config_rejects_non_sequence_prefixes(runtime_config, prefixes):
    runtime_config["profiles"]["approved"]["allowed_prefixes"] = prefixes
    with pytest.raises(ValueError, match="configuration is invalid"):
        ArtifactRuntime(runtime_config, Fernet.generate_key())


@pytest.mark.asyncio
async def test_real_memory_operator_transfer_verify_and_ready(runtime, managed):
    data = b"opaque byte fixture"
    result = await runtime.submit(data, [], managed)
    assert result["state"] == "pending"
    await runtime.service.drain()
    terminal = runtime.status(result["operation_id"])
    assert terminal["state"] == "ready"
    assert terminal["actual_bytes"] == len(data)
    assert terminal["sha256"] == hashlib.sha256(data).hexdigest()
    assert re.fullmatch(r"obj_[0-9a-f]{64}", terminal["object_id"])
    assert "approved" not in terminal["object_id"]
    assert "artifacts/result" not in terminal["object_id"]
    assert (await runtime._verify(data, managed, {}))["object_id"] == terminal[
        "object_id"
    ]
    assert await runtime.profiles["approved"].operator.read("artifacts/result") == data


@pytest.mark.asyncio
async def test_transfer_receipt_does_not_expose_outcome_as_ready(runtime, managed):
    receipt = await runtime._transfer(b"opaque", managed)
    assert set(receipt) == {"actual_bytes", "sha256", "object_id", "provider_identity"}
    assert receipt["actual_bytes"] == 6
    assert re.fullmatch(r"obj_[0-9a-f]{32}", receipt["object_id"])
    assert "state" not in receipt


@pytest.mark.asyncio
async def test_different_stored_bytes_leave_commit_unconfirmed(runtime, managed):
    await runtime.profiles["approved"].operator.write("artifacts/result", b"wrong!")
    result = await runtime.submit(b"opaque", [], managed)
    await runtime.service.drain()
    terminal = runtime.status(result["operation_id"])
    assert terminal["state"] == "failed"
    assert terminal["error_code"] == "commit_unknown"
    assert "sha256" not in terminal


@pytest.mark.asyncio
async def test_verifier_reads_exact_bounded_length(runtime, managed):
    operator = SimpleNamespace(
        stat=AsyncMock(return_value=SimpleNamespace(content_length=6)),
        read=AsyncMock(return_value=b"opaque"),
    )
    runtime.profiles = {"approved": ManagedProfile(operator, ("artifacts",))}
    result = await runtime._verify(
        b"opaque", managed, {"provider_identity": "version-1"}
    )
    operator.stat.assert_awaited_once_with("artifacts/result", version="version-1")
    operator.read.assert_awaited_once_with(
        "artifacts/result", offset=0, size=6, version="version-1"
    )
    assert result["actual_bytes"] == 6
    assert result["provider_identity"] == "version-1"
    assert result["sha256"] == hashlib.sha256(b"opaque").hexdigest()


@pytest.mark.asyncio
async def test_verifier_stat_length_mismatch_prevents_read(runtime, managed):
    operator = SimpleNamespace(
        stat=AsyncMock(return_value=SimpleNamespace(content_length=1000)),
        read=AsyncMock(),
    )
    runtime.profiles = {"approved": ManagedProfile(operator, ("artifacts",))}
    with pytest.raises(ValueError, match="unavailable"):
        await runtime._verify(b"opaque", managed, {})
    operator.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_verifier_provider_error_is_sanitized(runtime, managed):
    operator = SimpleNamespace(
        stat=AsyncMock(side_effect=opendal.exceptions.Unexpected("SECRET_SENTINEL"))
    )
    runtime.profiles = {"approved": ManagedProfile(operator, ("artifacts",))}
    with pytest.raises(ValueError, match="unavailable") as error:
        await runtime._verify(b"opaque", managed, {})
    assert "SECRET_SENTINEL" not in str(error.value)


@pytest.mark.asyncio
async def test_verifier_rejects_nonconforming_provider_read_size(runtime, managed):
    operator = SimpleNamespace(
        stat=AsyncMock(return_value=SimpleNamespace(content_length=6)),
        read=AsyncMock(return_value=b"too-long"),
    )
    runtime.profiles = {"approved": ManagedProfile(operator, ("artifacts",))}
    with pytest.raises(ValueError, match="bound"):
        await runtime._verify(b"opaque", managed, {})


@pytest.mark.asyncio
async def test_configured_runtime_is_cached(tmp_path, runtime_config, monkeypatch):
    path = tmp_path / "artifact-config.json"
    path.write_text(json.dumps(runtime_config))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(path))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CAPABILITY_KEY", Fernet.generate_key().decode())
    first = configured_artifact_runtime()
    assert isinstance(first, ArtifactRuntime)
    assert configured_artifact_runtime() is first
    path.write_text("invalid config after startup")
    assert configured_artifact_runtime() is first
    await first.close()


@pytest.mark.asyncio
async def test_startup_initializer_fails_before_first_artifact_request(
    tmp_path, runtime_config, monkeypatch
):
    path = tmp_path / "artifact-config.json"
    path.write_text(json.dumps(runtime_config))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(path))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CAPABILITY_KEY", Fernet.generate_key().decode())
    runtime = await initialize_configured_artifact_runtime()
    assert runtime is configured_artifact_runtime()
    await runtime.close()
    configured_artifact_runtime.cache_clear()
    path.write_text("invalid config")
    with pytest.raises(ValueError, match="configuration is invalid"):
        await initialize_configured_artifact_runtime()


@pytest.mark.asyncio
async def test_only_one_authority_can_recover_database(runtime_config):
    key = Fernet.generate_key()
    first = ArtifactRuntime(runtime_config, key)
    try:
        pending = first.store.create("test-domain", [], {})
        with pytest.raises(ValueError, match="configuration is invalid"):
            ArtifactRuntime(runtime_config, key)
        assert first.status(pending["operation_id"])["state"] == "pending"
    finally:
        await first.close()
    reopened = ArtifactRuntime(runtime_config, key)
    try:
        result = reopened.status(pending["operation_id"])
        assert result["state"] == "failed"
        assert result["error_code"] == "source_lost"
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_close_drains_transfers_before_releasing_authority(
    runtime_config, managed
):
    key = Fernet.generate_key()
    runtime = ArtifactRuntime(runtime_config, key)
    pending = await runtime.submit(b"opaque", [], managed)
    await runtime.close()
    assert runtime.status(pending["operation_id"])["state"] == "ready"
    reopened = ArtifactRuntime(runtime_config, key)
    try:
        assert reopened.status(pending["operation_id"])["state"] == "ready"
    finally:
        await reopened.close()
    await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_close_releases_authority_after_drain(runtime_config):
    key = Fernet.generate_key()
    runtime = ArtifactRuntime(runtime_config, key)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def drain():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    runtime.service.drain = drain
    try:
        closing = asyncio.create_task(runtime.close())
        await entered.wait()
        closing.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        reopened = ArtifactRuntime(runtime_config, key)
        await reopened.close()
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_closed_runtime_cannot_submit_without_authority(runtime_config, managed):
    runtime = ArtifactRuntime(runtime_config, Fernet.generate_key())
    await runtime.close()
    try:
        with pytest.raises((ValueError, RuntimeError), match="closed|authority"):
            await runtime.submit(b"opaque", [], managed)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_closing_runtime_rejects_new_work_while_draining(runtime_config, managed):
    runtime = ArtifactRuntime(runtime_config, Fernet.generate_key())
    entered = asyncio.Event()
    release = asyncio.Event()
    original_drain = runtime.service.drain

    async def drain():
        entered.set()
        await release.wait()

    runtime.service.drain = drain
    closing = asyncio.create_task(runtime.close())
    try:
        await entered.wait()
        submitting = asyncio.create_task(runtime.submit(b"opaque", [], managed))
        await asyncio.sleep(0)
        release.set()
        await closing
        with pytest.raises(
            (ValueError, RuntimeError), match="closed|closing|authority"
        ):
            await submitting
    finally:
        release.set()
        await closing
        await original_drain()


@pytest.mark.parametrize("document", ["invalid-json", "{}", "null", "[]"])
def test_configured_invalid_config_is_safe(tmp_path, monkeypatch, document):
    path = tmp_path / "invalid.json"
    path.write_text(document)
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(path))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CAPABILITY_KEY", Fernet.generate_key().decode())
    with pytest.raises(ValueError, match="configuration is invalid"):
        configured_artifact_runtime()


def test_configured_missing_key_is_safe(tmp_path, runtime_config, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(runtime_config))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(path))
    monkeypatch.delenv("DYNAMO_ARTIFACT_CAPABILITY_KEY", raising=False)
    with pytest.raises(ValueError, match="configuration is invalid"):
        configured_artifact_runtime()


@pytest.mark.parametrize(
    "profiles",
    [
        "SECRET_SENTINEL",
        {
            "bad": {
                "immutable": True,
                "service": "unsupported",
                "options": {},
                "allowed_prefixes": ["artifacts"],
            }
        },
    ],
)
def test_configured_provider_errors_are_sanitized(
    tmp_path, runtime_config, monkeypatch, profiles
):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**runtime_config, "profiles": profiles}))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CONFIG", str(path))
    monkeypatch.setenv("DYNAMO_ARTIFACT_CAPABILITY_KEY", Fernet.generate_key().decode())
    with pytest.raises(ValueError, match="configuration is invalid") as error:
        configured_artifact_runtime()
    assert "SECRET_SENTINEL" not in str(error.value)
