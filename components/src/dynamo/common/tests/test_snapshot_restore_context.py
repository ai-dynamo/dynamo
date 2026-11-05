# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import pytest

from dynamo.common.snapshot import restore_context
from dynamo.common.snapshot.constants import (
    KUBERNETES_OPTIONAL_ENV_NAMES,
    KUBERNETES_REQUIRED_ENV_NAMES,
    RESTORE_RUNTIME_ENV_NAMES,
    SNAPSHOT_CONTROL_DIR_ENV,
    SNAPSHOT_RESTORE_CONTEXT_FILE,
    SNAPSHOT_RESTORE_STANDBY_ENV,
)
from dynamo.common.snapshot.restore_context import (
    apply_snapshot_restore_env,
    refresh_snapshot_restore_config,
    write_snapshot_restore_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def clean_restore_env(monkeypatch, tmp_path):
    monkeypatch.setattr(
        restore_context, "_PID_ONE", tmp_path / "proc" / "1", raising=False
    )
    restore_env_names = {
        *KUBERNETES_REQUIRED_ENV_NAMES,
        *KUBERNETES_OPTIONAL_ENV_NAMES,
        *RESTORE_RUNTIME_ENV_NAMES,
        SNAPSHOT_CONTROL_DIR_ENV,
        SNAPSHOT_RESTORE_STANDBY_ENV,
    }
    for name in restore_env_names:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in restore_env_names:
        os.environ.pop(name, None)


def write_restore_context(monkeypatch, tmp_path, env):
    context_path = tmp_path / SNAPSHOT_RESTORE_CONTEXT_FILE
    context_path.write_text(json.dumps({"env": env}), encoding="utf-8")
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))
    return context_path


@pytest.fixture
def restore_proxy(monkeypatch, tmp_path):
    control_path = tmp_path / "custom-control"
    control_path.mkdir()
    proxy_binary = control_path / "snapshot-restore-proxy"
    proxy_binary.touch()
    proxy = restore_context._PID_ONE
    proxy.mkdir(parents=True)
    (proxy / "cmdline").write_bytes(os.fsencode(proxy_binary) + b"\0")
    (proxy / "exe").symlink_to(proxy_binary)
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(control_path))
    return proxy, control_path


@pytest.mark.parametrize(
    "existing_context",
    [None, '{"env":{"POD_UID":"stale-uid"}}', "invalid JSON"],
    ids=["missing-context", "stale-context", "invalid-context"],
)
@pytest.mark.parametrize("same_pod", [False, True])
async def test_proxy_refresh_uses_destination_identity_before_parsing(
    monkeypatch, restore_proxy, existing_context, same_pod
):
    proxy, control_path = restore_proxy
    destination = {
        name: f"destination-{name}" for name in KUBERNETES_REQUIRED_ENV_NAMES
    }
    destination["DYN_DISCOVERY_BACKEND"] = "kubernetes"
    destination["DYN_REQUEST_PLANE"] = "tcp"
    destination["DYN_EVENT_PLANE"] = "nats"
    monkeypatch.setenv("POD_UID", destination["POD_UID"] if same_pod else "capture-uid")
    monkeypatch.setenv("DYN_TCP_RPC_HOST", "capture-host")
    monkeypatch.setenv("HF_TOKEN", "capture-token")
    monkeypatch.setenv("UNRELATED", "capture-unrelated")
    context_path = control_path / SNAPSHOT_RESTORE_CONTEXT_FILE
    if existing_context is not None:
        context_path.write_text(existing_context, encoding="utf-8")
    (proxy / "environ").write_bytes(
        b"\0".join(f"{name}={value}".encode() for name, value in destination.items())
        + b"\0HF_TOKEN=secret\xff\0UNRELATED=\xff\0"
    )

    def parse_config():
        for name, value in destination.items():
            assert os.environ[name] == value
        assert "DYN_TCP_RPC_HOST" not in os.environ
        assert os.environ["HF_TOKEN"] == "capture-token"
        assert os.environ["UNRELATED"] == "capture-unrelated"
        return SimpleNamespace(
            namespace=os.environ["DYN_NAMESPACE"],
            discovery_backend=os.environ["DYN_DISCOVERY_BACKEND"],
            request_plane=os.environ["DYN_REQUEST_PLANE"],
            event_plane=os.environ["DYN_EVENT_PLANE"],
        )

    config = await refresh_snapshot_restore_config(SimpleNamespace(), parse_config)

    assert config.namespace == destination["DYN_NAMESPACE"]
    if existing_context is None:
        assert not context_path.exists()
    else:
        assert context_path.read_text(encoding="utf-8") == existing_context


@pytest.mark.parametrize(
    "environment",
    [b"POD_UID=one\0POD_UID=two\0", b"POD_UID\0", b"POD_UID=\xff\0"],
)
def test_proxy_rejects_invalid_supported_env_without_overwriting_context(
    restore_proxy, environment
):
    proxy, control_path = restore_proxy
    context_path = control_path / SNAPSHOT_RESTORE_CONTEXT_FILE
    context_path.write_text("existing", encoding="utf-8")
    (proxy / "environ").write_bytes(environment)

    with pytest.raises(RuntimeError, match="Snapshot proxy env POD_UID"):
        apply_snapshot_restore_env()

    assert context_path.read_text(encoding="utf-8") == "existing"


def test_proxy_rejects_spoofed_command(restore_proxy):
    proxy, control_path = restore_proxy
    other_binary = control_path / "other-binary"
    other_binary.touch()
    (proxy / "exe").unlink()
    (proxy / "exe").symlink_to(other_binary)

    with pytest.raises(RuntimeError, match="proxy executable does not match"):
        apply_snapshot_restore_env()


def test_legacy_standby_context_is_used_for_non_proxy_pid_one(
    monkeypatch, restore_proxy
):
    proxy, _ = restore_proxy
    (proxy / "cmdline").write_bytes(b"sleep\0infinity\0")
    monkeypatch.setenv("POD_UID", "destination-uid")
    write_snapshot_restore_context()
    monkeypatch.setenv("POD_UID", "capture-uid")

    apply_snapshot_restore_env()

    assert os.environ["POD_UID"] == "destination-uid"


def test_apply_snapshot_restore_env_applies_and_clears_values(monkeypatch, tmp_path):
    monkeypatch.setenv("DYN_REQUEST_PLANE", "tcp")
    monkeypatch.setenv("DYN_EVENT_PLANE_HOST", "192.0.2.10")
    monkeypatch.setenv("DYN_TCP_RPC_HOST", "192.0.2.15")
    monkeypatch.setenv("DYN_TCP_RPC_PORT", "25000")
    monkeypatch.setenv("DYN_TCP_RESPONSE_STREAM_HOST", "192.0.2.20")
    monkeypatch.setenv("DYN_TCP_RESPONSE_STREAM_PORT", "8080")
    monkeypatch.setenv("DYN_TCP_LISTEN_BACKLOG", "128")
    write_restore_context(
        monkeypatch,
        tmp_path,
        {
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_REQUEST_PLANE": None,
            "DYN_EVENT_PLANE_HOST": None,
            "DYN_TCP_RPC_HOST": "198.51.100.15",
            "DYN_TCP_RPC_PORT": "25001",
            "DYN_TCP_RESPONSE_STREAM_HOST": "198.51.100.20",
            "DYN_TCP_RESPONSE_STREAM_PORT": None,
            "DYN_TCP_LISTEN_BACKLOG": "8192",
            "UNSUPPORTED_ENV": "ignored",
        },
    )

    restored = apply_snapshot_restore_env()

    assert restored == {
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": None,
        "DYN_EVENT_PLANE_HOST": None,
        "DYN_TCP_RPC_HOST": "198.51.100.15",
        "DYN_TCP_RPC_PORT": "25001",
        "DYN_TCP_RESPONSE_STREAM_HOST": "198.51.100.20",
        "DYN_TCP_RESPONSE_STREAM_PORT": None,
        "DYN_TCP_LISTEN_BACKLOG": "8192",
    }
    assert os.environ["DYN_DISCOVERY_BACKEND"] == "etcd"
    assert "DYN_REQUEST_PLANE" not in os.environ
    assert "DYN_EVENT_PLANE_HOST" not in os.environ
    assert os.environ["DYN_TCP_RPC_HOST"] == "198.51.100.15"
    assert os.environ["DYN_TCP_RPC_PORT"] == "25001"
    assert os.environ["DYN_TCP_RESPONSE_STREAM_HOST"] == "198.51.100.20"
    assert "DYN_TCP_RESPONSE_STREAM_PORT" not in os.environ
    assert os.environ["DYN_TCP_LISTEN_BACKLOG"] == "8192"
    assert "UNSUPPORTED_ENV" not in os.environ


async def test_refresh_snapshot_restore_config_reparses_runtime_fields(
    monkeypatch, tmp_path
):
    config = SimpleNamespace(
        namespace="snapshot-ns",
        discovery_backend="file",
        request_plane="nats",
        event_plane="zmq",
        engine_args=SimpleNamespace(enable_sleep_mode=True),
    )
    write_restore_context(
        monkeypatch,
        tmp_path,
        {
            "DYN_NAMESPACE": "restore-ns",
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats",
        },
    )

    def parse_config():
        assert os.environ["DYN_NAMESPACE"] == "restore-ns"
        assert os.environ["DYN_DISCOVERY_BACKEND"] == "etcd"
        return SimpleNamespace(
            namespace=os.environ["DYN_NAMESPACE"],
            discovery_backend=os.environ["DYN_DISCOVERY_BACKEND"],
            request_plane=os.environ["DYN_REQUEST_PLANE"],
            event_plane=os.environ["DYN_EVENT_PLANE"],
        )

    refreshed = await refresh_snapshot_restore_config(config, parse_config)

    assert refreshed is config
    assert config.namespace == "restore-ns"
    assert config.discovery_backend == "etcd"
    assert config.request_plane == "tcp"
    assert config.event_plane == "nats"
    # Preserve the existing backend/engine config object; only runtime fields
    # are refreshed from the parser.
    assert config.engine_args.enable_sleep_mode is True


async def test_refresh_snapshot_restore_config_supports_async_parser(
    monkeypatch, tmp_path
):
    config = SimpleNamespace(
        namespace="snapshot-ns",
        discovery_backend="file",
        request_plane="nats",
        event_plane="zmq",
        server_args=SimpleNamespace(enable_memory_saver=True),
    )
    write_restore_context(
        monkeypatch,
        tmp_path,
        {
            "DYN_NAMESPACE": "restore-ns",
            "DYN_DISCOVERY_BACKEND": "mem",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": None,
        },
    )

    async def parse_config():
        return SimpleNamespace(
            namespace=os.environ["DYN_NAMESPACE"],
            discovery_backend=os.environ["DYN_DISCOVERY_BACKEND"],
            request_plane=os.environ["DYN_REQUEST_PLANE"],
            event_plane=os.environ.get("DYN_EVENT_PLANE"),
        )

    refreshed = await refresh_snapshot_restore_config(
        config,
        parse_config,
    )

    assert refreshed is config
    assert config.namespace == "restore-ns"
    assert config.discovery_backend == "mem"
    assert config.request_plane == "tcp"
    assert config.event_plane is None
    assert config.server_args.enable_memory_saver is True


async def test_refresh_snapshot_restore_config_supports_runtime_selector(
    monkeypatch, tmp_path
):
    config = SimpleNamespace(
        dynamo_args=SimpleNamespace(
            namespace="snapshot-ns",
            discovery_backend="file",
            request_plane="nats",
            event_plane="zmq",
        ),
        server_args=SimpleNamespace(enable_memory_saver=True),
    )
    context_path = tmp_path / SNAPSHOT_RESTORE_CONTEXT_FILE
    context_path.write_text(
        json.dumps({"env": {"DYN_NAMESPACE": "restore-ns"}}), encoding="utf-8"
    )
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))

    refreshed = await refresh_snapshot_restore_config(
        config,
        lambda: SimpleNamespace(
            dynamo_args=SimpleNamespace(
                namespace="restore-ns",
                discovery_backend="mem",
                request_plane="tcp",
                event_plane=None,
            )
        ),
        runtime_config=lambda parsed_config: parsed_config.dynamo_args,
    )

    assert refreshed is config
    assert config.dynamo_args.namespace == "restore-ns"
    assert config.dynamo_args.discovery_backend == "mem"
    assert config.server_args.enable_memory_saver is True


async def test_refresh_snapshot_restore_config_validates_kubernetes_env(
    monkeypatch, tmp_path
):
    config = SimpleNamespace(
        namespace="snapshot-ns",
        discovery_backend="file",
        request_plane="nats",
        event_plane="zmq",
    )
    write_restore_context(
        monkeypatch,
        tmp_path,
        {
            "DYN_NAMESPACE": "restore-ns",
            "DYN_DISCOVERY_BACKEND": "kubernetes",
            "DYN_REQUEST_PLANE": "tcp",
        },
    )

    with pytest.raises(RuntimeError, match="snapshot restore context requires"):
        await refresh_snapshot_restore_config(
            config,
            lambda: SimpleNamespace(
                namespace="restore-ns",
                discovery_backend="kubernetes",
                request_plane="tcp",
                event_plane=None,
            ),
        )
