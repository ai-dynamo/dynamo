# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import pytest

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
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def clean_restore_env(monkeypatch):
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


def test_apply_snapshot_restore_env_applies_and_clears_values(monkeypatch, tmp_path):
    monkeypatch.setenv("DYN_REQUEST_PLANE", "tcp")
    write_restore_context(
        monkeypatch,
        tmp_path,
        {
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_REQUEST_PLANE": None,
            "UNSUPPORTED_ENV": "ignored",
        },
    )

    restored = apply_snapshot_restore_env()

    assert restored == {
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": None,
    }
    assert os.environ["DYN_DISCOVERY_BACKEND"] == "etcd"
    assert "DYN_REQUEST_PLANE" not in os.environ
    assert "UNSUPPORTED_ENV" not in os.environ


def test_apply_snapshot_restore_env_replaces_main_container_name(monkeypatch, tmp_path):
    # Rename between checkpoint and restore: the restore context value wins.
    monkeypatch.setenv("DYN_MAIN_CONTAINER_NAME", "engine-old")
    write_restore_context(
        monkeypatch,
        tmp_path,
        {"DYN_MAIN_CONTAINER_NAME": "engine-new"},
    )

    restored = apply_snapshot_restore_env()

    assert restored == {"DYN_MAIN_CONTAINER_NAME": "engine-new"}
    assert os.environ["DYN_MAIN_CONTAINER_NAME"] == "engine-new"


def test_apply_snapshot_restore_env_clears_stale_main_container_name(
    monkeypatch, tmp_path
):
    # Restored into a default-named pod: the stale snapshot-time value must be
    # cleared so the runtime falls back to "main".
    monkeypatch.setenv("DYN_MAIN_CONTAINER_NAME", "engine-old")
    write_restore_context(
        monkeypatch,
        tmp_path,
        {"DYN_MAIN_CONTAINER_NAME": None},
    )

    restored = apply_snapshot_restore_env()

    assert restored == {"DYN_MAIN_CONTAINER_NAME": None}
    assert "DYN_MAIN_CONTAINER_NAME" not in os.environ


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
