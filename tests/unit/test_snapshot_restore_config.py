# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for snapshot restore config helpers."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_PACKAGE = _REPO_ROOT / "components/src/dynamo/common/utils/snapshot"


def _load_snapshot_module():
    """Load snapshot helpers without importing the native dynamo package."""
    stub_names = (
        "dynamo",
        "dynamo.common",
        "dynamo.common.utils",
        "dynamo.common.utils.namespace",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    dynamo_stub = types.ModuleType("dynamo")
    common_stub = types.ModuleType("dynamo.common")
    utils_stub = types.ModuleType("dynamo.common.utils")
    namespace_stub = types.ModuleType("dynamo.common.utils.namespace")

    def get_worker_namespace(namespace=None):
        import os

        namespace = namespace or os.environ.get("DYN_NAMESPACE", "dynamo")
        suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
        if suffix:
            return f"{namespace}-{suffix}"
        return namespace

    namespace_stub.get_worker_namespace = get_worker_namespace

    sys.modules["dynamo"] = dynamo_stub
    sys.modules["dynamo.common"] = common_stub
    sys.modules["dynamo.common.utils"] = utils_stub
    sys.modules["dynamo.common.utils.namespace"] = namespace_stub

    spec = importlib.util.spec_from_file_location(
        "_dynamo_snapshot_under_test",
        _SNAPSHOT_PACKAGE / "__init__.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


snapshot = _load_snapshot_module()
snapshot_constants = sys.modules[f"{snapshot.__name__}.constants"]
restore_context = sys.modules[f"{snapshot.__name__}.restore_context"]


@pytest.fixture(autouse=True)
def clean_restore_env(monkeypatch, tmp_path):
    env_names = set(snapshot_constants.RESTORE_RUNTIME_ENV_NAMES)
    env_names.update(snapshot_constants.KUBERNETES_REQUIRED_PODINFO_FILES)
    env_names.update(snapshot_constants.KUBERNETES_OPTIONAL_PODINFO_FILES)
    env_names.add(snapshot_constants.SNAPSHOT_CONTROL_DIR_ENV)
    env_names.add(snapshot_constants.SNAPSHOT_RESTORE_PLACEHOLDER_ENV)
    for name in env_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        restore_context,
        "SNAPSHOT_CONTROL_DIR",
        str(tmp_path / "unused-snapshot-control"),
    )


@pytest.fixture()
def podinfo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(restore_context, "PODINFO_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def control_dir(tmp_path, monkeypatch):
    path = tmp_path / "snapshot-control"
    monkeypatch.setenv(snapshot_constants.SNAPSHOT_CONTROL_DIR_ENV, str(path))
    return path


def _write_podinfo(podinfo_root: Path, name: str, value: str) -> None:
    (podinfo_root / name).write_text(value, encoding="utf-8")


def test_restore_runtime_env_names_are_minimal():
    assert {
        "DYN_DISCOVERY_BACKEND",
        "DYN_REQUEST_PLANE",
        "DYN_EVENT_PLANE",
        "NATS_SERVER",
        "ETCD_ENDPOINTS",
        "DYN_SYSTEM_PORT",
        "DYN_HEALTH_CHECK_ENABLED",
        "DYN_SYSTEM_STARTING_HEALTH_STATUS",
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
        "DYN_SYSTEM_HOST",
        "DYN_SYSTEM_HEALTH_PATH",
        "DYN_SYSTEM_LIVE_PATH",
        "DYN_KUBE_DISCOVERY_MODE",
        "CONTAINER_NAME",
        "MODEL_EXPRESS_URL",
        "PROMETHEUS_ENDPOINT",
    }.issubset(snapshot_constants.RESTORE_RUNTIME_ENV_NAMES)

    assert "DYN_SYSTEM_ENABLED" not in snapshot_constants.RESTORE_RUNTIME_ENV_NAMES


def test_placeholder_captures_and_reload_refreshes_runtime_env(
    control_dir,
    monkeypatch,
):
    restore_env = {
        "DYN_NAMESPACE": "restore-ns",
        "DYN_NAMESPACE_WORKER_SUFFIX": "abc123",
        "DYN_COMPONENT": "decode",
        "DYN_PARENT_DGD_K8S_NAME": "llama",
        "DYN_PARENT_DGD_K8S_NAMESPACE": "serving",
        "DYN_DISCOVERY_BACKEND": "kubernetes",
        "DYN_REQUEST_PLANE": "nats",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_SYSTEM_PORT": "9090",
        "NATS_SERVER": "nats://nats:4222",
        "MODEL_EXPRESS_URL": "http://model-express:8000",
        "PROMETHEUS_ENDPOINT": "http://prometheus:9090",
    }
    for name, value in restore_env.items():
        monkeypatch.setenv(name, value)
    restore_context.write_snapshot_restore_context()

    context_path = control_dir / snapshot_constants.SNAPSHOT_RESTORE_CONTEXT_FILE
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["env"]["DYN_NAMESPACE"] == "restore-ns"
    assert context["env"]["NATS_SERVER"] == "nats://nats:4222"
    assert context["env"]["ETCD_ENDPOINTS"] is None

    # Simulate checkpoint-time env restored by CRIU. Reload must apply the
    # captured restore-time env from the placeholder.
    monkeypatch.setenv("DYN_NAMESPACE", "checkpoint-ns")
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "old")
    monkeypatch.setenv("ETCD_ENDPOINTS", "http://checkpoint-etcd:2379")

    restored = restore_context.reload_snapshot_restore_config(
        namespace="checkpoint-ns",
        discovery_backend="etcd",
        request_plane="tcp",
        event_plane=None,
    )

    assert restored.namespace == "restore-ns-abc123"
    assert restored.discovery_backend == "kubernetes"
    assert restored.request_plane == "nats"
    assert restored.event_plane == "zmq"
    assert os.environ["DYN_SYSTEM_PORT"] == "9090"
    assert os.environ["NATS_SERVER"] == "nats://nats:4222"
    assert "ETCD_ENDPOINTS" not in os.environ
    assert os.environ["MODEL_EXPRESS_URL"] == "http://model-express:8000"
    assert os.environ["PROMETHEUS_ENDPOINT"] == "http://prometheus:9090"


def test_restore_placeholder_mode_and_runner(monkeypatch):
    calls = []

    def fake_write_snapshot_restore_context():
        calls.append(("write",))

    def fake_execvp(file, args):
        calls.append(("execvp", file, args))
        raise SystemExit(0)

    monkeypatch.setattr(
        restore_context,
        "write_snapshot_restore_context",
        fake_write_snapshot_restore_context,
    )
    monkeypatch.setattr(restore_context.os, "execvp", fake_execvp)

    assert not restore_context.is_restore_placeholder_mode()
    monkeypatch.setenv(snapshot_constants.SNAPSHOT_RESTORE_PLACEHOLDER_ENV, "1")
    assert restore_context.is_restore_placeholder_mode()

    with pytest.raises(SystemExit):
        restore_context.run_restore_placeholder()

    assert calls == [
        ("write",),
        ("execvp", "sleep", ["sleep", "infinity"]),
    ]


def test_null_restore_env_preserves_parsed_cli_fallbacks(control_dir):
    control_dir.mkdir(parents=True, exist_ok=True)
    context_path = control_dir / snapshot_constants.SNAPSHOT_RESTORE_CONTEXT_FILE
    context_path.write_text(
        json.dumps(
            {
                "version": 1,
                "env": {
                    "DYN_DISCOVERY_BACKEND": None,
                    "DYN_REQUEST_PLANE": None,
                    "DYN_EVENT_PLANE": None,
                },
            }
        ),
        encoding="utf-8",
    )

    restored = restore_context.reload_snapshot_restore_config(
        namespace="cli-ns",
        discovery_backend="file",
        request_plane="nats",
        event_plane="zmq",
    )

    assert restored.namespace == "cli-ns"
    assert restored.discovery_backend == "file"
    assert restored.request_plane == "nats"
    assert restored.event_plane == "zmq"


def test_reload_restore_config_falls_back_to_legacy_kubernetes_podinfo(
    podinfo_root,
):
    _write_podinfo(podinfo_root, "dyn_namespace", "legacy-ns")
    _write_podinfo(podinfo_root, "dyn_namespace_worker_suffix", "def456")
    _write_podinfo(podinfo_root, "dyn_component", "decode")
    _write_podinfo(podinfo_root, "dyn_parent_dgd_k8s_name", "legacy")
    _write_podinfo(podinfo_root, "dyn_parent_dgd_k8s_namespace", "serving")

    restored = restore_context.reload_snapshot_restore_config(
        namespace="checkpoint-ns",
        discovery_backend="kubernetes",
    )

    assert restored.namespace == "legacy-ns-def456"
    assert restored.discovery_backend == "kubernetes"
    assert os.environ["DYN_NAMESPACE"] == "legacy-ns"
