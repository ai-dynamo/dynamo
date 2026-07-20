# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ModelExpress GMS snapshot restore adapter."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

try:
    from gpu_memory_service.snapshot.transfer import (
        FileTransferSource,
        GMSSnapshotConfig,
        GMSTransferTarget,
        TransferBackendKind,
        create_transfer_backend,
    )
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]

_MX_BACKEND_MODULE = "gpu_memory_service.snapshot.backends.modelexpress"


def _source() -> FileTransferSource:
    return FileTransferSource(
        allocation_id="allocation-0",
        file_path="/checkpoint/shard_0000.bin",
        file_offset=4096,
        byte_count=8192,
    )


def _target(
    *,
    allocation_id: str = "allocation-0",
    device: int = 3,
    byte_count: int = 8192,
) -> GMSTransferTarget:
    return GMSTransferTarget(
        allocation_id=allocation_id,
        va=0x100000,
        device=device,
        byte_count=byte_count,
    )


@pytest.fixture
def fake_modelexpress(monkeypatch):
    """Install the MX descriptor and restore modules used by the adapter."""

    state = SimpleNamespace(
        from_env_calls=[],
        from_env_error=None,
        run_calls=[],
        run_error=None,
        run_result={
            "total_bytes": 8192,
            "elapsed_s": 0.25,
            "selected_strategy": "gds",
            "file_count": 1,
        },
    )

    @dataclass(frozen=True)
    class FakeMxFileReadSource:
        allocation_id: str
        file_path: str
        file_offset: int
        byte_count: int

    @dataclass(frozen=True)
    class FakeMxDeviceReadTarget:
        allocation_id: str
        va: int
        device: int
        byte_count: int

    @dataclass(frozen=True)
    class FakeGmsRestoreContext:
        sources: object
        targets: object
        grouped_sources: object
        device: int
        max_workers: int
        backend_config: object
        gds_chunk_size: object | None
        gds_max_inflight: object | None

        @classmethod
        def from_env(cls, **kwargs):
            state.from_env_calls.append(dict(kwargs))
            if state.from_env_error is not None:
                raise state.from_env_error
            return cls(**kwargs)

    class FakeMxGmsRestoreStrategyChain:
        @staticmethod
        def run(ctx):
            state.run_calls.append(ctx)
            if state.run_error is not None:
                raise state.run_error
            return dict(state.run_result)

    package = ModuleType("modelexpress")
    package.__path__ = []
    gds_loader = ModuleType("modelexpress.gds_loader")
    gds_loader.MxFileReadSource = FakeMxFileReadSource
    gds_loader.MxDeviceReadTarget = FakeMxDeviceReadTarget
    restore_strategy = ModuleType("modelexpress.restore_strategy")
    restore_strategy.GmsRestoreContext = FakeGmsRestoreContext
    restore_strategy.MxGmsRestoreStrategyChain = FakeMxGmsRestoreStrategyChain

    package.gds_loader = gds_loader
    package.restore_strategy = restore_strategy
    monkeypatch.setitem(sys.modules, "modelexpress", package)
    monkeypatch.setitem(sys.modules, "modelexpress.gds_loader", gds_loader)
    monkeypatch.setitem(
        sys.modules,
        "modelexpress.restore_strategy",
        restore_strategy,
    )

    state.file_source_type = FakeMxFileReadSource
    state.device_target_type = FakeMxDeviceReadTarget
    state.context_type = FakeGmsRestoreContext
    state.restore_strategy = restore_strategy

    import gpu_memory_service.snapshot.backends as backends_package

    missing = object()
    previous_backend_module = sys.modules.get(_MX_BACKEND_MODULE, missing)
    previous_backend_attribute = vars(backends_package).get(
        "modelexpress",
        missing,
    )
    if previous_backend_attribute is not missing:
        delattr(backends_package, "modelexpress")
    sys.modules.pop(_MX_BACKEND_MODULE, None)
    try:
        yield state
    finally:
        sys.modules.pop(_MX_BACKEND_MODULE, None)
        if "modelexpress" in vars(backends_package):
            delattr(backends_package, "modelexpress")
        if previous_backend_module is not missing:
            sys.modules[_MX_BACKEND_MODULE] = previous_backend_module
        if previous_backend_attribute is not missing:
            setattr(
                backends_package,
                "modelexpress",
                previous_backend_attribute,
            )


@pytest.fixture
def mx_backend(fake_modelexpress):
    return importlib.import_module(_MX_BACKEND_MODULE)


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_storage_client(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.client.memory_manager",
        _module(
            "gpu_memory_service.client.memory_manager",
            GMSClientMemoryManager=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.common.locks",
        _module(
            "gpu_memory_service.common.locks",
            RequestedLockType=SimpleNamespace(RO="RO", RW="RW"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.common.protocol.messages",
        _module(
            "gpu_memory_service.common.protocol.messages",
            GetAllocationResponse=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.snapshot.disk",
        _module(
            "gpu_memory_service.snapshot.disk",
            DeviceToFileWriter=object,
            load_manifest_and_metadata=lambda _path: None,
            plan_shard_layout=lambda _allocations, _shard_size: [],
        ),
    )

    storage_path = Path(__file__).resolve().parents[1] / "snapshot/storage_client.py"
    return _load_source_module("_gms_storage_client_under_test", storage_path)


def test_transfer_backend_kinds_include_supported_backends():
    values = {backend.value for backend in TransferBackendKind}

    assert TransferBackendKind.MODELEXPRESS.value == "modelexpress"
    assert TransferBackendKind.SHARDED_SSD.value == "sharded-ssd"
    assert {"modelexpress", "sharded-ssd"} <= values


def test_dynamo_backend_uses_modelexpress_restore_strategies(
    mx_backend,
    fake_modelexpress,
):
    assert mx_backend.GmsRestoreContext is fake_modelexpress.context_type
    assert (
        mx_backend.MxGmsRestoreStrategyChain
        is fake_modelexpress.restore_strategy.MxGmsRestoreStrategyChain
    )
    assert not hasattr(mx_backend, "P2PRestoreStrategy")
    assert not hasattr(mx_backend, "GdsRestoreStrategy")
    assert not hasattr(mx_backend, "PosixRestoreStrategy")


def test_create_modelexpress_backend_imports_backend_on_selection(fake_modelexpress):
    assert _MX_BACKEND_MODULE not in sys.modules
    backend = create_transfer_backend(
        "modelexpress",
        GMSSnapshotConfig(device=3, max_workers=4),
    )

    mx_backend = sys.modules[_MX_BACKEND_MODULE]
    assert isinstance(backend, mx_backend.ModelExpressTransferBackend)


def test_non_mx_dispatch_does_not_import_mx_backend_in_fresh_process():
    package_parent = Path(__file__).resolve().parents[2]
    script = f"""
import sys

sys.path.insert(0, {str(package_parent)!r})
sys.modules["modelexpress"] = None

from gpu_memory_service.snapshot.transfer import (
    GMSSnapshotConfig,
    create_transfer_backend,
)

mx_backend_name = "gpu_memory_service.snapshot.backends.modelexpress"
assert mx_backend_name not in sys.modules
backend = create_transfer_backend(
    "sharded-ssd",
    GMSSnapshotConfig(
        device=0,
        max_workers=1,
        backend_config={{"sharded_ssd_roots": ["/tmp"]}},
    ),
)
assert type(backend).__name__ == "ShardedSSDTransferBackend"
assert mx_backend_name not in sys.modules
backend.close()
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_missing_mx_dependency_error_is_clear_in_fresh_process():
    package_parent = Path(__file__).resolve().parents[2]
    script = f"""
import sys

sys.path.insert(0, {str(package_parent)!r})
sys.modules["modelexpress"] = None

from gpu_memory_service.snapshot.transfer import (
    GMSSnapshotConfig,
    create_transfer_backend,
)

try:
    create_transfer_backend(
        "modelexpress",
        GMSSnapshotConfig(device=0, max_workers=1),
    )
except ImportError as exc:
    message = str(exc).lower()
    assert isinstance(exc.__cause__, ImportError)
    assert "modelexpress" in message
    assert "newer than 0.4.0" in message
    assert "modelexpress.restore_strategy" in message
    assert "--transfer-backend=modelexpress" in message
else:
    raise AssertionError("missing modelexpress dependency did not fail import")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_mx_api_error_names_required_release(monkeypatch):
    package = ModuleType("modelexpress")
    package.__path__ = []
    legacy_gds_loader = ModuleType("modelexpress.gds_loader")
    package.gds_loader = legacy_gds_loader
    monkeypatch.setitem(sys.modules, "modelexpress", package)
    monkeypatch.setitem(
        sys.modules,
        "modelexpress.gds_loader",
        legacy_gds_loader,
    )
    monkeypatch.delitem(
        sys.modules,
        "modelexpress.restore_strategy",
        raising=False,
    )
    monkeypatch.delitem(sys.modules, _MX_BACKEND_MODULE, raising=False)

    with pytest.raises(ImportError, match="newer than 0.4.0") as exc:
        importlib.import_module(_MX_BACKEND_MODULE)

    message = str(exc.value)
    assert "modelexpress.restore_strategy" in message
    assert "--transfer-backend=modelexpress" in message
    assert isinstance(exc.value.__cause__, ImportError)


def test_group_restore_pairs_sorts_each_path_and_totals_bytes(
    mx_backend,
    fake_modelexpress,
):
    sources = [
        fake_modelexpress.file_source_type("a1", "/checkpoint/a.bin", 4096, 32),
        fake_modelexpress.file_source_type("b0", "/checkpoint/b.bin", 0, 64),
        fake_modelexpress.file_source_type("a0", "/checkpoint/a.bin", 0, 16),
    ]
    targets = {
        source.allocation_id: fake_modelexpress.device_target_type(
            source.allocation_id,
            0x100000 + index * 0x1000,
            3,
            source.byte_count,
        )
        for index, source in enumerate(sources)
    }

    grouped, total_bytes = mx_backend._group_restore_pairs(sources, targets)

    assert list(grouped) == ["/checkpoint/a.bin", "/checkpoint/b.bin"]
    assert [
        source.allocation_id for source, _target in grouped["/checkpoint/a.bin"]
    ] == ["a0", "a1"]
    assert grouped["/checkpoint/b.bin"] == [(sources[1], targets["b0"])]
    assert total_bytes == 112


def test_group_restore_pairs_handles_empty_input(mx_backend):
    assert mx_backend._group_restore_pairs([], {}) == ({}, 0)


def test_mx_transfer_session_validates_converts_and_runs_mx_chain(
    monkeypatch,
    mx_backend,
    fake_modelexpress,
):
    validations = []
    chain_calls = []
    real_validate = mx_backend.validate_transfer_targets

    def validate(sources, targets, *, device):
        validations.append((sources, targets, device))
        real_validate(sources, targets, device=device)

    def run(ctx):
        chain_calls.append(ctx)
        return {
            "total_bytes": 8192,
            "elapsed_s": 0.25,
            "selected_strategy": "gds",
            "file_count": 1,
        }

    monkeypatch.setattr(mx_backend, "validate_transfer_targets", validate)
    monkeypatch.setattr(
        fake_modelexpress.restore_strategy.MxGmsRestoreStrategyChain,
        "run",
        staticmethod(run),
    )
    backend = mx_backend.ModelExpressTransferBackend(
        config=GMSSnapshotConfig(
            device=3,
            max_workers=7,
            backend_config={
                "mx_gds_chunk_size_bytes": "4096",
                "mx_gds_max_inflight_batches": "2",
            },
        )
    )

    backend.start_restore([_source()]).restore({"allocation-0": _target()})

    assert validations == [([_source()], {"allocation-0": _target()}, 3)]
    assert len(chain_calls) == 1
    ctx = chain_calls[0]
    assert isinstance(ctx, fake_modelexpress.context_type)
    assert ctx.sources == [
        fake_modelexpress.file_source_type(
            allocation_id="allocation-0",
            file_path="/checkpoint/shard_0000.bin",
            file_offset=4096,
            byte_count=8192,
        )
    ]
    assert ctx.targets == {
        "allocation-0": fake_modelexpress.device_target_type(
            allocation_id="allocation-0",
            va=0x100000,
            device=3,
            byte_count=8192,
        )
    }
    assert ctx.grouped_sources == {
        "/checkpoint/shard_0000.bin": [(ctx.sources[0], ctx.targets["allocation-0"])]
    }
    assert ctx.device == 3
    assert ctx.max_workers == 7
    assert ctx.backend_config == {
        "mx_gds_chunk_size_bytes": "4096",
        "mx_gds_max_inflight_batches": "2",
    }
    assert ctx.gds_chunk_size == "4096"
    assert ctx.gds_max_inflight == "2"
    assert fake_modelexpress.from_env_calls == [
        {
            "sources": ctx.sources,
            "targets": ctx.targets,
            "grouped_sources": ctx.grouped_sources,
            "device": 3,
            "max_workers": 7,
            "backend_config": ctx.backend_config,
            "gds_chunk_size": "4096",
            "gds_max_inflight": "2",
        }
    ]


def test_mx_transfer_session_passes_unset_gds_config_to_from_env(
    mx_backend,
    fake_modelexpress,
):
    session = mx_backend.ModelExpressTransferBackend(
        config=GMSSnapshotConfig(
            device=3,
            max_workers=9,
            backend_config={
                "mx_gds_chunk_size_bytes": None,
                "mx_gds_max_inflight_batches": None,
            },
        )
    ).start_restore([_source()])

    session.restore({"allocation-0": _target()})

    assert fake_modelexpress.run_calls[0].gds_chunk_size is None
    assert fake_modelexpress.run_calls[0].gds_max_inflight is None
    assert fake_modelexpress.from_env_calls[0]["gds_chunk_size"] is None
    assert fake_modelexpress.from_env_calls[0]["gds_max_inflight"] is None


def test_mx_transfer_session_passes_offset_sorted_groups_to_from_env(
    mx_backend,
    fake_modelexpress,
):
    sources = [
        FileTransferSource("a1", "/checkpoint/a.bin", 4096, 32),
        FileTransferSource("b0", "/checkpoint/b.bin", 0, 64),
        FileTransferSource("a0", "/checkpoint/a.bin", 0, 16),
    ]
    targets = {
        source.allocation_id: GMSTransferTarget(
            allocation_id=source.allocation_id,
            va=0x100000 + index * 0x1000,
            device=3,
            byte_count=source.byte_count,
        )
        for index, source in enumerate(sources)
    }
    session = mx_backend.ModelExpressTransferBackend(
        config=GMSSnapshotConfig(device=3, max_workers=2)
    ).start_restore(sources)

    session.restore(targets)

    grouped = fake_modelexpress.from_env_calls[0]["grouped_sources"]
    assert list(grouped) == ["/checkpoint/a.bin", "/checkpoint/b.bin"]
    assert [
        source.allocation_id for source, _target in grouped["/checkpoint/a.bin"]
    ] == ["a0", "a1"]
    assert [
        source.allocation_id for source, _target in grouped["/checkpoint/b.bin"]
    ] == ["b0"]


@pytest.mark.parametrize(
    ("sources", "targets", "message"),
    [
        ([_source()], {}, "Missing GMS transfer target"),
        (
            [_source()],
            {"allocation-0": _target(allocation_id="allocation-1")},
            "GMS target allocation mismatch",
        ),
        (
            [_source()],
            {"allocation-0": _target(byte_count=4096)},
            "GMS target size mismatch",
        ),
        (
            [_source()],
            {"allocation-0": _target(device=4)},
            "GMS target device mismatch",
        ),
        (
            [_source(), _source()],
            {"allocation-0": _target()},
            "duplicate GMS transfer source allocation allocation-0",
        ),
        (
            [_source()],
            {
                "allocation-0": _target(),
                "allocation-1": _target(allocation_id="allocation-1"),
            },
            r"GMS transfer targets contain unknown allocations: \['allocation-1'\]",
        ),
    ],
)
def test_mx_transfer_session_validates_before_running_mx(
    mx_backend,
    fake_modelexpress,
    sources,
    targets,
    message,
):
    session = mx_backend.ModelExpressTransferBackend(
        config=GMSSnapshotConfig(device=3, max_workers=2)
    ).start_restore(sources)

    with pytest.raises(RuntimeError, match=message):
        session.restore(targets)

    assert fake_modelexpress.run_calls == []


def test_mx_transfer_session_wraps_mx_failure_with_gms_context(
    mx_backend,
    fake_modelexpress,
):
    fake_modelexpress.run_error = ValueError("P2P transfer failed")
    session = mx_backend.ModelExpressTransferBackend(
        config=GMSSnapshotConfig(device=3, max_workers=2)
    ).start_restore([_source()])

    with pytest.raises(
        RuntimeError, match="device=3.*rdma->gds->posix.*sources=1"
    ) as exc:
        session.restore({"allocation-0": _target()})

    assert isinstance(exc.value.__cause__, ValueError)
    assert "P2P transfer failed" in str(exc.value)


def test_mx_transfer_session_wraps_from_env_failure_with_gms_context(
    mx_backend,
    fake_modelexpress,
):
    error = ValueError("gds_chunk_size argument must be an integer")
    fake_modelexpress.from_env_error = error
    session = mx_backend.ModelExpressTransferBackend(
        config=GMSSnapshotConfig(
            device=3,
            max_workers=2,
            backend_config={"mx_gds_chunk_size_bytes": "invalid"},
        )
    ).start_restore([_source()])

    with pytest.raises(
        RuntimeError,
        match=(
            "MX GMS restore failed: device=3.*"
            "strategy_chain=rdma->gds->posix.*sources=1"
        ),
    ) as exc:
        session.restore({"allocation-0": _target()})

    assert exc.value.__cause__ is error
    assert "gds_chunk_size argument must be an integer" in str(exc.value)
    assert fake_modelexpress.run_calls == []


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("mx_gds_chunk_size_bytes", 0),
        ("mx_gds_chunk_size_bytes", -1),
        ("mx_gds_max_inflight_batches", 0),
        ("mx_gds_max_inflight_batches", -1),
    ],
)
def test_storage_client_rejects_nonpositive_gds_options(
    monkeypatch,
    option,
    value,
):
    storage_client = _import_storage_client(monkeypatch)
    with pytest.raises(ValueError, match=f"{option} must be positive"):
        storage_client.GMSStorageClient(
            socket_path="/tmp/gms-unused",
            **{option: value},
        )


def test_storage_client_passes_gds_options_to_backend_config(monkeypatch):
    storage_client = _import_storage_client(monkeypatch)
    captured = {}

    class FactoryCalled(Exception):
        pass

    def capture_factory(name, config):
        captured["name"] = name
        captured["config"] = config
        raise FactoryCalled

    monkeypatch.setattr(storage_client, "create_transfer_backend", capture_factory)
    client = storage_client.GMSStorageClient(
        socket_path="/tmp/gms-unused",
        device=3,
        mx_gds_chunk_size_bytes=1048576,
        mx_gds_max_inflight_batches=6,
    )

    with pytest.raises(FactoryCalled):
        client.load_to_gms("/checkpoint/device-3", max_workers=9)

    assert captured["name"] == "modelexpress"
    config = captured["config"]
    assert config.device == 3
    assert config.max_workers == 9
    assert config.backend_config["mx_gds_chunk_size_bytes"] == 1048576
    assert config.backend_config["mx_gds_max_inflight_batches"] == 6
