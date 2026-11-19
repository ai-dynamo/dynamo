# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip("gpu_memory_service is required", allow_module_level=True)
if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)
pytest.importorskip("vllm")

from gpu_memory_service.common.locks import GrantedLockType  # noqa: E402
from gpu_memory_service.integrations.common.utils import (  # noqa: E402
    FIXED_VA_WINDOWS_OPTION,
    get_gms_fixed_va_window,
)
from gpu_memory_service.integrations.vllm import graph_archive  # noqa: E402
from gpu_memory_service.integrations.vllm import patches  # noqa: E402
from gpu_memory_service.integrations.vllm import worker as gms_worker  # noqa: E402
from gpu_memory_service.v1.integrations.vllm import (  # noqa: E402
    worker as gms_v1_worker,
)
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.worker import utils as vllm_worker_utils  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


_SETTINGS = (
    "DYN_GMS_SCRATCH_KV_ENABLED",
    "DYN_GMS_VLLM_LOG_LEVEL",
    "MX_ENABLED",
)


@pytest.fixture
def start_worker(monkeypatch, tmp_path):
    """Run GMSWorker.init_device without a GPU, GMS server or vLLM parent.

    Returns a function that starts a worker and returns the weights manager
    factory mock. Its ``order`` lists the two patches that read no settings
    (stubbed out) and vLLM's own ``init_device``, as they ran, with the
    ``request_memory`` vLLM would have called. Process-wide vLLM state the
    start may patch is restored after the test.
    """
    for name in _SETTINGS:
        monkeypatch.delenv(name, raising=False)
    manager_factory = Mock()
    monkeypatch.setattr(
        gms_worker, "get_or_create_gms_client_memory_manager", manager_factory
    )
    monkeypatch.setattr(gms_worker, "_get_dp_adjusted_local_rank", lambda *_: 0)
    monkeypatch.setattr(
        gms_worker, "get_vmm_device_type", lambda: SimpleNamespace(value="cuda")
    )
    monkeypatch.setattr(
        gms_worker, "get_socket_path", lambda *_: str(tmp_path / "weights.sock")
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(set_device=lambda _: None),
    )
    base_module = sys.modules[gms_worker._BaseWorker.__module__]
    order = []
    monkeypatch.setattr(
        gms_worker._BaseWorker,
        "init_device",
        lambda _: order.append(("vllm init_device", base_module.request_memory)),
    )
    monkeypatch.setattr(
        gms_worker, "patch_empty_cache", lambda: order.append("empty_cache")
    )
    monkeypatch.setattr(
        gms_worker, "patch_memory_snapshot", lambda: order.append("memory_snapshot")
    )

    monkeypatch.setattr(base_module, "request_memory", base_module.request_memory)
    monkeypatch.setattr(
        vllm_worker_utils, "request_memory", vllm_worker_utils.request_memory
    )
    monkeypatch.setattr(patches, "_request_memory_patched", False)
    monkeypatch.setattr(patches, "_register_kv_caches_patched", False)
    nixl = ModuleType(patches._NIXL_MODULE)
    nixl.NixlBaseConnector = type(
        "NixlBaseConnector", (), {"register_kv_caches": lambda self, kv: None}
    )
    monkeypatch.setitem(sys.modules, patches._NIXL_MODULE, nixl)

    gms_root = logging.getLogger("gpu_memory_service")
    # A copy: the start appends to the list in place, and the original has to
    # come back unchanged.
    monkeypatch.setattr(gms_root, "handlers", list(gms_root.handlers))
    for attr in ("level", "propagate"):
        monkeypatch.setattr(gms_root, attr, getattr(gms_root, attr))

    def start(extra_config=None):
        worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
        worker.local_rank = 0
        worker.parallel_config = None
        worker.vllm_config = SimpleNamespace(
            load_config=SimpleNamespace(model_loader_extra_config=extra_config or {})
        )
        worker.init_device()
        return manager_factory

    start.order = order
    return start


def test_init_device_forwards_ro_connect_timeout(start_worker):
    manager_factory = start_worker(
        {"gms_read_only": True, "gms_ro_connect_timeout_ms": 4200}
    )

    assert manager_factory.call_args.kwargs["timeout_ms"] == 4200


def test_init_device_refuses_graph_archive_engine_granted_the_write_lock(start_worker):
    options = {graph_archive.LOADER_OPTION: "load"}
    weights_manager = gms_worker.get_or_create_gms_client_memory_manager.return_value
    weights_manager.granted_lock_type = GrantedLockType.RO
    start_worker(options)

    weights_manager.granted_lock_type = GrantedLockType.RW
    with pytest.raises(graph_archive.GraphArchiveError, match="read-only weights"):
        start_worker(options)
    # An engine without the option may be the writer.
    start_worker()


@pytest.mark.parametrize("scratch_kv", [True, False])
def test_init_device_reserves_the_fixed_va_windows(
    monkeypatch, start_worker, scratch_kv
):
    windows = {"weights": ["0x420000000000", 1 << 30], "kv_cache": [1 << 46, 1 << 30]}
    scratch_factory = Mock()
    monkeypatch.setattr(gms_worker, "get_or_create_scratch_manager", scratch_factory)
    monkeypatch.setattr(gms_worker, "is_scratch_kv_enabled", lambda: scratch_kv)

    manager_factory = start_worker({FIXED_VA_WINDOWS_OPTION: windows})

    assert manager_factory.call_args.kwargs["fixed_va_window"] == (
        0x420000000000,
        1 << 30,
    )
    if scratch_kv:
        assert scratch_factory.call_args.kwargs["fixed_va_window"] == (
            1 << 46,
            1 << 30,
        )
    else:
        # Without scratch KV the window is reserved when the KV cache is created.
        scratch_factory.assert_not_called()
    assert get_gms_fixed_va_window({}, "weights") is None
    with pytest.raises(ValueError, match="must be \\[base, size\\]"):
        get_gms_fixed_va_window({FIXED_VA_WINDOWS_OPTION: {"weights": [1]}}, "weights")


@pytest.mark.parametrize("scratch_kv", [True, False])
def test_kv_cache_manager_is_created_in_the_fixed_va_window(
    monkeypatch, tmp_path, scratch_kv
):
    scratch_factory = Mock(return_value=Mock(scratch_summary=lambda: (1, 0, 0)))
    manager_factory = Mock()
    monkeypatch.setattr(gms_worker, "get_or_create_scratch_manager", scratch_factory)
    monkeypatch.setattr(
        gms_worker, "get_or_create_gms_client_memory_manager", manager_factory
    )
    monkeypatch.setattr(gms_worker, "is_scratch_kv_enabled", lambda: scratch_kv)
    monkeypatch.setattr(
        gms_worker, "get_socket_path", lambda *_: str(tmp_path / "kv.sock")
    )
    monkeypatch.setattr(
        gms_worker._BaseWorker, "initialize_from_config", lambda *_: None
    )
    worker = gms_worker.GMSWorker.__new__(gms_worker.GMSWorker)
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(
            model_loader_extra_config={
                FIXED_VA_WINDOWS_OPTION: {"kv_cache": [1 << 46, 1 << 30]}
            }
        ),
        model_config=SimpleNamespace(enable_sleep_mode=True),
    )

    worker.initialize_from_config(SimpleNamespace(kv_cache_tensors=[object()]))

    # Scratch KV in shadow mode, else the sleep-mode KV manager.
    factory = scratch_factory if scratch_kv else manager_factory
    assert factory.call_args.kwargs["fixed_va_window"] == (1 << 46, 1 << 30)


def test_graph_archive_identity_reads_the_pinned_engine_args(tmp_path):
    def key(**kwargs):
        # A local model path: vLLM resolves a model name on the hub.
        engine_args = AsyncEngineArgs(
            model=str(tmp_path), num_gpu_blocks_override=64, **kwargs
        )
        identity = graph_archive.engine_identity(engine_args, {}, ["vllm"])
        return graph_archive.archive_key(identity, [0])

    default = AsyncEngineArgs(model=str(tmp_path))
    assert key() == key(master_port=default.master_port + 100)
    assert key() != key(max_num_seqs=8)
    assert key() != key(attention_backend="FLASHINFER")


def test_scratch_kv_follows_the_setting_when_the_worker_starts(
    monkeypatch, start_worker
):
    """Scratch KV switched on after this module was imported (the engine sets
    it just before building the engine) must still take effect, including in
    the vLLM worker module that bound request_memory at its own import."""
    base_module = sys.modules[gms_worker._BaseWorker.__module__]
    original = base_module.request_memory
    connector = sys.modules[patches._NIXL_MODULE].NixlBaseConnector
    original_register = connector.register_kv_caches
    monkeypatch.setenv("DYN_GMS_SCRATCH_KV_ENABLED", "1")

    start_worker()

    patched = base_module.request_memory
    assert patched is not original
    assert patched is vllm_worker_utils.request_memory
    connector = sys.modules[patches._NIXL_MODULE].NixlBaseConnector
    assert connector.register_kv_caches is not original_register
    # vLLM's init_device measures memory and calls request_memory, so every
    # patch has to be in place before it runs.
    assert sorted(start_worker.order[:-1]) == ["empty_cache", "memory_snapshot"]
    assert start_worker.order[-1] == ("vllm init_device", patched)


def test_gms_logging_follows_the_setting_when_the_worker_starts(
    monkeypatch, start_worker
):
    vllm_handler = logging.NullHandler()
    monkeypatch.setattr(logging.getLogger("vllm"), "handlers", [vllm_handler])
    gms_root = logging.getLogger("gpu_memory_service")
    gms_root.handlers = []
    monkeypatch.setenv("DYN_GMS_VLLM_LOG_LEVEL", "DEBUG")

    start_worker()

    assert gms_root.level == logging.DEBUG
    assert gms_root.handlers == [vllm_handler]


def test_modelexpress_logging_configured_when_enabled_at_worker_start(
    monkeypatch, start_worker
):
    modelexpress = ModuleType("modelexpress")
    modelexpress.configure_vllm_logging = Mock()
    lifecycle = ModuleType("modelexpress.lifecycle")
    lifecycle.pause_serving = Mock()
    lifecycle.resume_serving = Mock()
    monkeypatch.setitem(sys.modules, "modelexpress", modelexpress)
    monkeypatch.setitem(sys.modules, "modelexpress.lifecycle", lifecycle)
    monkeypatch.setenv("MX_ENABLED", "1")

    start_worker()

    modelexpress.configure_vllm_logging.assert_called_once_with()

    # Pause and resume are first used when the engine sleeps and wakes; an
    # install without them has to fail the start instead.
    monkeypatch.delattr(lifecycle, "resume_serving")
    with pytest.raises(ImportError, match="has no resume_serving"):
        start_worker()
    monkeypatch.setitem(sys.modules, "modelexpress.lifecycle", None)
    with pytest.raises(ImportError, match="modelexpress is not installed"):
        start_worker()


def test_v1_worker_uses_upstream_sleep_backend_accessor(monkeypatch):
    monkeypatch.setattr(gms_v1_worker.Worker, "init_device", lambda _: None)
    backend = Mock()
    worker = gms_v1_worker.GMSV1Worker.__new__(gms_v1_worker.GMSV1Worker)
    worker._sleep_mode_backend = backend
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=True)
    )
    worker.model_runner = SimpleNamespace(get_model=Mock())

    worker.init_device()

    assert (
        worker.vllm_config.model_config.sleep_mode_backend == gms_v1_worker.BACKEND_NAME
    )
    assert (
        worker._maybe_get_memory_pool_context("weights")
        is backend.capture_weights.return_value
    )
    backend.capture_weights.assert_called_once_with(worker.model_runner.get_model)
    assert (
        worker._maybe_get_memory_pool_context("kv_cache")
        is backend.capture_kv_cache.return_value
    )
