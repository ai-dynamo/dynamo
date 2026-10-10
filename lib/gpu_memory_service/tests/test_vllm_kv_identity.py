# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from gpu_memory_service.client.rpc import GMS_ERR_IDENTITY_MISMATCH, GmsRemoteError
from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv, kv_identity

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.fixture(autouse=True)
def _clear_dynamic_gms_role_env(monkeypatch):
    monkeypatch.delenv("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", raising=False)
    yield
    monkeypatch.delenv("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", raising=False)


def test_v3_semantic_kv_tags_include_model_layers_and_size():
    tensor_a = SimpleNamespace(
        shared_by=["model.layers.1.self_attn", "model.layers.0.self_attn"],
        size=123,
    )
    same_a_different_order = SimpleNamespace(
        shared_by=["model.layers.0.self_attn", "model.layers.1.self_attn"],
        size=123,
    )
    resized_a = SimpleNamespace(shared_by=tensor_a.shared_by, size=789)

    def config(tensor):
        return SimpleNamespace(kv_cache_tensors=[tensor])

    tag_a = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(tensor_a), "model=org/model\0revision=abc"
    )[0]
    same_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(same_a_different_order), "model=org/model\0revision=abc"
    )[0]
    resized_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(resized_a), "model=org/model\0revision=abc"
    )[0]
    different_model_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(tensor_a), "model=org/model\0revision=def"
    )[0]

    assert tag_a.startswith("kv_pool:v4:")
    assert tag_a == same_tag
    assert tag_a != resized_tag
    assert tag_a != different_model_tag


def test_model_identity_includes_resolved_model_revision():
    commit = "a" * 40
    identity = install_vmm_ipc_kv._model_identity(
        SimpleNamespace(
            model="org/model",
            revision="main",
            code_revision=None,
            quantization="fp8",
            hf_config=SimpleNamespace(_commit_hash=commit),
        )
    )

    assert identity == f"model=org/model\0artifact={commit}\0quantization=fp8"


def test_model_identity_uses_hf_snapshot_directory_commit(monkeypatch):
    monkeypatch.delenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", raising=False)
    commit = "c" * 40
    model = f"/hf/hub/models--org--model/snapshots/{commit}"
    identity = install_vmm_ipc_kv._model_identity(
        SimpleNamespace(
            model=model,
            revision=None,
            code_revision=None,
            hf_config=SimpleNamespace(),
        )
    )

    assert identity == f"model={model}\0artifact={commit}"


@pytest.mark.parametrize(
    "model",
    [
        f"/models/snapshots/{'c' * 40}",
        f"/hf/hub/models--org--model/snapshots/{'c' * 39}",
        "/hf/hub/models--org--model/snapshots/main",
    ],
)
def test_model_identity_ignores_paths_that_are_not_hf_snapshots(monkeypatch, model):
    monkeypatch.delenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="immutable resolved model revision"):
        install_vmm_ipc_kv._model_identity(
            SimpleNamespace(model=model, revision=None, hf_config=SimpleNamespace())
        )


def test_model_identity_accepts_explicit_artifact_digest(monkeypatch):
    monkeypatch.setenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", "image-sha256:abc")
    identity = install_vmm_ipc_kv._model_identity(
        SimpleNamespace(model="/models/current", revision=None, code_revision="main")
    )

    assert identity == "model=/models/current\0artifact=image-sha256:abc"


def test_model_identity_rejects_mutable_revision(monkeypatch):
    monkeypatch.delenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="immutable resolved model revision"):
        install_vmm_ipc_kv._model_identity(
            SimpleNamespace(model="org/model", revision="main")
        )


def test_model_identity_fails_closed_when_unavailable():
    with pytest.raises(RuntimeError, match="immutable resolved model revision"):
        install_vmm_ipc_kv._model_identity(SimpleNamespace())


def test_semantic_kv_tags_fail_closed_without_model_identity():
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=["layer.0"], size=123)]
    )

    with pytest.raises(RuntimeError, match="stable model identity"):
        install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(config)


def test_semantic_kv_tags_follow_physical_packed_allocations():
    packed_a = SimpleNamespace(
        shared_by=["layer.0"], size=1024, offset=0, block_stride=128
    )
    packed_b = SimpleNamespace(
        shared_by=["layer.1"], size=1024, offset=64, block_stride=128
    )
    unpacked = SimpleNamespace(shared_by=["mamba"], size=256, block_stride=0)
    config = SimpleNamespace(kv_cache_tensors=[packed_a, packed_b, unpacked])

    packed_tag, unpacked_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config, "model=org/model"
    )
    changed_packed_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        SimpleNamespace(
            kv_cache_tensors=[
                packed_a,
                SimpleNamespace(
                    shared_by=["layer.changed"],
                    size=1024,
                    offset=64,
                    block_stride=128,
                ),
                unpacked,
            ]
        ),
        "model=org/model",
    )[0]

    assert packed_tag != unpacked_tag
    assert packed_tag != changed_packed_tag


def test_semantic_kv_tags_disambiguate_duplicate_layer_identity():
    tensor_a = SimpleNamespace(shared_by=["model.layers.0.self_attn"], size=123)
    tensor_b = SimpleNamespace(shared_by=["model.layers.0.self_attn"], size=123)

    tag_a, tag_b = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        SimpleNamespace(kv_cache_tensors=[tensor_a, tensor_b]), "model=org/model"
    )

    assert tag_a.startswith("kv_pool:v4:")
    assert tag_b.startswith("kv_pool:v4:")
    assert tag_a != tag_b
    assert tag_a.endswith(":dup0")
    assert tag_b.endswith(":dup1")


def test_persistent_tag_plan_recognizes_complete_reattach():
    class Manager:
        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            assert engine_id == "engine"
            assert include_unclaimed is True
            return [SimpleNamespace(tag=tag) for tag in ("kv:a", "kv:b")]

    assert install_vmm_ipc_kv._persistent_tag_plan_reattaches(
        Manager(), "engine", ["kv:a", "kv:b"]
    )


def test_persistent_tag_plan_releases_only_stale_unclaimed_kv():
    allocations = [
        SimpleNamespace(tag="kv_pool:v4:planned", claimed=False, allocation_id="a"),
        SimpleNamespace(tag="kv_pool:v3:stale", claimed=False, allocation_id="b"),
        SimpleNamespace(tag="weights:v1:unrelated", claimed=False, allocation_id="c"),
    ]

    class Manager:
        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            assert engine_id == "engine"
            assert include_unclaimed is True
            return allocations

        def release_persistent(self, engine_id, tag, allocation_id=None):
            assert engine_id == "engine"
            released.append((tag, allocation_id))
            return True

    released = []
    assert install_vmm_ipc_kv._persistent_tag_plan_reattaches(
        Manager(), "engine", ["kv_pool:v4:planned"]
    )
    # Cleanup names the incarnation it listed.
    assert released == [("kv_pool:v3:stale", "b")]


def test_stale_cleanup_preserves_allocation_claimed_during_release():
    stale = SimpleNamespace(tag="kv_pool:v3:old", claimed=False)
    claimed = SimpleNamespace(tag="kv_pool:v3:old", claimed=True)
    listings = [[stale], [claimed]]

    class Manager:
        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            return listings.pop(0)

        def release_persistent(self, engine_id, tag, allocation_id=None):
            raise RuntimeError("persistent allocation claimed by another session")

    with pytest.raises(RuntimeError, match="incompatible layout are still claimed"):
        install_vmm_ipc_kv._persistent_tag_plan_reattaches(
            Manager(), "engine", ["kv_pool:v4:new"]
        )


def test_stale_cleanup_preserves_allocation_recreated_during_release():
    stale = SimpleNamespace(tag="kv_pool:v3:old", claimed=False, allocation_id="old")

    class Manager:
        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            return [stale]

        def release_persistent(self, engine_id, tag, allocation_id=None):
            assert allocation_id == "old"
            raise GmsRemoteError(
                "persistent allocation identity mismatch", GMS_ERR_IDENTITY_MISMATCH
            )

    with pytest.raises(RuntimeError, match="incompatible layout are still claimed"):
        install_vmm_ipc_kv._persistent_tag_plan_reattaches(
            Manager(), "engine", ["kv_pool:v4:new"]
        )


def test_claimed_stale_layout_fails_closed():
    manager = SimpleNamespace(
        list_persistent=lambda engine_id=None, include_unclaimed=False: [
            SimpleNamespace(tag="kv_pool:v3:live", claimed=True)
        ]
    )

    with pytest.raises(RuntimeError, match="second KV pool"):
        install_vmm_ipc_kv._persistent_tag_plan_reattaches(
            manager, "engine", ["kv_pool:v4:new"]
        )


def test_fresh_layout_reclaims_obsolete_unclaimed_kv():
    allocations = [SimpleNamespace(tag="kv_pool:v3:old", claimed=False)]
    released = []

    class Manager:
        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            return allocations

        def release_persistent(self, engine_id, tag, allocation_id=None):
            released.append((engine_id, tag))
            return True

    assert not install_vmm_ipc_kv._persistent_tag_plan_reattaches(
        Manager(), "engine", ["kv_pool:v4:new"]
    )
    assert released == [("engine", "kv_pool:v3:old")]


def test_persistent_tag_plan_rejects_partial_reattach():
    manager = SimpleNamespace(
        list_persistent=lambda engine_id=None, include_unclaimed=False: [
            SimpleNamespace(tag="kv:a")
        ]
    )

    with pytest.raises(RuntimeError, match="only partially present"):
        install_vmm_ipc_kv._persistent_tag_plan_reattaches(
            manager, "engine", ["kv:a", "kv:b"]
        )


@pytest.mark.parametrize(
    ("reattaching", "released"),
    [(False, [("engine", "kv:a"), ("engine", "kv:b")]), (True, [])],
)
def test_failed_kv_allocation_rolls_back_only_fresh_plan(
    monkeypatch, reattaching, released
):
    from gpu_memory_service.client.torch import allocator

    events = []

    @contextmanager
    def passthrough(*_args):
        yield

    manager = SimpleNamespace(
        release_persistent=lambda engine_id, tag: (
            events.append((engine_id, tag)) or True
        )
    )
    monkeypatch.setattr(
        install_vmm_ipc_kv,
        "_semantic_kv_tensor_tag_plan",
        lambda *_args: ["kv:a", "kv:b"],
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_model_identity", lambda *_: "model")
    monkeypatch.setattr(
        install_vmm_ipc_kv,
        "_persistent_tag_plan_reattaches",
        lambda *_args: reattaching,
    )
    monkeypatch.setattr(allocator, "set_persistent_allocator_tag_plan", lambda *_: None)
    monkeypatch.setattr(
        allocator, "clear_persistent_allocator_tag_plan", lambda *_: None
    )
    monkeypatch.setattr(allocator, "gms_use_persistent_pool", passthrough)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_persistent_kv_zeros_as_empty", passthrough
    )

    with pytest.raises(RuntimeError, match="allocation failed"):
        with install_vmm_ipc_kv.persistent_kv_allocation_context(
            manager, "engine", SimpleNamespace(), SimpleNamespace(), 0
        ):
            raise RuntimeError("allocation failed")

    assert events == released


def test_persistent_kv_zeros_as_empty_is_context_local(monkeypatch):
    import sys

    int8_marker = object()
    fp16_marker = object()
    calls = []

    def fake_zeros(*args, **kwargs):
        calls.append(("zeros", args, dict(kwargs)))
        return ("zeros", kwargs.get("dtype"))

    def fake_empty(*args, **kwargs):
        calls.append(("empty", args, dict(kwargs)))
        return ("empty", kwargs.get("dtype"))

    fake_torch = SimpleNamespace(
        int8=int8_marker,
        float16=fp16_marker,
        zeros=fake_zeros,
        empty=fake_empty,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    other_thread_result = []
    with install_vmm_ipc_kv._persistent_kv_zeros_as_empty(True):
        assert fake_torch.zeros((16,), dtype=int8_marker, device="cuda") == (
            "empty",
            int8_marker,
        )
        thread = threading.Thread(
            target=lambda: other_thread_result.append(
                fake_torch.zeros((16,), dtype=int8_marker)
            )
        )
        thread.start()
        thread.join()
        assert fake_torch.zeros((16,), dtype=fp16_marker) == (
            "zeros",
            fp16_marker,
        )

    assert other_thread_result == [("zeros", int8_marker)]
    assert fake_torch.zeros((16,), dtype=int8_marker) == ("zeros", int8_marker)
    assert [kind for kind, _, _ in calls] == ["empty", "zeros", "zeros", "zeros"]


def test_generic_failover_shadow_mode_enables_shared_geometry(monkeypatch):
    monkeypatch.delenv("DYN_VLLM_GMS_SHADOW_MODE", raising=False)
    monkeypatch.delenv("GMS_VLLM_SHARED_KV", raising=False)
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")

    assert kv_identity.shared_kv_enabled()
    assert kv_identity.allocation_shared()
    assert kv_identity.use_existing_shared_geometry()


@pytest.mark.parametrize(
    ("mode", "blocks", "allowed"),
    [("granular", 16, True), ("granular", 15, False)],
)
def test_standby_uses_existing_geometry_without_directory_ownership(
    monkeypatch, mode, blocks, allowed
):
    from gms_kv_ring.daemon import client as client_module

    monkeypatch.setenv("GMS_VLLM_KV_RECOVERY_MODE", mode)
    monkeypatch.setenv("GMS_VLLM_SHARED_KV", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/directory.sock")
    monkeypatch.setenv("GMS_VLLM_ENGINE_ID", "pool-0")
    monkeypatch.setenv("ENGINE_ID", "1")

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def directory_pool_geometry(self, manifest_id, pool_id):
            assert (manifest_id, pool_id) == ("manifest", "pool-0")
            return {"total_blocks": 16}, 7, "engine-0"

        def directory_register_pool(self, *_args, **_kwargs):
            raise AssertionError("standby must not publish pool geometry")

        def close(self):
            pass

    monkeypatch.setattr(client_module, "DaemonClient", Client)
    if allowed:
        install_vmm_ipc_kv._register_shared_kv_blocks(blocks)
    else:
        with pytest.raises(RuntimeError, match="without directory ownership"):
            install_vmm_ipc_kv._register_shared_kv_blocks(blocks)


def test_cold_start_claims_empty_directory_before_registering_geometry(monkeypatch):
    from gms_kv_ring.daemon import client as client_module

    monkeypatch.setenv("GMS_VLLM_SHARED_KV", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/directory.sock")
    monkeypatch.setenv("GMS_VLLM_ENGINE_ID", "pool-0")
    monkeypatch.setenv("ENGINE_ID", "primary")
    monkeypatch.delenv("GMS_KV_DIRECTORY_STANDBY", raising=False)
    calls = []

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def directory_pool_geometry(self, manifest_id, pool_id):
            assert (manifest_id, pool_id) == ("manifest", "pool-0")
            return None, 7, None

        def directory_promote(self, epoch, writer_id):
            calls.append(("promote", epoch, writer_id))
            return True, 8, writer_id

        def directory_register_pool(
            self, manifest_id, writer_id, pool_id, total_blocks, *, expected_epoch
        ):
            calls.append(("register", expected_epoch, writer_id))
            assert (manifest_id, pool_id, total_blocks) == ("manifest", "pool-0", 16)
            return True, False, expected_epoch

        def close(self):
            pass

    monkeypatch.setattr(client_module, "DaemonClient", Client)
    install_vmm_ipc_kv._register_shared_kv_blocks(16)
    assert calls == [
        ("promote", 7, "engine-primary"),
        ("register", 8, "engine-primary"),
    ]


def test_standby_cannot_claim_empty_directory_geometry(monkeypatch):
    from gms_kv_ring.daemon import client as client_module

    monkeypatch.setenv("GMS_VLLM_SHARED_KV", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/directory.sock")
    monkeypatch.setenv("GMS_VLLM_ENGINE_ID", "pool-0")
    monkeypatch.setenv("ENGINE_ID", "shadow")
    monkeypatch.setenv("GMS_KV_DIRECTORY_STANDBY", "1")

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def directory_pool_geometry(self, *_args):
            return None, 7, None

        def directory_promote(self, *_args):
            raise AssertionError("standby must not promote")

        def directory_register_pool(self, *_args, **_kwargs):
            raise AssertionError("standby must not publish pool geometry")

        def close(self):
            pass

    monkeypatch.setattr(client_module, "DaemonClient", Client)
    with pytest.raises(RuntimeError, match="without directory ownership"):
        install_vmm_ipc_kv._register_shared_kv_blocks(16)


def test_native_kv_allocation_context_check_detects_worker_drift(monkeypatch):
    from vllm.v1.worker.gpu_worker import Worker

    monkeypatch.setattr(Worker, "initialize_from_config", lambda self, config: None)

    assert not install_vmm_ipc_kv.native_kv_allocation_hook_available()


def test_geometry_wait_honors_vllm_specific_timeout(monkeypatch):
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    monkeypatch.setenv("GMS_KV_LEASE_GEOMETRY_WAIT_MS", "300000")
    monkeypatch.setenv("GMS_VLLM_KV_GEOMETRY_WAIT_MS", "42")

    assert install_vmm_ipc_kv._geometry_wait_ms(-1) == 42


def test_register_rank_binding_publishes_actual_allocations(monkeypatch):
    from gms_kv_ring.daemon import client as client_module

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "manifest")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/directory.sock")
    monkeypatch.setenv(
        "GMS_VLLM_TP_COORDINATOR_DIRECTORY_SOCKET", "/tmp/coordinator.sock"
    )
    monkeypatch.setenv("GMS_VLLM_ENGINE_ID", "pool-0")
    monkeypatch.setenv("ENGINE_ID", "0")
    monkeypatch.setenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", "artifact")
    tensor = SimpleNamespace(
        size=4096,
        offset=0,
        block_stride=0,
        shared_by=("layer.0",),
    )
    config = SimpleNamespace(kv_cache_tensors=[tensor], kv_cache_groups=[])
    model = SimpleNamespace(
        model="model",
        model_weights=None,
        hf_config_path=None,
        revision=None,
        hf_config=None,
        code_revision=None,
        quantization=None,
    )
    tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config, install_vmm_ipc_kv._model_identity(model)
    )[0]
    manager = SimpleNamespace(
        list_persistent=lambda **_kwargs: [
            SimpleNamespace(
                tag=tag,
                allocation_id="allocation-1",
                aligned_size=8192,
                claimed=True,
            )
        ]
    )
    calls = []
    sockets = []

    class Client:
        def __init__(self, socket, **_kwargs):
            sockets.append(socket)

        def directory_pool_binding(self, manifest_id, pool_id):
            assert (manifest_id, pool_id) == ("manifest", "pool-0")
            return None, 7, "engine-0"

        def directory_register_pool_rank(self, *args, **kwargs):
            calls.append((args, kwargs))
            return True, False, 7

        def close(self):
            pass

    monkeypatch.setattr(client_module, "DaemonClient", Client)

    install_vmm_ipc_kv.register_persistent_kv_rank_binding(
        manager,
        "device-1",
        rank=1,
        tensor_parallel_size=2,
        kv_cache_config=config,
        model_config=model,
    )

    args, kwargs = calls[0]
    assert args[:5] == ("manifest", "engine-0", "pool-0", 1, 2)
    assert args[6] == [
        {
            "engine_id": "device-1",
            "tag": tag,
            "allocation_id": "allocation-1",
            "aligned_size": 8192,
        }
    ]
    assert kwargs == {"expected_epoch": 7}
    assert sockets == ["/tmp/directory.sock", "/tmp/coordinator.sock"]
    assert len(calls) == 2


def test_geometry_patch_does_not_wait_with_explicit_block_override(monkeypatch):
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    waits = []
    monkeypatch.setattr(
        install_vmm_ipc_kv,
        "_existing_shared_kv_blocks",
        lambda *, wait_ms: waits.append(wait_ms) or None,
    )

    def original(_vllm_config, _kv_cache_specs, available_memory):
        return [SimpleNamespace(num_blocks=available_memory)]

    patched = install_vmm_ipc_kv._wrap_get_kv_cache_configs(original)
    config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=4096))

    assert patched(config, object(), -1)[0].num_blocks == -1
    assert waits == [0]


def test_vllm_geometry_patch_updates_late_engine_core_alias(monkeypatch):
    import sys
    import types

    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    def original(_vllm_config, _kv_cache_specs, _available_memory):
        return "original"

    vllm_mod = types.ModuleType("vllm")
    v1_mod = types.ModuleType("vllm.v1")
    core_pkg = types.ModuleType("vllm.v1.core")
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.get_kv_cache_configs = original
    core_pkg.kv_cache_utils = kv_cache_utils

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.core", core_pkg)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", kv_cache_utils)
    monkeypatch.delitem(sys.modules, "vllm.v1.engine.core", raising=False)
    monkeypatch.setattr(install_vmm_ipc_kv, "_GEOMETRY_PATCH_INSTALLED", False)

    assert install_vmm_ipc_kv.install_geometry_patch()
    patched = kv_cache_utils.get_kv_cache_configs
    assert getattr(patched, "_gms_geometry_patched", False)

    engine_core = types.ModuleType("vllm.v1.engine.core")
    engine_core.get_kv_cache_configs = original
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", engine_core)

    assert install_vmm_ipc_kv.install_geometry_patch()
    assert engine_core.get_kv_cache_configs is patched
