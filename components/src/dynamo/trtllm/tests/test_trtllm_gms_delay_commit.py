# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM delayed GMS publish isolate."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.fixture(autouse=True)
def clear_delay_commit_state():
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    model_loader.set_delay_commit_until_engine_init(False)
    model_loader._pending_gms_write = None
    model_loader._last_imported_weights_bytes = 0
    yield
    model_loader.set_delay_commit_until_engine_init(False)
    model_loader._pending_gms_write = None
    model_loader._last_imported_weights_bytes = 0


def test_setup_gms_plumbs_delay_commit_flag(monkeypatch):
    import gpu_memory_service.integrations.trtllm as trtllm_gms
    from gpu_memory_service.common.locks import RequestedLockType

    captured: dict[str, object] = {}

    monkeypatch.setattr(trtllm_gms, "patch_empty_cache", lambda: None)
    monkeypatch.setattr(trtllm_gms, "patch_model_loader", lambda: None)
    monkeypatch.setattr(trtllm_gms, "set_gms_enabled", lambda enabled: None)
    monkeypatch.setattr(
        trtllm_gms,
        "set_gms_lock_mode",
        lambda mode: captured.setdefault("lock_mode", mode),
    )
    monkeypatch.setattr(
        trtllm_gms,
        "set_delay_commit_until_engine_init",
        lambda enabled: captured.setdefault("delay", enabled),
    )
    monkeypatch.setattr(
        trtllm_gms,
        "set_extra_config",
        lambda extra: captured.setdefault("extra", extra),
    )
    monkeypatch.setattr(trtllm_gms, "install_mpi_worker_bootstrap", lambda: None)

    trtllm_gms.setup_gms({"gms_delay_commit_until_engine_init": True})

    assert captured["delay"] is True
    assert captured["extra"] == {"gms_delay_commit_until_engine_init": True}
    assert captured["lock_mode"] == RequestedLockType.RW_OR_RO


def test_setup_gms_rejects_non_bool_delay_commit():
    import gpu_memory_service.integrations.trtllm as trtllm_gms

    with pytest.raises(
        ValueError, match="gms_delay_commit_until_engine_init must be a boolean"
    ):
        trtllm_gms.setup_gms({"gms_delay_commit_until_engine_init": "true"})


def test_load_rw_defers_publish_until_finalize(monkeypatch):
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    fake_client = object()
    fake_model = object()
    fake_balancer = object()

    monkeypatch.setattr(
        model_loader, "gms_use_mem_pool", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        model_loader, "_move_untracked_params", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(model_loader.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model_loader,
        "finalize_gms_write",
        lambda *_args, **_kwargs: pytest.fail("finalize_gms_write should be deferred"),
    )

    model_loader.set_delay_commit_until_engine_init(True)

    model, moe_load_balancer = model_loader._load_rw(
        self=None,
        checkpoint_dir="checkpoint",
        checkpoint_loader=None,
        gms_client=fake_client,
        device_index=0,
        original_load=lambda *_args, **_kwargs: (fake_model, fake_balancer),
    )

    assert model is fake_model
    assert moe_load_balancer is fake_balancer
    assert model_loader._pending_gms_write == (fake_client, fake_model)


def test_finalize_pending_gms_write_commits_and_clears_slot(monkeypatch):
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    fake_client = object()
    fake_model = object()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        model_loader,
        "finalize_gms_write",
        lambda client, model: calls.append((client, model)) or 123,
    )

    model_loader._pending_gms_write = (fake_client, fake_model)

    assert model_loader.finalize_pending_gms_write() == 123
    assert calls == [(fake_client, fake_model)]
    assert model_loader._pending_gms_write is None
    assert model_loader._last_imported_weights_bytes == 123
