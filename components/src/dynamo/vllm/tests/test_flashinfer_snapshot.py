# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest

from dynamo.vllm import flashinfer_snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _worker():
    return SimpleNamespace(rank=3, local_rank=1)


def test_flashinfer_snapshot_no_resources_is_noop(monkeypatch, caplog):
    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", lambda _: None)

    with caplog.at_level(logging.INFO):
        pause_report = flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())
        resume_report = flashinfer_snapshot.resume_flashinfer_peer_resources(_worker())

    assert "No active FlashInfer peer resources to pause" in caplog.text
    assert "No active FlashInfer peer resources to resume" in caplog.text
    assert pause_report.count == 0
    assert resume_report.count == 0


def test_flashinfer_snapshot_require_resources_raises_on_no_resources(monkeypatch):
    monkeypatch.setenv(
        flashinfer_snapshot.DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES, "1"
    )
    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", lambda _: None)

    with pytest.raises(RuntimeError, match="No active FlashInfer peer resources"):
        flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())


class _FakeResource:
    def __init__(self):
        self.calls = []

    def pause(self, **kwargs):
        self.calls.append(("pause", kwargs))

    def resume(self, **kwargs):
        self.calls.append(("resume", kwargs))


def test_flashinfer_snapshot_pauses_and_resumes_fake_resource(monkeypatch):
    resource = _FakeResource()
    fake_ar_module = SimpleNamespace(
        _fi_ar_workspace=resource,
        _fi_ar_quant_workspace=resource,
    )

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FI_AR_MODULE:
            return fake_ar_module
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    pause_report = flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())
    resume_report = flashinfer_snapshot.resume_flashinfer_peer_resources(_worker())

    assert resource.calls == [
        ("pause", {"synchronize": True, "barrier": True}),
        ("resume", {"synchronize": True, "barrier": True}),
    ]
    assert pause_report.count == 1
    assert resume_report.count == 1
    assert pause_report.resources[0].name.endswith("._fi_ar_workspace")
    assert pause_report.resources[0].kind == "generic"
    assert pause_report.resources[0].class_name.endswith("._FakeResource")


def test_flashinfer_snapshot_inspect_discovers_without_mutating(monkeypatch):
    resource = _FakeResource()
    fake_ar_module = SimpleNamespace(
        _fi_ar_workspace=resource,
        _fi_ar_quant_workspace=None,
    )

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FI_AR_MODULE:
            return fake_ar_module
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    report = flashinfer_snapshot.inspect_flashinfer_peer_resources(_worker())

    assert resource.calls == []
    assert report.operation == "inspect"
    assert report.count == 1
    assert report.resources[0].name.endswith("._fi_ar_workspace")


def test_flashinfer_snapshot_unsupported_active_resource_raises(monkeypatch):
    fake_ar_module = SimpleNamespace(
        _fi_ar_workspace=object(),
        _fi_ar_quant_workspace=None,
    )

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FI_AR_MODULE:
            return fake_ar_module
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    with pytest.raises(RuntimeError, match="does not support pause"):
        flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())


class _FakeParallelState:
    def __init__(self, fi_ar_comm):
        self.fi_ar_comm = fi_ar_comm

    def get_ep_group(self):
        return None

    def get_tp_group(self):
        return SimpleNamespace(
            device_communicator=SimpleNamespace(fi_ar_comm=self.fi_ar_comm)
        )


@pytest.mark.parametrize(
    "fake_ar_module",
    [
        SimpleNamespace(),
        SimpleNamespace(_fi_ar_workspace=None, _fi_ar_quant_workspace=None),
    ],
)
def test_flashinfer_snapshot_enabled_fi_ar_without_workspace_raises(
    monkeypatch, fake_ar_module
):
    parallel_state = _FakeParallelState(SimpleNamespace(disabled=False))

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FI_AR_MODULE:
            return fake_ar_module
        if module_name == flashinfer_snapshot._PARALLEL_STATE_MODULE:
            return parallel_state
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    with pytest.raises(RuntimeError, match="has no supported pause path"):
        flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())


def test_flashinfer_snapshot_enabled_fi_ar_with_workspace_is_handled(monkeypatch):
    workspace = _FakeResource()
    fake_ar_module = SimpleNamespace(
        _fi_ar_workspace=workspace,
        _fi_ar_quant_workspace=None,
    )
    parallel_state = _FakeParallelState(SimpleNamespace(disabled=False))

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FI_AR_MODULE:
            return fake_ar_module
        if module_name == flashinfer_snapshot._PARALLEL_STATE_MODULE:
            return parallel_state
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())

    assert workspace.calls == [("pause", {"synchronize": True, "barrier": True})]


def test_static_two_sided_workspace_without_manager_fails_closed(monkeypatch):
    workspace = object()
    mnnvl_moe = SimpleNamespace(moe_workspace=workspace, moe_prepare_workspace=None)
    two_sided_module = SimpleNamespace(MnnvlMoe=mnnvl_moe)

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FLASHINFER_TWO_SIDED_MODULE:
            return two_sided_module
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    with pytest.raises(RuntimeError, match="MnnvlMoe\\.moe_workspace"):
        flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())


def test_static_two_sided_workspace_with_manager_uses_supported_resource(monkeypatch):
    manager = type("FlashInferNVLinkTwoSidedManager", (), {})()
    manager.initialized = True
    manager.cpu_group = object()
    mnnvl_moe = SimpleNamespace(moe_workspace=object(), moe_prepare_workspace=None)
    parallel_state = SimpleNamespace(
        get_ep_group=lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(all2all_manager=manager)
        ),
        get_tp_group=lambda: None,
    )

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._PARALLEL_STATE_MODULE:
            return parallel_state
        if module_name == flashinfer_snapshot._FLASHINFER_TWO_SIDED_MODULE:
            return SimpleNamespace(MnnvlMoe=mnnvl_moe)
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)
    monkeypatch.setattr(
        flashinfer_snapshot,
        "_require_attr_from_module",
        lambda module_name, attr, reason: SimpleNamespace(pause=lambda **kwargs: None),
    )

    report = flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())

    assert report.count == 1
    assert report.resources[0].kind == "two_sided_manager"


def test_one_sided_workspace_cache_without_manager_fails_closed(monkeypatch):
    cache = {"rank0": object()}
    moe_alltoall = SimpleNamespace(_WORKSPACE_CACHE=cache)
    one_sided_module = SimpleNamespace(MoeAlltoAll=moe_alltoall)

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FLASHINFER_ONE_SIDED_MODULE:
            return one_sided_module
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    with pytest.raises(RuntimeError, match="MoeAlltoAll\\._WORKSPACE_CACHE"):
        flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())


def test_flashinfer_snapshot_disabled_fi_ar_without_workspace_is_noop(monkeypatch):
    fake_ar_module = SimpleNamespace(
        _fi_ar_workspace=None,
        _fi_ar_quant_workspace=None,
    )
    parallel_state = _FakeParallelState(SimpleNamespace(disabled=True))

    def optional_import(module_name):
        if module_name == flashinfer_snapshot._FI_AR_MODULE:
            return fake_ar_module
        if module_name == flashinfer_snapshot._PARALLEL_STATE_MODULE:
            return parallel_state
        return None

    monkeypatch.setattr(flashinfer_snapshot, "_optional_import", optional_import)

    flashinfer_snapshot.pause_flashinfer_peer_resources(_worker())
