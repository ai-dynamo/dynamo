# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of PdConnector's request-phase and event contract.

Native vLLM/KVBM dependencies are replaced at the import boundary. The wrapper
under test is loaded from its real source; these tests do not certify GPU DMA.
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def pd_types(monkeypatch):
    class Child:
        def __init__(self):
            self.get_num_new_matched_tokens = MagicMock(return_value=(128, True))
            self.update_state_after_alloc = MagicMock()
            self.get_kv_connector_kv_cache_events = MagicMock(return_value=object())
            self.take_events = MagicMock(return_value=iter([object(), object()]))

    class OffloadingConnector(Child):
        pass

    class DynamoConnector(Child):
        pass

    class NixlConnector(Child):
        pass

    class MultiConnector:
        def __init__(self, vllm_config, role, kv_cache_config):
            self._connectors = vllm_config.children
            self._requests_to_connector = {}

    @dataclass
    class MultiKVConnectorMetadata:
        metadata: tuple = ()

    def install(name, **values):
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent not in sys.modules:
                package = ModuleType(parent)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    install(
        "kvbm.vllm_integration.connector.dynamo_connector",
        DynamoConnector=DynamoConnector,
    )
    install(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorHandshakeMetadata=object,
        KVConnectorRole=object,
    )
    install(
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
        MultiConnector=MultiConnector,
        MultiKVConnectorMetadata=MultiKVConnectorMetadata,
    )
    install(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
        OffloadingConnector=OffloadingConnector,
    )
    install(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
        NixlConnector=NixlConnector,
        NixlHandshakePayload=object,
    )
    install("vllm.v1.core.sched.output", SchedulerOutput=object)
    install("vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector")
    install("vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector")

    source = (
        Path(__file__).parents[5]
        / "lib/bindings/kvbm/python/kvbm/vllm_integration/connector/pd_connector.py"
    )
    spec = importlib.util.spec_from_file_location("_pd_connector_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        wrapper=module.PdConnector,
        prefix=OffloadingConnector,
        nixl=NixlConnector,
    )


@pytest.fixture
def pd_connector(pd_types):
    children = [pd_types.prefix(), pd_types.nixl()]
    return pd_types.wrapper(SimpleNamespace(children=children), object(), object())


@pytest.mark.parametrize("params", [None, {}, {"do_remote_decode": True}])
def test_prefill_uses_only_prefix_loader(pd_connector, params):
    request = SimpleNamespace(request_id="prefill-request", kv_transfer_params=params)
    assert pd_connector.get_num_new_matched_tokens(request, 32) == (128, True)
    pd_connector._connectors[0].get_num_new_matched_tokens.assert_called_once_with(
        request, 32
    )
    pd_connector._connectors[1].get_num_new_matched_tokens.assert_not_called()
    assert pd_connector._requests_to_connector[request.request_id] == 0


def test_post_prefill_decode_uses_only_nixl_despite_prefix_hit(pd_connector):
    request = SimpleNamespace(
        request_id="decode-request", kv_transfer_params={"do_remote_prefill": True}
    )
    pd_connector._connectors[1].get_num_new_matched_tokens.return_value = (64, True)
    assert pd_connector.get_num_new_matched_tokens(request, 0) == (64, True)
    pd_connector._connectors[0].get_num_new_matched_tokens.assert_not_called()
    assert pd_connector._requests_to_connector[request.request_id] == 1
    # NIXL clears this flag after allocation; the loader cannot change mid-request.
    request.kv_transfer_params["do_remote_prefill"] = False
    assert pd_connector.get_num_new_matched_tokens(request, 0) == (64, True)


def test_prefix_async_lookup_remains_deferred(pd_connector):
    request = SimpleNamespace(request_id="waiting-request", kv_transfer_params=None)
    pd_connector._connectors[0].get_num_new_matched_tokens.return_value = (None, False)
    assert pd_connector.get_num_new_matched_tokens(request, 0) == (None, False)
    pd_connector._connectors[1].get_num_new_matched_tokens.assert_not_called()


@pytest.mark.parametrize("is_handoff", [False, True])
def test_only_selected_child_claims_external_prefix_both_see_store_blocks(
    pd_connector, is_handoff
):
    request = SimpleNamespace(
        request_id="allocated-request",
        kv_transfer_params={"do_remote_prefill": is_handoff},
    )
    blocks = object()
    pd_connector.update_state_after_alloc(request, blocks, 64)
    chosen = 1 if is_handoff else 0
    for index, child in enumerate(pd_connector._connectors):
        child.update_state_after_alloc.assert_called_once_with(
            request, blocks, 64 if index == chosen else 0
        )


def test_worker_and_scheduler_inventory_events_forwarded(pd_connector):
    prefix, nixl = pd_connector._connectors
    assert (
        pd_connector.get_kv_connector_kv_cache_events()
        is prefix.get_kv_connector_kv_cache_events.return_value
    )
    scheduler_events = list(prefix.take_events.return_value)
    prefix.take_events.return_value = iter(scheduler_events)
    assert list(pd_connector.take_events()) == scheduler_events
    nixl.get_kv_connector_kv_cache_events.assert_not_called()
    nixl.take_events.assert_not_called()


def test_pd_connector_rejects_reversed_children(pd_types):
    with pytest.raises(TypeError, match="Expected first connector"):
        pd_types.wrapper(
            SimpleNamespace(children=[pd_types.nixl(), pd_types.prefix()]),
            object(),
            object(),
        )
