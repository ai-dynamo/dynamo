# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import yaml

try:
    from dynamo.vllm.omni.connectors import nixl_connector as nixl_module
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]


class _FakeDescriptor:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


class _FakeMeta:
    def __init__(self, token: str):
        self._token = token

    def model_dump(self):
        return {"token": self._token}


class _FakeReadableOp:
    def __init__(self, token: str, registry: dict[str, dict]):
        self._token = token
        self._registry = registry

    def metadata(self):
        return _FakeMeta(self._token)

    async def wait_for_completion(self):
        await self._registry[self._token]["read_done"].wait()


class _FakeReadOp:
    async def wait_for_completion(self):
        return None


class _FakeConnector:
    def __init__(self):
        self._registry: dict[str, dict] = {}
        self._counter = 0

    async def create_readable(self, descriptors):
        desc_list = descriptors if isinstance(descriptors, list) else [descriptors]
        self._counter += 1
        token = f"tok-{self._counter}"
        self._registry[token] = {
            "src": [d.tensor for d in desc_list],
            "read_done": nixl_module.asyncio.Event(),
        }
        return _FakeReadableOp(token, self._registry)

    async def begin_read(self, rdma_meta, local_descriptors):
        token = rdma_meta["token"]
        local_list = (
            local_descriptors
            if isinstance(local_descriptors, list)
            else [local_descriptors]
        )
        src_list = self._registry[token]["src"]
        for src, dst in zip(src_list, [d.tensor for d in local_list], strict=False):
            dst.copy_(src)
        self._registry[token]["read_done"].set()
        return _FakeReadOp()


class _FakeRdmaMetadata:
    @classmethod
    def model_validate(cls, value):
        return value


@pytest.fixture
def fake_nixl(monkeypatch):
    fake_mod = SimpleNamespace(
        Connector=_FakeConnector,
        Descriptor=_FakeDescriptor,
        RdmaMetadata=_FakeRdmaMetadata,
    )
    monkeypatch.setattr(nixl_module, "_nixl_connect", fake_mod)
    monkeypatch.setattr(nixl_module, "_NIXL_AVAILABLE", True)
    return fake_mod


def test_nixl_connector_tensor_roundtrip(fake_nixl):
    connector = nixl_module.DynamoOmniNixlConnector(config={})
    try:
        payload = torch.arange(8, dtype=torch.float32)
        ok, size, metadata = connector.put("0", "1", "req-tensor", payload)

        assert ok is True
        assert metadata is not None
        assert size == payload.numel() * payload.element_size()
        assert metadata["kind"] == "tensor_list"

        out = connector.get("0", "1", "req-tensor", metadata=metadata)
        assert out is not None
        received, rx_size = out
        assert rx_size == size
        assert isinstance(received, torch.Tensor)
        assert torch.equal(received, payload)
    finally:
        connector.close()


def test_nixl_connector_object_roundtrip(fake_nixl):
    connector = nixl_module.DynamoOmniNixlConnector(config={})
    try:
        payload = {
            "kind": "text",
            "outputs": [{"token_ids": [1, 2], "text": "hi", "finish_reason": "stop"}],
            "multimodal_output": {"sr": 16000},
        }
        ok, size, metadata = connector.put("1", "router", "req-obj", payload)

        assert ok is True
        assert metadata is not None
        assert size > 0
        assert metadata["kind"] == "serialized_obj"

        out = connector.get("1", "router", "req-obj", metadata=metadata)
        assert out is not None
        received, _ = out
        assert received == payload
    finally:
        connector.close()


def test_nixl_connector_cleanup_clears_pending(fake_nixl):
    connector = nixl_module.DynamoOmniNixlConnector(config={"pending_timeout_s": 60})
    try:
        ok, _, _ = connector.put(
            "0", "1", "req-clean", torch.tensor([1], dtype=torch.int64)
        )
        assert ok is True
        assert "req-clean" in connector._pending

        connector.cleanup("req-clean")
        assert "req-clean" not in connector._pending
    finally:
        connector.close()


@pytest.mark.parametrize("is_inline", [False, True])
def test_nixl_connector_initializes_router_edge(
    monkeypatch, tmp_path, fake_nixl, is_inline
):
    from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors

    from dynamo.vllm.omni.stage_worker import (
        _ensure_stage_connectors,
        _uses_nixl_connector,
    )

    factory = nixl_module.OmniConnectorFactory
    monkeypatch.setattr(factory, "_registry", dict(factory._registry))
    builtin = factory._registry.get("NixlConnector")
    config_path = tmp_path / "connectors.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "connectors": {"output": {"name": "NixlConnector", "extra": {}}},
                "stages": [
                    {
                        "stage_id": 0,
                        "output_connectors": {
                            "to_stage_router": (
                                {"name": "NixlConnector", "extra": {}}
                                if is_inline
                                else "output"
                            )
                        },
                    }
                ],
            }
        )
    )
    connector_path = _ensure_stage_connectors(str(config_path), [])
    assert _uses_nixl_connector(connector_path, [])
    nixl_module.register_dynamoomni_nixl_connector()
    nixl_module.register_dynamoomni_nixl_connector()
    _, connectors = initialize_orchestrator_connectors(connector_path)
    try:
        assert isinstance(
            connectors[("0", "router")], nixl_module.DynamoOmniNixlConnector
        )
        assert factory._registry.get("NixlConnector") is builtin
    finally:
        for connector in connectors.values():
            connector.close()
