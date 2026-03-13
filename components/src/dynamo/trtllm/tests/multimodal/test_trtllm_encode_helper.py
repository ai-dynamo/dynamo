# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EncodeHelper._process_embedding_path_flow and
read_embeddings_from_encode_response, focusing on multi-embedding support."""

import sys
from contextlib import asynccontextmanager
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

# Stub out dynamo.nixl_connect so encode_helper can be imported without the
# native extension. Real NIXL operations are mocked per-test.
_nixl_stub = ModuleType("dynamo.nixl_connect")
_nixl_stub.Connector = object
_nixl_stub.Descriptor = MagicMock
_nixl_stub.RdmaMetadata = MagicMock
sys.modules.setdefault("dynamo.nixl_connect", _nixl_stub)

from dynamo.trtllm.encode_helper import EncodeHelper  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(tensors_by_path: dict):
    """Return a mock multimodal_processor that loads tensors by path."""
    proc = MagicMock()
    proc.load_tensor_from_path_or_url = MagicMock(
        side_effect=lambda path: tensors_by_path[path]
    )
    return proc


def _make_connector(captured: list):
    """Return a mock NIXL connector that records the tensor passed to it."""

    @asynccontextmanager
    async def create_readable(descriptor):
        # Record whatever tensor was wrapped
        captured.append(descriptor)
        op = MagicMock()
        op.metadata = MagicMock(return_value=MagicMock(model_dump=lambda: {}))
        op.wait_for_completion = AsyncMock()
        yield op

    conn = MagicMock()
    conn.create_readable = create_readable
    return conn


def _make_begin_read_connector(tensor_to_fill):
    """Return a connector whose begin_read fills tensor_to_fill with given data."""

    async def begin_read(metadata, descriptor):
        # Simulate NIXL filling the output tensor
        descriptor.copy_(tensor_to_fill)
        op = MagicMock()
        op.__enter__ = lambda s: s
        op.__exit__ = MagicMock(return_value=False)
        op.wait_for_completion = AsyncMock()
        return op

    conn = MagicMock()
    conn.begin_read = begin_read
    return conn


# ---------------------------------------------------------------------------
# _process_embedding_path_flow — single path (unchanged behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_plain_tensor_unchanged():
    """Single plain tensor: shape and per_embedding_shapes=[original_shape] in response."""
    t = torch.rand(1, 576, 512)
    captured = []
    connector = _make_connector(captured)
    processor = _make_processor({"/tmp/emb.pt": t})

    responses = []
    async for resp in EncodeHelper._process_embedding_path_flow(
        ["/tmp/emb.pt"], processor, connector
    ):
        responses.append(resp)

    assert len(responses) == 1
    resp = responses[0]
    assert resp["embeddings_shape"] == list(t.shape)
    assert resp["per_embedding_shapes"] == [list(t.shape)]
    # The tensor fed to NIXL should be the original (not concatenated)
    assert captured[0].shape == t.shape


# ---------------------------------------------------------------------------
# _process_embedding_path_flow — multiple plain tensors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_plain_tensors_concatenated():
    """Two plain tensors of the same shape are cat'd along dim=0 for one NIXL op."""
    t1 = torch.rand(1, 576, 512)
    t2 = torch.rand(1, 576, 512)
    captured = []
    connector = _make_connector(captured)
    processor = _make_processor({"/tmp/a.pt": t1, "/tmp/b.pt": t2})

    responses = []
    async for resp in EncodeHelper._process_embedding_path_flow(
        ["/tmp/a.pt", "/tmp/b.pt"], processor, connector
    ):
        responses.append(resp)

    assert len(responses) == 1
    resp = responses[0]
    # Concatenated shape: [576+576, 512] = [1152, 512]
    assert resp["embeddings_shape"] == [1152, 512]
    assert resp["per_embedding_shapes"] == [[1, 576, 512], [1, 576, 512]]
    assert captured[0].shape == torch.Size([1152, 512])


@pytest.mark.asyncio
async def test_multiple_plain_tensors_variable_seq_len():
    """Two plain tensors with different seq_len are handled via flatten+cat."""
    t1 = torch.rand(1, 400, 256)
    t2 = torch.rand(1, 600, 256)
    captured = []
    connector = _make_connector(captured)
    processor = _make_processor({"/a.pt": t1, "/b.pt": t2})

    responses = []
    async for resp in EncodeHelper._process_embedding_path_flow(
        ["/a.pt", "/b.pt"], processor, connector
    ):
        responses.append(resp)

    resp = responses[0]
    assert resp["embeddings_shape"] == [1000, 256]  # 400 + 600
    assert resp["per_embedding_shapes"] == [[1, 400, 256], [1, 600, 256]]


# ---------------------------------------------------------------------------
# _process_embedding_path_flow — dict format with multiple paths (warns + uses first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dict_format_multiple_paths_uses_first(caplog):
    """Dict-format with multiple paths: warns and uses only the first embedding."""
    t1 = {"mm_embeddings": torch.rand(1, 576, 512), "special": [1, 2]}
    t2 = {"mm_embeddings": torch.rand(1, 576, 512), "special": [3, 4]}
    captured = []
    connector = _make_connector(captured)
    processor = _make_processor({"/a.pt": t1, "/b.pt": t2})

    import logging

    with caplog.at_level(logging.WARNING, logger="dynamo.trtllm.encode_helper"):
        responses = []
        async for resp in EncodeHelper._process_embedding_path_flow(
            ["/a.pt", "/b.pt"], processor, connector
        ):
            responses.append(resp)

    assert any("Only the first embedding will be used" in r.message for r in caplog.records)
    resp = responses[0]
    # Shape should be from first tensor only
    assert resp["embeddings_shape"] == [1, 576, 512]
    assert resp["per_embedding_shapes"] == [[1, 576, 512]]


# ---------------------------------------------------------------------------
# read_embeddings_from_encode_response — single tensor (unchanged behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_single_tensor_returns_tensor():
    """Single plain tensor: read returns a torch.Tensor (unchanged)."""
    t = torch.rand(1, 576, 512)
    encode_response = {
        "embeddings_shape": list(t.shape),
        "embeddings_dtype": str(t.dtype),
        "auxiliary_data": {},
        "per_embedding_shapes": [[1, 576, 512]],
        "nixl_readable_metadata": {},
    }
    connector = _make_begin_read_connector(t)

    with patch("dynamo.nixl_connect.RdmaMetadata") as mock_meta:
        mock_meta.model_validate = MagicMock(return_value=MagicMock())
        result = await EncodeHelper.read_embeddings_from_encode_response(
            encode_response, connector
        )

    assert isinstance(result, torch.Tensor)
    assert result.shape == t.shape


# ---------------------------------------------------------------------------
# read_embeddings_from_encode_response — multiple tensors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_multiple_tensors_splits_correctly():
    """Multiple plain tensors: read splits the concatenated tensor back into a list."""
    t1 = torch.rand(1, 400, 256)
    t2 = torch.rand(1, 600, 256)

    # Simulate what the encode worker would send
    flat1 = t1.reshape(-1, 256)  # [400, 256]
    flat2 = t2.reshape(-1, 256)  # [600, 256]
    concatenated = torch.cat([flat1, flat2], dim=0)  # [1000, 256]

    encode_response = {
        "embeddings_shape": list(concatenated.shape),
        "embeddings_dtype": str(concatenated.dtype),
        "auxiliary_data": {},
        "per_embedding_shapes": [[1, 400, 256], [1, 600, 256]],
        "nixl_readable_metadata": {},
    }
    connector = _make_begin_read_connector(concatenated)

    with patch("dynamo.nixl_connect.RdmaMetadata") as mock_meta:
        mock_meta.model_validate = MagicMock(return_value=MagicMock())
        result = await EncodeHelper.read_embeddings_from_encode_response(
            encode_response, connector
        )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == torch.Size([1, 400, 256])
    assert result[1].shape == torch.Size([1, 600, 256])
    assert torch.allclose(result[0], t1)
    assert torch.allclose(result[1], t2)


@pytest.mark.asyncio
async def test_read_multiple_same_shape_splits_correctly():
    """Two embeddings of identical shape are split and reshaped back correctly."""
    t1 = torch.rand(1, 576, 512)
    t2 = torch.rand(1, 576, 512)

    flat1 = t1.reshape(-1, 512)  # [576, 512]
    flat2 = t2.reshape(-1, 512)  # [576, 512]
    concatenated = torch.cat([flat1, flat2], dim=0)  # [1152, 512]

    encode_response = {
        "embeddings_shape": [1152, 512],
        "embeddings_dtype": str(t1.dtype),
        "auxiliary_data": {},
        "per_embedding_shapes": [[1, 576, 512], [1, 576, 512]],
        "nixl_readable_metadata": {},
    }
    connector = _make_begin_read_connector(concatenated)

    with patch("dynamo.nixl_connect.RdmaMetadata") as mock_meta:
        mock_meta.model_validate = MagicMock(return_value=MagicMock())
        result = await EncodeHelper.read_embeddings_from_encode_response(
            encode_response, connector
        )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == torch.Size([1, 576, 512])
    assert result[1].shape == torch.Size([1, 576, 512])
    assert torch.allclose(result[0], t1)
    assert torch.allclose(result[1], t2)


# ---------------------------------------------------------------------------
# Backward compatibility: old response without per_embedding_shapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_backward_compat_no_per_embedding_shapes():
    """Old encode worker responses without per_embedding_shapes still return a tensor."""
    t = torch.rand(1, 576, 512)
    encode_response = {
        "embeddings_shape": list(t.shape),
        "embeddings_dtype": str(t.dtype),
        "auxiliary_data": {},
        # no 'per_embedding_shapes' key
        "nixl_readable_metadata": {},
    }
    connector = _make_begin_read_connector(t)

    with patch("dynamo.nixl_connect.RdmaMetadata") as mock_meta:
        mock_meta.model_validate = MagicMock(return_value=MagicMock())
        result = await EncodeHelper.read_embeddings_from_encode_response(
            encode_response, connector
        )

    assert isinstance(result, torch.Tensor)
    assert result.shape == t.shape
