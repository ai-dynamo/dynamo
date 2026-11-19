# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM read-only attach: post-processing on meta and KV-cache scales."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)

pytest.importorskip("vllm")

import torch
import vllm.distributed.parallel_state as parallel_state
from gpu_memory_service.integrations.vllm import model_loader
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsKVCacheMethod,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader import utils as vllm_loader_utils

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

_KV_METHODS = {
    "base": lambda: BaseKVCacheMethod(SimpleNamespace()),
    "compressed_tensors": lambda: CompressedTensorsKVCacheMethod(
        SimpleNamespace(kv_cache_scheme=None)
    ),
}


class _Recorder(QuantizeMethodBase):
    def __init__(self):
        self.processed = 0

    def create_weights(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, layer):
        self.processed += 1


def _kv_layer(method: BaseKVCacheMethod) -> torch.nn.Module:
    """An attention layer's KV-scale state, as vLLM's _init_kv_cache_quant sets it."""
    layer = torch.nn.Module()
    layer.kv_cache_dtype = "fp8"
    layer.num_kv_heads = 1
    set_default_quant_scales(layer, register_buffer=True)
    layer.quant_method = method
    method.create_weights(layer)
    return layer


class _Model(torch.nn.Module):
    """Tied-embedding checkpoint shape: an lm_head and an embedding, then a
    KV-cache layer, then a quantized layer that post-processing must reach."""

    def __init__(self, kv_method: BaseKVCacheMethod):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(16, 4)
        self.attn = _kv_layer(kv_method)
        self.proj = torch.nn.Linear(4, 4)
        self.proj.quant_method = _Recorder()
        self.lm_head = ParallelLMHead(16, 4)


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )


@pytest.mark.parametrize("kv_method", sorted(_KV_METHODS))
def test_meta_post_processing_reaches_layers_after_retie_and_kv_scales(
    monkeypatch, single_rank, kv_method
):
    """On main, the lm_head retie (and, without it, the first KV-cache layer)
    raised on meta tensors and stopped post-processing for every layer."""
    method = _KV_METHODS[kv_method]()
    monkeypatch.setattr(
        vllm_loader_utils, "initialize_model", lambda **kwargs: _Model(method)
    )
    config = SimpleNamespace(
        dtype=torch.float32,
        quantization=None,
        word_embeddings_untied_by_checkpoint=True,
        model="tiny",
    )

    model = model_loader._create_meta_model(object(), config)

    assert model.proj.quant_method.processed == 1
    assert model.attn.quant_method is method
    assert config.word_embeddings_untied_by_checkpoint
    assert model.lm_head.weight is not model.embed_tokens.weight


class _Failing(QuantizeMethodBase):
    """A quant method whose post-processing needs tensor values."""

    def create_weights(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        raise NotImplementedError("Cannot copy out of meta tensor")


def test_failed_meta_post_processing_refuses_the_attach(monkeypatch, single_rank):
    """Layers after the failing one would keep their construction-time state."""
    model = _Model(_KV_METHODS["base"]())
    model.proj.quant_method = _Failing()
    monkeypatch.setattr(vllm_loader_utils, "initialize_model", lambda **kwargs: model)
    config = SimpleNamespace(
        dtype=torch.float32,
        quantization=None,
        word_embeddings_untied_by_checkpoint=False,
        model="tiny",
    )

    with pytest.raises(RuntimeError, match="post-processing on meta tensors failed"):
        model_loader._create_meta_model(object(), config)


def _writer_and_reader(kv_method: str, scales: dict[str, float]):
    """A writer layer post-processed with real scales, and a meta reader layer
    with the writer's published tensors attached (as materialization does)."""
    writer = _kv_layer(_KV_METHODS[kv_method]())
    with torch.no_grad():
        for name, value in scales.items():
            getattr(writer, name).fill_(value)
    writer.quant_method.process_weights_after_loading(writer)

    with torch.device("meta"):
        reader = _kv_layer(_KV_METHODS[kv_method]())
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        reader._buffers[name] = writer._buffers[name].clone()
    return writer, reader


@pytest.mark.parametrize("kv_method", sorted(_KV_METHODS))
def test_kv_cache_scales_are_rebuilt_from_the_attached_tensors(kv_method):
    """compressed-tensors overrides the KV-cache post-processing without a
    pre-processed branch on vLLM 0.33; calling it on the reader replaced the
    attached scale with a meta placeholder and raised."""
    writer, reader = _writer_and_reader(
        kv_method, {"k_scale": 0.5, "v_scale": 0.25, "q_scale": 0.125}
    )
    attached = reader._k_scale

    model_loader._rebuild_kv_cache_scales(torch.nn.Sequential(reader))

    for name in ("_k_scale_float", "_v_scale_float", "_q_scale_float"):
        assert getattr(reader, name) == getattr(writer, name)
    assert (reader._k_scale_float, reader._v_scale_float) == (0.5, 0.25)
    assert float(reader._k_scale_cpu) == 0.5
    assert reader._k_scale is attached
    assert not [n for n, p in reader.named_parameters() if p.is_meta]


def test_read_mode_rebuilds_host_scales_after_the_attach(monkeypatch):
    """The attention backends read the host copies. Left at their default of
    1.0 next to the writer's device scales, the engine produces garbage."""
    writer, _ = _writer_and_reader(
        "compressed_tensors", {"k_scale": 0.5, "v_scale": 0.25}
    )

    def attach(client, model, *, device_index):
        for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
            model[0]._buffers[name] = writer._buffers[name].clone()

    monkeypatch.setattr(
        vllm_loader_utils,
        "initialize_model",
        lambda **kwargs: torch.nn.Sequential(
            _kv_layer(_KV_METHODS["compressed_tensors"]())
        ),
    )
    monkeypatch.setattr(model_loader, "materialize_module_from_gms", attach)
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *args: None)
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)
    config = SimpleNamespace(
        dtype=torch.float32,
        quantization=None,
        word_embeddings_untied_by_checkpoint=False,
    )
    client = SimpleNamespace(total_bytes=0, close=lambda: None)

    reader = model_loader._load_read_mode(client, SimpleNamespace(), config, 0)

    assert (reader[0]._k_scale_float, reader[0]._v_scale_float) == (0.5, 0.25)


def test_per_head_kv_cache_scales_fail_closed():
    writer, reader = _writer_and_reader("compressed_tensors", {"k_scale": 0.5})
    reader._buffers["_k_scale"] = torch.tensor([0.5, 0.75])

    with pytest.raises(RuntimeError, match="per-head KV-cache scales"):
        model_loader._rebuild_kv_cache_scales(torch.nn.Sequential(reader))
