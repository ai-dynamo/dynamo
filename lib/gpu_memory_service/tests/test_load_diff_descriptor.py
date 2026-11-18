# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: descriptor storage in GMS metadata and the identity check."""

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import tensor as gms_tensor
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata
from gpu_memory_service.common.protocol.messages import MetadataPutRequest
from gpu_memory_service.common.protocol.wire import _frame_message
from gpu_memory_service.integrations.vllm.load_diff import descriptor

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

# GMS V1 caps a frame at 1 MiB. This protocol has no cap, so chunking is what
# keeps a large descriptor's frames below that.
_RPC_FRAME_LIMIT = 1 << 20


class _Metadata:
    """The metadata half of a GMS client, recording the frame of every put."""

    def __init__(self):
        self.entries = {}
        self.frames = []

    def metadata_put(self, key, allocation_id, offset_bytes, value):
        request = MetadataPutRequest(key, allocation_id, offset_bytes, value)
        self.frames.append(len(_frame_message(request)))
        self.entries[key] = (allocation_id, offset_bytes, bytes(value))
        return True

    def metadata_get(self, key):
        return self.entries.get(key)

    def metadata_list(self, prefix=""):
        return [k for k in self.entries if k.startswith(prefix)]


def test_large_descriptor_is_chunked_into_small_rpc_frames():
    rng = random.Random(0)
    big = {
        "final": {
            f"m{i}|module": {"x": {"v": f"{rng.getrandbits(128):032x}"}}
            for i in range(60000)
        }
    }
    manager = _Metadata()
    stored = descriptor.put(manager, "a0", big)
    assert stored > 2 * descriptor._CHUNK_BYTES
    assert len(manager.metadata_list(descriptor.KEY + "/")) >= 3
    assert max(manager.frames) <= _RPC_FRAME_LIMIT
    assert {alloc for alloc, _, _ in manager.entries.values()} == {"a0"}
    assert descriptor.get(manager) == big


def test_reader_rejects_missing_foreign_and_corrupt_descriptors(monkeypatch):
    manager = _Metadata()
    with pytest.raises(RuntimeError, match="no load-diff descriptor"):
        descriptor.get(manager)

    descriptor.put(manager, "a0", {"final": {}})
    header = json.loads(manager.entries[descriptor.KEY][2])
    foreign = dict(header, schema=descriptor.SCHEMA + 1)
    manager.metadata_put(descriptor.KEY, "a0", 0, json.dumps(foreign).encode())
    with pytest.raises(RuntimeError, match="schema"):
        descriptor.get(manager)

    manager.metadata_put(descriptor.KEY, "a0", 0, json.dumps(header).encode())
    with monkeypatch.context() as patch:
        patch.setattr(descriptor, "_MAX_DESCRIPTOR_BYTES", 4)
        with pytest.raises(RuntimeError, match="too large"):
            descriptor.get(manager)

    alloc, offset, chunk = manager.entries[f"{descriptor.KEY}/0"]
    manager.metadata_put(f"{descriptor.KEY}/0", alloc, offset, chunk[:-1] + b"\0")
    with pytest.raises(RuntimeError, match="corrupt"):
        descriptor.get(manager)
    del manager.entries[f"{descriptor.KEY}/0"]
    with pytest.raises(RuntimeError, match="chunk 0 is missing"):
        descriptor.get(manager)


def test_tensor_loading_skips_the_descriptor():
    """A default-path reader of a load-diff writer's layout sees only tensors."""
    manager = _Metadata()
    meta = TensorMetadata.from_tensor(torch.zeros(2, 3), "parameter")
    manager.metadata_put("layer.weight", "a0", 0, meta.to_bytes())
    descriptor.put(manager, "a0", {"final": {}})
    specs = GMSTensorSpec.load_all(manager)
    assert list(specs) == ["layer.weight"]
    assert specs["layer.weight"].meta.shape == (2, 3)


def test_mismatch_names_writer_and_reader_values():
    with pytest.raises(RuntimeError, match=r"'vllm': \('0.32.0', '0.33.0'\)"):
        descriptor.check_match(
            {"vllm": "0.32.0", "tp": 4}, {"vllm": "0.33.0", "tp": 4}, "identity"
        )
    descriptor.check_match({"tp": 4}, {"tp": 4}, "identity")


def test_source_hashes_cover_only_the_source_packages():
    name = gms_tensor.__name__
    hashes = descriptor.source_hashes({name, "json", "not_a_package.module"})
    expected = hashlib.sha256(Path(gms_tensor.__file__).read_bytes()).hexdigest()
    assert hashes == {name: expected}


def test_source_modules_name_the_module_of_every_class(monkeypatch):
    """Owners, and the classes of the objects a snapshot describes: a nested
    class is named by its module, and classes of other packages are left out."""

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = TensorMetadata.from_tensor(torch.zeros(1), "buffer")

    nested = f"{descriptor.__name__}.Outer.Inner"
    final = {"|module": {"a": {"o": nested}, "b": {"o": "json.JSONDecoder"}}}
    monkeypatch.setattr(descriptor, "Outer", type("Outer", (), {}), raising=False)
    names = descriptor.source_modules(Layer(), final)
    assert names == {gms_tensor.__name__, descriptor.__name__}


def _identity(attention_backend=None):
    hf_config = SimpleNamespace(
        architectures=["Toy"], to_dict=lambda: {"hidden_size": 8}
    )
    model_config = SimpleNamespace(
        model="toy",
        revision=None,
        hf_config=hf_config,
        dtype=torch.bfloat16,
        quantization=None,
        max_model_len=128,
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        expert_placement_strategy="linear",
        enable_eplb=False,
        rank=0,
        data_parallel_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        attention_config=SimpleNamespace(backend=attention_backend),
    )
    # As the reader compares it: the writer's identity after a JSON round trip.
    return json.loads(json.dumps(descriptor.identity(vllm_config, model_config)))


@pytest.fixture
def no_kernel_env(monkeypatch):
    monkeypatch.setattr(
        descriptor,
        "torch_device",
        lambda: SimpleNamespace(get_device_name=lambda: "toy-gpu"),
    )
    for name in descriptor._KERNEL_ENV:
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "name, writer_value, reader_value",
    [
        ("VLLM_USE_DEEP_GEMM", "0", "1"),
        ("VLLM_BATCH_INVARIANT", None, "1"),
        ("VLLM_ALLREDUCE_USE_FLASHINFER", "1", None),
    ],
)
def test_kernel_env_difference_is_refused(
    monkeypatch, no_kernel_env, name, writer_value, reader_value
):
    _set_env(monkeypatch, name, writer_value)
    writer = _identity()
    assert writer == _identity()
    _set_env(monkeypatch, name, reader_value)
    with pytest.raises(RuntimeError, match=f"kernel_env.*{name}"):
        descriptor.check_match(writer, _identity(), "identity")


def test_attention_backend_difference_is_refused(no_kernel_env):
    writer = _identity("FLASH_ATTN")
    with pytest.raises(RuntimeError, match="attention_config"):
        descriptor.check_match(writer, _identity("FLASHINFER"), "identity")
