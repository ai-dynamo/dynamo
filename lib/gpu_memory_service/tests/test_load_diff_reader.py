# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: the read-only attach steps that run without a GPU engine."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)
# Not importorskip("vllm"): in an image without vLLM, pytest puts
# components/src/dynamo on sys.path while it collects the component tests,
# and "vllm" then resolves to Dynamo's own vllm component.
pytest.importorskip("vllm.model_executor")

import torch  # noqa: E402
from gpu_memory_service.integrations.vllm import model_loader, patches  # noqa: E402
from gpu_memory_service.integrations.vllm.load_diff import reader, state  # noqa: E402
from gpu_memory_service.integrations.vllm.load_diff.writer import (  # noqa: E402
    module_row,
)
from vllm.model_executor.layers.fused_moe import expert_map_manager  # noqa: E402
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizeMethodBase,
)
from vllm.model_executor.utils import is_weights_pre_processed  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


class _Method(QuantizeMethodBase):
    supports_pre_processed_weights = True

    def create_weights(self, layer, *args, **kwargs):
        pass

    def apply(self, layer, *args, **kwargs):
        pass

    def process_weights_after_loading(self, layer):
        layer.kernel_built_pre_processed = is_weights_pre_processed()


class _Unsupported(_Method):
    supports_pre_processed_weights = False


def _layer(method=None):
    layer = torch.nn.Linear(2, 2, bias=False)
    if method is not None:
        layer.quant_method = method
    return layer


def test_rebuild_runs_quant_methods_in_pre_processed_mode():
    modules = {"a": _layer(_Method()), "b": _layer()}
    reader._rebuild(modules)
    assert modules["a"].kernel_built_pre_processed is True
    with pytest.raises(RuntimeError, match=r"_Unsupported \(a\)"):
        reader._rebuild({"a": _layer(_Unsupported())})


def test_module_checks_drop_deleted_parameters_and_refuse_differences():
    writer_side = _layer(_Method())
    del writer_side._parameters["weight"]
    published = {"": module_row(writer_side)}

    model = _layer(_Method())
    reader._check_modules({"": model}, published)
    assert "weight" not in model._parameters

    other = _layer(_Unsupported())
    with pytest.raises(
        RuntimeError, match=r"quant '\S*_Method' .writer. != '\S*_Unsupported'"
    ):
        reader._check_modules({"": other}, published)


def test_buffers_the_writer_registered_while_loading_attach_as_buffers(monkeypatch):
    source = torch.arange(4.0)
    spec = SimpleNamespace(
        meta=SimpleNamespace(tensor_type="buffer"),
        materialize=lambda manager, device_index: source,
    )
    monkeypatch.setattr(
        reader.GMSTensorSpec, "load_all", lambda manager: {"gathered": spec}
    )
    model = _layer()
    published = {"aliases": {"gathered": "gathered"}, "storage_aliases": {}}
    reader._attach(SimpleNamespace(), model, published, 0)
    assert "gathered" in model._buffers
    assert torch.equal(model.gathered, source)
    assert model.gathered.data_ptr() != source.data_ptr()


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_expert_map", torch.zeros(4), persistent=False)


class _Runner(torch.nn.Module):
    """Like vLLM's MoERunner: a read-only property over a submodule's buffer."""

    def __init__(self):
        super().__init__()
        self.experts = _Experts()

    @property
    def expert_map(self):
        return self.experts._expert_map


def test_read_only_property_is_attached_through_its_source(monkeypatch):
    source = torch.arange(4.0)

    def spec(kind):
        return SimpleNamespace(
            meta=SimpleNamespace(tensor_type=kind),
            materialize=lambda manager, device_index: source,
        )

    specs = {"expert_map": spec("tensor_attr"), "experts._expert_map": spec("buffer")}
    monkeypatch.setattr(reader.GMSTensorSpec, "load_all", lambda manager: specs)
    published = {
        "aliases": {"expert_map": "expert_map", "experts._expert_map": "expert_map"},
        "storage_aliases": {},
    }
    model = _Runner()
    constructed = model.experts._expert_map
    reader._attach(SimpleNamespace(), model, published, 0)
    assert model.expert_map is constructed and torch.equal(constructed, source)

    # Writer-side, the names were two tensors: the property's value is not
    # the tensor attached under its name, so the attach fails.
    published["aliases"]["experts._expert_map"] = "experts._expert_map"
    model = _Runner()
    model.experts._expert_map = torch.empty(4, device="meta")
    with pytest.raises(RuntimeError, match="read-only attribute mismatch"):
        reader._attach(SimpleNamespace(), model, published, 0)


def test_tuple_elements_and_parameters_first_seen_under_another_name(monkeypatch):
    source = torch.arange(4.0)

    def spec(kind):
        return SimpleNamespace(
            meta=SimpleNamespace(tensor_type=kind),
            materialize=lambda manager, device_index: source,
        )

    # Order as the writer's walk yields them: a parent's attributes first.
    specs = {
        "pair.0": spec("tensor_attr"),
        "rows.0": spec("tensor_attr"),
        "child.weight": spec("parameter"),
    }
    monkeypatch.setattr(reader.GMSTensorSpec, "load_all", lambda manager: specs)
    published = {
        "aliases": {"pair.0": "pair.0", "rows.0": "rows.0", "child.weight": "rows.0"},
        "storage_aliases": {},
    }
    model = torch.nn.Module()
    model.child = torch.nn.Module()
    model.child.weight = torch.nn.Parameter(torch.empty(4, device="meta"))
    model.pair = (torch.empty(4, device="meta"),)
    model.rows = [model.child.weight]
    reader._attach(SimpleNamespace(), model, published, 0)
    assert type(model.pair) is tuple and torch.equal(model.pair[0], source)
    assert isinstance(model.child.weight, torch.nn.Parameter)
    assert model.rows[0] is model.child.weight
    assert model.child.weight.data_ptr() == source.data_ptr()


@pytest.mark.parametrize("bulk_import", [True, False, None])
def test_attach_imports_every_allocation_first_when_bulk_import_is_on(
    monkeypatch, bulk_import
):
    calls = []

    def spec(allocation_id):
        def materialize(manager, device_index):
            calls.append(("materialize", allocation_id))
            return torch.arange(4.0)

        return SimpleNamespace(
            meta=SimpleNamespace(tensor_type="parameter"),
            allocation_id=allocation_id,
            materialize=materialize,
        )

    specs = {"a.weight": spec("x"), "b.weight": spec("x"), "c.weight": spec("y")}
    monkeypatch.setattr(reader.GMSTensorSpec, "load_all", lambda manager: specs)
    model = torch.nn.Module()
    for name in "abc":
        setattr(model, name, torch.nn.Module())
        getattr(model, name).weight = torch.nn.Parameter(torch.empty(4, device="meta"))
    published = {"aliases": {name: name for name in specs}, "storage_aliases": {}}
    if bulk_import is None:  # a manager without bulk import
        manager = SimpleNamespace()
    else:
        manager = SimpleNamespace(
            bulk_import=bulk_import,
            import_allocations=lambda ids: calls.append(("import", sorted(ids))),
        )
    reader._attach(manager, model, published, 0)
    materialized = [call for call in calls if call[0] == "materialize"]
    assert len(materialized) == 3
    if bulk_import:
        assert calls[0] == ("import", ["x", "x", "y"])
        assert calls[1:] == materialized
    else:
        assert calls == materialized
    assert not model.c.weight.is_meta


def test_zero_element_meta_tensors_are_materialized_and_others_reported():
    model = _layer()
    model.register_buffer("rope", torch.empty(8, 0, device="meta"), persistent=False)
    model.register_buffer("real", torch.empty(8, 2, device="meta"), persistent=False)
    placeholders = [torch.empty(0, device="meta")]
    model.kv_cache = placeholders
    assert reader._remaining_meta(model, torch.device("cpu")) == ["real"]
    assert model.rope.device.type == "cpu" and tuple(model.rope.shape) == (8, 0)
    assert model.kv_cache is placeholders
    assert placeholders[0].device.type == "cpu"


def _toy(method_cls=_Method):
    layer = _layer(method_cls())
    layer.register_buffer("counter", torch.zeros(1))
    return layer


def _load(layer):
    """What the writer's loading leaves outside the tensors."""
    layer.scale_float = 0.5
    layer.kernel_built_pre_processed = True


def _published(load=_load):
    """A writer's model after ``load``, and the descriptor it publishes."""
    writer = _toy()
    construct = state.snapshot(writer)
    load(writer)
    final = state.snapshot(writer)
    tensors = {
        "weight": ("parameter", writer.weight),
        "counter": ("buffer", writer.counter),
    }
    published = {
        "identity": {"model": "toy"},
        "sources": {},
        "modules": {name: module_row(m) for name, m in writer.named_modules()},
        "aliases": {name: name for name in tensors},
        "storage_aliases": {},
        "final": final,
        "changed": state.diff(construct, final),
    }
    return writer, published, tensors


@pytest.fixture
def restore(monkeypatch):
    """Run ``reader.restore`` on the CPU: ``restore(published, tensors)``."""
    cls = expert_map_manager.ExpertMapManager
    monkeypatch.setattr(cls, "_calculate_expert_maps", cls._calculate_expert_maps)
    monkeypatch.setattr(patches, "_expert_map_device_patched", False)
    monkeypatch.setattr(
        reader, "get_vmm_device_type", lambda: SimpleNamespace(value="cpu")
    )

    def run(published, tensors, identity=None, build=_toy):
        identity = identity or published["identity"]
        monkeypatch.setattr(reader.descriptor, "get", lambda manager: published)
        monkeypatch.setattr(reader.descriptor, "identity", lambda *args: identity)
        monkeypatch.setattr(reader, "initialize_model", lambda **kwargs: build())
        specs = {
            name: SimpleNamespace(
                meta=SimpleNamespace(tensor_type=kind),
                materialize=lambda manager, device, t=tensor: t.detach().clone(),
            )
            for name, (kind, tensor) in tensors.items()
        }
        monkeypatch.setattr(reader.GMSTensorSpec, "load_all", lambda manager: specs)
        model_config = SimpleNamespace(dtype=torch.float32)
        return reader.restore(SimpleNamespace(), None, model_config, 0)

    return run


def test_restore_rebuilds_the_writers_state_on_a_meta_model(restore):
    writer, published, tensors = _published()

    model = restore(published, tensors)

    assert torch.equal(model.weight, writer.weight)
    assert isinstance(model.weight, torch.nn.Parameter) and not model.training
    assert model.kernel_built_pre_processed is True  # rebuilt by the quant method
    assert model.scale_float == 0.5  # applied from the diff
    # The engine writes its private tensors after the attach.
    assert model.counter.add_(1).tolist() == [1.0]


@pytest.mark.parametrize(
    "case, error",
    [
        ("another identity", "identity mismatch"),
        ("a tensor the writer did not describe", "tensor schema mismatch"),
        ("another parameter set", "parameter schema mismatch"),
        ("another quant method", "module mismatch"),
        ("an object the reader does not rebuild", "could not reproduce"),
    ],
)
def test_restore_refuses_what_it_cannot_reproduce(restore, case, error):
    def load(layer):
        _load(layer)
        if case == "an object the reader does not rebuild":
            layer.handle = _Method()

    _, published, tensors = _published(load)
    identity, build = None, _toy
    if case == "another identity":
        identity = {"model": "other"}
    elif case == "a tensor the writer did not describe":
        del published["aliases"]["counter"]
    elif case == "another parameter set":
        published["modules"][""]["params"] = []
    elif case == "another quant method":

        def build():
            return _toy(_Unsupported)

    with pytest.raises(RuntimeError, match=error):
        restore(published, tensors, identity, build)


def test_read_mode_uses_the_load_diff_reader_only_when_selected(monkeypatch):
    calls = []
    model = _layer()
    monkeypatch.setattr(
        reader, "restore", lambda *args: calls.append("load diff") or model
    )
    monkeypatch.setattr(
        model_loader, "_create_meta_model", lambda *args: calls.append("meta") or model
    )
    monkeypatch.setattr(
        model_loader,
        "materialize_module_from_gms",
        lambda *args, **kwargs: calls.append("attach"),
    )
    monkeypatch.setattr(model_loader, "get_mx_load_context", lambda *args: None)
    monkeypatch.setattr(model_loader, "_last_imported_weights_bytes", 0)
    monkeypatch.setattr(model_loader, "_last_model_memory_usage_offset_bytes", 0)
    closed = []
    client = SimpleNamespace(total_bytes=0, close=lambda: closed.append(True))

    def read(extra_config):
        load_config = SimpleNamespace(model_loader_extra_config=extra_config)
        vllm_config = SimpleNamespace(load_config=load_config)
        return model_loader._load_read_mode(client, vllm_config, None, 0)

    read({})
    assert calls == ["meta", "attach"]
    read({"gms_reconstruction_mode": "load_diff"})
    assert calls == ["meta", "attach", "load diff"] and not closed

    def refuse(*args):
        raise RuntimeError("could not reproduce")

    monkeypatch.setattr(reader, "restore", refuse)
    with pytest.raises(RuntimeError, match="could not reproduce"):
        read({"gms_reconstruction_mode": "load_diff"})
    assert closed == [True]


def test_expert_maps_are_built_on_the_given_device_under_meta(monkeypatch):
    """vLLM builds EP expert maps on the CPU when the default device is meta;
    the reader needs them where the writer's are."""
    cls = expert_map_manager.ExpertMapManager
    monkeypatch.setattr(cls, "_calculate_expert_maps", cls._calculate_expert_maps)
    monkeypatch.setattr(patches, "_expert_map_device_patched", False)
    seen = []

    def determine_expert_map(**kwargs):
        seen.append(torch.get_default_device().type)
        return 1, None, None

    monkeypatch.setattr(
        expert_map_manager, "determine_expert_map", determine_expert_map
    )
    stub = SimpleNamespace(
        ep_size=2,
        ep_rank=0,
        global_num_experts=2,
        _placement_strategy="linear",
        num_fused_shared_experts=0,
        rocm_aiter_enabled=False,
    )
    with torch.device("meta"):
        with patches.expert_maps_on(torch.device("cuda", 0)):
            cls._calculate_expert_maps(stub)
        cls._calculate_expert_maps(stub)
    assert seen == ["cuda", "cpu"]
