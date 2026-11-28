# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: snapshots, the diff, and applying it to another model."""

import enum
import json
import math
import sys

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.integrations.vllm.load_diff import state

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


class Kernel:
    pass


class Method:
    def __init__(self):
        self.backend_ready = False


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.quant_method = Method()
        self.calculate_scales = True
        self.placeholder = 1.0
        self.dims = (2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()


class Backend(enum.Enum):
    TRITON = "triton"
    FLASHINFER = "flashinfer"


def _handler(*args, **kwargs):
    return None


def _load(model):
    """What a writer's load_weights + process_weights_after_loading does."""
    layer = model.layer
    layer._scale_float = 0.5
    layer.calculate_scales = False
    del layer.placeholder
    layer.dims = (4, 2)
    layer.quant_method.kernel = Kernel()
    layer.quant_method.backend_ready = True


def _rebuild(model):
    """What a reader's quant-method rebuild recreates."""
    model.layer.quant_method.kernel = Kernel()
    model.layer.quant_method.backend_ready = True


def _published(writer, load):
    construct = state.snapshot(writer)
    load(writer)
    final = state.snapshot(writer)
    # The diff travels as JSON with sorted keys, so that is the order apply
    # sees attributes in.
    published = {"final": final, "changed": state.diff(construct, final)}
    return json.loads(json.dumps(published, sort_keys=True))


def _apply(published, reader):
    return state.apply(reader, published["final"], published["changed"])


def test_diff_lists_only_what_loading_changed():
    changed = _published(Model(), _load)["changed"]
    assert set(changed["layer|module"]) == {
        "_scale_float",
        "calculate_scales",
        "placeholder",
        "dims",
    }
    assert set(changed["layer|quant_method"]) == {"kernel", "backend_ready"}
    assert "|module" not in changed


def test_apply_sets_load_time_values_and_deletions():
    published = _published(Model(), _load)
    reader = Model()
    _rebuild(reader)
    report = _apply(published, reader)
    assert not report.fatal and not report.nondeterministic
    assert reader.layer._scale_float == 0.5
    assert reader.layer.calculate_scales is False
    assert reader.layer.dims == (4, 2) and isinstance(reader.layer.dims, tuple)
    assert not hasattr(reader.layer, "placeholder")
    assert report.deleted == ["layer|module.placeholder"]


def test_numeric_types_are_told_apart():
    """1, 1.0 and True are equal in Python, but not the same value."""

    def build():
        model = Model()
        model.layer.flag = 1
        model.layer.ratio = 1
        model.layer.pair = [0, 1]
        return model

    def load(model):
        model.layer.flag = True
        model.layer.ratio = 1.0
        model.layer.pair = [False, 1.0]

    published = _published(build(), load)
    assert set(published["changed"]["layer|module"]) == {"flag", "ratio", "pair"}
    reader = build()
    report = _apply(published, reader)
    assert not report.fatal and len(report.applied) == 3
    assert reader.layer.flag is True and type(reader.layer.ratio) is float
    assert [type(x) for x in reader.layer.pair] == [bool, float]

    # Nor is True the construction value 1: the target rebuilt it.
    reader = build()
    reader.layer.ratio = True
    report = _apply(published, reader)
    assert [r[0] for r in report.conflicts] == ["layer|module.ratio"]
    assert reader.layer.ratio is True


def test_object_loading_created_that_the_target_lacks_is_fatal():
    report = _apply(_published(Model(), _load), Model())
    assert report.missing_objects == [
        ("layer|quant_method.kernel", {"absent": True}, {"o": f"{__name__}.Kernel"})
    ]
    assert report.fatal


def test_owner_the_target_lacks_is_fatal():
    published = _published(Model(), _load)
    published["final"]["gone|module"] = {}
    reader = Model()
    del reader.layer.quant_method
    report = _apply(published, reader)
    assert sorted(report.missing_owners) == ["gone|module", "layer|quant_method"]
    assert report.fatal


def test_rebuilt_value_that_differs_from_the_writer_is_a_conflict():
    """Neither value can be trusted (e.g. a process-local handle): fail."""
    published = _published(Model(), _load)
    reader = Model()
    _rebuild(reader)
    reader.layer._scale_float = 0.25
    report = _apply(published, reader)
    assert report.conflicts == [("layer|module._scale_float", {"v": 0.25}, {"v": 0.5})]
    assert report.fatal and reader.layer._scale_float == 0.25


def test_rebuilt_value_that_loading_deleted_is_a_conflict():
    def build(device="cpu", index=0):
        model = Model()
        model.layer.scratch = torch.zeros(2, device=device)
        model.layer.cache = {"k": torch.zeros(2, device=device)}
        model.layer.target = torch.device("cuda", index)
        return model

    def load(model):
        del model.layer.placeholder
        del model.layer.scratch, model.layer.cache, model.layer.target

    published = _published(build(), load)
    # What differs between two processes is not a rebuild: tensors that meta
    # construction left on meta, also inside a container, and a device index.
    reader = build("meta", 1)
    reader.layer.placeholder = 9.0
    report = _apply(published, reader)
    assert report.conflicts == [
        ("layer|module.placeholder", {"v": 9.0}, {"absent": True})
    ]
    assert report.fatal and reader.layer.placeholder == 9.0
    assert report.deleted == [
        "layer|module.cache",
        "layer|module.scratch",
        "layer|module.target",
    ]

    # A target that never had the attribute needs nothing.
    reader = build()
    del reader.layer.placeholder
    report = _apply(published, reader)
    assert not report.fatal and len(report.deleted) == 3


def test_difference_loading_did_not_make_is_reported_not_applied():
    published = _published(Model(), _load)
    reader = Model()
    _rebuild(reader)
    reader.flag = 3
    published["final"]["|module"]["flag"] = {"v": 7}
    report = _apply(published, reader)
    assert report.nondeterministic == [("|module.flag", {"v": 3}, {"v": 7})]
    assert reader.flag == 3 and not report.fatal


def test_object_loading_wired_into_a_new_attribute_is_relinked():
    """E.g. a kernel stamping itself onto its layer during loading."""

    def build():
        model = Model()
        model.layer.quant_method.linear = Kernel()
        return model

    def load(model):
        model.layer.provider = model.layer.quant_method.linear

    published = _published(build(), load)
    assert published["final"]["layer|module"]["provider"]["ref"] == (
        "quant_method.linear"
    )
    reader = build()
    report = _apply(published, reader)
    assert not report.fatal
    assert report.linked == ["layer|module.provider -> quant_method.linear"]
    assert reader.layer.provider is reader.layer.quant_method.linear


def test_rebuilt_value_is_not_relinked_over():
    def build(device="cpu"):
        model = Model()
        model.layer.provider = None
        model.layer.workspace = torch.zeros(2, device=device)
        model.layer.quant_method.linear = Kernel()
        return model

    def load(model):
        linear = model.layer.quant_method.linear
        model.layer.provider = model.layer.workspace = linear

    published = _published(build(), load)
    reader = build()
    reader.layer.provider = 7
    report = _apply(published, reader)
    assert [r[0] for r in report.conflicts] == ["layer|module.provider"]
    assert report.fatal and reader.layer.provider == 7

    # The construction value is still re-linked over, and so is a tensor that
    # meta construction left on meta: the rebuild did not make it.
    reader = build("meta")
    report = _apply(published, reader)
    assert not report.fatal and len(report.linked) == 2
    linear = reader.layer.quant_method.linear
    assert reader.layer.provider is linear and reader.layer.workspace is linear


def test_two_objects_where_loading_left_one_are_fatal():
    """Two objects of one class have the same description as one object held
    twice, so the link is checked by identity. Nothing says which of the two
    the writer kept, so they are not re-linked."""

    def build():
        model = Model()
        model.layer.provider = Kernel()
        model.layer.quant_method.linear = Kernel()
        return model

    def load(model):
        model.layer.provider = model.layer.quant_method.linear

    published = _published(build(), load)
    reader = build()
    provider = reader.layer.provider
    report = _apply(published, reader)
    assert [r[0] for r in report.missing_objects] == [
        "layer|module.provider",
        "layer|quant_method.linear",
    ]
    assert report.fatal and not report.linked and reader.layer.provider is provider

    # A target that holds one object in both places matches.
    reader = build()
    load(reader)
    report = _apply(published, reader)
    assert not report.fatal and not report.nondeterministic

    # A link that loading did not make is reported only.
    published["changed"] = {}
    report = _apply(published, build())
    assert not report.fatal
    assert [r[0] for r in report.nondeterministic] == [
        "layer|module.provider",
        "layer|quant_method.linear",
    ]


def test_object_at_its_home_is_linked_to_two_more_attributes():
    def build():
        model = Model()
        model.layer.quant_method.linear = Kernel()
        return model

    def load(model):
        linear = model.layer.quant_method.linear
        model.layer.provider = model.layer.quant_method.fallback = linear

    reader = build()
    report = _apply(_published(build(), load), reader)
    assert not report.fatal and len(report.linked) == 2
    linear = reader.layer.quant_method.linear
    assert reader.layer.provider is linear
    assert reader.layer.quant_method.fallback is linear


def test_owner_relinked_during_the_pass_is_described_as_it_is_then():
    """Loading can replace an owner with an object the module already holds.
    Its row then describes the new object, not the one that was there."""

    def build():
        model = Model()
        model.layer.spare = Kernel()
        return model

    def load(model):
        model.layer.quant_method = model.layer.spare

    reader = build()
    report = _apply(_published(build(), load), reader)
    assert not report.fatal
    assert report.linked == ["layer|module.quant_method -> module.spare"]
    assert reader.layer.quant_method is reader.layer.spare


def test_shared_objects_compare_without_their_reference():
    """A shared object records where else it is held; the target's own
    description of it does not, and must still match."""

    def build():
        model = Model()
        model.layer.config = model.layer.quant_method.config = Kernel()
        model.layer.cached = model.layer.quant_method.cached = Kernel()
        return model

    def load(model):
        model.layer.cached = None

    published = _published(build(), load)
    assert "ref" in published["final"]["layer|quant_method"]["config"]
    reader = build()
    report = _apply(published, reader)
    assert not report.fatal and not report.nondeterministic
    assert report.applied == ["layer|module.cached"] and reader.layer.cached is None


def test_containers_loading_changed_in_place_are_applied():
    def build():
        model = Model()
        model.layer.ids = list(range(100))
        model.layer.table = {"a": 1}
        model.layer.names = {"x"}
        model.layer.pending = None
        return model

    def load(model):
        model.layer.ids[3] = -1
        model.layer.table["a"] = 2
        model.layer.names.add("y")
        model.layer.pending = {}

    reader = build()
    report = _apply(_published(build(), load), reader)
    assert not report.fatal
    assert reader.layer.ids[3] == -1 and reader.layer.table == {"a": 2}
    assert reader.layer.names == {"x", "y"} and reader.layer.pending == {}


def test_applied_containers_are_not_shared_with_the_snapshot():
    def build():
        model = Model()
        model.layer.ids = [1, 2]
        model.layer.table = {"a": 1}
        return model

    def load(model):
        model.layer.ids[0] = 9
        model.layer.table["a"] = 2

    published = _published(build(), load)
    first = build()
    assert not _apply(published, first).fatal
    first.layer.ids.append(7)
    first.layer.table["a"] = 3
    second = build()
    assert not _apply(published, second).fatal
    assert second.layer.ids == [9, 2] and second.layer.table == {"a": 2}


def test_hook_registries_are_compared_by_position_not_by_handle_id():
    """Hook registries are keyed by process-global handle ids, so a hook is
    identified by its position, also where a registry flags hooks. Any other
    table keeps its keys, also one from ints to callables."""

    def build():
        model = Model()
        model.layer.dispatch = {1: _handler}
        return model

    def load(model, kwargs_first=True):
        model.layer.dispatch = {2: _handler}
        model.layer.register_forward_hook(_handler, with_kwargs=kwargs_first)
        model.layer.register_forward_hook(_handler, with_kwargs=not kwargs_first)

    published = _published(build(), load)
    assert "dispatch" in published["changed"]["layer|module"]
    report = _apply(published, build())
    assert "layer|module.dispatch" in [r[0] for r in report.missing_objects]

    rebuilt = build()
    # Another registration first, so the handle ids differ from the writer's.
    rebuilt.register_forward_hook(_handler).remove()
    load(rebuilt)
    assert not _apply(published, rebuilt).fatal

    swapped = build()
    load(swapped, kwargs_first=False)
    report = _apply(published, swapped)
    assert [r[0] for r in report.missing_objects] == [
        "layer|module._forward_hooks_with_kwargs"
    ]

    # A flag whose hook is gone is ignored, as torch ignores it.
    stale = build()
    stale.layer.register_forward_hook(_handler, with_kwargs=True)
    stale.layer._forward_hooks.clear()
    load(stale)
    assert not _apply(published, stale).fatal


def test_container_state_that_cannot_travel_must_be_rebuilt():
    """A hook registered during loading holds a callable: detected by digest."""

    def load(model):
        model.layer.register_forward_pre_hook(lambda module, args: None)

    published = _published(Model(), load)
    report = _apply(published, Model())
    assert [r[0] for r in report.missing_objects] == ["layer|module._forward_pre_hooks"]
    assert report.fatal
    rebuilt = Model()
    load(rebuilt)
    assert not _apply(published, rebuilt).fatal


def test_cpu_tensors_travel_bit_exact_and_large_ones_by_digest():
    def build():
        model = Model()
        model.layer.fp8 = torch.zeros(4, dtype=torch.float8_e4m3fn)
        model.layer.bf16 = torch.zeros(3, dtype=torch.bfloat16)
        model.layer.big = torch.zeros(state._MAX_CPU_VALUES + 1)
        return model

    fp8 = torch.tensor([0.5, -1.0, 448.0, 0.0]).to(torch.float8_e4m3fn)

    def load(model):
        model.layer.fp8 = fp8.clone()
        model.layer.bf16.copy_(torch.tensor([1.0, 2.5, -3.0]))
        model.layer.big[0] = 7.0

    reader = build()
    report = _apply(_published(build(), load), reader)
    assert sorted(report.restored) == ["layer|module.bf16", "layer|module.fp8"]
    assert torch.equal(reader.layer.fp8.view(torch.uint8), fp8.view(torch.uint8))
    assert reader.layer.bf16.tolist() == [1.0, 2.5, -3.0]
    assert [r[0] for r in report.missing_tensors] == ["layer|module.big"]
    assert report.fatal

    # A tensor the target rebuilt to a third value is kept and reported.
    rebuilt = build()
    rebuilt.layer.bf16.fill_(9.0)
    report = _apply(_published(build(), load), rebuilt)
    assert "layer|module.bf16" in [r[0] for r in report.missing_tensors]
    assert rebuilt.layer.bf16.tolist() == [9.0, 9.0, 9.0]


def test_tensor_restored_in_place_must_suit_every_holder():
    """A CPU tensor that two attributes share is restored in place, so both
    see it. If loading gave only one of them a new tensor, the other would be
    left with a value the writer never had."""

    def build():
        model = Model()
        model.layer.table = model.layer.quant_method.table = torch.zeros(2)
        return model

    def load_in_place(model):
        model.layer.table.copy_(torch.tensor([5.0, 6.0]))

    def load_one_holder(model):
        model.layer.table = torch.tensor([5.0, 6.0])

    reader = build()
    assert not _apply(_published(build(), load_in_place), reader).fatal
    assert reader.layer.quant_method.table is reader.layer.table
    assert reader.layer.table.tolist() == [5.0, 6.0]

    report = _apply(_published(build(), load_one_holder), build())
    assert [r[0] for r in report.conflicts] == ["layer|quant_method.table"]
    assert report.fatal

    # A holder that loading gave another value still held the construction
    # tensor before the restore, so the value is applied.
    def load_and_drop_a_holder(model):
        model.layer.table.copy_(torch.tensor([5.0, 6.0]))
        model.layer.quant_method.table = None

    reader = build()
    assert not _apply(_published(build(), load_and_drop_a_holder), reader).fatal
    assert reader.layer.table.tolist() == [5.0, 6.0]
    assert reader.layer.quant_method.table is None


def test_holders_that_still_share_a_tensor_are_not_split():
    """Loading gives one of three holders a new tensor and fills the old one.
    Nothing says which holders still share it, so none gets its own tensor."""

    def build():
        model = Model()
        model.layer.a = model.layer.b = model.layer.c = torch.zeros(2)
        return model

    def load(model):
        model.layer.a = torch.tensor([5.0, 6.0])
        model.layer.b.fill_(9.0)

    reader = build()
    report = _apply(_published(build(), load), reader)
    assert [r[0] for r in report.conflicts] == ["layer|module.b", "layer|module.c"]
    assert report.fatal and reader.layer.b is reader.layer.c


@pytest.mark.parametrize("view", ["head", "tail"])
def test_view_of_a_tensor_restored_in_place_must_suit_too(view):
    """A view shares its base's storage, so an in-place restore of the base
    changes it like any other holder. ``head`` is applied before ``table``
    and ``tail`` after it."""

    def build():
        model = Model()
        model.layer.table = torch.zeros(2)
        setattr(model.layer, view, model.layer.table[:1])
        # Described by a digest of the tensor's contents.
        model.layer.columns = [model.layer.table, "x"]
        return model

    def load_in_place(model):
        model.layer.table.copy_(torch.tensor([5.0, 6.0]))

    def load_base_only(model):
        model.layer.table = torch.tensor([5.0, 6.0])

    reader = build()
    assert not _apply(_published(build(), load_in_place), reader).fatal
    assert reader.layer.table.tolist() == [5.0, 6.0]
    assert vars(reader.layer)[view].tolist() == [5.0]

    writer = build()
    published = _published(writer, load_base_only)
    assert vars(writer.layer)[view].tolist() == [0.0]
    report = _apply(published, build())
    assert [r[0] for r in report.conflicts] == [f"layer|module.{view}"]
    assert report.fatal


def test_constructor_cpu_tensor_left_on_meta_is_restored():
    """Meta construction leaves constructor-made CPU tensors on meta."""
    writer = Model()
    writer.layer._k_scale_cpu = torch.tensor([1.0])
    published = _published(writer, lambda model: None)
    reader = Model()
    reader.layer._k_scale_cpu = torch.empty(1, device="meta")
    report = _apply(published, reader)
    assert report.restored == ["layer|module._k_scale_cpu"] and not report.fatal
    assert reader.layer._k_scale_cpu.tolist() == [1.0]


def test_enum_and_dtype_tokens_are_applied_or_fatal(tmp_path, monkeypatch):
    def build():
        model = Model()
        model.layer.quant_method.backend = Backend.TRITON
        model.layer.out_dtype = torch.float16
        return model

    def load(model):
        model.layer.quant_method.backend = Backend.FLASHINFER
        model.layer.out_dtype = torch.bfloat16

    published = _published(build(), load)
    reader = build()
    assert not _apply(published, reader).fatal
    assert reader.layer.quant_method.backend is Backend.FLASHINFER
    assert reader.layer.out_dtype is torch.bfloat16

    # An enum whose module this process has not imported is never imported,
    # even when the module is importable.
    (tmp_path / "load_diff_not_imported.py").write_text(
        "import enum\n\n\nclass Backend(enum.Enum):\n    FLASHINFER = 'flashinfer'\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    desc = published["final"]["layer|quant_method"]["backend"]
    desc["token"] = desc["token"].replace(__name__, "load_diff_not_imported")
    report = _apply(published, build())
    assert [r[0] for r in report.unapplied_tokens] == ["layer|quant_method.backend"]
    assert report.fatal and "load_diff_not_imported" not in sys.modules


def test_nan_and_device_tokens():
    def build():
        model = Model()
        model.layer.eps = 0.0
        model.layer.target = None
        return model

    def load(model):
        model.layer.eps = float("nan")
        model.layer.target = torch.device("cuda:0")

    published = _published(build(), load)
    reader = build()
    reader.layer.target = torch.device("cuda:1")  # the index is process-local
    assert not _apply(published, reader).fatal
    assert math.isnan(reader.layer.eps)
    assert reader.layer.target == torch.device("cuda:1")

    # A device is never applied: the target has to have set its own.
    report = _apply(published, build())
    assert [r[0] for r in report.conflicts] == ["layer|module.target"]
    assert report.fatal
