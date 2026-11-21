# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reader side: attach a model read-only from GMS and apply the load diff.

1. Fetch the descriptor; refuse a different identity or different sources.
2. Construct the model on meta. Check module classes and expert routing, and
   drop the parameters that the writer's loading deleted.
3. Attach the GMS tensors. Parameters alias the read-only weights; buffers and
   tensor attributes are private copies, because the engine writes them.
4. Rebuild Python-side objects with each quant method's
   ``process_weights_after_loading`` under vLLM's
   ``weights_already_processed()``, which skips the tensor rewrites.
5. Apply the diff and verify. Anything loading created that the reader does
   not have, a rebuilt value that differs from the writer's, or a tensor left
   on meta fails the load. There is no fallback: a read-only engine has no
   weights of its own to post-process.

Imports vLLM; the GMS model loader imports this module only in load-diff mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.module import _resolve_module_attr
from gpu_memory_service.client.torch.tensor import GMSTensorSpec
from gpu_memory_service.common.vmm import get_vmm_device_type
from gpu_memory_service.integrations.vllm.load_diff import descriptor
from gpu_memory_service.integrations.vllm.load_diff.state import apply, owners
from gpu_memory_service.integrations.vllm.load_diff.writer import module_row
from gpu_memory_service.integrations.vllm.patches import expert_maps_on
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.utils import weights_already_processed
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)


def restore(
    manager: "GMSClientMemoryManager", vllm_config, model_config, device_index: int
) -> torch.nn.Module:
    """Build the model and attach it read-only from the writer's descriptor."""
    published = descriptor.get(manager)
    descriptor.check_match(
        published["identity"],
        descriptor.identity(vllm_config, model_config),
        "identity",
    )
    # Before construction: a source mismatch is cheap to find here.
    sources = published["sources"]
    descriptor.check_match(sources, descriptor.source_hashes(sources), "source")

    device = torch.device(get_vmm_device_type().value, device_index)
    with expert_maps_on(device), set_default_torch_dtype(model_config.dtype):
        with torch.device("meta"):
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)
    modules = dict(model.named_modules())
    _check_modules(modules, published["modules"])
    _attach(manager, model, published, device_index)
    _rebuild(modules)
    _materialize_empty(model, device)
    report = apply(model, published["final"], published["changed"])
    loaded_sources = set(sources) | descriptor.source_modules(model)
    descriptor.check_match(sources, descriptor.source_hashes(loaded_sources), "source")

    for name, module in modules.items():
        params = {k for k, v in module._parameters.items() if v is not None}
        if params != set(published["modules"][name]["params"]):
            raise RuntimeError(f"Load-diff parameter schema mismatch: {name}")
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()
    still_meta = _remaining_meta(model, device)

    logger.info("[GMS] load diff: %s", report.summary())
    if report.nondeterministic:
        logger.info(
            "[GMS] load diff: %d values differ from the writer's where loading "
            "did not change them; kept: %s",
            len(report.nondeterministic),
            report.nondeterministic[:10],
        )
    logger.debug(
        "[GMS] load diff: applied %s, restored %s, linked %s, deleted %s",
        report.applied,
        report.restored,
        report.linked,
        report.deleted,
    )
    if report.fatal or still_meta:
        raise RuntimeError(
            "Load-diff attach could not reproduce what the writer's loading "
            f"created: owners {report.missing_owners[:5]}, objects "
            f"{report.missing_objects[:10]}, tensors {report.missing_tensors[:10]}, "
            f"conflicts {report.conflicts[:10]}, unapplied tokens "
            f"{report.unapplied_tokens[:10]}, still on meta {still_meta[:10]}"
        )
    return model.eval()


def _check_modules(modules: dict, published: dict) -> None:
    """Same modules, classes and expert routing as the writer; drop the
    parameters that the writer's loading deleted."""
    if set(modules) != set(published):
        extra = sorted(set(modules) ^ set(published))
        raise RuntimeError(f"Load-diff module schema mismatch: {extra[:10]}")
    for name, module in modules.items():
        row, mine = published[name], module_row(module)
        for key in ("cls", "quant", "impl", "routing"):
            if row[key] != mine[key]:
                raise RuntimeError(
                    f"Load-diff module mismatch at {name!r}: {key} "
                    f"{row[key]!r} (writer) != {mine[key]!r} (reader)"
                )
        for key in mine["params"]:
            if key not in row["params"]:
                del module._parameters[key]


def _current(owner, key: str):
    if isinstance(owner, (list, tuple)):
        return owner[int(key)]
    # Absent until attach for buffers that the writer's loading registered.
    return getattr(owner, key, None)


def _attach(manager, model: torch.nn.Module, published: dict, device_index: int):
    specs = GMSTensorSpec.load_all(manager)
    aliases = published["aliases"]
    if set(specs) != set(aliases):
        extra = sorted(set(specs) ^ set(aliases))
        raise RuntimeError(f"Load-diff tensor schema mismatch: {extra[:10]}")
    # As the default attach does: with bulk import on, import every referenced
    # allocation in one call before any tensor is bound, and materialize below
    # reuses the mappings. A manager without bulk import maps one allocation per
    # tensor in materialize.
    if getattr(manager, "bulk_import", False):
        manager.import_allocations(spec.allocation_id for spec in specs.values())
    # A name the writer registered as a parameter makes its whole alias
    # group one parameter, whichever name comes first.
    parameters = {
        aliases[name]
        for name, spec in specs.items()
        if spec.meta.tensor_type == "parameter"
    }
    attached, derived = {}, []
    # Not inference_mode: the engine later writes private tensors in place.
    with torch.no_grad():
        for name, spec in specs.items():
            owner, key = _resolve_module_attr(model, name)
            tensor = attached.get(aliases[name])
            if tensor is None:
                current = _current(owner, key)
                as_parameter = aliases[name] in parameters
                tensor = _materialize(
                    manager, spec, current, name, device_index, as_parameter
                )
                attached[aliases[name]] = tensor
            if isinstance(owner, list):
                owner[int(key)] = tensor
            elif isinstance(owner, tuple):
                if owner[int(key)] is not tensor:
                    _replace_in_tuple(model, name, tensor)
            elif spec.meta.tensor_type == "parameter":
                owner._parameters[key] = tensor
            elif spec.meta.tensor_type == "buffer" or key in owner._buffers:
                # Includes buffers that the writer's loading registered.
                owner._buffers[key] = tensor
            elif _read_only_property(owner, key):
                # Derived from a tensor attached under its own name (e.g.
                # MoERunner.expert_map); checked once everything is attached.
                derived.append((name, owner, key, tensor))
            else:
                setattr(owner, key, tensor)
    for name, owner, key, tensor in derived:
        if _current(owner, key) is not tensor:
            raise RuntimeError(f"Load-diff read-only attribute mismatch: {name}")
    for name, (canonical, offset) in published["storage_aliases"].items():
        a, b = _resolve_module_attr(model, name), _resolve_module_attr(model, canonical)
        if isinstance(a[0], (list, tuple)) or isinstance(b[0], (list, tuple)):
            continue  # list elements are attached one by one
        if _current(*a).data_ptr() - _current(*b).data_ptr() != offset:
            raise RuntimeError(f"Load-diff private storage alias mismatch: {name}")


def _read_only_property(owner, key: str) -> bool:
    attr = getattr(type(owner), key, None)
    return isinstance(attr, property) and attr.fset is None


def _replace_in_tuple(model: torch.nn.Module, name: str, tensor) -> None:
    container, _, index = name.rpartition(".")
    owner, key = _resolve_module_attr(model, container)
    if _read_only_property(owner, key):
        raise RuntimeError(f"Load diff cannot attach into a read-only tuple: {name}")
    items = list(_current(owner, key))
    items[int(index)] = tensor
    setattr(owner, key, tuple(items))


def _materialize(
    manager,
    spec: GMSTensorSpec,
    current,
    name: str,
    device_index: int,
    as_parameter: bool,
) -> torch.Tensor:
    source = spec.materialize(manager, device_index)
    if as_parameter:
        return torch.nn.Parameter(source, requires_grad=False)
    if current is None or current.is_meta:
        return source.clone()  # private: the engine may write it
    if current.shape != source.shape or current.dtype != source.dtype:
        raise RuntimeError(f"Load-diff private tensor layout mismatch: {name}")
    # In place, so that other holders of the constructed tensor see it.
    current.copy_(source)
    return current


def _rebuild(modules: dict) -> None:
    methods = {
        name: vars(m)["quant_method"]
        for name, m in modules.items()
        if isinstance(vars(m).get("quant_method"), QuantizeMethodBase)
    }
    unsupported = sorted(
        f"{type(method).__name__} ({name})"
        for name, method in methods.items()
        if not method.supports_pre_processed_weights
    )
    if unsupported:
        raise RuntimeError(
            "Load diff needs quant methods that support pre-processed "
            f"weights; unsupported: {unsupported[:10]}"
        )
    with torch.no_grad(), weights_already_processed():
        for name, method in methods.items():
            method.process_weights_after_loading(modules[name])


def _is_empty_meta(value) -> bool:
    return (
        torch.is_tensor(value)
        and value.is_meta
        and value.numel() == 0
        and not isinstance(value, torch.nn.Parameter)
    )


def _materialize_empty(model: torch.nn.Module, device: torch.device) -> None:
    """Materialize zero-element meta tensors on ``device``: they hold no data,
    so GMS publishes nothing for them."""
    for module in model.modules():
        for table in (module._parameters, module._buffers):
            for key, value in list(table.items()):
                if value is not None and value.is_meta and value.numel() == 0:
                    empty = torch.empty_like(value, device=device)
                    table[key] = (
                        torch.nn.Parameter(empty, requires_grad=False)
                        if isinstance(value, torch.nn.Parameter)
                        else empty
                    )
        for _, owner in owners(module):
            for key, value in list(vars(owner).items()):
                if _is_empty_meta(value):
                    setattr(owner, key, torch.empty_like(value, device=device))
                elif type(value) in (list, tuple) and any(map(_is_empty_meta, value)):
                    # E.g. attention kv_cache placeholders, bound later.
                    items = [
                        torch.empty_like(x, device=device) if _is_empty_meta(x) else x
                        for x in value
                    ]
                    if type(value) is list:
                        value[:] = items  # in place: other holders see it too
                    else:
                        setattr(owner, key, tuple(items))


def _remaining_meta(model: torch.nn.Module, device: torch.device) -> list[str]:
    """Names of tensors still on meta, after zero-element ones are materialized."""
    _materialize_empty(model, device)
    named = list(model.named_parameters()) + list(model.named_buffers())
    missing = [name for name, tensor in named if tensor.is_meta]
    for name, module in model.named_modules():
        for owner_name, owner in owners(module):
            for key, value in vars(owner).items():
                items = value if type(value) in (list, tuple) else (value,)
                if any(torch.is_tensor(x) and x.is_meta for x in items):
                    missing.append(f"{name}|{owner_name}.{key}")
    return missing
