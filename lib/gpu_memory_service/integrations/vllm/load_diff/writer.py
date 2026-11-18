# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writer side: publish what loading changed, next to the weights.

Selected with ``model_loader_extra_config={"gms_reconstruction_mode":
"load_diff"}``. The GMS writer then snapshots the model right after
construction and again after loading, and publishes the descriptor before the
layout is committed.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.module import _iter_module_tensors
from gpu_memory_service.integrations.vllm.load_diff import descriptor
from gpu_memory_service.integrations.vllm.load_diff.state import _cls, diff, snapshot

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

OPTION = "gms_reconstruction_mode"
MODE = "load_diff"


def selected(vllm_config) -> bool:
    """Whether the loader option selects load diff. Unknown values raise."""
    extra_config = vllm_config.load_config.model_loader_extra_config or {}
    mode = extra_config.get(OPTION)
    if mode is not None and mode != MODE:
        raise ValueError(f"{OPTION} must be {MODE!r} or unset, not {mode!r}")
    return mode == MODE


def module_row(module: torch.nn.Module) -> dict:
    """Classes, parameter names and expert routing of one module, compared
    by the reader before it attaches anything."""
    impl = vars(module).get("impl")
    return dict(
        cls=_cls(module),
        quant=_cls(vars(module).get("quant_method")),
        impl=_cls(impl) if impl is not None else None,
        params=[k for k, v in module._parameters.items() if v is not None],
        routing=routing_digest(module),
    )


def routing_digest(module: torch.nn.Module) -> dict | None:
    """Digest of the module's EP expert map, if it has one off meta."""
    expert_map = module._buffers.get("_expert_map")
    if expert_map is None or expert_map.is_meta:
        return None
    raw = expert_map.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return {
        "shape": list(expert_map.shape),
        "dtype": str(expert_map.dtype),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def descriptor_allocation(manager: "GMSClientMemoryManager", model) -> str:
    """An allocation that outlives the write: one a parameter lives in.

    The writer releases allocations that hold only non-parameter storage once
    it has copied that storage out, and GMS drops an allocation's metadata
    with it. The first mapping can be such an allocation.
    """
    spans = [
        (va, va + m.aligned_size, m.allocation_id) for va, m in manager.mappings.items()
    ]
    for param in model.parameters():
        ptr = param.data_ptr()
        for start, end, allocation_id in spans:
            if start <= ptr < end:
                return allocation_id
    raise RuntimeError("Load diff needs a model parameter in GMS memory")


def publish(
    manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    construct: dict,
    identity: dict,
) -> None:
    """Publish the load diff of ``model`` against its ``construct`` snapshot.

    Runs after the model's tensors are registered and before the writer
    copies non-parameters out of GMS.
    """
    final = snapshot(model)
    changed = diff(construct, final)
    registered = set(manager.metadata_list())
    aliases, storage_aliases = {}, {}
    objects, storages = {}, {}
    # Hold every tensor until the walk ends: properties can return new
    # tensors, whose ids would otherwise be reused by later ones.
    tensors = list(_iter_module_tensors(model))
    for name, tensor, kind in tensors:
        if name not in registered:
            continue
        # Names that were one tensor object stay one tensor on the reader.
        aliases[name] = objects.setdefault(int(tensor._cdata), name)
        if kind != "parameter":
            canonical, base = storages.setdefault(
                int(tensor.untyped_storage()._cdata), (name, tensor.data_ptr())
            )
            storage_aliases[name] = [canonical, tensor.data_ptr() - base]
    body = dict(
        identity=identity,
        sources=descriptor.source_hashes(descriptor.source_modules(model, final)),
        modules={name: module_row(m) for name, m in model.named_modules()},
        aliases=aliases,
        storage_aliases=storage_aliases,
        final=final,
        changed=changed,
    )
    stored = descriptor.put(manager, descriptor_allocation(manager, model), body)
    logger.info(
        "[GMS] load diff published: %d modules, %d attributes changed by loading "
        "in %d owners, %d source modules, %.1f KiB compressed",
        len(body["modules"]),
        sum(len(row) for row in changed.values()),
        len(changed),
        len(body["sources"]),
        stored / 1024,
    )
