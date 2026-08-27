# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module tensor operations for GPU Memory Service.

This module provides module-level tensor operations:
- Module tensor iteration
- Tensor registration (write path)
- Tensor materialization (read path)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Tuple

import torch
from gpu_memory_service.client.torch.tensor import GMSTensorSpec, TensorMetadata

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)


# =============================================================================
# Module Tensor Iteration
# =============================================================================


def _iter_module_tensors(
    module: torch.nn.Module,
    prefix: str = "",
) -> Iterator[Tuple[str, torch.Tensor, str]]:
    """Iterate over all CUDA tensors in a module tree.

    Yields (qualified_name, tensor, tensor_type) for:
    - Parameters (tensor_type="parameter")
    - Buffers (tensor_type="buffer")
    - Other tensor attributes like _k_scale (tensor_type="tensor_attr")

    Args:
        module: The nn.Module to iterate.
        prefix: Prefix for qualified names (used in recursion).

    Yields:
        (name, tensor, tensor_type) tuples for each CUDA tensor.
    """
    # Parameters
    for name, param in module._parameters.items():
        if param is not None and param.is_cuda:
            qualified = f"{prefix}{name}" if prefix else name
            yield (qualified, param, "parameter")

    # Buffers
    for name, buf in module._buffers.items():
        if buf is not None and buf.is_cuda:
            qualified = f"{prefix}{name}" if prefix else name
            yield (qualified, buf, "buffer")

    # Other tensor attributes (not params/buffers/submodules)
    skip = (
        set(module._parameters.keys())
        | set(module._buffers.keys())
        | set(module._modules.keys())
    )
    for attr_name in dir(module):
        if attr_name in skip or attr_name.startswith("__"):
            continue
        try:
            attr_val = getattr(module, attr_name, None)
        except Exception:
            continue

        if torch.is_tensor(attr_val) and attr_val.is_cuda:
            qualified = f"{prefix}{attr_name}" if prefix else attr_name
            yield (qualified, attr_val, "tensor_attr")
        elif isinstance(attr_val, (list, tuple)) and attr_val:
            if all(torch.is_tensor(x) and x.is_cuda for x in attr_val):
                for i, x in enumerate(attr_val):
                    qualified = (
                        f"{prefix}{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                    )
                    yield (qualified, x, "tensor_attr")

    # Recurse into submodules
    for name, submodule in module._modules.items():
        if submodule is not None:
            subprefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from _iter_module_tensors(submodule, subprefix)


_MLA_SCALE_LEAVES = frozenset({"q_scale", "k_scale", "v_scale", "prob_scale"})

# Buffers that several modules share on purpose, as a communication channel
# rather than as private state. Cloning these per published name severs the
# link between producer and consumer.
_SHARED_CHANNEL_LEAVES = frozenset({"topk_indices_buffer"})

# Tensors vLLM caches on plain (non-``nn.Module``) helper objects as aliases
# of the owning module's tensor. Name-keyed materialization cannot reach
# them, so ``_repoint_non_module_tensor_caches`` re-establishes the alias --
# but only for these, since a same-name match is not by itself evidence of
# an intended alias.
_REPOINTED_CACHE_LEAVES = frozenset(
    {
        # `MLAAttention.impl` -- consumer end of the DSA top-k channel.
        "topk_indices_buffer",
        # `FusedMoE.expert_map_manager` -- MoE expert routing map.
        "_expert_map",
    }
)


def _tensor_in_gms_mappings(
    gms_client_memory_manager: "GMSClientMemoryManager",
    tensor: torch.Tensor,
) -> bool:
    ptr = int(tensor.data_ptr())
    return any(
        va <= ptr < va + mapping.aligned_size
        for va, mapping in gms_client_memory_manager.mappings.items()
    )


def _bind_module_tensor(
    model: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    tensor_type: str,
) -> None:
    """Write ``tensor`` onto ``model`` at ``name``.

    If the destination is already a Parameter, replace the Parameter even
    when the GMS spec called it a tensor_attr. GLM MLA q/k/v/prob scales
    are Parameters on the RO meta model and plain CUDA tensors on the
    writer; treating the spec type as gospel left the Parameter on meta
    and zero-filled it.
    """
    mod, attr = _resolve_module_attr(model, name)
    if hasattr(mod, "_parameters") and attr in mod._parameters:
        param = mod._parameters[attr]
        requires_grad = bool(param.requires_grad) if param is not None else False
        mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return
    if (
        tensor_type == "buffer"
        and hasattr(mod, "_buffers")
        and attr in mod._buffers
    ):
        mod._buffers[attr] = tensor
        return
    if attr.isdigit() and not isinstance(mod, torch.nn.Module):
        if isinstance(mod, list):
            mod[int(attr)] = tensor
            return
        if isinstance(mod, tuple):
            container_name, _ = name.rsplit(".", 1)
            owner, container_attr = _resolve_module_attr(model, container_name)
            if isinstance(getattr(type(owner), container_attr, None), property):
                return
            elements = list(mod)
            elements[int(attr)] = tensor
            setattr(owner, container_attr, tuple(elements))
            return
        logger.debug("[GMS] Cannot bind container element %r", name)
        return
    if isinstance(getattr(type(mod), attr, None), property):
        logger.debug("[GMS] Skipping property attribute %r", name)
        return
    setattr(mod, attr, tensor)


def _resolve_module_attr(
    root: torch.nn.Module, qualified_name: str
) -> Tuple[torch.nn.Module, str]:
    """Resolve a dotted name to (parent_module, leaf_attr).

    Handles ModuleList/Sequential (numeric indices) and ModuleDict (key access).
    """
    parts = qualified_name.split(".")
    mod = root
    for p in parts[:-1]:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif hasattr(mod, "__getitem__"):
            try:
                mod = mod[int(p)] if p.isdigit() else mod[p]
            except Exception:
                raise AttributeError(f"Cannot resolve {p!r} in {qualified_name!r}")
        else:
            raise AttributeError(f"Cannot resolve {p!r} in {qualified_name!r}")
    return mod, parts[-1]


# =============================================================================
# Public API - Registration and Materialization
# =============================================================================


def register_module_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
) -> set[str]:
    """Register all model tensors into the GMS metadata store.

    Args:
        gms_client_memory_manager: GMS client memory manager in write mode.
        model: PyTorch model to register.

    Returns:
        Allocation IDs referenced by registered tensors.
    """
    referenced_allocation_ids: set[str] = set()
    for name, tensor, tensor_type in _iter_module_tensors(model):
        ptr = int(tensor.data_ptr())

        # Find allocation containing this tensor
        for va, mapping in gms_client_memory_manager.mappings.items():
            if va <= ptr < va + mapping.aligned_size:
                offset = ptr - va
                meta = TensorMetadata.from_tensor(tensor, tensor_type)
                gms_client_memory_manager.metadata_put(
                    key=name,
                    allocation_id=mapping.allocation_id,
                    offset_bytes=offset,
                    value=meta.to_bytes(),
                )
                referenced_allocation_ids.add(mapping.allocation_id)
                break
        else:
            # No mapping matched - tensor pointer not in any GMS allocation
            if tensor_type == "parameter":
                # Parameters are model weights - must be in GMS allocations
                raise RuntimeError(f"Tensor {name!r} not found in any GMS allocation")
            # Buffers and tensor_attrs may be dynamically allocated (e.g., KV cache)
            logger.debug(
                "[GMS] Skipping %s %r - not in GMS allocations", tensor_type, name
            )
    return referenced_allocation_ids


def _repoint_non_module_tensor_caches(model: torch.nn.Module) -> None:
    """Repoint tensors cached on plain objects hanging off the model.

    Materialization binds tensors by qualified name, which only reaches
    what ``named_modules`` walks. vLLM also caches tensors on plain
    objects -- ``MLAAttention.impl`` is an ``AttentionImpl``, not an
    ``nn.Module`` -- and those keep whatever the meta constructor
    produced. The DSA top-k channel shows the failure clearly: the
    indexer writes its indices into the layer's materialized
    ``topk_indices_buffer`` while ``impl`` still reads its own
    uninitialized one, so attention scores against arbitrary KV slots.

    The repair is deliberately restricted to ``_REPOINTED_CACHE_LEAVES``.
    Matching on attribute name alone is not evidence of an intended
    alias -- a helper object may hold its own ``weight`` or ``scale`` --
    and silently rebinding serving state on a false positive would be a
    correctness bug. Add a leaf here only with evidence that vLLM
    constructs it as an alias of the module's tensor.

    The writer avoids needing this by swapping storage under the existing
    TensorImpl (see ``rebind_nonparameter_tensors``). The reader cannot:
    it starts from a meta model, and PyTorch rejects both ``set_`` and
    ``.data =`` when they would move a tensor off the meta device, so
    materialization has to bind new objects and then repair the holders.
    """
    repointed: list[str] = []
    for mod_name, module in model.named_modules():
        for holder_attr, holder in list(vars(module).items()):
            if holder is None or isinstance(holder, (torch.nn.Module, torch.Tensor)):
                continue
            holder_vars = getattr(holder, "__dict__", None)
            if not holder_vars:
                continue
            for attr in _REPOINTED_CACHE_LEAVES & holder_vars.keys():
                current = holder_vars[attr]
                if not torch.is_tensor(current):
                    continue
                # Prefer the layer's own indexer: vLLM sources the sparse
                # buffers from there (sparse_mla_attention.py binds
                # `indexer.topk_indices_buffer`), else the module itself.
                replacement = None
                for source in (getattr(module, "indexer", None), module):
                    if source is None or source is holder:
                        continue
                    cand = getattr(source, attr, None)
                    if torch.is_tensor(cand):
                        replacement = cand
                        break
                if replacement is None or replacement is current:
                    continue
                if (
                    replacement.shape != current.shape
                    or replacement.dtype != current.dtype
                ):
                    continue
                if (
                    not current.is_meta
                    and replacement.data_ptr() == current.data_ptr()
                ):
                    continue
                setattr(holder, attr, replacement)
                repointed.append(f"{mod_name}.{holder_attr}.{attr}")
    if repointed:
        logger.info(
            "[GMS] Re-pointed %d tensors cached on non-Module objects: %s",
            len(repointed),
            repointed[:6],
        )


def _shared_clone(
    cache: dict[tuple[str, int], torch.Tensor],
    alias_counts: dict[tuple[str, int], int],
    spec: "GMSTensorSpec",
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Clone ``tensor`` off GMS memory, reusing one clone per location.

    vLLM shares a single storage across several module paths, and GMS
    publishes one key per path. Cloning each key separately hands every
    alias a private copy, which silently severs buffers that are shared on
    purpose -- notably ``topk_indices_buffer``, the channel by which the
    DSA indexer passes top-k indices to sparse attention.

    Unaliased tensors keep the plain per-name clone. Aliased ones are
    cloned once and every name gets a view carrying its own shape and
    stride, so the data lives off the VMM mapping and names that shared
    storage on the writer still share it here.

    Scope: this preserves aliases that start at the *same* published
    location, which is what the keying ``(allocation_id, offset_bytes)``
    expresses. Views of one writer storage at different byte offsets are
    published as different locations and are not rejoined -- the metadata
    does not carry source-storage identity. Aliases at one location must
    also agree on dtype and fit the first-seen extent, since the shared
    clone is created from whichever alias arrives first; both are asserted
    rather than left to silently produce a wrong view.
    """
    loc = (spec.allocation_id, spec.offset_bytes)
    if alias_counts.get(loc, 1) <= 1:
        return tensor.detach().clone()

    base = cache.get(loc)
    if base is None:
        # Clone the whole storage this location spans so every aliased view
        # (which may be longer or differently strided than `tensor`) stays
        # in range.
        storage_elems = tensor.untyped_storage().size() // tensor.element_size()
        base = torch.as_strided(tensor, (storage_elems,), (1,), 0).detach().clone()
        cache[loc] = base
    if base.dtype != tensor.dtype:
        raise ValueError(
            f"GMS aliases at {loc} disagree on dtype "
            f"({base.dtype} vs {tensor.dtype}); cannot share one clone"
        )
    span = tensor.storage_offset() + sum(
        (d - 1) * s for d, s in zip(tensor.shape, tensor.stride())
    )
    if span >= base.numel():
        raise ValueError(
            f"GMS alias at {loc} spans {span + 1} elements but the shared "
            f"clone holds {base.numel()}"
        )
    return torch.as_strided(
        base, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset()
    )


def materialize_module_from_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    device_index: int,
) -> None:
    """Materialize model tensors from GMS.

    Args:
        gms_client_memory_manager: GMS client memory manager in read mode.
        model: Model to populate with tensors.
        device_index: CUDA device index.
    """
    specs = GMSTensorSpec.load_all(gms_client_memory_manager)

    # vLLM shares one storage across several module paths: `W_UK_T`/`W_UV`
    # are views into `kv_b_proj.weight` (process_weights_after_loading uses
    # replace_parameter(prefer_copy=True), which keeps the storage address),
    # and one `cos_sin_cache` / `topk_indices_buffer` is referenced by every
    # layer. `materialize` already applies each name's own shape and stride
    # over the shared address, so the views themselves are right -- but the
    # buffer/tensor_attr branch below used to `.clone()` every name, which
    # hands each alias a private copy. For a producer/consumer channel such
    # as `topk_indices_buffer` that severs the link: the DSA indexer writes
    # its top-k indices into one copy while attention reads another.
    #
    # Count how many names map to each physical location so shared ones can
    # keep aliasing instead of being cloned apart.
    alias_counts: dict[tuple[str, int], int] = {}
    for _spec in specs.values():
        _loc = (_spec.allocation_id, _spec.offset_bytes)
        alias_counts[_loc] = alias_counts.get(_loc, 0) + 1

    # One clone per shared location, reused by every name that maps to it.
    # Cloning is what moves a buffer off the GMS VMM mapping (Triton and
    # some CUDA kernels illegal-address against VMM), so the clone must
    # stay -- but it has to be made once and shared, not once per name.
    shared_clones: dict[tuple[str, int], torch.Tensor] = {}

    n_shared = sum(1 for v in alias_counts.values() if v > 1)
    if n_shared:
        logger.info(
            "[GMS] Preserving aliasing for %d shared locations "
            "(%d of %d published names)",
            n_shared,
            sum(v for v in alias_counts.values() if v > 1),
            len(specs),
        )

    for name, spec in specs.items():
        tensor = spec.materialize(gms_client_memory_manager, device_index)
        try:
            mod, attr = _resolve_module_attr(model, name)
        except AttributeError:
            logger.warning(
                "[GMS] Cannot resolve %s on RO model; attaching at root skip",
                name,
            )
            continue
        tensor_type = spec.meta.tensor_type

        # Parameters first: writer may have registered MLA scales as
        # tensor_attr while the RO meta model holds them as Parameters.
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            param = mod._parameters[attr]
            if param is not None:
                if param.shape != tensor.shape or param.dtype != tensor.dtype:
                    logger.warning(
                        "[GMS] Replacing %s: param=%s/%s -> gms=%s/%s",
                        name,
                        tuple(param.shape),
                        param.dtype,
                        tuple(tensor.shape),
                        tensor.dtype,
                    )
                payload = (
                    _shared_clone(shared_clones, alias_counts, spec, tensor)
                    if tensor_type in ("tensor_attr", "buffer")
                    else tensor
                )
                mod._parameters[attr] = torch.nn.Parameter(
                    payload, requires_grad=param.requires_grad
                )
            continue

        # Tensor attrs and buffers: clone since they may be mutated, and
        # because the clone is what lifts them off the GMS VMM mapping.
        # Skip read-only properties (e.g. MoERunner.expert_map) that vLLM
        # exposes via getattr during write-side registration.
        if tensor_type in ("tensor_attr", "buffer"):
            cloned = _shared_clone(shared_clones, alias_counts, spec, tensor)
            if (
                tensor_type == "buffer"
                and hasattr(mod, "_buffers")
                and attr in mod._buffers
            ):
                mod._buffers[attr] = cloned
            else:
                try:
                    setattr(mod, attr, cloned)
                except (AttributeError, TypeError):
                    logger.debug(
                        "[GMS] Skipping unsettable %s %r on %s",
                        tensor_type,
                        name,
                        type(mod).__name__,
                    )
            continue

        # Fallback: set as attribute
        setattr(mod, attr, tensor)

    _repoint_non_module_tensor_caches(model)

    # Leftover meta params/buffers were never in the writer's GMS layout: the
    # writer registers the MLA scales under their `_`-prefixed buffer names,
    # so the reader's `q_scale`/`k_scale`/`v_scale`/`prob_scale` Parameters
    # find no match. They have to leave the meta device -- dummy profile_run
    # and kernel warmup illegal-address otherwise -- and the fill value is
    # load-bearing. A scale is a multiplicative factor, so its neutral value
    # is 1.0, which is also what vLLM's `set_default_quant_scales` uses when
    # a checkpoint carries no calibrated scale. Zero-filling instead silently
    # scales attention to nothing. `init_fp8_kv_scales` masks half of it on
    # wake by resetting `k_scale`/`v_scale` to 1.0, which is why only
    # `q_scale` and `prob_scale` stayed at zero and the damage looked like a
    # gradual quality decay rather than an outright failure.
    device = torch.device(f"cuda:{device_index}")
    filled: list[str] = []

    def _neutral(name: str, shape, dtype) -> torch.Tensor:
        leaf = name.rsplit(".", 1)[-1].lstrip("_")
        value = 1.0 if leaf in _MLA_SCALE_LEAVES else 0.0
        return torch.full(shape, value, dtype=dtype, device=device)

    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        try:
            mod, attr = _resolve_module_attr(model, name)
        except AttributeError:
            continue
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            mod._parameters[attr] = torch.nn.Parameter(
                _neutral(name, param.shape, param.dtype),
                requires_grad=param.requires_grad,
            )
            filled.append(name)
    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        try:
            mod, attr = _resolve_module_attr(model, name)
        except AttributeError:
            continue
        if hasattr(mod, "_buffers") and attr in mod._buffers:
            mod._buffers[attr] = _neutral(name, buf.shape, buf.dtype)
            filled.append(name)
    if filled:
        logger.warning(
            "[GMS] Materialized %d leftover meta tensors to neutral values: %s",
            len(filled),
            filled[:10],
        )


def copy_unmapped_tensors_into_gms(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    leaves: frozenset[str] = _MLA_SCALE_LEAVES,
) -> int:
    """Clone CUDA tensors that missed the GMS pool into the active mempool.

    Must run under ``gms_use_mem_pool("weights")`` so the clone lands in a
    GMS allocation and ``prepare_gms_write`` can register it. GLM MLA
    q/k/v/prob scales are allocated by ``process_weights_after_loading``
    outside the pool; without this, RO import zero-fills 312 leftover
    Parameters.
    """
    copied: list[str] = []
    seen: set[str] = set()
    for name, tensor, tensor_type in list(_iter_module_tensors(model)):
        if name.rsplit(".", 1)[-1] not in leaves:
            continue
        seen.add(name)
        if _tensor_in_gms_mappings(gms_client_memory_manager, tensor):
            continue
        gms_copy = tensor.detach().contiguous().clone()
        _bind_module_tensor(model, name, gms_copy, tensor_type)
        copied.append(name)
    for name, param in list(model.named_parameters()):
        if name in seen or param is None or not param.is_cuda:
            continue
        if name.rsplit(".", 1)[-1] not in leaves:
            continue
        if _tensor_in_gms_mappings(gms_client_memory_manager, param):
            continue
        gms_copy = param.detach().contiguous().clone()
        _bind_module_tensor(model, name, gms_copy, "parameter")
        copied.append(name)
    if copied:
        logger.info(
            "[GMS] Copied %d tensors into the GMS pool for publish: %s",
            len(copied),
            copied[:8],
        )
    return len(copied)


def rebind_nonparameter_tensors(
    gms_client_memory_manager: "GMSClientMemoryManager",
    model: torch.nn.Module,
    *,
    retain_gms_tensors: list[torch.Tensor] | None = None,
) -> int:
    """Move GMS-resident non-parameter tensors onto private storage.

    The publisher builds the whole model inside the GMS memory pool, so
    buffers and tensor attributes (fp8 KV scales, quantization ranges, ...)
    land in the same committed allocations as the weights, which are
    remapped read-only after publish. Unlike parameters, these tensors can
    be written after load (for example ``init_fp8_kv_scales`` on wake),
    which faults on the read-only mapping.

    The swap is done **in place** with ``Tensor.set_``, one private storage
    per source storage. Rebinding by name instead -- assigning freshly
    cloned tensors back onto module attributes -- looks equivalent but is
    not: vLLM caches some of these tensors on plain objects that module
    traversal never visits. ``MLAAttention.impl`` is not an ``nn.Module``,
    and it holds the consumer end of the DSA ``topk_indices_buffer``
    channel, so a name-based rebind leaves the indexer writing top-k
    indices into the new buffer while attention keeps reading the old one.
    Swapping storage under the existing TensorImpl makes every holder
    follow, including the ones that cannot be enumerated.

    Must run before CUDA graph capture: the private storage is at a new
    address.

    Returns the number of bytes rebound, i.e. how much memory is duplicated
    between the read-only GMS copies and the private storage.

    If ``retain_gms_tensors`` is provided, copies of the original
    GMS-backed tensors are appended to it. A deferred writer that rebinds
    before commit must keep those alive so the underlying GMS pool
    allocations are not freed before the layout is published.
    """
    mappings = gms_client_memory_manager.mappings

    def _in_gms(t: torch.Tensor) -> bool:
        ptr = int(t.data_ptr())
        return any(
            va <= ptr < va + m.aligned_size for va, m in mappings.items()
        )

    # Collect the GMS-resident non-parameter tensors, grouped by storage.
    # Rebinding must preserve TensorImpl identity: vLLM caches some of
    # these on plain (non-Module) objects -- notably
    # `MLAAttention.impl.topk_indices_buffer`, the consumer end of the DSA
    # top-k channel -- which no name-based traversal can reach. Swapping
    # each tensor's storage in place makes every holder follow, including
    # the ones we cannot see.
    by_storage: dict[int, list[torch.Tensor]] = {}
    for name, tensor, tensor_type in list(_iter_module_tensors(model)):
        if tensor_type == "parameter":
            continue
        if not _in_gms(tensor):
            # Allocated outside the GMS pool; already private.
            continue
        by_storage.setdefault(int(tensor.untyped_storage()._cdata), []).append(
            tensor
        )

    rebound_bytes = 0
    with torch.inference_mode():
        for tensors in by_storage.values():
            src = tensors[0].untyped_storage()
            if retain_gms_tensors is not None:
                # Keep the GMS storage alive for the deferred publish. This
                # must be a tensor that still points AT `src`: once every
                # live TensorImpl has been redirected by `set_` below, this
                # byte view is the only remaining owner. Cloning instead
                # would retain a private copy and let the pool allocation go.
                retain_gms_tensors.append(
                    torch.empty(0, dtype=torch.uint8, device=tensors[0].device)
                    .set_(src, 0, (src.size(),), (1,))
                )
            private = torch.empty(
                src.size(), dtype=torch.uint8, device=tensors[0].device
            )
            private.untyped_storage().copy_(src)
            dst = private.untyped_storage()
            for t in tensors:
                t.set_(
                    dst,
                    int(t.storage_offset()),
                    tuple(t.shape),
                    tuple(t.stride()),
                )
            rebound_bytes += src.size()

    return rebound_bytes
