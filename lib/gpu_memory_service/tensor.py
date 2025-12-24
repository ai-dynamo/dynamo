"""Tensor utilities for GPU Memory Service.

This module consolidates all tensor-related functionality for GPU Memory Service:
- Tensor metadata serialization/deserialization
- Module tree extraction (params, buffers, tensor_attrs)
- Registry operations (reading/writing tensor specs)
- Tensor materialization from mapped memory

This provides a unified interface for both write (registration) and read
(materialization) paths.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from gpu_memory_service.allocator import RPCCumemAllocator

logger = logging.getLogger(__name__)


# =============================================================================
# Tensor Metadata - serialization format for registry storage
# =============================================================================


@dataclass(frozen=True)
class TensorMeta:
    """Metadata for a tensor stored in the GPU Memory Service registry (serialization format).

    This is the canonical format for tensor metadata stored in the artifact
    registry. The dtype is stored as a string for JSON serialization.
    """

    shape: Tuple[int, ...]
    dtype: str  # String representation like "torch.float16"
    nbytes: int  # Logical size (numel * element_size)
    stride: Tuple[int, ...]
    span_bytes: int  # Actual memory span needed for strided view
    tensor_type: str = "parameter"  # "parameter", "buffer", or "tensor_attr"


@dataclass(frozen=True)
class ParsedTensorMeta:
    """Parsed tensor metadata with torch.dtype (for materialization).

    This is the parsed form of TensorMeta, with dtype converted from
    string to actual torch.dtype for use in tensor creation.
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype
    stride: Optional[Tuple[int, ...]]
    span_bytes: int
    tensor_type: str = "parameter"


def compute_span_bytes(
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    element_size: int,
) -> int:
    """Compute the byte span needed to cover a strided tensor view.

    For a tensor starting at data_ptr, this returns the total bytes from
    data_ptr to the last element of the tensor (inclusive).

    Args:
        shape: Tensor shape tuple.
        stride: Tensor stride tuple (in elements, not bytes).
        element_size: Size of one element in bytes.

    Returns:
        Total byte span from data_ptr to cover all elements.
    """
    if len(shape) == 0:
        return int(element_size)
    if any(int(d) == 0 for d in shape):
        return 0
    if len(shape) != len(stride):
        raise ValueError(f"stride rank mismatch: shape={shape} stride={stride}")

    max_offset_elems = 0
    for d, st in zip(shape, stride):
        d = int(d)
        st = int(st)
        if d <= 0:
            continue
        max_offset_elems += (d - 1) * st

    return int((max_offset_elems + 1) * int(element_size))


def tensor_meta_from_tensor(
    tensor: torch.Tensor, tensor_type: str = "parameter"
) -> TensorMeta:
    """Create TensorMeta from an existing tensor.

    Args:
        tensor: The tensor to extract metadata from.
        tensor_type: Type classification ("parameter", "buffer", "tensor_attr").

    Returns:
        TensorMeta with all fields populated.
    """
    shape = tuple(tensor.shape)
    stride = tuple(int(s) for s in tensor.stride())
    nbytes = int(tensor.numel() * tensor.element_size())
    span_bytes = compute_span_bytes(shape, stride, tensor.element_size())

    return TensorMeta(
        shape=shape,
        dtype=str(tensor.dtype),
        nbytes=nbytes,
        stride=stride,
        span_bytes=span_bytes,
        tensor_type=tensor_type,
    )


def serialize_tensor_meta(meta: TensorMeta) -> bytes:
    """Serialize TensorMeta to JSON bytes for registry storage.

    Args:
        meta: The tensor metadata to serialize.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    return json.dumps(
        {
            "shape": list(meta.shape),
            "dtype": meta.dtype,
            "nbytes": meta.nbytes,
            "stride": list(meta.stride),
            "span_bytes": meta.span_bytes,
            "tensor_type": meta.tensor_type,
        },
        sort_keys=True,
    ).encode("utf-8")


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string (e.g., 'torch.float16') to torch.dtype."""
    s = str(dtype_str)
    if s.startswith("torch."):
        s = s.split(".", 1)[1]
    try:
        dt = getattr(torch, s)
    except Exception as e:
        raise ValueError(f"Unknown dtype string: {dtype_str!r}") from e
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Invalid dtype string: {dtype_str!r}")
    return dt


def parse_tensor_meta(value: bytes) -> ParsedTensorMeta:
    """Parse registry metadata blob into ParsedTensorMeta.

    Supports both legacy and current formats:
    - legacy: {"shape": [...], "dtype": "torch.float16", "nbytes": ...}
    - current: {"shape": [...], "dtype": "...", "stride": [...], "span_bytes": ..., "tensor_type": ...}

    Args:
        value: UTF-8 encoded JSON bytes from registry.

    Returns:
        ParsedTensorMeta with torch.dtype and computed span_bytes.
    """
    obj = json.loads(value.decode("utf-8"))
    shape = tuple(int(x) for x in obj["shape"])
    dtype = _parse_dtype(obj["dtype"])

    stride: Optional[Tuple[int, ...]] = None
    if "stride" in obj and obj["stride"] is not None:
        stride = tuple(int(x) for x in obj["stride"])

    # Prefer span_bytes if present; otherwise compute it
    span_bytes = obj.get("span_bytes")
    if span_bytes is None:
        span_bytes = obj.get("nbytes")  # Legacy fallback
    if span_bytes is None or span_bytes < 0:
        elem_size = int(torch.tensor([], dtype=dtype).element_size())
        if stride is not None:
            span_bytes = compute_span_bytes(shape, stride, elem_size)
        else:
            numel = 1
            for d in shape:
                numel *= int(d)
            span_bytes = numel * elem_size

    tensor_type = obj.get("tensor_type", "parameter")

    return ParsedTensorMeta(
        shape=shape,
        dtype=dtype,
        stride=stride,
        span_bytes=int(span_bytes),
        tensor_type=tensor_type,
    )


# =============================================================================
# Module Tree Extraction - captures ALL tensor attributes
# =============================================================================


@dataclass
class TensorIPCInfo:
    """Information about a tensor for IPC/registry purposes."""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    stride: Optional[Tuple[int, ...]]
    span_bytes: int
    data_ptr: Optional[int] = None  # Set during registration (write mode)


@dataclass
class ModuleTreeNode:
    """Hierarchical representation of module weights.

    Mirrors PyTorch's nn.Module structure for complete weight extraction.
    This captures parameters, buffers, AND other tensor attributes.
    """

    parameters: Dict[str, TensorIPCInfo] = field(default_factory=dict)
    buffers: Dict[str, TensorIPCInfo] = field(default_factory=dict)
    tensor_attrs: Dict[str, Union[TensorIPCInfo, List[TensorIPCInfo]]] = field(
        default_factory=dict
    )
    submodules: Dict[str, "ModuleTreeNode"] = field(default_factory=dict)


def _tensor_to_ipc_info(t: torch.Tensor) -> TensorIPCInfo:
    """Convert a tensor to TensorIPCInfo."""
    stride = tuple(int(s) for s in t.stride()) if t.dim() > 0 else None
    if t.is_meta:
        span_bytes = 0
    else:
        elem_size = t.element_size()
        if len(t.shape) == 0:
            span_bytes = elem_size
        elif stride is None:
            span_bytes = int(t.numel()) * elem_size
        else:
            max_offset_elems = 0
            for d, st in zip(t.shape, stride):
                d = int(d)
                st = int(st)
                if d > 0:
                    max_offset_elems += (d - 1) * st
            span_bytes = (max_offset_elems + 1) * elem_size
    return TensorIPCInfo(
        shape=tuple(t.shape),
        dtype=t.dtype,
        stride=stride,
        span_bytes=span_bytes,
        data_ptr=int(t.data_ptr()) if t.is_cuda and not t.is_meta else None,
    )


def extract_module_tree(
    module: torch.nn.Module, include_non_cuda: bool = False
) -> ModuleTreeNode:
    """Extract all tensors from a module tree.

    This captures:
    - Parameters (via module._parameters)
    - Buffers (via module._buffers)
    - Other tensor attributes (like _k_scale, _v_scale set directly on modules)

    Args:
        module: The nn.Module to extract from.
        include_non_cuda: If True, include CPU/meta tensors; otherwise only CUDA tensors.

    Returns:
        ModuleTreeNode with hierarchical tensor information.
    """
    node = ModuleTreeNode()

    # Extract parameters (direct, not recursive)
    for name, param in module._parameters.items():
        if param is not None:
            if include_non_cuda or param.is_cuda:
                node.parameters[name] = _tensor_to_ipc_info(param)

    # Extract buffers (direct, not recursive)
    for name, buf in module._buffers.items():
        if buf is not None:
            if include_non_cuda or buf.is_cuda:
                node.buffers[name] = _tensor_to_ipc_info(buf)

    # Extract other tensor attributes (not params or buffers)
    param_names = set(module._parameters.keys())
    buffer_names = set(module._buffers.keys())
    submodule_names = set(module._modules.keys())

    for attr_name in dir(module):
        if attr_name.startswith("_") and attr_name not in (
            "_parameters",
            "_buffers",
            "_modules",
        ):
            if attr_name in (
                "__class__",
                "__dict__",
                "__doc__",
                "__module__",
                "__weakref__",
            ):
                continue
        if (
            attr_name in param_names
            or attr_name in buffer_names
            or attr_name in submodule_names
        ):
            continue
        try:
            attr_val = getattr(module, attr_name, None)
        except Exception:
            continue

        if torch.is_tensor(attr_val):
            if include_non_cuda or attr_val.is_cuda:
                node.tensor_attrs[attr_name] = _tensor_to_ipc_info(attr_val)
        elif isinstance(attr_val, (list, tuple)) and len(attr_val) > 0:
            if all(torch.is_tensor(x) for x in attr_val):
                infos = []
                for x in attr_val:
                    if include_non_cuda or x.is_cuda:
                        infos.append(_tensor_to_ipc_info(x))
                if infos:
                    node.tensor_attrs[attr_name] = infos

    # Recurse into submodules
    for name, submodule in module._modules.items():
        if submodule is not None:
            node.submodules[name] = extract_module_tree(submodule, include_non_cuda)

    return node


def iter_module_tree_tensors(
    node: ModuleTreeNode,
    prefix: str = "",
) -> Iterator[Tuple[str, TensorIPCInfo, str]]:
    """Iterate over all tensors in a module tree.

    Yields:
        (qualified_name, TensorIPCInfo, tensor_type) where tensor_type is
        "parameter", "buffer", or "tensor_attr"
    """
    for name, info in node.parameters.items():
        qualified = f"{prefix}{name}" if prefix else name
        yield (qualified, info, "parameter")

    for name, info in node.buffers.items():
        qualified = f"{prefix}{name}" if prefix else name
        yield (qualified, info, "buffer")

    for name, info_or_list in node.tensor_attrs.items():
        qualified = f"{prefix}{name}" if prefix else name
        if isinstance(info_or_list, list):
            for i, info in enumerate(info_or_list):
                yield (f"{qualified}.{i}", info, "tensor_attr")
        else:
            yield (qualified, info_or_list, "tensor_attr")

    for subname, subnode in node.submodules.items():
        subprefix = f"{prefix}{subname}." if prefix else f"{subname}."
        yield from iter_module_tree_tensors(subnode, subprefix)


def resolve_module_attr(
    root: torch.nn.Module, qualified_name: str
) -> tuple[torch.nn.Module, str]:
    """Resolve a dotted parameter/buffer name to (parent_module, leaf_attr).

    Handles common container modules where named_parameters() uses numeric / key
    path segments:
    - ModuleList / Sequential: "layers.0.attn.q_proj.weight"
    - ModuleDict: "experts.3.weight" (key access)
    """
    parts = qualified_name.split(".")
    mod: torch.nn.Module = root
    for p in parts[:-1]:
        if hasattr(mod, p):
            mod = getattr(mod, p)
            continue

        if hasattr(mod, "__getitem__"):
            try:
                if p.isdigit():
                    mod = mod[int(p)]
                else:
                    mod = mod[p]
                continue
            except Exception:
                pass

        raise AttributeError(
            f"Could not resolve submodule path segment {p!r} in {qualified_name!r}"
        )

    return mod, parts[-1]


# =============================================================================
# Registry Operations - reading/writing tensor specs
# =============================================================================


@dataclass(frozen=True)
class RegistryTensorSpec:
    """A single tensor entry from the Artifact Registry."""

    key: str
    name: str
    allocation_id: str
    offset_bytes: int
    meta: ParsedTensorMeta


def load_registry_tensor_specs(
    allocator: "RPCCumemAllocator", prefix: str
) -> Dict[str, RegistryTensorSpec]:
    """Load and parse all registry entries under a given prefix.

    Args:
        allocator: The GPU Memory Service allocator with registry access.
        prefix: Registry key prefix (e.g., "abc123:").

    Returns:
        Mapping of tensor name (without prefix) -> RegistryTensorSpec.
    """
    keys = allocator.registry_list(prefix)
    specs: Dict[str, RegistryTensorSpec] = {}

    for key in keys:
        got = allocator.registry_get(key)
        if got is None:
            raise RuntimeError(f"Registry key disappeared during read: {key}")
        allocation_id, offset_bytes, value = got
        name = key[len(prefix) :] if key.startswith(prefix) else key
        meta = parse_tensor_meta(value)
        if name in specs:
            raise RuntimeError(f"Duplicate registry tensor name: {name}")
        specs[name] = RegistryTensorSpec(
            key=key,
            name=name,
            allocation_id=str(allocation_id),
            offset_bytes=int(offset_bytes),
            meta=meta,
        )

    return specs


def register_tensor(
    allocator: "RPCCumemAllocator",
    name: str,
    tensor: torch.Tensor,
    allocation_id: str,
    base_va: int,
    registry_prefix: str,
    tensor_type: str = "parameter",
) -> int:
    """Register a tensor's metadata in the GPU Memory Service registry.

    Args:
        allocator: The GPU Memory Service allocator (must be in write mode).
        name: The tensor name (e.g., "model.layers.0.self_attn.q_proj.weight").
        tensor: The CUDA tensor to register.
        allocation_id: The GPU Memory Service allocation ID containing this tensor.
        base_va: The base virtual address of the allocation.
        registry_prefix: Prefix for registry keys (e.g., "abc123:").
        tensor_type: Type classification ("parameter", "buffer", "tensor_attr").

    Returns:
        The tensor's nbytes.
    """
    ptr = int(tensor.data_ptr())
    offset = ptr - base_va

    meta = tensor_meta_from_tensor(tensor, tensor_type)
    allocator.registry_put(
        key=f"{registry_prefix}{name}",
        allocation_id=allocation_id,
        offset_bytes=int(offset),
        value=serialize_tensor_meta(meta),
    )

    return meta.nbytes


def register_module_tensors(
    allocator: "RPCCumemAllocator",
    model: torch.nn.Module,
    registry_prefix: str,
) -> int:
    """Register all tensors from a model into the GPU Memory Service registry.

    This extracts all parameters, buffers, and tensor attributes from the model
    and registers them in the GPU Memory Service registry.

    Args:
        allocator: The GPU Memory Service allocator (must be in write mode).
        model: The PyTorch model to register.
        registry_prefix: Prefix for registry keys (e.g., "abc123:").

    Returns:
        Total bytes registered.
    """
    module_tree = extract_module_tree(model, include_non_cuda=False)

    # Count tensors by type for logging
    param_count = 0
    buffer_count = 0
    attr_count = 0
    for _, _, tensor_type in iter_module_tree_tensors(module_tree):
        if tensor_type == "parameter":
            param_count += 1
        elif tensor_type == "buffer":
            buffer_count += 1
        else:
            attr_count += 1

    logger.info(
        "[GPU Memory Service] Registering tensors: %d params, %d buffers, %d tensor_attrs",
        param_count,
        buffer_count,
        attr_count,
    )

    total_bytes = 0

    for name, _, tensor_type in iter_module_tree_tensors(module_tree):
        try:
            mod, attr = resolve_module_attr(model, name)
            if hasattr(mod, "_parameters") and attr in mod._parameters:
                t = mod._parameters[attr]
            elif hasattr(mod, "_buffers") and attr in mod._buffers:
                t = mod._buffers[attr]
            else:
                t = getattr(mod, attr, None)

            if t is None or not torch.is_tensor(t) or not t.is_cuda:
                continue

            ptr = int(t.data_ptr())

            # Find which allocation contains this tensor
            alloc_info = None
            for va, mapping in allocator._mappings.items():
                if va <= ptr < va + mapping.aligned_size:
                    alloc_info = (va, mapping.allocation_id)
                    break

            if alloc_info is None:
                logger.debug(
                    "[GPU Memory Service] Skipping non-GPU Memory Service tensor %s (type=%s) ptr=0x%x",
                    name,
                    tensor_type,
                    ptr,
                )
                continue

            base_va, alloc_id = alloc_info
            nbytes = register_tensor(
                allocator,
                name,
                t,
                alloc_id,
                base_va,
                registry_prefix,
                tensor_type,
            )
            total_bytes += nbytes

        except Exception as e:
            logger.debug(
                "[GPU Memory Service] Could not register tensor %s: %s", name, e
            )

    return total_bytes


# =============================================================================
# Tensor Materialization - creating tensors from mapped memory
# =============================================================================


def tensor_from_registry_spec(
    allocator: "RPCCumemAllocator",
    spec: RegistryTensorSpec,
    *,
    device_index: int,
) -> torch.Tensor:
    """Create a torch.Tensor that aliases mapped CUDA memory for a registry spec.

    Args:
        allocator: The GPU Memory Service allocator (imports the allocation if needed).
        spec: The registry tensor specification.
        device_index: CUDA device index.

    Returns:
        A tensor aliasing the mapped memory.
    """
    try:
        import gpu_memory_service.extensions._tensor_from_pointer as tfp
    except Exception as e:
        raise RuntimeError(
            "Missing gpu_memory_service.core.csrc._tensor_from_pointer extension "
            "(required for import-only loader)."
        ) from e

    base_va = allocator.import_allocation(spec.allocation_id)
    ptr = int(base_va) + int(spec.offset_bytes)

    if spec.meta.stride is None:
        return tfp.tensor_from_pointer_contiguous(
            ptr, list(spec.meta.shape), spec.meta.dtype, int(device_index)
        )
    return tfp.tensor_from_pointer(
        ptr,
        list(spec.meta.shape),
        list(spec.meta.stride),
        spec.meta.dtype,
        int(device_index),
    )


def materialize_module_from_registry(
    allocator: "RPCCumemAllocator",
    model: torch.nn.Module,
    *,
    prefix: str,
    device_index: int,
    strict: bool = True,
) -> int:
    """Materialize model tensors by importing from registry metadata.

    This handles:
    - Parameters (via module._parameters)
    - Buffers (via module._buffers)
    - Tensor attributes (via setattr, for things like _k_scale, _v_scale)

    Args:
        allocator: The GPU Memory Service allocator in read mode.
        model: The model to populate with tensors.
        prefix: Registry key prefix (e.g., "abc123:").
        device_index: CUDA device index.
        strict: If True, warn about remaining meta tensors.

    Returns:
        Total imported tensor byte span.
    """
    specs = load_registry_tensor_specs(allocator, prefix)

    # Categorize registry entries by tensor_type
    param_specs = {k: v for k, v in specs.items() if v.meta.tensor_type == "parameter"}
    buffer_specs = {k: v for k, v in specs.items() if v.meta.tensor_type == "buffer"}
    attr_specs = {k: v for k, v in specs.items() if v.meta.tensor_type == "tensor_attr"}

    model_params = set(n for n, _ in model.named_parameters())
    model_buffers = set(n for n, _ in model.named_buffers())
    registry_params = set(param_specs.keys())
    registry_buffers = set(buffer_specs.keys())

    logger.info(
        "[GPU Memory Service] Registry contains: %d params, %d buffers, %d tensor_attrs",
        len(param_specs),
        len(buffer_specs),
        len(attr_specs),
    )

    in_registry_not_model = (
        (registry_params | registry_buffers) - model_params - model_buffers
    )
    in_model_not_registry = (
        (model_params | model_buffers) - registry_params - registry_buffers
    )

    if in_registry_not_model:
        logger.warning(
            "[GPU Memory Service] Registry params/buffers NOT in model (%d): %s",
            len(in_registry_not_model),
            list(in_registry_not_model)[:10],
        )
    if in_model_not_registry:
        logger.warning(
            "[GPU Memory Service] Model params/buffers NOT in registry (%d): %s",
            len(in_model_not_registry),
            list(in_model_not_registry)[:10],
        )

    imported_span_bytes = 0
    tensor_attr_count = 0

    for name, spec in specs.items():
        t = tensor_from_registry_spec(allocator, spec, device_index=device_index)
        imported_span_bytes += int(spec.meta.span_bytes)

        mod, attr = resolve_module_attr(model, name)
        tensor_type = spec.meta.tensor_type

        # Handle tensor_attrs - copy since they may be mutated
        if tensor_type == "tensor_attr":
            setattr(mod, attr, t.detach().clone())
            tensor_attr_count += 1
            continue

        # Parameters: prefer in-place updates
        if (
            hasattr(mod, "_parameters")
            and attr in mod._parameters
            and mod._parameters[attr] is not None
        ):
            p = mod._parameters[attr]
            assert p is not None
            if tuple(p.shape) != tuple(t.shape) or p.dtype != t.dtype:
                raise RuntimeError(
                    f"Parameter mismatch for {name}: "
                    f"param(shape={tuple(p.shape)}, dtype={p.dtype}) "
                    f"registry(shape={tuple(t.shape)}, dtype={t.dtype})"
                )
            needs_replace = bool(getattr(p, "is_meta", False)) or (p.device != t.device)
            if needs_replace:
                req = bool(p.requires_grad)
                if not (t.is_floating_point() or t.is_complex()):
                    req = False
                mod._parameters[attr] = torch.nn.Parameter(t, requires_grad=req)
            else:
                p.data = t
            continue

        # Buffers - copy since they may be mutated
        if (
            hasattr(mod, "_buffers")
            and attr in mod._buffers
            and mod._buffers[attr] is not None
        ):
            b = mod._buffers[attr]
            assert b is not None
            if tuple(b.shape) != tuple(t.shape) or b.dtype != t.dtype:
                raise RuntimeError(
                    f"Buffer mismatch for {name}: "
                    f"buf(shape={tuple(b.shape)}, dtype={b.dtype}) "
                    f"registry(shape={tuple(t.shape)}, dtype={t.dtype})"
                )
            mod._buffers[attr] = t.detach().clone()
            continue

        # Fallback
        existing = getattr(mod, attr, None)
        if isinstance(existing, torch.nn.Parameter):
            if existing.is_meta:
                req = bool(existing.requires_grad)
                if not (t.is_floating_point() or t.is_complex()):
                    req = False
                mod._parameters[attr] = torch.nn.Parameter(t, requires_grad=req)
            else:
                existing.data = t
        elif torch.is_tensor(existing):
            setattr(mod, attr, t if tensor_type == "parameter" else t.detach().clone())
        else:
            logger.debug(
                "[GPU Memory Service] Creating new %s for registry entry not in model: %s",
                tensor_type,
                name,
            )
            if tensor_type == "parameter":
                new_param = torch.nn.Parameter(t, requires_grad=False)
                if hasattr(mod, "_parameters"):
                    mod._parameters[attr] = new_param
                else:
                    setattr(mod, attr, new_param)
            elif tensor_type == "buffer":
                if hasattr(mod, "_buffers"):
                    mod._buffers[attr] = t.detach().clone()
                else:
                    setattr(mod, attr, t.detach().clone())
            else:
                setattr(mod, attr, t.detach().clone())

    if tensor_attr_count > 0:
        logger.info(
            "[GPU Memory Service] Materialized %d tensor_attrs from registry",
            tensor_attr_count,
        )

    # Handle remaining meta tensors
    meta_initialized = 0
    cuda_device = torch.device("cuda", device_index)
    for n, p in list(model.named_parameters()):
        if p.is_meta:
            is_scale = "scale" in n.lower()
            fill_value = 1.0 if is_scale else 0.0
            new_tensor = torch.full(
                p.shape,
                fill_value,
                dtype=p.dtype,
                device=cuda_device,
                requires_grad=False,
            )
            mod, attr = resolve_module_attr(model, n)
            if hasattr(mod, "_parameters") and attr in mod._parameters:
                mod._parameters[attr] = torch.nn.Parameter(
                    new_tensor, requires_grad=False
                )
            else:
                setattr(mod, attr, torch.nn.Parameter(new_tensor, requires_grad=False))
            meta_initialized += 1

    for n, b in list(model.named_buffers()):
        if b.is_meta:
            is_scale = "scale" in n.lower()
            fill_value = 1.0 if is_scale else 0.0
            new_tensor = torch.full(
                b.shape,
                fill_value,
                dtype=b.dtype,
                device=cuda_device,
                requires_grad=False,
            )
            mod, attr = resolve_module_attr(model, n)
            if hasattr(mod, "_buffers") and attr in mod._buffers:
                mod._buffers[attr] = new_tensor
            else:
                setattr(mod, attr, new_tensor)
            meta_initialized += 1

    if meta_initialized > 0:
        logger.info(
            "[GPU Memory Service] Initialized %d meta tensors with default values",
            meta_initialized,
        )

    if strict:
        meta_params: list[str] = []
        for n, p in model.named_parameters():
            if p.is_meta:
                meta_params.append(n)
        for n, b in model.named_buffers():
            if b.is_meta:
                meta_params.append(n)
        if meta_params:
            logger.warning(
                "[GPU Memory Service] Model still has %d meta tensors after initialization: %s",
                len(meta_params),
                meta_params[:10],
            )

    return imported_span_bytes
