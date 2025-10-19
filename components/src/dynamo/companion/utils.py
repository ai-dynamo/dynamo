# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for companion server and client."""

import base64
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Tuple, Any

import cloudpickle
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

logger = logging.getLogger(__name__)


# Type aliases for clarity
RebuildArgs = Tuple[Any, ...]  # CUDA IPC rebuild arguments
EmptyTensorMarker = Tuple[str, int, torch.dtype]  # ("empty", device_idx, dtype)
TensorData = Union[RebuildArgs, EmptyTensorMarker]
TensorListData = List[Optional[TensorData]]


@dataclass
class ModuleTreeNode:
    """
    Represents a module in the model tree with its tensors and submodules.

    This structure mirrors PyTorch's module hierarchy and contains all the
    information needed to reconstruct tensors via CUDA IPC.
    """
    # Direct tensor storage
    parameters: Dict[str, TensorData] = field(default_factory=dict)
    buffers: Dict[str, TensorData] = field(default_factory=dict)
    tensor_attrs: Dict[str, Union[TensorData, TensorListData]] = field(default_factory=dict)

    # Nested submodules
    submodules: Dict[str, 'ModuleTreeNode'] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModuleTreeNode':
        """Reconstruct from dictionary format."""
        # Recursively convert submodules
        submodules = {
            name: cls.from_dict(submodule_data)
            for name, submodule_data in data.get('submodules', {}).items()
        }

        return cls(
            parameters=data.get('parameters', {}),
            buffers=data.get('buffers', {}),
            tensor_attrs=data.get('tensor_attrs', {}),
            submodules=submodules
        )


def count_tensors_in_tree(tree: ModuleTreeNode) -> tuple[int, int, int]:
    """
    Count total parameters, buffers, and tensor attributes in the tree.

    Args:
        tree: Module tree node

    Returns:
        Tuple of (param_count, buffer_count, tensor_attr_count)
    """
    param_count = len(tree.parameters)
    buffer_count = len(tree.buffers)

    # Count tensor attributes (including list items)
    tensor_attr_count = 0
    for attr_value in tree.tensor_attrs.values():
        if isinstance(attr_value, list):
            tensor_attr_count += sum(1 for item in attr_value if item is not None)
        else:
            tensor_attr_count += 1

    for submodule_tree in tree.submodules.values():
        sub_params, sub_buffers, sub_attrs = count_tensors_in_tree(submodule_tree)
        param_count += sub_params
        buffer_count += sub_buffers
        tensor_attr_count += sub_attrs

    return param_count, buffer_count, tensor_attr_count


def encode_response(response) -> str:
    """
    Encode a response dataclass as base64-encoded cloudpickle for transmission.

    Args:
        response: Dataclass response object

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(cloudpickle.dumps(asdict(response))).decode('utf-8')


def reconstruct_tensor_from_ipc(rebuild_args, target_device: Optional[int] = None) -> torch.Tensor:
    """
    Reconstruct a tensor from IPC rebuild args, mapping to target device.

    Args:
        rebuild_args: Tuple or special marker for tensor reconstruction
        target_device: Target CUDA device index (defaults to current device)

    Returns:
        Reconstructed CUDA tensor
    """
    # Handle empty tensor markers
    if isinstance(rebuild_args, tuple) and rebuild_args[0] == "empty":
        _, device_idx, dtype = rebuild_args
        if target_device is not None:
            device_idx = target_device
        return torch.tensor([], device=f"cuda:{device_idx}", dtype=dtype)

    # Reconstruct from IPC args
    if target_device is None:
        target_device = torch.cuda.current_device()

    # Update device in rebuild args (device is at index 6)
    rebuild_args_list = list(rebuild_args)
    rebuild_args_list[6] = target_device
    rebuild_args = tuple(rebuild_args_list)

    # Reconstruct the tensor
    tensor = rebuild_cuda_tensor(*rebuild_args)
    return tensor


def extract_module_tree_ipc_info(model: torch.nn.Module, prefix: str = "") -> ModuleTreeNode:
    """
    Extract CUDA IPC rebuild info for the entire module tree.

    Args:
        model: PyTorch module to extract from
        prefix: Prefix for logging (used in recursion)

    Returns:
        ModuleTreeNode containing all tensor data and submodules
    """
    node = ModuleTreeNode()

    # Extract parameters at this level (recurse=False)
    for name, param in model.named_parameters(recurse=False):
        if not isinstance(param, torch.nn.Parameter):
            continue

        tensor = param.data

        # Ensure tensor is on CUDA
        if tensor.device.type != "cuda":
            logger.warning(f"Parameter {prefix}{name} is not on CUDA, skipping")
            continue

        # Get rebuild info for IPC sharing
        _, rebuild_args = reduce_tensor(tensor)
        node.parameters[name] = rebuild_args

    # Extract buffers at this level (recurse=False)
    for name, buffer in model.named_buffers(recurse=False):
        if buffer is None:
            continue

        # Ensure buffer is on CUDA
        if buffer.device.type != "cuda":
            logger.warning(f"Buffer {prefix}{name} is not on CUDA, skipping")
            continue

        # Get rebuild info for IPC sharing
        _, rebuild_args = reduce_tensor(buffer)
        node.buffers[name] = rebuild_args

    # Extract tensor attributes that aren't parameters or buffers
    # Get all parameters and buffers names to exclude them
    param_names = {name for name, _ in model.named_parameters(recurse=False)}
    buffer_names = {name for name, _ in model.named_buffers(recurse=False)}
    submodule_names = {name for name, _ in model.named_children()}

    for name in dir(model):
        # Skip already processed params/buffers/submodules
        if name in param_names or name in buffer_names or name in submodule_names:
            continue

        try:
            attr = getattr(model, name)

            # Handle tensor attributes
            if isinstance(attr, torch.Tensor):
                if attr.device.type == "cuda":
                    if attr.numel() > 0:
                        _, rebuild_args = reduce_tensor(attr)
                        node.tensor_attrs[name] = rebuild_args
                        logger.debug(f"Extracted tensor attribute {prefix}{name} on {attr.device}")
                    else:
                        # Store empty tensors too - they need to be CUDA, not meta
                        node.tensor_attrs[name] = ("empty", attr.device.index, attr.dtype)
                        logger.debug(f"Extracted empty tensor attribute {prefix}{name} on {attr.device}")
                elif attr.device.type == "meta":
                    logger.warning(f"Skipping meta tensor attribute {prefix}{name} - model may have tensors still on meta device!")

            # Handle lists of tensors (like kv_cache)
            elif isinstance(attr, list) and len(attr) > 0 and isinstance(attr[0], torch.Tensor):
                tensor_list = []
                all_empty = True
                for i, tensor in enumerate(attr):
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                        if tensor.numel() > 0:
                            _, rebuild_args = reduce_tensor(tensor)
                            tensor_list.append(rebuild_args)
                            all_empty = False
                        else:
                            # Store marker for empty tensor with device info
                            tensor_list.append(("empty", tensor.device.index, tensor.dtype))
                    else:
                        tensor_list.append(None)  # Placeholder for non-cuda tensors
                # Include list if it has at least one cuda tensor (even if empty)
                if not all_empty or any(t is not None for t in tensor_list):
                    node.tensor_attrs[name] = tensor_list
                    logger.debug(f"Extracted tensor list attribute {prefix}{name} with {len(tensor_list)} tensors")
        except Exception as e:
            # Skip attributes that can't be accessed
            logger.debug(f"Skipping attribute {prefix}{name}: {e}")
            continue

    # Recursively handle submodules
    for name, submodule in model.named_children():
        new_prefix = f"{prefix}{name}."
        node.submodules[name] = extract_module_tree_ipc_info(submodule, new_prefix)

    return node


def import_weights_from_tree(
    target_module: torch.nn.Module,
    tree: ModuleTreeNode,
    prefix: str = ""
) -> None:
    """
    Import weights from the module tree into the target module.

    Uses the same pattern as meta_load.py's import_weights function.
    Recursively merges parameters, buffers, tensor attributes, and submodules.

    Args:
        target_module: The module to import weights into
        tree: The module tree node from companion server with IPC rebuild info
        prefix: Current prefix for logging (internal use)
    """
    # Import parameters at this level
    for name, rebuild_args in tree.parameters.items():
        logger.debug(f"Importing parameter: {prefix}{name}")
        tensor = reconstruct_tensor_from_ipc(rebuild_args)

        if hasattr(target_module, name):
            delattr(target_module, name)  # Remove the existing parameter
        target_module.register_parameter(name, torch.nn.Parameter(tensor))

    # Import buffers at this level
    for name, rebuild_args in tree.buffers.items():
        logger.debug(f"Importing buffer: {prefix}{name}")
        tensor = reconstruct_tensor_from_ipc(rebuild_args)

        if hasattr(target_module, name):
            delattr(target_module, name)  # Remove the existing buffer
        target_module.register_buffer(name, tensor)

    # Import tensor attributes (non-parameter, non-buffer tensors)
    for name, attr_value in tree.tensor_attrs.items():
        if isinstance(attr_value, list):
            # Handle list of tensors (like kv_cache)
            tensor_list = []
            for item in attr_value:
                if item is not None:
                    tensor = reconstruct_tensor_from_ipc(item)
                    tensor_list.append(tensor)
                else:
                    # Keep non-cuda placeholder tensors as is
                    tensor_list.append(torch.tensor([]))
            logger.debug(f"Importing tensor list attribute: {prefix}{name} with {len(tensor_list)} tensors")
            setattr(target_module, name, tensor_list)
        else:
            # Handle single tensor
            tensor = reconstruct_tensor_from_ipc(attr_value)
            logger.debug(f"Importing tensor attribute: {prefix}{name}")
            setattr(target_module, name, tensor)

    # Recursively handle submodules
    for name, submodule_tree in tree.submodules.items():
        logger.debug(f"Importing submodule: {prefix}{name}")

        if hasattr(target_module, name):
            submodule = getattr(target_module, name)
            import_weights_from_tree(submodule, submodule_tree, f"{prefix}{name}.")
        else:
            logger.warning(f"Submodule {prefix}{name} not found in target model, skipping")


def check_for_meta_tensors(model: torch.nn.Module) -> List[str]:
    """
    Check if any tensors in the model are still on meta device.

    This checks parameters, buffers, and all tensor attributes in all modules.

    Args:
        model: PyTorch model to check

    Returns:
        List of tensor names/paths that are still on meta device
    """
    meta_tensors = []

    # Check parameters
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_tensors.append(f"param:{name}")

    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            meta_tensors.append(f"buffer:{name}")

    # Check tensor attributes in all modules
    for module_name, module in model.named_modules():
        # Get all parameters and buffers names to exclude them
        param_names = {name for name, _ in module.named_parameters(recurse=False)}
        buffer_names = {name for name, _ in module.named_buffers(recurse=False)}
        submodule_names = {name for name, _ in module.named_children()}

        for attr_name in dir(module):
            # Skip already processed params/buffers/submodules
            if attr_name in param_names or attr_name in buffer_names or attr_name in submodule_names:
                continue

            try:
                attr = getattr(module, attr_name)

                # Check single tensors
                if isinstance(attr, torch.Tensor) and attr.device.type == "meta":
                    full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                    meta_tensors.append(f"attr:{full_name}")

                # Check lists of tensors
                elif isinstance(attr, list):
                    for i, item in enumerate(attr):
                        if isinstance(item, torch.Tensor) and item.device.type == "meta":
                            full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                            meta_tensors.append(f"attr:{full_name}[{i}]")
            except Exception:
                # Skip attributes that can't be accessed
                continue

    return meta_tensors

