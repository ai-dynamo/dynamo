#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Assumption tests for KVBM connector's expectations of vLLM interfaces.

These unit tests validate that KVBM's assumptions about vLLM's internal
interfaces remain stable across vLLM releases. They do NOT test functional
correctness of KVBM or vLLM logic, but rather ensure the API contract remains
intact to prevent silent breakage.

Inspired by vLLM's test_lmcache_integration.py approach to interface testing.
"""

from typing import Any

import pytest

# Skip if vLLM is not available
pytest.importorskip("vllm", reason="vLLM not available")

# ruff: noqa: E402
# Imports must be after pytest.importorskip() to handle missing vLLM gracefully
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

# Test markers
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
    pytest.mark.nightly,
    pytest.mark.pre_merge,
]


def _get_obj_name(obj: Any) -> str:
    """Get a readable name for an object (class name or repr)."""
    return getattr(obj, "__name__", None) or obj.__class__.__name__


def _assert_attr_exists(obj: Any, attr: str) -> None:
    """Assert that an attribute exists on an object or dataclass."""
    obj_name = _get_obj_name(obj)
    # Check __dataclass_fields__ directly - works for both classes and instances,
    # and handles decorated dataclasses (e.g., @config @dataclass)
    dataclass_fields = getattr(obj, "__dataclass_fields__", None)
    if dataclass_fields is not None:
        assert attr in dataclass_fields, f"Dataclass {obj_name} missing field '{attr}'"
    else:
        assert hasattr(obj, attr), f"Object {obj_name} missing attribute '{attr}'"


def _get_property_return_type(prop: property) -> Any:
    """Extract return type from a property's fget annotations."""
    fget = prop.fget
    if fget is None or not hasattr(fget, "__annotations__"):
        return None
    annotations = fget.__annotations__
    if "return" not in annotations:
        return None
    return_type = annotations["return"]
    # Handle Optional types by extracting the inner type
    if hasattr(return_type, "__origin__") and return_type.__origin__ is type(None):
        return_type = return_type.__args__[0]
    return return_type


def _assert_instance_of(obj: Any, attr: str, value: Any, expected_type: Any) -> None:
    """Assert that value matches expected type, handling properties specially."""
    prop = type(obj).__dict__.get(attr)

    if isinstance(prop, property):
        return_type = _get_property_return_type(prop)
        if return_type is not None:
            is_match = return_type == expected_type or (
                isinstance(return_type, type) and issubclass(return_type, expected_type)
            )
            assert (
                is_match
            ), f"Property '{attr}' return type {return_type} is not {expected_type}"
            return

    assert isinstance(
        value, expected_type
    ), f"Attribute '{attr}' value {type(value)} is not instance of {expected_type}"


def _get_type_origin(t: Any) -> Any:
    """Extract the origin type from a potentially parameterized generic.

    e.g., list[int] -> list, set[str] -> set, dict[str, Any] -> dict
    """
    origin = getattr(t, "__origin__", None)
    return origin if origin is not None else t


def _check_dataclass_field_type(obj: type, attr: str, expected_type: Any) -> None:
    """Check dataclass field type annotation matches expected type."""
    field = getattr(obj, "__dataclass_fields__")[attr]
    field_type = field.type

    # Handle generic types (e.g., list[int] -> list, set[str] -> set)
    field_type_origin = _get_type_origin(field_type)
    expected_type_origin = _get_type_origin(expected_type)

    obj_name = _get_obj_name(obj)

    # First check exact match (including parameterized generics)
    if field_type == expected_type:
        return

    # Then check origin types match (e.g., set[str] vs set[int] both have origin set)
    if field_type_origin == expected_type_origin:
        return

    # Finally check subclass relationship (only works with actual types, not generics)
    if isinstance(field_type_origin, type) and isinstance(expected_type_origin, type):
        if issubclass(field_type_origin, expected_type_origin):
            return

    raise AssertionError(
        f"Dataclass {obj_name}.{attr} type {field_type} is not {expected_type}"
    )


def assumes(obj: Any, attr: str, is_callable: bool = False, is_instance_of: Any = None):
    """
    Helper function to validate interface assumptions.

    Checks that an object has the expected attribute with correct type and callability.
    Used to guard against breaking changes in vLLM's internal interfaces.

    Args:
        obj: The object to check
        attr: The attribute name to validate
        is_callable: If True, verify the attribute is callable
        is_instance_of: If provided, verify the attribute is an instance of this type
    """
    _assert_attr_exists(obj, attr)

    # For dataclass classes (not instances), fields with default_factory don't exist
    # as class attributes, so check field type annotation instead of getattr
    dataclass_fields = getattr(obj, "__dataclass_fields__", None)
    is_dataclass_class = dataclass_fields is not None and isinstance(obj, type)

    if is_dataclass_class:
        if is_instance_of is not None:
            _check_dataclass_field_type(obj, attr, is_instance_of)
        # Note: is_callable check not supported for dataclass class fields
        return

    value = getattr(obj, attr)

    if is_callable:
        assert callable(
            value
        ), f"Attribute '{attr}' on {_get_obj_name(obj)} is not callable"

    if is_instance_of is not None:
        _assert_instance_of(obj, attr, value, is_instance_of)


def test_config_interface():
    assumes(VllmConfig, "model_config")
    assumes(VllmConfig, "cache_config")
    assumes(VllmConfig, "parallel_config")
    assumes(VllmConfig, "kv_transfer_config")
    assumes(VllmConfig, "kv_events_config")

    assumes(KVTransferConfig, "kv_role")
    assumes(KVTransferConfig, "kv_load_failure_policy")
    assumes(KVTransferConfig, "kv_connector_module_path")
    assumes(KVTransferConfig, "engine_id")
    assumes(KVTransferConfig, "kv_connector")
    assumes(KVTransferConfig, "kv_connector_extra_config")

    assumes(ModelConfig, "dtype")

    assumes(ParallelConfig, "world_size")
    assumes(ParallelConfig, "data_parallel_rank")

    assumes(CacheConfig, "cache_dtype")
    assumes(CacheConfig, "block_size")
    assumes(CacheConfig, "gpu_memory_utilization")
    assumes(CacheConfig, "enable_prefix_caching")


def test_scheduler_output_interface():
    """
    Test SchedulerOutput interface expectations for KVBM vLLM integration.
    Protects against interface changes in vLLM's SchedulerOutput object.
    """
    assumes(SchedulerOutput, "finished_req_ids", is_instance_of=set[str])
    assumes(SchedulerOutput, "scheduled_new_reqs", is_instance_of=list[NewRequestData])
    assumes(SchedulerOutput, "num_scheduled_tokens", is_instance_of=dict)
    assumes(SchedulerOutput, "total_num_scheduled_tokens")


def test_request_interface():
    """
    Test Request interface expectations for KVBM vLLM integration.
    Protects against interface changes in vLLM's Request object.
    """
    req = Request(
        request_id="test_request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        eos_token_id=100,
        lora_request=LoRARequest(
            lora_name="test_lora", lora_int_id=1, lora_path="test_path"
        ),
        cache_salt="test_salt",
    )

    assumes(req, "request_id", is_instance_of=str)
    assumes(req, "all_token_ids")  # ConstantList
    assumes(req, "num_tokens", is_instance_of=int)
    assumes(req, "num_computed_tokens", is_instance_of=int)
    assumes(req, "cache_salt", is_instance_of=str)
    assumes(req, "lora_request", is_instance_of=LoRARequest)
    assumes(req, "priority", is_instance_of=int)
    assumes(req, "sampling_params", is_instance_of=SamplingParams)


def test_new_request_interface():
    """
    Test NewRequestData interface expectations for KVBM vLLM integration.
    Protects against interface changes in vLLM's NewRequestData object.
    """
    assumes(NewRequestData, "req_id", is_instance_of=str)
    assumes(NewRequestData, "block_ids", is_instance_of=tuple[list[int], ...])
    assumes(NewRequestData, "prompt_token_ids", is_instance_of=(list[int] | None))
    assumes(NewRequestData, "num_computed_tokens", is_instance_of=int)


def test_cached_request_interface():
    assumes(CachedRequestData, "resumed_req_ids", is_instance_of=set[str])
    assumes(CachedRequestData, "req_ids", is_instance_of=list[str])
    assumes(CachedRequestData, "new_token_ids", is_instance_of=list[list[int]])
    assumes(
        CachedRequestData,
        "new_block_ids",
        is_instance_of=list[tuple[list[int], ...] | None],
    )
    assumes(CachedRequestData, "num_computed_tokens", is_instance_of=list[int])
