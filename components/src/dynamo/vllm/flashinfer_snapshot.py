# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer peer-resource pause/resume hooks for vLLM Snapshot.

This module intentionally avoids importing vLLM, torch, or FlashInfer at module
import time. Snapshot workers call these helpers after vLLM has initialized
distributed state inside each rank process.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("vllm.dynamo.flashinfer_snapshot")

_FI_AR_MODULE = "vllm.distributed.device_communicators.flashinfer_all_reduce"
_PARALLEL_STATE_MODULE = "vllm.distributed.parallel_state"
_MNNVL_COMPAT_MODULE = "vllm.distributed.device_communicators.mnnvl_compat"
_FLASHINFER_TWO_SIDED_MODULE = "flashinfer.comm.trtllm_alltoall"
_FLASHINFER_ONE_SIDED_MODULE = "flashinfer.comm.trtllm_moe_alltoall"
DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES = (
    "DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES"
)


@dataclass(frozen=True)
class _Resource:
    name: str
    obj: Any
    kind: str


@dataclass(frozen=True)
class ResourceSummary:
    """Log-safe summary of a discovered FlashInfer peer resource."""

    name: str
    kind: str
    class_name: str


@dataclass(frozen=True)
class FlashInferResourceReport:
    """FlashInfer resource report returned by pause/resume hooks."""

    operation: str
    resources: tuple[ResourceSummary, ...]

    @property
    def count(self) -> int:
        return len(self.resources)


def pause_flashinfer_peer_resources(worker: Any) -> FlashInferResourceReport:
    """Pause active FlashInfer peer resources before checkpoint."""
    try:
        return _apply_flashinfer_operation(worker, "pause")
    except Exception:
        logger.exception("Failed to pause FlashInfer peer resources.")
        raise


def resume_flashinfer_peer_resources(worker: Any) -> FlashInferResourceReport:
    """Resume active FlashInfer peer resources after restore."""
    try:
        return _apply_flashinfer_operation(worker, "resume")
    except Exception:
        logger.exception("Failed to resume FlashInfer peer resources.")
        raise


def inspect_flashinfer_peer_resources(worker: Any) -> FlashInferResourceReport:
    """Inspect active FlashInfer peer resources without mutating them."""
    try:
        _, report = _discover_flashinfer_resource_report(worker, "inspect")
        return report
    except Exception:
        logger.exception("Failed to inspect FlashInfer peer resources.")
        raise


def _apply_flashinfer_operation(
    worker: Any, operation: str
) -> FlashInferResourceReport:
    resources, report = _discover_flashinfer_resource_report(worker, operation)
    if not resources:
        return report

    context = _worker_context(worker)
    for resource in resources:
        logger.info(
            "FlashInfer peer resource %s start rank=%s local_rank=%s pid=%s "
            "resource=%s class=%s",
            operation,
            context["rank"],
            context["local_rank"],
            context["pid"],
            resource.name,
            _qualified_class_name(resource.obj),
        )
        if resource.kind == "two_sided_manager":
            _apply_two_sided_manager(resource.obj, operation)
        elif resource.kind == "one_sided_manager":
            _apply_one_sided_manager(resource.obj, operation)
        elif resource.kind == "generic":
            _apply_generic_resource(resource, operation)
        else:
            raise RuntimeError(
                f"Active FlashInfer resource {resource.name} "
                f"({_qualified_class_name(resource.obj)}) has no supported "
                f"{operation} path."
            )
        logger.info(
            "FlashInfer peer resource %s complete rank=%s local_rank=%s "
            "pid=%s resource=%s class=%s",
            operation,
            context["rank"],
            context["local_rank"],
            context["pid"],
            resource.name,
            _qualified_class_name(resource.obj),
        )
    return report


def _discover_flashinfer_resource_report(
    worker: Any, operation: str
) -> tuple[list[_Resource], FlashInferResourceReport]:
    resources = _discover_resources(worker)
    context = _worker_context(worker)
    report = _resource_report(operation, resources)
    logger.info(
        "Discovered FlashInfer peer resources for %s rank=%s local_rank=%s "
        "pid=%s resource_count=%s resources=%s",
        operation,
        context["rank"],
        context["local_rank"],
        context["pid"],
        report.count,
        _resources_for_log(report),
    )
    if not resources:
        if _require_flashinfer_snapshot_resources():
            raise RuntimeError(
                "No active FlashInfer peer resources discovered for Snapshot "
                f"{operation}, but "
                f"{DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES}=1."
            )
        logger.info(
            "No active FlashInfer peer resources to %s " "rank=%s local_rank=%s pid=%s",
            operation,
            context["rank"],
            context["local_rank"],
            context["pid"],
        )
        return resources, report

    return resources, report


def _discover_resources(worker: Any) -> list[_Resource]:
    resources: list[_Resource] = []
    seen: set[int] = set()
    found_allreduce_workspace = _discover_vllm_allreduce_workspaces(resources, seen)
    _discover_worker_parallel_groups(resources, seen, found_allreduce_workspace)
    supported_kinds = {resource.kind for resource in resources}
    found_two_sided_manager = "two_sided_manager" in supported_kinds
    found_one_sided_manager = "one_sided_manager" in supported_kinds
    _discover_static_two_sided_workspaces(resources, seen, found_two_sided_manager)
    _discover_one_sided_workspace_cache(resources, seen, found_one_sided_manager)
    return resources


def _discover_vllm_allreduce_workspaces(
    resources: list[_Resource], seen: set[int]
) -> bool:
    module = _optional_import(_FI_AR_MODULE)
    if module is None:
        return False

    found_workspace = False
    for attr in ("_fi_ar_workspace", "_fi_ar_quant_workspace"):
        if not hasattr(module, attr):
            logger.warning(
                "vLLM FlashInfer all-reduce module %s has no %s attribute; "
                "cannot inspect that Snapshot peer resource.",
                _FI_AR_MODULE,
                attr,
            )
            continue
        workspace = getattr(module, attr)
        if workspace is not None:
            found_workspace |= _add_resource(
                resources, seen, f"{_FI_AR_MODULE}.{attr}", workspace
            )
    return found_workspace


def _discover_worker_parallel_groups(
    resources: list[_Resource], seen: set[int], found_allreduce_workspace: bool
) -> None:
    parallel_state = _optional_import(_PARALLEL_STATE_MODULE)
    if parallel_state is None:
        return

    ep_group = _get_parallel_group(parallel_state, "get_ep_group")
    if ep_group is not None:
        device_communicator = _get_expected_attr(
            ep_group, "device_communicator", "EP group"
        )
        if device_communicator is not None:
            all2all_manager = getattr(device_communicator, "all2all_manager", None)
            _discover_all2all_manager(resources, seen, all2all_manager)

    tp_group = _get_parallel_group(parallel_state, "get_tp_group")
    if tp_group is not None:
        device_communicator = _get_expected_attr(
            tp_group, "device_communicator", "TP group"
        )
        if device_communicator is not None:
            fi_ar_comm = getattr(device_communicator, "fi_ar_comm", None)
            _discover_tp_flashinfer_allreduce(
                resources, seen, fi_ar_comm, found_allreduce_workspace
            )


def _discover_all2all_manager(
    resources: list[_Resource], seen: set[int], manager: Any
) -> None:
    if manager is None:
        return

    class_name = manager.__class__.__name__
    if class_name == "FlashInferNVLinkTwoSidedManager":
        if getattr(manager, "initialized", False):
            _add_resource(
                resources,
                seen,
                "get_ep_group().device_communicator.all2all_manager",
                manager,
                kind="two_sided_manager",
            )
        else:
            logger.debug("FlashInfer two-sided all2all manager is not initialized.")
        return

    if class_name == "FlashInferNVLinkOneSidedManager":
        if getattr(manager, "initialized", False) or (
            getattr(manager, "moe_alltoall", None) is not None
        ):
            _add_resource(
                resources,
                seen,
                "get_ep_group().device_communicator.all2all_manager",
                manager,
                kind="one_sided_manager",
            )
        else:
            logger.debug("FlashInfer one-sided all2all manager is not initialized.")
        return

    if _supports_generic_pause_resume(manager):
        _add_resource(
            resources,
            seen,
            "get_ep_group().device_communicator.all2all_manager",
            manager,
        )
        return

    if "FlashInfer" in class_name and getattr(manager, "initialized", True):
        _add_resource(
            resources,
            seen,
            "get_ep_group().device_communicator.all2all_manager",
            manager,
            kind="unsupported",
        )


def _discover_static_two_sided_workspaces(
    resources: list[_Resource], seen: set[int], found_two_sided_manager: bool
) -> None:
    module = _optional_import(_FLASHINFER_TWO_SIDED_MODULE)
    if module is None:
        return

    mnnvl_moe = getattr(module, "MnnvlMoe", None)
    if mnnvl_moe is None:
        logger.warning(
            "%s has no MnnvlMoe attribute; cannot inspect static FlashInfer "
            "two-sided Snapshot resources.",
            _FLASHINFER_TWO_SIDED_MODULE,
        )
        return

    for attr in ("moe_workspace", "moe_prepare_workspace"):
        workspace = getattr(mnnvl_moe, attr, None)
        if workspace is None:
            continue
        name = f"{_FLASHINFER_TWO_SIDED_MODULE}.MnnvlMoe.{attr}"
        if found_two_sided_manager:
            logger.debug(
                "Discovered static FlashInfer two-sided workspace %s; supported "
                "manager resource is already present.",
                name,
            )
            continue
        _add_resource(resources, seen, name, workspace, kind="unsupported")


def _discover_one_sided_workspace_cache(
    resources: list[_Resource], seen: set[int], found_one_sided_manager: bool
) -> None:
    module = _optional_import(_FLASHINFER_ONE_SIDED_MODULE)
    if module is None:
        return

    moe_alltoall = getattr(module, "MoeAlltoAll", None)
    if moe_alltoall is None:
        logger.warning(
            "%s has no MoeAlltoAll attribute; cannot inspect cached FlashInfer "
            "one-sided Snapshot resources.",
            _FLASHINFER_ONE_SIDED_MODULE,
        )
        return

    workspace_cache = getattr(moe_alltoall, "_WORKSPACE_CACHE", None)
    if not workspace_cache:
        return
    if found_one_sided_manager:
        logger.debug(
            "Discovered FlashInfer one-sided _WORKSPACE_CACHE with %s entries; "
            "supported manager resource is already present.",
            _safe_len(workspace_cache),
        )
        return
    _add_resource(
        resources,
        seen,
        f"{_FLASHINFER_ONE_SIDED_MODULE}.MoeAlltoAll._WORKSPACE_CACHE",
        workspace_cache,
        kind="unsupported",
    )


def _discover_tp_flashinfer_allreduce(
    resources: list[_Resource],
    seen: set[int],
    fi_ar_comm: Any,
    found_allreduce_workspace: bool,
) -> None:
    if fi_ar_comm is None:
        return

    if getattr(fi_ar_comm, "disabled", False):
        logger.debug("FlashInfer TP all-reduce communicator is disabled.")
        return

    if _supports_generic_pause_resume(fi_ar_comm):
        _add_resource(
            resources,
            seen,
            "get_tp_group().device_communicator.fi_ar_comm",
            fi_ar_comm,
        )
        return

    if not found_allreduce_workspace:
        _add_resource(
            resources,
            seen,
            "get_tp_group().device_communicator.fi_ar_comm",
            fi_ar_comm,
            kind="unsupported",
        )
        return

    logger.debug(
        "Discovered %s without direct pause/resume hooks; initialized "
        "FlashInfer all-reduce workspaces are handled through %s globals.",
        _qualified_class_name(fi_ar_comm),
        _FI_AR_MODULE,
    )


def _add_resource(
    resources: list[_Resource],
    seen: set[int],
    name: str,
    obj: Any,
    kind: str = "generic",
) -> bool:
    key = id(obj)
    if key in seen:
        return False
    seen.add(key)
    resources.append(_Resource(name=name, obj=obj, kind=kind))
    return True


def _resource_report(
    operation: str, resources: list[_Resource]
) -> FlashInferResourceReport:
    return FlashInferResourceReport(
        operation=operation,
        resources=tuple(
            ResourceSummary(
                name=resource.name,
                kind=resource.kind,
                class_name=_qualified_class_name(resource.obj),
            )
            for resource in resources
        ),
    )


def _resources_for_log(report: FlashInferResourceReport) -> list[dict[str, str]]:
    return [
        {
            "name": resource.name,
            "kind": resource.kind,
            "class": resource.class_name,
        }
        for resource in report.resources
    ]


def _apply_generic_resource(resource: _Resource, operation: str) -> None:
    method = getattr(resource.obj, operation, None)
    if not callable(method):
        raise RuntimeError(
            f"Active FlashInfer resource {resource.name} "
            f"({_qualified_class_name(resource.obj)}) does not support "
            f"{operation}()."
        )
    method(synchronize=True, barrier=True)


def _apply_two_sided_manager(manager: Any, operation: str) -> None:
    mnnvl_moe = _require_attr_from_module(
        _FLASHINFER_TWO_SIDED_MODULE,
        "MnnvlMoe",
        "FlashInfer two-sided all2all Snapshot hook",
    )
    if operation == "pause":
        mnnvl_moe.pause(synchronize=True, barrier=True)
        return

    mnnvl_config, custom_communicator = _mnnvl_resume_deps(
        "FlashInfer two-sided all2all Snapshot hook"
    )
    cpu_group = _require_resource_attr(manager, "cpu_group")
    config = mnnvl_config(
        comm_backend=custom_communicator(cpu_group),
        fabric_page_size=1 << 29,
        allocation_granularity=0,
    )
    mnnvl_moe.resume(
        config=config,
        synchronize=True,
        barrier=True,
        zero_local=True,
    )


def _apply_one_sided_manager(manager: Any, operation: str) -> None:
    moe_alltoall = _require_resource_attr(manager, "moe_alltoall")
    if operation == "pause":
        pause = getattr(moe_alltoall, "pause", None)
        if not callable(pause):
            raise RuntimeError(
                "Active FlashInfer one-sided MoeAlltoAll resource "
                f"({_qualified_class_name(moe_alltoall)}) does not support pause()."
            )
        pause(synchronize=True, barrier=True)
        return

    resume = getattr(moe_alltoall, "resume", None)
    if not callable(resume):
        raise RuntimeError(
            "Active FlashInfer one-sided MoeAlltoAll resource "
            f"({_qualified_class_name(moe_alltoall)}) does not support resume()."
        )
    mnnvl_config, custom_communicator = _mnnvl_resume_deps(
        "FlashInfer one-sided all2all Snapshot hook"
    )
    cpu_group = _require_resource_attr(manager, "cpu_group")
    config = mnnvl_config(comm_backend=custom_communicator(cpu_group))
    resume(
        config=config,
        synchronize=True,
        barrier=True,
        reinitialize=True,
    )


def _mnnvl_resume_deps(reason: str) -> tuple[Any, Any]:
    mnnvl_config = _require_attr_from_module(
        "flashinfer.comm.mnnvl", "MnnvlConfig", reason
    )
    custom_communicator = _require_attr_from_module(
        _MNNVL_COMPAT_MODULE, "CustomCommunicator", reason
    )
    return mnnvl_config, custom_communicator


def _supports_generic_pause_resume(obj: Any) -> bool:
    return callable(getattr(obj, "pause", None)) or callable(
        getattr(obj, "resume", None)
    )


def _require_flashinfer_snapshot_resources() -> bool:
    return os.environ.get(DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES) == "1"


def _safe_len(obj: Any) -> str:
    try:
        return str(len(obj))
    except Exception:
        return "unknown"


def _get_parallel_group(parallel_state: Any, getter_name: str) -> Any:
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        logger.warning(
            "vLLM parallel_state has no %s(); cannot inspect Snapshot "
            "FlashInfer resources for that group.",
            getter_name,
        )
        return None
    try:
        return getter()
    except AssertionError as exc:
        logger.debug(
            "vLLM %s unavailable for Snapshot FlashInfer hook: %s", getter_name, exc
        )
        return None
    except Exception as exc:
        logger.warning(
            "vLLM %s failed while inspecting Snapshot FlashInfer resources: %s",
            getter_name,
            exc,
        )
        return None


def _get_expected_attr(obj: Any, attr: str, description: str) -> Any:
    if not hasattr(obj, attr):
        logger.warning(
            "%s has no %s attribute; cannot inspect Snapshot FlashInfer resources.",
            description,
            attr,
        )
        return None
    return getattr(obj, attr)


def _optional_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logger.warning(
            "Could not import %s while inspecting Snapshot FlashInfer resources: %s",
            module_name,
            exc,
        )
        return None


def _require_attr_from_module(module_name: str, attr: str, reason: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"{reason} requires importable {module_name}; import failed: {exc}"
        ) from exc

    value = getattr(module, attr, None)
    if value is None:
        raise RuntimeError(f"{reason} requires {module_name}.{attr}.")
    return value


def _require_resource_attr(obj: Any, attr: str) -> Any:
    value = getattr(obj, attr, None)
    if value is None:
        raise RuntimeError(
            f"Active FlashInfer resource {_qualified_class_name(obj)} has no "
            f"required {attr!r} attribute."
        )
    return value


def _worker_context(worker: Any) -> dict[str, Any]:
    return {
        "rank": getattr(worker, "rank", os.environ.get("RANK", "unknown")),
        "local_rank": getattr(
            worker, "local_rank", os.environ.get("LOCAL_RANK", "unknown")
        ),
        "pid": os.getpid(),
    }


def _qualified_class_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"
