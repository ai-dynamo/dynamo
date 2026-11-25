# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retune's semantic launch profile and public graph constraints.

The operator supplies the immutable process binding. Status acknowledges only
its target hash; the frontend separately acknowledges compatible running routers.
No function inspects Pod templates or reconstructs their launch semantics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional

from aisim_core.sdk.common import DefaultHFModels

from dynamo.llm import KvRouterConfig
from dynamo.planner.retune.controller import startup_router_domain
from dynamo.planner.retune.live.contracts import (
    FrontendStartup,
    PremiseReport,
    RetuneCapability,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import (
        DgdrExecutorConfig,
        PlannerConfig,
        RetuneConfig,
    )

FRONTEND_TYPE = "frontend"
PLANNER_TYPE = "planner"
WORKER_TYPES = frozenset({"worker", "prefill", "decode", ""})


def worker_types(planner_mode: str) -> tuple[str, ...]:
    return ("prefill", "decode") if planner_mode == "disagg" else ("aggregated",)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def find_frontend(dgd: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    components = _components(dgd)
    frontends = [
        component for component in components if _type(component) == FRONTEND_TYPE
    ]
    if len(frontends) != 1:
        raise ValueError(
            f"P7: the DGD has {len(frontends)} frontend components; live retune needs exactly one"
        )
    frontend = frontends[0]
    name = frontend.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("P7: the frontend has no component name")
    return name, dict(frontend)


def served_model_names(
    dgd: Mapping[str, Any],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Read existing semantic model status, never model-flag guesses."""

    projected = _projected_status(dgd)
    models: dict[str, str] = {}
    unknown: list[str] = []
    for component in _components(dgd):
        if _type(component) not in WORKER_TYPES:
            continue
        name = str(component.get("name", ""))
        facts = projected.get(name)
        model = facts.get("servedModelName") if isinstance(facts, Mapping) else None
        if isinstance(model, str) and model:
            models[name] = model
        else:
            unknown.append(name)
    return models, tuple(sorted(unknown))


def served_model_name(dgd: Mapping[str, Any]) -> Optional[str]:
    names = set(served_model_names(dgd)[0].values())
    return next(iter(names)) if len(names) == 1 else None


def retune_capability(
    dgd: Mapping[str, Any], process_capability: Optional[RetuneCapability]
) -> RetuneCapability:
    """Accept a process binding only while the current target acknowledges it."""

    if process_capability is None:
        raise ValueError(
            "this Planner process has no supported operator-provided Retune binding"
        )
    projected = _projected_status(dgd)
    published = [
        status.get("retune")
        for component in _components(dgd)
        if _type(component) == PLANNER_TYPE
        and isinstance(name := component.get("name"), str)
        and isinstance(status := projected.get(name), Mapping)
        and status.get("retune") is not None
    ]
    if len(published) != 1 or not isinstance(published[0], Mapping):
        raise ValueError(
            "exactly one current Planner Retune target must be acknowledged"
        )
    if published[0].get("configHash") != process_capability.config_hash:
        raise ValueError(
            "the published Retune target differs from this Planner process's binding; wait for its replacement"
        )
    frontend_name, _ = find_frontend(dgd)
    if frontend_name != process_capability.frontend_component_name:
        raise ValueError("the Retune binding refers to a different frontend")
    return process_capability


def resolve_startup(
    dgd: Mapping[str, Any],
    *,
    planner_mode: str,
    capability: Optional[RetuneCapability] = None,
) -> FrontendStartup | str:
    """Resolve the shared runtime profile without interpreting a launch recipe."""

    try:
        name, spec = find_frontend(dgd)
        if capability is None:
            raise ValueError("the semantic Retune launch binding is unavailable")
        if name != capability.frontend_component_name:
            raise ValueError("the Retune binding refers to a different frontend")
        # Missing profile support in an older SDK makes this opt-in unavailable.
        resolver = getattr(KvRouterConfig, "retune_profile", None)
        if resolver is None:
            raise ValueError("the runtime does not support the Retune launch profile")
        config = resolver(
            capability.profile,
            capability.overlap_score_credit,
            capability.prefill_load_scale,
            capability.temperature,
        )
        router_config = dict(config.frontend_kwargs())
        maps = {
            worker_type: {
                key: float(value)
                for key, value in config.policy_parameters(worker_type).items()
            }
            for worker_type in worker_types(planner_mode)
        }
        return FrontendStartup(
            router_mode="kv_router",
            router_config=router_config,
            startup_maps=maps,
            frontend_component=name,
            replicas=_replicas(spec),
            replica_sync=bool(router_config["router_replica_sync"]),
            fingerprint_source=canonical_json(
                {
                    "configHash": capability.config_hash,
                    "router_config": router_config,
                    "startup_maps": maps,
                }
            ),
        )
    except (ValueError, TypeError) as error:
        return f"P4: {error}"


def resolve_executor_config(
    config: DgdrExecutorConfig, capability: RetuneCapability
) -> DgdrExecutorConfig:
    """Bind storage to this process, treating legacy declarations as assertions."""

    wiring = {
        "claim_name": capability.claim_name,
        "mount_path": capability.mount_path,
        "segments_dir": capability.segments_dir,
        "windows_dir": capability.windows_dir,
    }
    legacy = config.model_dump(include=set(wiring))
    for name, value in wiring.items():
        if name in config.model_fields_set and legacy[name] != value:
            raise ValueError(
                f"retune.dgdr.{name} conflicts with the operator's Retune capability"
            )
    return config.model_copy(update=wiring)


def evaluate_premises(
    dgd: Mapping[str, Any],
    *,
    planner_config: PlannerConfig,
    capability: Optional[RetuneCapability] = None,
) -> PremiseReport:
    retune = planner_config.retune
    if retune is None or retune.dgdr is None:
        raise ValueError("evaluate_premises requires planner_config.retune.dgdr")
    parsed = resolve_startup(
        dgd, planner_mode=str(planner_config.mode), capability=capability
    )
    startup = parsed if isinstance(parsed, FrontendStartup) else None
    p4 = (
        [parsed]
        if isinstance(parsed, str)
        else _startup_errors(parsed, retune, str(planner_config.mode))
    )
    others: list[str] = []
    warnings: list[str] = []
    components = _components(dgd)
    models, unknown = served_model_names(dgd)
    if len(set(models.values())) > 1:
        others.append("P7: the DGD serves multiple models; live retune needs one")
    if unknown:
        warnings.append(
            f"P7: cannot determine the served model of {', '.join(unknown)}"
        )
    if retune.dgdr.model not in DefaultHFModels:
        warnings.append(
            f"P6: retune.dgdr.model {retune.dgdr.model!r} is not an AISimulate default HF model; each DGDR Run must be able to download it"
        )
    if any(_type(component) == "epp" for component in components):
        others.append("P7: the DGD routes through an EPP component (GAIE)")
    sidecars = sorted(
        str(component.get("name"))
        for component in components
        if component.get("frontendSidecar")
    )
    if sidecars:
        others.append(f"P7: {', '.join(sidecars)} run a frontendSidecar")
    if startup is not None and startup.replicas > 1 and not startup.replica_sync:
        others.append("P8: multiple frontend replicas require synchronized routers")
    for component in components:
        if (
            _type(component) in WORKER_TYPES
            and component.get("scalingAdapter") is not None
        ):
            others.append(
                f"P9: worker component {component.get('name')} has a scalingAdapter; its replicas could change during a window"
            )
    projected = _projected_status(dgd)
    for component in components:
        name = component.get("name")
        if not isinstance(name, str):
            continue
        status = projected.get(name)
        if (
            _type(component) == PLANNER_TYPE
            and _replicas(component) != 1
            and isinstance(status, Mapping)
            and status.get("retune") is not None
        ):
            others.append(
                f"P9: planner component {component.get('name')} has {_replicas(component)} replicas; live retune needs 1"
            )
    try:
        accepted = retune_capability(dgd, capability)
        resolve_executor_config(retune.dgdr, accepted)
    except ValueError as error:
        others.append(f"P10: {error}")
    reasons = tuple(_unique([*p4, *others]))
    return PremiseReport(
        p4_ok=not p4,
        spec_ok=not reasons,
        reasons=reasons,
        warnings=tuple(warnings),
        startup=startup,
    )


def _startup_errors(
    startup: FrontendStartup, retune: RetuneConfig, planner_mode: str
) -> list[str]:
    try:
        startup_router_domain(retune, startup.router_config, worker_types(planner_mode))
    except ValueError as error:
        return [f"P4: retune.router does not fit the launch profile: {error}"]
    return []


def _components(dgd: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    components = (dgd.get("spec") or {}).get("components")
    return (
        [component for component in components if isinstance(component, Mapping)]
        if isinstance(components, list)
        else []
    )


def _type(component: Mapping[str, Any]) -> str:
    return str(component.get("type") or "")


def _replicas(component: Mapping[str, Any]) -> int:
    replicas = component.get("replicas")
    try:
        return 1 if replicas is None else int(replicas)
    except (TypeError, ValueError):
        return 1


def _projected_status(dgd: Mapping[str, Any]) -> Mapping[str, Any]:
    status = dgd.get("status")
    metadata = dgd.get("metadata")
    if not isinstance(status, Mapping) or not isinstance(metadata, Mapping):
        return {}
    generation = metadata.get("generation")
    observed = status.get("observedGeneration")
    components = status.get("components")
    if (
        type(generation) is not int
        or generation < 1
        or type(observed) is not int
        or generation != observed
        or not isinstance(components, Mapping)
    ):
        return {}
    return components


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
