# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM capability publication for Dynamo KV hints."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs

from dynamo.common.constants import (
    KV_HINT_GPU_DIRECT_CAPABILITY_KEY,
    KV_HINT_GPU_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import ModelRuntimeConfig, WorkerType
from dynamo.vllm.kv_connector_protocols import MULTI_CONNECTOR_WRAPPERS

_KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY = "kv_transfer_params"
_KV_HINT_EXTRA_ARGS_KEY = "kv_hint"


@dataclass(frozen=True)
class KvTransferHintSource:
    source_control_endpoint: str
    worker_type: str


def _apply_kv_hint(sampling_params: Any, kv_hint: Any) -> None:
    """Attach the complete Dynamo KV hint message to vLLM's private input."""
    if not isinstance(kv_hint, Mapping):
        return

    extra_args = (
        dict(sampling_params.extra_args)
        if isinstance(sampling_params.extra_args, dict)
        else {}
    )
    existing_kv_transfer_params = extra_args.get(_KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY)
    kv_transfer_params = (
        dict(existing_kv_transfer_params)
        if isinstance(existing_kv_transfer_params, dict)
        else {}
    )
    kv_transfer_params[_KV_HINT_EXTRA_ARGS_KEY] = dict(kv_hint)
    extra_args[_KV_TRANSFER_PARAMS_EXTRA_ARGS_KEY] = kv_transfer_params
    sampling_params.extra_args = extra_args


def _secondary_tiers(engine_args: AsyncEngineArgs) -> list[Mapping[str, Any]]:
    """Find secondary tiers in supported connector wrappers, in child order.

    Only known wrappers have a defined ``connectors`` contract. In particular,
    wrapper-owned settings must not be mistaken for a child's KVCR settings.
    Discovery does not certify a wrapper's load arbitration or event forwarding.
    """

    def visit(kv_config: Any) -> list[Mapping[str, Any]]:
        if isinstance(kv_config, Mapping):
            name = kv_config.get("kv_connector")
            extra = kv_config.get("kv_connector_extra_config")
        else:
            name = getattr(kv_config, "kv_connector", None)
            extra = getattr(kv_config, "kv_connector_extra_config", None)
        if not isinstance(extra, Mapping):
            return []
        if name in MULTI_CONNECTOR_WRAPPERS:
            children = extra.get("connectors", [])
            if not isinstance(children, list) or any(
                not isinstance(child, Mapping) for child in children
            ):
                raise ValueError(
                    f"{name} requires connectors to be a list of connector configs"
                )
            return [tier for child in children for tier in visit(child)]
        tiers = extra.get("secondary_tiers")
        return (
            [tier for tier in tiers if isinstance(tier, Mapping)]
            if isinstance(tiers, list)
            else []
        )

    return visit(getattr(engine_args, "kv_transfer_config", None))


def _transfer_hint_domain(engine_args: AsyncEngineArgs) -> str | None:
    """Read an operator-certified model, tokenizer, namespace and layout domain.

    This is deliberately distinct from ``kv_transfer_domain``, which is a
    topology label key. No domain is inferred from a model name or connector.
    Keep it in the outer connector extra config, never inside a secondary tier:
    vLLM forwards secondary-tier settings as constructor keyword arguments.
    """
    kv_config = getattr(engine_args, "kv_transfer_config", None)
    extra = getattr(kv_config, "kv_connector_extra_config", None)
    domain = (
        extra.get(KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY)
        if isinstance(extra, Mapping)
        else None
    )
    if domain is None:
        domain = os.environ.get("DYN_KV_HINT_TRANSFER_DOMAIN")
    if domain is None:
        return None
    if not isinstance(domain, str) or not domain.strip() or domain != domain.strip():
        raise ValueError("router_hint_transfer_domain must be a nonempty string")
    return domain


def _supports_transfer_hint(tier: Mapping[str, Any]) -> bool:
    """Return whether a secondary tier advertises TRANSFER capability."""
    capabilities = tier.get("router_capabilities")
    if not isinstance(capabilities, list):
        return False
    return KV_HINT_TRANSFER_CAPABILITY_KEY in capabilities


def _transfer_hint_tiers(engine_args: AsyncEngineArgs) -> list[Mapping[str, Any]]:
    """Return secondary tiers that opt in to TRANSFER hint support."""
    transfer_hint_tiers: list[Mapping[str, Any]] = []
    for tier in _secondary_tiers(engine_args):
        if _supports_transfer_hint(tier):
            transfer_hint_tiers.append(tier)
    return transfer_hint_tiers


def _transfer_hint_source_host(host: str | None) -> str | None:
    """Normalize an advertisable host for a tcp:// endpoint."""
    if not host:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Reject URI fragments, wildcard binds, malformed IP literals and
        # whitespace without performing a blocking DNS lookup at registration.
        if len(host) > 253 or not re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.?", host
        ):
            return None
        if all(character.isdigit() or character == "." for character in host):
            return None
        if any(
            len(label) > 63
            or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
            for label in host.rstrip(".").split(".")
        ):
            return None
        return host
    if (
        address.is_unspecified
        or address.is_multicast
        or str(address) == "255.255.255.255"
    ):
        return None
    if address.version == 6:
        return f"[{address.compressed}]"
    return address.compressed


def _transfer_hint_source_port(configured_port: object) -> int | None:
    """Normalize one configured source-control port."""
    if isinstance(configured_port, bool) or not isinstance(configured_port, (int, str)):
        return None
    try:
        control_port = int(configured_port)
    except ValueError:
        return None
    if not 0 < control_port <= 65535:
        return None
    return control_port


def _transfer_hint_source_control_endpoints(
    tier: Mapping[str, Any], dp_range: tuple[int, int]
) -> dict[str, str] | None:
    """Build source-control endpoints keyed by global DP rank.

    ``control_ports`` is local to this worker: entry 0 belongs to dp_start,
    entry 1 belongs to dp_start + 1, and so on. vLLM/KVCC selects from the
    same list with data_parallel_rank_local.
    """
    dp_start, dp_size = dp_range
    if dp_start < 0 or dp_size <= 0:
        return None
    control_ports = tier.get("control_ports")
    if not isinstance(control_ports, list):
        raise ValueError(  # noqa: TRY004 - preserve the config-validation contract
            "P2P KV transfer hint support requires control_ports to be a list"
        )
    if len(control_ports) != dp_size:
        raise ValueError(
            "P2P KV transfer hint support requires control_ports to contain exactly "
            f"{dp_size} entries for the worker-local DP ranks; "
            f"got {len(control_ports)}"
        )
    configured_host = tier.get("control_advertise_host")
    host = _transfer_hint_source_host(
        configured_host if isinstance(configured_host, str) else None
    )
    if host is None:
        return None
    endpoints: dict[str, str] = {}
    for local_dp_rank, global_dp_rank in enumerate(range(dp_start, dp_start + dp_size)):
        control_port = _transfer_hint_source_port(control_ports[local_dp_rank])
        if control_port is None:
            return None
        endpoints[str(global_dp_rank)] = f"tcp://{host}:{control_port}"
    if len(set(endpoints.values())) != dp_size:
        raise ValueError(
            "P2P KV transfer hint support requires unique per-DP control ports"
        )
    return endpoints


def _transfer_hint_worker_type(worker_type: WorkerType) -> str | None:
    """Normalize a Dynamo WorkerType value into TRANSFER runtime metadata."""
    role = getattr(worker_type, "value", None)
    if not isinstance(role, str):
        role = str(worker_type)
    if role == "agg":
        role = "aggregated"
    if role not in {"aggregated", "prefill", "decode"}:
        return None
    return role


def resolve_kv_transfer_hint_sources(
    engine_args: AsyncEngineArgs,
    worker_type: WorkerType,
    dp_range: tuple[int, int] = (0, 1),
) -> dict[int, KvTransferHintSource] | None:
    """Resolve per-rank source metadata for registration or state-agent publication."""
    transfer_hint_worker_type = _transfer_hint_worker_type(worker_type)
    if transfer_hint_worker_type is None:
        return None

    transfer_hint_tiers = _transfer_hint_tiers(engine_args)
    if not transfer_hint_tiers:
        return None
    # kv.fetch@1.0 describes one P2P source and cannot disambiguate tiers.
    if len(transfer_hint_tiers) > 1:
        raise ValueError(
            "P2P KV transfer hint support requires exactly one capable "
            "secondary tier; found multiple tiers advertising "
            f"{KV_HINT_TRANSFER_CAPABILITY_KEY}"
        )

    endpoints = _transfer_hint_source_control_endpoints(
        transfer_hint_tiers[0], dp_range
    )
    if endpoints is None:
        raise ValueError(
            "P2P KV transfer hint support requires advertisable source control endpoints "
            "for all managed DP ranks"
        )

    return {
        int(rank): KvTransferHintSource(endpoint, transfer_hint_worker_type)
        for rank, endpoint in endpoints.items()
    }


def _publish_kv_transfer_hint_capability(
    runtime_config: ModelRuntimeConfig,
    engine_args: AsyncEngineArgs,
    worker_type: WorkerType,
    dp_range: tuple[int, int] = (0, 1),
    *,
    publish_source_endpoints: bool = True,
) -> None:
    """Publish vLLM's TRANSFER capability and runtime metadata."""
    sources = resolve_kv_transfer_hint_sources(engine_args, worker_type, dp_range)
    if sources is None:
        return
    transfer_hint_worker_type = next(iter(sources.values())).worker_type
    domain = _transfer_hint_domain(engine_args)

    # set_engine_specific expects JSON text; the PyO3 binding parses each value
    # with serde_json::from_str, so Python strings must be json.dumps'ed first.
    # Publish capability last so partial metadata is not capability-only if this
    # setup is ever observed before model registration completes.
    if publish_source_endpoints:
        endpoints = {
            str(rank): source.source_control_endpoint
            for rank, source in sources.items()
        }
        runtime_config.set_engine_specific(
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
            json.dumps(endpoints),
        )
    runtime_config.set_engine_specific(
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
        json.dumps(transfer_hint_worker_type),
    )
    if domain is not None:
        runtime_config.set_engine_specific(
            KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY, json.dumps(domain)
        )
    runtime_config.set_engine_specific(
        KV_HINT_TRANSFER_CAPABILITY_KEY, json.dumps(True)
    )


def publish_kv_hint_capabilities(
    runtime_config: ModelRuntimeConfig,
    engine_args: AsyncEngineArgs,
    worker_type: WorkerType,
    dp_range: tuple[int, int] = (0, 1),
    *,
    publish_source_endpoints: bool = True,
) -> None:
    """Publish each versioned KV hint capability that vLLM supports."""
    _publish_kv_transfer_hint_capability(
        runtime_config,
        engine_args,
        worker_type,
        dp_range,
        publish_source_endpoints=publish_source_endpoints,
    )
    _publish_gpu_transfer_hint_capability(
        runtime_config,
        engine_args,
        worker_type,
        dp_range,
        publish_source_endpoints=publish_source_endpoints,
    )


def _publish_gpu_transfer_hint_capability(
    runtime_config: ModelRuntimeConfig,
    engine_args: AsyncEngineArgs,
    worker_type: WorkerType,
    dp_range: tuple[int, int],
    *,
    publish_source_endpoints: bool,
) -> None:
    tiers = [
        tier
        for tier in _secondary_tiers(engine_args)
        if KV_HINT_GPU_DIRECT_CAPABILITY_KEY in (tier.get("router_capabilities") or ())
    ]
    if not tiers:
        return
    if len(tiers) != 1 or dp_range[1] != 1:
        raise ValueError(
            "Direct GPU KV transfer currently requires one TP1 GPU tier/rank"
        )
    tier = tiers[0]
    if tier.get("type") != "GPUKVCRSecondaryTierManager":
        raise ValueError("Direct GPU capability requires the GPU-only KVCR tier")
    domain = _transfer_hint_domain(engine_args)
    if domain is None or tier.get("compatibility_domain") != domain:
        raise ValueError("Direct GPU capability requires one certified layout domain")
    role = _transfer_hint_worker_type(worker_type)
    endpoints = _transfer_hint_source_control_endpoints(tier, dp_range)
    if role is None or endpoints is None or not publish_source_endpoints:
        raise ValueError(
            "Direct GPU transfer requires native worker-owned source endpoints"
        )
    # CacheOwner/state-agent sources use a different ownership contract; they
    # must not advertise this worker-owned retained-GPU inventory as CPU RAM.
    kv_config = getattr(engine_args, "kv_transfer_config", None)
    extra = getattr(kv_config, "kv_connector_extra_config", None) or {}
    if extra.get("dynamo_state_agent") is not None:
        raise ValueError(
            "Direct GPU KV transfer does not support CacheOwner attachments"
        )
    for key, value in (
        (KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, role),
        (KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY, domain),
        (KV_HINT_GPU_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, endpoints),
        (KV_HINT_GPU_DIRECT_CAPABILITY_KEY, True),
    ):
        runtime_config.set_engine_specific(key, json.dumps(value))
