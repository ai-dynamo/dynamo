# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-connector PD-coordination strategies for dynamo's vLLM prefill handler.

vLLM's KV connectors disagree on the shape of ``kv_transfer_params``:
NIXL is pull-based (decode reads block locations from the prefill
response), Mooncake is push-based (prefill pushes blocks under a
pre-allocated ``transfer_id``). This module isolates each protocol
behind :class:`KvConnectorProtocol` so the handler stays
connector-agnostic and new connectors are one class + one registry
entry.
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type


class KvConnectorProtocol(ABC):
    """One instance per prefill request; carries any per-request state."""

    def __init__(self, vllm_config: Any) -> None:
        self._vllm_config = vllm_config

    @abstractmethod
    def prefill_request_kv_transfer_params(self) -> Dict[str, Any]:
        """``kv_transfer_params`` for the prefill request to vLLM."""

    @abstractmethod
    def decode_request_kv_transfer_params(
        self, prefill_response: Any
    ) -> Optional[Dict[str, Any]]:
        """``kv_transfer_params`` for the decode worker, derived from the
        prefill response. Return ``None`` if the protocol doesn't produce
        one."""


class NixlConnectorProtocol(KvConnectorProtocol):
    """Pull-based: decode-side params come straight off the engine response."""

    def prefill_request_kv_transfer_params(self) -> Dict[str, Any]:
        return {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }

    def decode_request_kv_transfer_params(
        self, prefill_response: Any
    ) -> Optional[Dict[str, Any]]:
        return prefill_response.kv_transfer_params


class MooncakeConnectorProtocol(KvConnectorProtocol):
    """Push-based: ``transfer_id`` is allocated up front and threaded
    through both sides; bootstrap address is published to the decode
    worker so it can pull from this prefill's bootstrap server."""

    def __init__(self, vllm_config: Any) -> None:
        super().__init__(vllm_config)
        # Resolve vLLM's canonical bootstrap-addr helper at construction so
        # missing-mooncake / renamed-path errors surface at request setup
        # rather than after the prefill has already run. Used over get_ip()
        # because the helper accounts for local_engines_only and
        # data_parallel_master_ip; an arbitrary local NIC only coincidentally
        # matches the bootstrap server.
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (  # noqa: E501
                get_mooncake_bootstrap_addr,
            )
        except ImportError as e:
            raise RuntimeError(
                "MooncakeConnector PD requires vLLM with the Mooncake KV "
                "connector available. Failed to import "
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
                "mooncake_connector.get_mooncake_bootstrap_addr"
            ) from e
        self._get_bootstrap_addr = get_mooncake_bootstrap_addr
        self._transfer_id: str = str(uuid.uuid4())

    def prefill_request_kv_transfer_params(self) -> Dict[str, Any]:
        return {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "transfer_id": self._transfer_id,
        }

    def decode_request_kv_transfer_params(
        self, prefill_response: Any
    ) -> Optional[Dict[str, Any]]:
        host, port = self._get_bootstrap_addr(self._vllm_config)
        return {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "transfer_id": self._transfer_id,
            # http:// is required: decode does `remote_bootstrap_addr + "/query"`.
            "remote_bootstrap_addr": f"http://{host}:{port}",
            "remote_engine_id": self._vllm_config.kv_transfer_config.engine_id,
        }


# Keyed by ``KVTransferConfig.kv_connector``. One entry per connector.
KV_CONNECTOR_PROTOCOLS: Dict[str, Type[KvConnectorProtocol]] = {
    "NixlConnector": NixlConnectorProtocol,
    "MooncakeConnector": MooncakeConnectorProtocol,
}

# Wrapper connectors that compose sub-connectors under
# ``kv_connector_extra_config["connectors"]``. ``PdConnector``
# (kvbm.vllm_integration.connector) subclasses vLLM's ``MultiConnector``
# with the same config shape, so both delegate PD coordination to their
# single PD-capable child.
MULTI_CONNECTOR_WRAPPERS: Tuple[str, ...] = ("MultiConnector", "PdConnector")


def disable_hybrid_kv_cache_manager_for_incompatible_pd_connector(
    vllm_config: Any,
) -> None:
    """Disable HMA before vLLM builds KV cache groups for incompatible PdConnector children."""
    kv_cfg = vllm_config.kv_transfer_config
    if kv_cfg is None or kv_cfg.kv_connector != "PdConnector":
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    if not MultiConnector.all_children_support_hma(kv_cfg):
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True


def make_kv_connector_protocol(vllm_config: Any) -> KvConnectorProtocol:
    """Resolve the PD protocol for the engine's configured KV connector.

    Defaults to NIXL when no ``KVTransferConfig`` is set (non-PD code paths).
    Raises ``ValueError`` when a configured connector name has no matching
    protocol — a mismatch between dynamo and the vLLM engine is a
    misconfiguration, not a benign default; silently falling back to NIXL
    would emit the wrong wire shape and surface as opaque decode failures.

    Wrapper connectors (``MultiConnector`` and dynamo's ``PdConnector``,
    see :data:`MULTI_CONNECTOR_WRAPPERS`) are special-cased: vLLM enforces
    that exactly one sub-connector produces ``kv_transfer_params`` (see
    ``MultiConnector.request_finished`` — "Only one connector can produce KV
    transfer params"), so we pick the single PD-capable sub-connector and
    delegate to its protocol. Non-PD sub-connectors (e.g. MooncakeStore in
    ``load_async`` mode, or KVBM's DynamoConnector) ride along as save-only
    side effects inside vLLM.
    """
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    name = getattr(kv_cfg, "kv_connector", None) if kv_cfg is not None else None
    if name is None:
        return NixlConnectorProtocol(vllm_config)
    if name in MULTI_CONNECTOR_WRAPPERS:
        return _resolve_multi_connector_protocol(vllm_config, kv_cfg)
    cls = KV_CONNECTOR_PROTOCOLS.get(name)
    if cls is None:
        raise ValueError(
            f"Unsupported kv_connector={name!r} for PD. Supported names: "
            f"{sorted(KV_CONNECTOR_PROTOCOLS)}. If this is a typo or a "
            f"renamed vLLM connector, fix the kv_transfer_config; if this "
            f"is a new connector, add it to KV_CONNECTOR_PROTOCOLS."
        )
    return cls(vllm_config)


def _resolve_multi_connector_protocol(
    vllm_config: Any, kv_cfg: Any
) -> KvConnectorProtocol:
    """Delegate a wrapper connector to its single PD-capable sub-connector.

    Validates the wrapper config shape up front so a malformed
    ``kv_connector_extra_config`` fails as a descriptive ``ValueError`` at
    request setup, not as an ``AttributeError`` mid-resolution. The
    resolved protocol is bound to the child's view of the config (see
    :func:`_child_vllm_config`), not the wrapper's.
    """
    wrapper = getattr(kv_cfg, "kv_connector", "MultiConnector")
    extra = getattr(kv_cfg, "kv_connector_extra_config", None) or {}
    if not isinstance(extra, dict):
        raise ValueError(
            f"{wrapper} requires kv_connector_extra_config to be a dict, "
            f"got {type(extra).__name__}."
        )
    sub_configs = extra.get("connectors") or []
    if not isinstance(sub_configs, list):
        raise ValueError(
            f"{wrapper} requires kv_connector_extra_config['connectors'] to "
            f"be a list of connector configs, got {type(sub_configs).__name__}."
        )
    sub_names = []
    for sub in sub_configs:
        if not isinstance(sub, dict):
            raise ValueError(
                f"Each entry of {wrapper}'s "
                f"kv_connector_extra_config['connectors'] must be a dict, "
                f"got {type(sub).__name__}."
            )
        sub_names.append(sub.get("kv_connector"))
    matches = [
        (name, sub)
        for name, sub in zip(sub_names, sub_configs)
        if name in KV_CONNECTOR_PROTOCOLS
    ]
    if not matches:
        raise ValueError(
            f"{wrapper} has no PD-capable sub-connector. Sub-connectors: "
            f"{sub_names}. Supported PD connectors: "
            f"{sorted(KV_CONNECTOR_PROTOCOLS)}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{wrapper} has multiple PD-capable sub-connectors "
            f"({[name for name, _ in matches]}); vLLM forbids more than one "
            f"connector from producing kv_transfer_params, so dynamo cannot "
            f"pick one unambiguously."
        )
    name, sub_config = matches[0]
    return KV_CONNECTOR_PROTOCOLS[name](
        _child_vllm_config(vllm_config, kv_cfg, sub_config)
    )


def _child_vllm_config(
    vllm_config: Any, kv_cfg: Any, sub_config: Dict[str, Any]
) -> Any:
    """Build the config the selected sub-connector actually runs under.

    vLLM's ``MultiConnector`` instantiates each child with a
    ``KVTransferConfig`` made from the child's own entry, with
    ``engine_id`` falling back to the wrapper's when the entry doesn't
    override it. The protocol must be bound the same way: binding to the
    wrapper config would leak the wrapper's ``engine_id`` into
    ``remote_engine_id``, and the decode side could never match the
    transfer to the child connector that owns it.
    """
    child_kv_cfg = copy.copy(kv_cfg)
    # Children don't inherit the wrapper's extra config — that's where the
    # "connectors" list itself lives.
    child_kv_cfg.kv_connector_extra_config = {}
    for key, value in sub_config.items():
        setattr(child_kv_cfg, key, value)
    child_vllm_config = copy.copy(vllm_config)
    child_vllm_config.kv_transfer_config = child_kv_cfg
    return child_vllm_config
