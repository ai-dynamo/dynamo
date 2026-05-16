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

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


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


def make_kv_connector_protocol(vllm_config: Any) -> KvConnectorProtocol:
    """Resolve the PD protocol for the engine's configured KV connector.

    Defaults to NIXL when no ``KVTransferConfig`` is set (non-PD code paths).
    Raises ``ValueError`` when a configured connector name has no matching
    protocol — a mismatch between dynamo and the vLLM engine is a
    misconfiguration, not a benign default; silently falling back to NIXL
    would emit the wrong wire shape and surface as opaque decode failures.
    """
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    name = getattr(kv_cfg, "kv_connector", None) if kv_cfg is not None else None
    if name is None:
        return NixlConnectorProtocol(vllm_config)
    cls = KV_CONNECTOR_PROTOCOLS.get(name)
    if cls is None:
        raise ValueError(
            f"Unsupported kv_connector={name!r} for PD. Supported names: "
            f"{sorted(KV_CONNECTOR_PROTOCOLS)}. If this is a typo or a "
            f"renamed vLLM connector, fix the kv_transfer_config; if this "
            f"is a new connector, add it to KV_CONNECTOR_PROTOCOLS."
        )
    return cls(vllm_config)
