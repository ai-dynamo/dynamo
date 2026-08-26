# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery advertisement for vLLM's push-mode NIXL KV connector.

Push mode inverts the transfer -- decode registers its blocks and prefill
WRITEs into them -- so decode must be able to name the prefill engine before
prefill has produced anything. Publishing these coordinates is what lets the
frontend dispatch both legs at once.

Advertising is optional: without it the frontend falls back to the sequential
handoff, which is correct but forfeits the overlap. The paths below that
decline log why.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from vllm import envs as vllm_envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from dynamo.llm import ModelRuntimeConfig, WorkerType
from dynamo.vllm.kv_connector_protocols import resolve_nixl_push_kv_transfer_config

logger = logging.getLogger(__name__)


def _side_channel_endpoint(vllm_config: VllmConfig) -> Optional[tuple[str, int]]:
    """Mirror ``NixlBaseConnectorScheduler.__init__``: host verbatim from the
    env var, port offset by the data-parallel index.

    Recomputed rather than read back off the connector, which is constructed
    inside the engine core process and unreachable from registration.
    """
    host = vllm_envs.VLLM_NIXL_SIDE_CHANNEL_HOST
    if not host:
        return None
    port = (
        vllm_envs.VLLM_NIXL_SIDE_CHANNEL_PORT
        + vllm_config.parallel_config.data_parallel_index
    )
    return host, port


def _nixl_agent_engine_id(engine_id: Any, parallel_config: Any) -> str:
    """Mirror how vLLM names the NIXL agent: ``kv_transfer_config.engine_id``
    is only the base identity, rewritten per rank to ``<base>_dp<rank>`` when
    the engine is data-parallel (``EngineCoreProc.run_engine_core``). A dense
    engine -- TP and TEP with one DP rank included -- keeps it unsuffixed.

    Wrong in either direction is silent: the peer rejects the handshake with
    "Remote NIXL agent engine ID mismatch" and the transfer falls back.
    """
    is_data_parallel = (
        parallel_config.data_parallel_size > 1
        or parallel_config.data_parallel_index > 0
    )
    if not is_data_parallel:
        return str(engine_id)
    return f"{engine_id}_dp{parallel_config.data_parallel_index}"


def publish_nixl_push_endpoint(
    runtime_config: ModelRuntimeConfig,
    vllm_config: VllmConfig,
    worker_type: WorkerType,
    dp_range: tuple[int, int] = (0, 1),
) -> bool:
    """Advertise this engine's NIXL push coordinates. Returns whether it did.

    A no-op unless this is a prefill worker running ``NixlPushConnector``:
    only the prefill side of a push transfer is named by its peer, and only
    push mode needs naming at all.
    """
    if worker_type != WorkerType.Prefill:
        return False

    kv_transfer_config = resolve_nixl_push_kv_transfer_config(vllm_config)
    if kv_transfer_config is None:
        return False

    # One advertised port cannot describe several engines. vLLM gives each
    # data-parallel rank its own side channel, so a worker fronting more than
    # one rank would publish coordinates that are wrong for all but the first.
    # Decline rather than mislead; the sequential handoff still works.
    if dp_range[1] > 1:
        logger.warning(
            "NixlPushConnector prefill worker manages %d data-parallel ranks, "
            "each with its own NIXL side channel. Not advertising push "
            "coordinates; the prefill/decode handoff will run sequentially "
            "instead of overlapped. Use external or hybrid DP load balancing "
            "to get one rank per worker.",
            dp_range[1],
        )
        return False

    engine_id: Optional[Any] = getattr(kv_transfer_config, "engine_id", None)
    if not engine_id:
        logger.warning(
            "NixlPushConnector prefill worker has no kv_transfer_config."
            "engine_id; not advertising push coordinates. The handoff will "
            "run sequentially."
        )
        return False

    endpoint = _side_channel_endpoint(vllm_config)
    if endpoint is None:
        logger.warning(
            "VLLM_NIXL_SIDE_CHANNEL_HOST is unset, so this prefill worker has "
            "no address to publish. Not advertising push coordinates; the "
            "handoff will run sequentially."
        )
        return False

    host, port = endpoint
    parallel_config = vllm_config.parallel_config

    # Dynamo assigns this worker a DP range; vLLM names its NIXL agent from
    # the rank it believes it is. If those disagree, the identity published
    # here would address an engine that does not exist, and a wrong identity
    # fails the handshake quietly. Decline rather than advertise a
    # plausible-looking lie.
    dp_index = parallel_config.data_parallel_index
    if dp_range[0] != dp_index:
        logger.warning(
            "NixlPushConnector prefill worker was assigned a DP range starting "
            "at %d but vLLM reports data_parallel_index=%d. Not advertising "
            "push coordinates whose engine identity may be wrong; the handoff "
            "will run sequentially instead of overlapped.",
            dp_range[0],
            dp_index,
        )
        return False

    nixl_engine_id = _nixl_agent_engine_id(engine_id, parallel_config)
    runtime_config.set_nixl_push_endpoint(
        nixl_engine_id,
        host,
        port,
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )
    logger.info(
        "Publishing NIXL push endpoint to discovery: engine_id=%s %s:%d "
        "(tp=%d, pp=%d)",
        nixl_engine_id,
        host,
        port,
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )
    return True
