# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight vLLM multimodal embedding-cache startup configuration."""

import logging

logger = logging.getLogger(__name__)


def configure_multimodal_embedding_cache(
    engine_args: object,
    *,
    route_to_encoder: bool,
    capacity_gb: float,
    namespace: str,
    component: str,
) -> None:
    """Configure vLLM's CPU embedding cache before engine creation.

    Separate encode-worker deployments use Dynamo's worker-layer cache. All
    other vLLM deployments use the EC connector owned by vLLM. The vLLM config
    import stays lazy so cache-disabled workers do not load EC-transfer code.
    """
    if route_to_encoder or capacity_gb <= 0:
        return

    from vllm.config import ECTransferConfig

    engine_id = f"{namespace}.{component}.backend.0"
    setattr(
        engine_args,
        "ec_transfer_config",
        ECTransferConfig(
            engine_id=engine_id,
            ec_role="ec_both",
            ec_connector="DynamoMultimodalEmbeddingCacheConnector",
            ec_connector_module_path=(
                "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
            ),
            ec_connector_extra_config={
                "multimodal_embedding_cache_capacity_gb": capacity_gb,
            },
        ),
    )
    logger.info(
        "Configured multimodal embedding cache: engine_id=%s, capacity=%.2f GB",
        engine_id,
        capacity_gb,
    )
