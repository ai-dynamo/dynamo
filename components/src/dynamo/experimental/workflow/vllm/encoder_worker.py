# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a configurable vision encoder as a remote workflow stage."""

from __future__ import annotations

import argparse
import asyncio

from dynamo.experimental.workflow import NixlTensorCarrier, RemoteStageServer
from dynamo.experimental.workflow.vllm.stages import EncoderStage
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.vllm.multimodal_utils.custom_encoder import (
    resolve_vision_encoder_backend_class,
)


@dynamo_worker()
async def encoder_worker(
    runtime: DistributedRuntime,
    endpoint_id: str,
    model: str,
    custom_encoder_class: str,
    stage_id: str = "encoder",
    nixl_send_pool_capacity: int = 0,
    nixl_send_pool_bytes: int = 0,
    batch_queue_wait_ms: float = 0.0,
) -> None:
    """Load and serve one configured remote encoder stage."""

    backend_class = resolve_vision_encoder_backend_class(custom_encoder_class)
    stage = EncoderStage.from_backend(
        backend_class(),
        model=model,
        batch_queue_wait_s=batch_queue_wait_ms / 1000.0,
        name=f"workflow-{stage_id}",
    )
    carrier: NixlTensorCarrier | None = None
    try:
        carrier = NixlTensorCarrier(
            send_pool_capacity=nixl_send_pool_capacity,
            send_pool_bytes=nixl_send_pool_bytes,
        )
        server = RemoteStageServer(stage_id, stage, carrier)
        await runtime.endpoint(endpoint_id).serve_endpoint(server.generate)
    finally:
        try:
            if carrier is not None:
                await carrier.close()
        finally:
            stage.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a custom vision encoder as a remote workflow stage",
        allow_abbrev=False,
    )
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--custom-encoder-class", required=True)
    parser.add_argument("--stage-id", default="encoder")
    parser.add_argument("--nixl-send-pool-capacity", type=int, default=0)
    parser.add_argument("--nixl-send-pool-bytes", type=int, default=0)
    parser.add_argument("--batch-queue-wait-ms", type=float, default=0.0)
    args = parser.parse_args()
    asyncio.run(
        encoder_worker(
            args.endpoint_id,
            args.model,
            args.custom_encoder_class,
            args.stage_id,
            args.nixl_send_pool_capacity,
            args.nixl_send_pool_bytes,
            args.batch_queue_wait_ms,
        )
    )


if __name__ == "__main__":
    main()
