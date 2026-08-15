# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a configurable vision encoder as a remote workflow stage."""

from __future__ import annotations

import argparse
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.vllm.multimodal_utils.custom_encoder import (
    resolve_vision_encoder_backend_class,
)
from dynamo.experimental.workflow.vllm.stages import EncoderStage
from dynamo.experimental.workflow import NixlTensorCarrier, RemoteStageServer


@dynamo_worker()
async def encoder_worker(
    runtime: DistributedRuntime,
    endpoint_id: str,
    model: str,
    custom_encoder_class: str,
    stage_id: str = "encoder",
) -> None:
    """Load and serve one configured remote encoder stage."""

    backend_class = resolve_vision_encoder_backend_class(custom_encoder_class)
    stage = EncoderStage.from_backend(
        backend_class(),
        model=model,
        name=f"workflow-{stage_id}",
    )
    carrier: NixlTensorCarrier | None = None
    try:
        carrier = NixlTensorCarrier()
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
    args = parser.parse_args()
    asyncio.run(
        encoder_worker(
            args.endpoint_id,
            args.model,
            args.custom_encoder_class,
            args.stage_id,
        )
    )


if __name__ == "__main__":
    main()
