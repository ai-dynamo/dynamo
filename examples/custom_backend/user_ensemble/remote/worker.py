# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the external encoder or classifier as one workflow stage process."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.vllm.multimodal_utils.custom_encoder import AsyncVisionEncoder
from dynamo.workflow import NixlTensorCarrier, RemoteStageServer, StageRunner
from examples.custom_backend.user_ensemble.config import (
    DEFAULT_ENCODER_CLASS,
    DEFAULT_MODEL,
    load_encoder_backend,
)
from examples.custom_backend.user_ensemble.remote.bindings import (
    CLASSIFIER_ENDPOINT,
    ENCODER_ENDPOINT,
)
from examples.custom_backend.user_ensemble.resources import build_encoder_stage
from examples.custom_backend.user_ensemble.stages import DummyClassifier


@dynamo_worker()
async def remote_worker(
    runtime: DistributedRuntime,
    stage_id: str,
    model: str,
    encoder_class: str,
) -> None:
    encoder: AsyncVisionEncoder[Any, Any, Any] | None = None
    carrier: NixlTensorCarrier | None = None
    try:
        runner: StageRunner
        endpoint_id: str
        if stage_id == "encoder":
            runner, encoder = build_encoder_stage(
                model,
                load_encoder_backend(encoder_class),
                name="remote-workflow-vision-encoder",
            )
            endpoint_id = ENCODER_ENDPOINT
        elif stage_id == "classifier":
            runner = DummyClassifier()
            endpoint_id = CLASSIFIER_ENDPOINT
        else:
            raise ValueError(f"unknown user ensemble stage {stage_id!r}")

        carrier = NixlTensorCarrier()
        server = RemoteStageServer(stage_id, runner, carrier)
        await runtime.endpoint(endpoint_id).serve_endpoint(server.generate)
    finally:
        try:
            if carrier is not None:
                await carrier.close()
        finally:
            if encoder is not None:
                encoder.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one remote user-ensemble workflow stage",
        allow_abbrev=False,
    )
    parser.add_argument("stage", choices=("encoder", "classifier"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--encoder-class", default=DEFAULT_ENCODER_CLASS)
    args = parser.parse_args()
    asyncio.run(
        remote_worker(
            args.stage,
            args.model,
            args.encoder_class,
        )
    )


if __name__ == "__main__":
    main()
