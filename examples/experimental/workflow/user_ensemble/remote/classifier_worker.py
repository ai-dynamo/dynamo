# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the application-owned classifier as one workflow stage process."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.experimental.workflow.vllm import NixlWriteTensorReceiverCarrier
from dynamo.experimental.workflow import NixlTensorCarrier, RemoteStageServer
from examples.experimental.workflow.user_ensemble.remote.bindings import CLASSIFIER_ENDPOINT
from examples.experimental.workflow.user_ensemble.stages import (
    DummyClassifier,
    DummyMetadataClassifier,
)

CLASSIFIER_INPUT_ENV = "DYN_BENCH_CLASSIFIER_INPUT"
CLASSIFIER_BUFFER_BYTES_ENV = "DYN_CLASSIFIER_NIXL_BUFFER_BYTES"
DEFAULT_CLASSIFIER_BUFFER_BYTES = 512 * 1024 * 1024


def _build_stage() -> tuple[Any, Any]:
    classifier_input = os.environ.get(CLASSIFIER_INPUT_ENV, "tensor")
    if classifier_input == "metadata":
        return DummyMetadataClassifier(), None
    if classifier_input != "tensor":
        raise ValueError(
            f"{CLASSIFIER_INPUT_ENV} must be 'metadata' or 'tensor', "
            f"got {classifier_input!r}"
        )

    transfer_mode = os.environ.get("DYN_VLLM_EMBEDDING_TRANSFER_MODE", "nixl-write")
    if transfer_mode == "nixl-read":
        return DummyClassifier(), NixlTensorCarrier()
    if transfer_mode != "nixl-write":
        raise ValueError(
            "classifier tensor input requires nixl-read or nixl-write, "
            f"got {transfer_mode!r}"
        )
    buffer_bytes = int(
        os.environ.get(
            CLASSIFIER_BUFFER_BYTES_ENV,
            str(DEFAULT_CLASSIFIER_BUFFER_BYTES),
        )
    )
    if buffer_bytes <= 0:
        raise ValueError(f"{CLASSIFIER_BUFFER_BYTES_ENV} must be positive")
    return DummyClassifier(), NixlWriteTensorReceiverCarrier(buffer_size=buffer_bytes)


@dynamo_worker()
async def classifier_worker(runtime: DistributedRuntime) -> None:
    stage, carrier = _build_stage()
    try:
        server = RemoteStageServer("classifier", stage, carrier)
        await runtime.endpoint(CLASSIFIER_ENDPOINT).serve_endpoint(server.generate)
    finally:
        if carrier is not None:
            await carrier.close()


def main() -> None:
    asyncio.run(classifier_worker())


if __name__ == "__main__":
    main()
