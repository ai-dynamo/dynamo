# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the application-owned classifier as one workflow stage process."""

from __future__ import annotations

import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.experimental.workflow import NixlTensorCarrier, RemoteStageServer
from examples.experimental.workflow.user_ensemble.remote.bindings import CLASSIFIER_ENDPOINT
from examples.experimental.workflow.user_ensemble.stages import DummyClassifier


@dynamo_worker()
async def classifier_worker(runtime: DistributedRuntime) -> None:
    carrier = NixlTensorCarrier()
    try:
        server = RemoteStageServer("classifier", DummyClassifier(), carrier)
        await runtime.endpoint(CLASSIFIER_ENDPOINT).serve_endpoint(server.generate)
    finally:
        await carrier.close()


def main() -> None:
    asyncio.run(classifier_worker())


if __name__ == "__main__":
    main()
