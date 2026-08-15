# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the application-owned classifier as one workflow stage process."""

from __future__ import annotations

import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.workflow import RemoteStageServer
from examples.custom_backend.user_ensemble.remote.bindings import CLASSIFIER_ENDPOINT
from examples.custom_backend.user_ensemble.stages import DummyClassifier


@dynamo_worker()
async def classifier_worker(runtime: DistributedRuntime) -> None:
    server = RemoteStageServer("classifier", DummyClassifier())
    await runtime.endpoint(CLASSIFIER_ENDPOINT).serve_endpoint(server.generate)


def main() -> None:
    asyncio.run(classifier_worker())


if __name__ == "__main__":
    main()
