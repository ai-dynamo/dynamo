# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Awaitable, Callable

from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.init_embedding import _init_pooling


async def init_rerank(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Serve and advertise a dedicated cross-encoder worker."""
    await _init_pooling(
        runtime,
        config,
        shutdown_event,
        shutdown_endpoints,
        run_deferred_handlers,
        rerank=True,
    )
