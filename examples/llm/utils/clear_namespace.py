#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

import argparse
import asyncio
import logging
import os

from utils.prefill_queue import PrefillQueue

from dynamo.runtime import DistributedRuntime, EtcdKvCache, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dynamo_worker()
async def clear_namespace(runtime: DistributedRuntime, namespace: str):
    etcd_kv_cache = await EtcdKvCache.create(
        runtime.etcd_client(),
        f"/{namespace}/",
        {},
    )
    await etcd_kv_cache.clear_all()
    logger.info(f"Cleared /{namespace} in EtcdKvCache")

    prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
    prefill_queue_stream_name = f"{namespace}_prefill_queue"
    async with PrefillQueue.get_instance(
        nats_server=prefill_queue_nats_server,
        stream_name=prefill_queue_stream_name,
        dequeue_timeout=3,
    ) as prefill_queue:
        cleared_count = await prefill_queue.clear_queue()
        logger.info(
            f"Cleared {cleared_count} requests from prefill queue{prefill_queue_stream_name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(clear_namespace(args.namespace))
