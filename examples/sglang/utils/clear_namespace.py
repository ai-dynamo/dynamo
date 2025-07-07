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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(clear_namespace(args.namespace))
