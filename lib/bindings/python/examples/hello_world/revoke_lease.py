# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import asyncio

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    parser = argparse.ArgumentParser()
    parser.add_argument("lease_id", type=int, help="Lease ID to revoke")
    args = parser.parse_args()
    await init(runtime, args.lease_id)


async def init(runtime: DistributedRuntime, lease_id: int):
    client = runtime.etcd_client()
    await client.revoke_lease(lease_id)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
