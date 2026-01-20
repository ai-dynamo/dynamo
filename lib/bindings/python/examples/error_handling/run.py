# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import random
import string

import uvloop
from client import init as client_init
from server import init as server_init

from dynamo.runtime import DistributedRuntime, dynamo_worker


def random_string(length=10):
    chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return "".join(random.choices(chars, k=length))


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    ns = random_string()
    task = asyncio.create_task(server_init(runtime, ns))
    await client_init(runtime, ns)
    runtime.shutdown()
    await task


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
