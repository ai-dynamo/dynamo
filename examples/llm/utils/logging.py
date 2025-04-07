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

import asyncio
import logging

from dynamo._core import Client

logger = logging.getLogger(__name__)


async def check_required_workers(
    workers_client: Client, required_workers: int, on_change=True, poll_interval=0.5
):
    """Wait until the minimum number of workers are ready."""
    worker_ids = workers_client.endpoint_ids()
    num_workers = len(worker_ids)

    while num_workers < required_workers:
        await asyncio.sleep(poll_interval)
        worker_ids = workers_client.endpoint_ids()
        new_count = len(worker_ids)

        if (not on_change) or new_count != num_workers:
            logger.info(
                f"Waiting for more workers to be ready.\n"
                f" Current: {new_count},"
                f" Required: {required_workers}"
            )
        num_workers = new_count

    print(f"Workers ready: {worker_ids}")
    return worker_ids
