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

import pytest

from dynamo._core import Context


@pytest.mark.asyncio
async def test_client_context_cancel(server, client):
    _, handler = server
    context = Context()
    stream = await client.generate("_generate_until_context_cancelled", context=context)

    iteration_count = 0
    async for annotated in stream:
        number = annotated.data()
        print(f"Received iteration: {number}")

        # Verify received valid number
        assert number == iteration_count

        # Break after receiving 2 responses
        if iteration_count >= 2:
            print("Cancelling after 2 responses...")
            context.stop_generating()
            break

        iteration_count += 1

    # Give server a moment to process the cancellation
    await asyncio.sleep(0.2)

    # Verify server detected the cancellation
    assert handler.context_is_stopped
    assert handler.context_is_killed

    # TODO: Test with _generate_until_asyncio_cancelled server handler
