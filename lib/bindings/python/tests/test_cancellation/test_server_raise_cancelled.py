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


import pytest


@pytest.mark.asyncio
async def test_server_raise_cancelled(server, client):
    _, handler = server
    stream = await client.generate("_generate_and_raise_cancelled")

    iteration_count = 0
    try:
        async for annotated in stream:
            number = annotated.data()
            print(f"Received iteration: {number}")
            assert number == iteration_count
            iteration_count += 1
        assert False, "Stream completed without cancellation"
    except ValueError as e:
        # Verify the expected cancellation exception is received
        # TODO: Should this be a asyncio.CancelledError?
        assert (
            str(e)
            == "a python exception was caught while processing the async generator: CancelledError: "
        )

    # Verify server context cancellation status
    # TODO: Server to gracefully stop the stream?
    assert not handler.context_is_stopped
    assert not handler.context_is_killed
