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
#

#
# This file is included as a string in lib.rs. Most work should be done in the Rust caller.
#

import json
import logging

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TokensPrompt

# TODO this should match DYN_LOG level
logging.basicConfig(level=logging.INFO)


async def main(request_queue, ready_event, extra_engine_args, **kwargs):
    arg_map = kwargs
    if extra_engine_args != "":
        json_map = {}
        # extra_engine_args is a filename
        try:
            with open(extra_engine_args) as f:
                json_map = json.load(f)
        except FileNotFoundError:
            logging.error(f"File {extra_engine_args} not found.")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {extra_engine_args}: {e}")
        logging.debug(f"Adding extra engine arguments: {json_map}")
        arg_map = {**arg_map, **json_map}  # json_map gets precedence
    engine_args = AsyncEngineArgs(**arg_map)

    # Main loop
    try:
        async with build_async_engine_client_from_engine_args(
            engine_args
        ) as engine_client:
            ready_event.set()
            while True:
                req = await request_queue.get()
                if req is None:  # Stop sentinel
                    break
                (request_id, request, sampling_params, response_queue) = req

                prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
                gen = engine_client.generate(prompt, sampling_params, request_id)
                async for res in gen:
                    await response_queue.put(res)
                await response_queue.put(None)

                request_queue.task_done()
    except Exception as e:
        logging.error(f"vllm init failed: {e}")
    finally:
        logging.debug("vllm worker stopped")


async def run_response(response_queue):
    try:
        while True:
            item = await response_queue.get()
            yield item
            response_queue.task_done()
            if item is None:
                return
    except Exception as e:
        logging.error(f"failed reading response from vllm: {e}")
