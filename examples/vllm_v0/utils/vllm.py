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

# TODO: rename to avoid ambiguity with vllm package
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

from dynamo.sdk.lib.config import ServiceConfig


class RouterType:
    RANDOM = "random"
    ROUND_ROBIN = "round-robin"
    KV = "kv"
    KV_LOAD = "kv-load"


def parse_vllm_args(service_name, prefix) -> AsyncEngineArgs:
    config = ServiceConfig.get_instance()
    vllm_args = config.as_args(service_name, prefix=prefix)
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--router",
        type=str,
        choices=[
            RouterType.RANDOM,
            RouterType.ROUND_ROBIN,
            RouterType.KV,
            RouterType.KV_LOAD,
        ],
        default=RouterType.RANDOM,
        help="Router type to use for scheduling requests to workers",
    )
    parser.add_argument(
        "--router-num-threads",
        type=int,
        default=4,
        help="Number of threads to use for the router to process the requests",
    )
    parser.add_argument(
        "--remote-prefill", action="store_true", help="Enable remote prefill"
    )
    parser.add_argument(
        "--conditional-disagg",
        action="store_true",
        help="Use disaggregated router to decide whether to prefill locally or remotely",
    )
    parser.add_argument(
        "--max-local-prefill-length",
        type=int,
        default=1000,
        help="Maximum length for local prefill. If remote prefill is enabled and the prefill length is greater than this value the request will be sent for remote prefill, otherwise prefill phase will run locally.",
    )
    parser.add_argument(
        "--max-prefill-queue-size",
        type=int,
        default=3,
        help="Maximum queue size for remote prefill. If the prefill queue size is greater than this value, prefill phase of the incoming request will be executed locally.",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args(vllm_args)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.router = args.router
    engine_args.router_num_threads = args.router_num_threads
    engine_args.remote_prefill = args.remote_prefill
    engine_args.conditional_disagg = args.conditional_disagg
    engine_args.max_local_prefill_length = args.max_local_prefill_length
    engine_args.max_prefill_queue_size = args.max_prefill_queue_size
    return engine_args
