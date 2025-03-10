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


def parse_vllm_args() -> AsyncEngineArgs:
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--random-router",
        action="store_true",
        help="Randomly schedule request to workers",
    )
    parser.add_argument(
        "--round-robin-router",
        action="store_true",
        help="Round robin schedule request to workers",
    )
    parser.add_argument(
        "--kv-router",
        action="store_true",
        help="Enable metric publishing for kv router",
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
        "--custom-disagg-router",
        action="store_true",
        help="Use custom python implementation of disaggregated router instead of the default rust one",
    )
    parser.add_argument(
        "--max-local-prefill-length",
        type=int,
        default=1000,
        help="Maximum length of local prefill",
    )
    parser.add_argument(
        "--max-remote-prefill-cache-hit-ratio",
        type=float,
        default=0.5,
        help="Maximum cache hit ratio for remote prefill "
        "(only applicable to custom python implementation of disaggregated router)",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.random_router = args.random_router
    engine_args.round_robin_router = args.round_robin_router
    engine_args.kv_router = args.kv_router
    engine_args.remote_prefill = args.remote_prefill
    engine_args.conditional_disagg = args.conditional_disagg
    engine_args.custom_disagg_router = args.custom_disagg_router
    engine_args.max_local_prefill_length = args.max_local_prefill_length
    engine_args.max_remote_prefill_cache_hit_ratio = (
        args.max_remote_prefill_cache_hit_ratio
    )
    return engine_args
