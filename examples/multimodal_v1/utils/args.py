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


def parse_vllm_args(service_name, prefix) -> AsyncEngineArgs:
    config = ServiceConfig.get_instance()
    vllm_args = config.as_args(service_name, prefix=prefix)
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--enable-disagg", action="store_true", help="Enable disaggregation"
    )
    parser.add_argument(
        "--image-token-id",
        type=int,
        default=32000,
        help="Image token ID used to represent image patches in the token sequence",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=576,
        help="Number of patches the input image is divided into (must be positive)",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="<prompt>",
        help="Prompt template to use for the model",
    )
    parser.add_argument(
        "--router",
        type=str,
        choices=["random", "round-robin", "kv"],
        default="random",
        help="Router type to use for scheduling requests to workers",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args(vllm_args)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.enable_disagg = args.enable_disagg
    engine_args.image_token_id = args.image_token_id
    engine_args.num_patches = args.num_patches
    engine_args.prompt_template = args.prompt_template
    engine_args.router = args.router
    return engine_args
