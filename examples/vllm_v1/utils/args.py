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
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args(vllm_args)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.enable_disagg = args.enable_disagg
    return engine_args
