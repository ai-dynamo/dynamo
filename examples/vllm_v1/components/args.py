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

import logging
import socket
import sys
from typing import Optional

from vllm.config import KVTransferConfig
from vllm.distributed.kv_events import KVEventsConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger(__name__)

# Only used if you run it manually from the command line
DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def find_free_port() -> int:
    """Find a free port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


class Config:
    """Command line parameters or defaults"""

    # dynamo specific
    namespace: str
    component: str
    endpoint: str
    kv_events_port: int
    is_prefill_worker: bool

    # mirror vLLM
    model: str
    served_model_name: Optional[str]

    # rest vLLM args
    engine_args: AsyncEngineArgs


def overwrite_args(config):
    defaults = {
        "task": "generate",
        "skip_tokenizer_init": True,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
        # Always set up KV Events for routing
        "kv_events_config": KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=f"tcp://*:{config.kv_events_port}",
        ),
        # Always setting up kv transfer for disagg
        "kv_transfer_config": KVTransferConfig(
            kv_connector="NixlConnector", kv_role="kv_both"
        ),
    }

    # Made decision to always overwrite.
    # Respecting users original cmd line args at all costs requires a bunch of arg parse work

    logger.debug("Setting Dynamo defaults for vLLM")
    for key, value in defaults.items():
        if hasattr(config.engine_args, key):
            setattr(config.engine_args, key, value)
            logger.debug(f" engine_args.{key} = {value}")
        else:
            raise ValueError(f"{key} not found in AsyncEngineArgs from vLLM.")


def parse_args() -> Config:
    parser = FlexibleArgumentParser(
        description="vLLM server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--is-prefill-worker",
        action="store_true",
        help="Enable prefill functionality for this worker. Currently overwrites the --endpoint to be a specially chosen dyn://dynamo.prefill.generate",
    )
    parser.add_argument(
        "--kv-events-port",
        type=int,
        default=find_free_port(),
        help="Endpoint where vLLM publishes metrics for dynamo. For DP, we handle the port iteration.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)

    config = Config()
    config.model = args.model
    if args.served_model_name:
        assert (
            len(args.served_model_name) <= 1
        ), "We do not support multiple model names."
        config.served_model_name = args.served_model_name[0]
    else:
        # This becomes an `Option` on the Rust side
        config.served_model_name = None

    if args.is_prefill_worker:
        args.endpoint = "dyn://dynamo.prefill.generate"

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logger.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.engine_args = engine_args
    config.is_prefill_worker = args.is_prefill_worker
    config.kv_events_port = args.kv_events_port

    if config.engine_args.block_size is None:
        config.engine_args.block_size = 16
        logger.debug(
            f"Setting reasonable default of {config.engine_args.block_size} for block_size"
        )

    overwrite_args(config)

    return config
