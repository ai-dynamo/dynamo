# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subprocess helper for test_router_per_worker_config.py.

Usage:
    python counter_worker.py <count_file> <device_type> <endpoint_path>
                              [--router-mode MODE] [--discovery-backend BACKEND]
                              [--request-plane PLANE]

    count_file:    path to file where the request count is written after each request
    device_type:   "cpu" sets CUDA_VISIBLE_DEVICES=""; "gpu" sets CUDA_VISIBLE_DEVICES="0"
    endpoint_path: dotted endpoint path, e.g. "test.counter.generate"
"""

import argparse
import asyncio
import os

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.kv_router_args import (
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.groups.router_args import (
    RouterArgGroup,
    RouterConfigBase,
)
from dynamo.common.configuration.utils import add_argument

request_count = 0

# Register needs a model path, so we use a HF model name here.
HF_MODEL_NAME = "Qwen/Qwen3-0.6B"

_ROUTER_MODE_MAP = {
    "round-robin": "RoundRobin",
    "random": "Random",
    "power-of-two": "PowerOfTwoChoices",
    "kv": "KV",
    "direct": "Direct",
    "least-loaded": "LeastLoaded",
    "device-aware-weighted": "DeviceAwareWeighted",
}


class CounterWorkerConfig(RouterConfigBase, KvRouterConfigBase):
    """Configuration for the counter worker subprocess."""

    count_file: str
    device_type: str
    endpoint_path: str
    discovery_backend: str
    request_plane: str


class CounterWorkerArgGroup(ArgGroup):
    """CLI arguments for the counter worker subprocess."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("count_file")
        parser.add_argument("device_type")
        parser.add_argument("endpoint_path")
        add_argument(
            parser,
            flag_name="--discovery-backend",
            env_var="DYN_DISCOVERY_BACKEND",
            default="etcd",
            help="Discovery backend.",
        )
        add_argument(
            parser,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            help="Request plane.",
        )
        RouterArgGroup().add_arguments(parser)
        KvRouterArgGroup().add_arguments(parser)


async def generate(request, context):
    global request_count
    request_count += 1
    with open(count_file, "w") as f:
        f.write(str(request_count))
    yield {"token_ids": [1, 2, 3]}


async def main():
    global count_file

    parser = argparse.ArgumentParser()
    CounterWorkerArgGroup().add_arguments(parser)
    config = CounterWorkerConfig.from_cli_args(parser.parse_args())

    count_file = config.count_file

    # Set device type BEFORE importing dynamo so the Rust side sees the correct
    # env var when the endpoint instance registers itself with the discovery system.
    if config.device_type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif config.device_type == "gpu":
        # Any non-empty, non-"-1", non-"none" value → Cuda in endpoint_device_type().
        # "0" works even without a physical GPU since detection is purely env-var based.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from dynamo.llm import (
        KvRouterConfig,
        ModelInput,
        ModelType,
        RouterConfig,
        RouterMode,
        register_model,
    )
    from dynamo.runtime import DistributedRuntime

    router_mode_attr = _ROUTER_MODE_MAP.get(config.router_mode, "DeviceAwareWeighted")
    router_mode = getattr(RouterMode, router_mode_attr)
    kv_router_config = (
        KvRouterConfig(**config.kv_router_kwargs())
        if router_mode == RouterMode.KV
        else None
    )
    router_config = RouterConfig(
        router_mode, kv_router_config, **config.router_kwargs()
    )

    loop = asyncio.get_event_loop()
    runtime = DistributedRuntime(loop, config.discovery_backend, config.request_plane)
    endpoint = runtime.endpoint(config.endpoint_path)

    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        HF_MODEL_NAME,
        "counter",
        router_config=router_config,
    )

    await endpoint.serve_endpoint(generate)


asyncio.run(main())
