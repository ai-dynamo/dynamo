#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.llama_cpp --model-path /data/models/Qwen3-0.6B-Q8_0.gguf [args]`

import argparse
import logging
import os
import sys
from typing import Optional

import uvloop
from llama_cpp import Llama

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from . import __version__

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"

configure_dynamo_logging()


class Config:
    """Command line parameters or defaults"""

    namespace: str
    component: str
    endpoint: str
    model_path: str
    model_name: Optional[str]
    context_length: int
    migration_limit: int


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    config = cmd_line_args()

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    endpoint = component.endpoint(config.endpoint)
    await register_llm(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        config.model_path,
        config.model_name,
        migration_limit=config.migration_limit,
    )

    # Initialize the engine
    # For more parameters see:
    # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api
    kwargs = {
        "model_path": config.model_path,
        "n_gpu_layers": -1,  # GPU if we can
        "n_threads": 16,  # Otherwise give it some CPU
    }
    if config.context_length:
        kwargs["n_ctx"] = config.context_length
    engine = Llama(**kwargs)

    await endpoint.serve_endpoint(RequestHandler(engine).generate)


class RequestHandler:
    def __init__(self, engine):
        self.engine_client = engine

    async def generate(self, request):
        gen = self.engine_client.create_chat_completion(
            request["messages"], stream=True
        )
        # TODO this is a synchronous generator in an async method.
        # Move it to a thread so it doesn't block the event loop.
        for res in gen:
            logging.debug(f"res: {res}")
            yield res


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="llama.cpp server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Backend llama.cpp {__version__}"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a local GGUF file.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Name to serve the model under. Defaults to deriving it from model path.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Max model context length. Defaults to models max, usually model_max_length from tokenizer_config.json. Reducing this reduces VRAM requirements.",
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=0,
        help="Maximum number of times a request may be migrated to a different engine worker. The number may be overridden by the engine.",
    )
    args = parser.parse_args()

    config = Config()
    config.model_path = args.model_path
    if args.model_name:
        config.model_name = args.model_name
    else:
        # This becomes an `Option` on the Rust side
        config.model_name = None

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.context_length = args.context_length
    config.migration_limit = args.migration_limit
    return config


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
