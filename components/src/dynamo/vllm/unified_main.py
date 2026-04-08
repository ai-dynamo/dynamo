# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the vLLM backend using DynamoPythonBackendModel.

Usage:
    python -m dynamo.vllm.unified_main <vllm args>
"""

import uvloop

from dynamo.common.backend import BackendConfig, DynamoPythonBackendModel
from dynamo.llm import ModelInput
from dynamo.vllm.args import parse_args
from dynamo.vllm.dynamo_engine import VllmDynamoEngine


async def worker():
    config = parse_args()

    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    engine = VllmDynamoEngine(config.engine_args)
    backend_config = BackendConfig.from_runtime_config(
        config,
        model_name=config.model,
        served_model_name=config.served_model_name,
        model_input=ModelInput.Tokens,
    )
    model = DynamoPythonBackendModel(backend_config, engine)
    await model.run()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
