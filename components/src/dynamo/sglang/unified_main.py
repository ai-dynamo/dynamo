# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the SGLang backend using DynamoPythonBackendModel.

Usage:
    python -m dynamo.sglang.unified_main <sglang args>

See dynamo/common/backend/README.md for architecture, response contract,
and feature gap details.
"""

import sys

import uvloop

from dynamo.common.backend import BackendConfig, DynamoPythonBackendModel
from dynamo.llm import ModelInput
from dynamo.sglang.args import parse_args
from dynamo.sglang.dynamo_engine import SglangDynamoEngine


async def worker():
    config = await parse_args(sys.argv[1:])
    server_args = config.server_args
    dynamo_args = config.dynamo_args

    model_input = (
        ModelInput.Text if not server_args.skip_tokenizer_init else ModelInput.Tokens
    )

    engine = SglangDynamoEngine(server_args)
    backend_config = BackendConfig.from_runtime_config(
        dynamo_args,
        model_name=server_args.model_path,
        served_model_name=server_args.served_model_name,
        model_input=model_input,
    )
    model = DynamoPythonBackendModel(backend_config, engine)
    await model.run()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
