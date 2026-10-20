# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use the same runtime arguments and Worker configuration as TokenSpeed."""

import argparse
from pathlib import Path

from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint


class Config(DynamoRuntimeConfig):
    model: str
    served_model_name: str
    component: str
    use_kv_events: bool = False
    mock_token_delay: float
    max_model_len: int


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Dynamo DeepInfer mock worker")
    DynamoRuntimeArgGroup().add_arguments(parser)
    parser.add_argument(
        "--model", required=True, help="Local tokenizer/config directory; no weights"
    )
    parser.add_argument("--served-model-name", default="deepinfer-mock")
    parser.add_argument("--mock-token-delay", type=float, default=0.001)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args(argv)
    if not Path(args.model, "tokenizer.json").is_file():
        parser.error(
            "--model must contain tokenizer.json; run the example's prepare_model.py"
        )
    if args.mock_token_delay < 0 or args.max_model_len <= 0:
        parser.error(
            "--mock-token-delay must be nonnegative and --max-model-len positive"
        )
    config = Config.from_cli_args(args)
    if config.dyn_structural_tag is None and config.dyn_enable_structural_tag is None:
        config.dyn_structural_tag = False
    config.validate()
    config.namespace, config.component, config.endpoint = parse_endpoint(
        config.endpoint or f"dyn://{config.namespace}.deepinfer.generate"
    )
    # These overrides are published with model registration and run on the frontend.
    config.dyn_tool_call_parser = config.dyn_tool_call_parser or "deepseek_v41"
    config.dyn_reasoning_parser = config.dyn_reasoning_parser or "deepseek_v41"
    if config.structural_tag is not None:
        parser.error("mock inference does not implement constrained decoding")
    return config
