#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across router benchmark scripts."""

import logging

# Default values
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_SEED = 0
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MOONCAKE_BLOCK_SIZE = 512


def setup_logger(name: str) -> logging.Logger:
    """Setup and return a logger with standard formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def add_common_args(parser):
    """Add common CLI arguments shared across benchmark scripts."""
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Server URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--use-expected-osl",
        action="store_true",
        help="Pass expected_output_tokens to nvext for router tracking",
    )


def resolve_tokenizer(args):
    """Set tokenizer to model if not specified."""
    if args.tokenizer is None:
        args.tokenizer = args.model


def get_common_aiperf_flags():
    """Return common aiperf flags used across benchmarks."""
    return [
        "--endpoint-type",
        "chat",
        "--endpoint",
        "v1/chat/completions",
        "--streaming",
        "--extra-inputs",
        "ignore_eos:true",
        "--no-gpu-telemetry",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]
