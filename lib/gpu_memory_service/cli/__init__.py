# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for GPU Memory Service."""

from gpu_memory_service.cli.args import Config, parse_args
from gpu_memory_service.cli.runner import main

__all__ = [
    "Config",
    "parse_args",
    "main",
]
