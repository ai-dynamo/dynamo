# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and CLI argument parsing for the myengine sample backend."""

import argparse

from dynamo.common.backend import BackendCommonArgGroup, BackendCommonConfig


class Config(BackendCommonConfig):
    """MyEngine configuration. Inherits all common fields; no extras needed."""

    pass


def parse_args() -> Config:
    """Parse command-line arguments for the myengine backend."""
    parser = argparse.ArgumentParser(description="Dynamo myengine sample worker")
    BackendCommonArgGroup().add_arguments(parser)
    args = parser.parse_args()
    config = Config.from_cli_args(args)
    config.validate()
    return config
