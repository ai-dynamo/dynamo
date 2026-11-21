# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for ``python -m dynamo.deepinfer``: the Dynamo runtime flags plus the
DeepInfer options of ``options.py``, which ``node.py`` shares."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from typing import Optional

from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.deepinfer.options import (  # noqa: F401 (re-exported)
    DeepInferOptions,
    add_deepinfer_arguments,
    apply_env_defaults,
    deepinfer_env,
    finalize_options,
)

logger = logging.getLogger(__name__)


class DeepInferArgs(DynamoRuntimeConfig, DeepInferOptions):
    component: str


def parse_args(argv: Optional[Sequence[str]] = None) -> DeepInferArgs:
    parser = argparse.ArgumentParser(
        description="Dynamo DeepInfer worker: all DP ranks of one DeepInfer engine "
        "(on a multi-node engine, the head node)",
        allow_abbrev=False,
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    add_deepinfer_arguments(parser)
    ns = parser.parse_args(argv)

    config = DeepInferArgs.from_cli_args(ns)
    finalize_options(config, ns)
    if config.node_rank != 0:
        raise ValueError(
            f"--node-rank {config.node_rank}: run python -m dynamo.deepinfer.node "
            "on the engine's other nodes"
        )
    config.validate()
    _validate(config)

    endpoint = config.endpoint or f"dyn://{config.namespace}.backend.generate"
    config.namespace, config.component, config.endpoint = parse_endpoint(endpoint)
    return config


def _validate(config: DeepInferArgs) -> None:
    if config.custom_jinja_template:
        path = os.path.expanduser(os.path.expandvars(config.custom_jinja_template))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Custom Jinja template file not found: {path}")
        config.custom_jinja_template = path
