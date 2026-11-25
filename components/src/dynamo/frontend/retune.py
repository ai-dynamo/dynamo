# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the operator's semantic launch profile before normal CLI parsing."""

import argparse
import json
from collections.abc import Mapping
from typing import Any

from dynamo.llm import KvRouterConfig


def apply_retune_defaults(
    parser: argparse.ArgumentParser,
    value: str,
    environ: Mapping[str, str],
    env_dests: Mapping[str, str],
) -> None:
    contract = json.loads(value)
    if not isinstance(contract, dict) or contract.get("version") != 1:
        raise ValueError("unsupported frontend Retune contract version")
    router = contract.get("router")
    if not isinstance(router, dict):
        raise ValueError("frontend Retune router profile is missing")
    try:
        profile = KvRouterConfig.retune_profile(
            router["profile"],
            router["overlapScoreCredit"],
            router["prefillLoadScale"],
            router["temperature"],
        )
    except KeyError as error:
        raise ValueError(
            f"missing frontend Retune router setting: {error.args[0]}"
        ) from error
    defaults: dict[str, Any] = profile.frontend_kwargs()
    defaults.update(
        router_mode="kv",
        session_affinity_ttl_secs=None,
        active_decode_blocks_threshold=None,
        active_prefill_tokens_threshold=None,
        active_prefill_tokens_threshold_frac=None,
    )
    explicit = {dest for name, dest in env_dests.items() if name in environ}
    registered = set(env_dests.values())
    parser.set_defaults(
        **{
            name: setting
            for name, setting in defaults.items()
            if name in registered and name not in explicit
        },
        retune_capability=value,
    )
