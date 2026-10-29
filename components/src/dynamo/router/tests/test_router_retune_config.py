# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""The classification of every KV router policy parameter for retune."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from dynamo.llm import KvRouterConfig
from dynamo.router.simulation.config import (
    ROUTER_KNOB_DEFAULTS,
    ROUTER_NOT_RETUNABLE,
    ROUTER_PUBLIC_TO_LIVE,
    ROUTER_RETUNE_EXTENSIONS,
    RouterRetuneConfig,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def test_every_policy_parameter_is_public_an_extension_or_not_retunable() -> None:
    # A new PolicyParameters field must be classified before retune can see it.
    policy_parameters = set(KvRouterConfig().policy_parameters("aggregated"))
    public = set(ROUTER_PUBLIC_TO_LIVE.values())
    extensions = set(ROUTER_RETUNE_EXTENSIONS)
    not_retunable = set(ROUTER_NOT_RETUNABLE)
    assert len(policy_parameters) == 8
    assert public | extensions | not_retunable == policy_parameters
    assert len(public) + len(extensions) + len(not_retunable) == 8
    assert list(ROUTER_KNOB_DEFAULTS) == list(ROUTER_PUBLIC_TO_LIVE)
    # The retune schema has no field for them; writing one names the reason.
    assert not not_retunable & set(RouterRetuneConfig.model_fields)
    for name, reason in ROUTER_NOT_RETUNABLE.items():
        with pytest.raises(
            ValidationError, match=re.escape(f"{name} is not retunable: {reason}")
        ):
            RouterRetuneConfig.model_validate({name: 1.0})
