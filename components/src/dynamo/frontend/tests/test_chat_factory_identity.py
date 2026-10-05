# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Factory identity crosses the real binding without callback-name or env inference."""

import os

import pytest

from dynamo._core import EngineType, EntrypointArgs

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


async def vllm_named_custom_callback(*args):
    raise AssertionError("construction must not invoke the factory")


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, "custom", "vllm", "sglang"])
async def test_factory_identity_roundtrip_is_explicit(identity, monkeypatch):
    monkeypatch.setenv("DYN_CHAT_PROCESSOR", "sglang")
    before = dict(os.environ)
    args = EntrypointArgs(
        EngineType.Dynamic,
        chat_engine_factory=vllm_named_custom_callback,
        chat_engine_factory_identity=identity,
    )
    assert args.chat_engine_factory_identity == (identity or "custom")
    assert dict(os.environ) == before
    with pytest.raises(AttributeError):
        args.chat_engine_factory_identity = "vllm"


def test_no_factory_has_no_identity():
    assert EntrypointArgs(EngineType.Dynamic).chat_engine_factory_identity is None


@pytest.mark.parametrize("identity", ["custom", "vllm", "sglang"])
def test_identity_requires_factory(identity):
    with pytest.raises(ValueError, match="requires chat_engine_factory"):
        EntrypointArgs(EngineType.Dynamic, chat_engine_factory_identity=identity)


@pytest.mark.parametrize("identity", ["", "VLLM", "dynamo", "arbitrary-payload"])
def test_unknown_identity_is_rejected_without_echo(identity):
    with pytest.raises(ValueError, match="must be custom, vllm, or sglang") as caught:
        EntrypointArgs(
            EngineType.Dynamic,
            chat_engine_factory=vllm_named_custom_callback,
            chat_engine_factory_identity=identity,
        )
    if identity:
        assert identity not in str(caught.value)


def test_identity_requires_dynamic_engine_and_callable():
    with pytest.raises(ValueError, match="requires a dynamic frontend"):
        EntrypointArgs(EngineType.Echo, chat_engine_factory=vllm_named_custom_callback)
    with pytest.raises(ValueError, match="must be callable"):
        EntrypointArgs(EngineType.Dynamic, chat_engine_factory=123)
