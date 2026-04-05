# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageWorker.

No GPU, no vllm_omni — uses mock StageEngine matching AsyncOmni.generate() signature.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.vllm.omni.stage_worker import OmniStageWorker

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _MockEngine:
    """Satisfies StageEngine Protocol — matches AsyncOmni.generate() signature."""

    def __init__(self, output=None):
        self.received_prompt = None
        self.received_request_id = None
        self._output = output or {"output": "mock", "finished": True}

    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        self.received_prompt = prompt
        self.received_request_id = request_id

        async def _gen():
            yield self._output

        return _gen()


class _ErrorEngine:
    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            raise RuntimeError("engine exploded")
            yield  # make it an async generator

        return _gen()


class _MockContext:
    def id(self):
        return "test-req-id"


def _make_stage_config(**overrides):
    defaults = dict(stage_type="llm", final_output=False, final_output_type="text")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_worker(engine=None, stage_config=None, connectors=None):
    return OmniStageWorker(
        engine=engine or _MockEngine(),
        stage_config=stage_config or _make_stage_config(),
        connectors=connectors or {},
        stage_id=0,
    )


@pytest.mark.asyncio
async def test_direct_input_path():
    """from_connector=False: engine receives request['engine_inputs'] as prompt."""
    engine = _MockEngine()
    worker = _make_worker(engine=engine)
    request = {"engine_inputs": {"prompt": "hello"}, "sampling_params_list": None}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_prompt == {"prompt": "hello"}
    assert any("shm_meta" in c for c in chunks)


@pytest.mark.asyncio
async def test_connector_input_path():
    """from_connector=True: try_recv_via_connector called, engine gets result as prompt."""
    engine = _MockEngine()
    worker = _make_worker(engine=engine)
    expected_prompt = {"prior_token_ids": [1, 2, 3]}
    request = {
        "from_connector": True,
        "from_stage": "0",
        "to_stage": "1",
        "request_id": "req-1",
    }

    with patch(
        "vllm_omni.distributed.omni_connectors.adapter.try_recv_via_connector",
        return_value=(expected_prompt, {}),
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_prompt == expected_prompt
    assert any("shm_meta" in c for c in chunks)


@pytest.mark.asyncio
async def test_engine_error_yields_error_chunk():
    """Engine raises → yields {error: ..., finished: True}, no crash."""
    worker = _make_worker(engine=_ErrorEngine())
    request = {"engine_inputs": {"prompt": "hello"}}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert any("error" in c for c in chunks)
    assert any(c.get("finished") for c in chunks)
