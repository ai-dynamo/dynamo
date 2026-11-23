# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vllm import SamplingParams

from dynamo.common.backend.health_check import HEALTH_CHECK_KEY
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm import handlers
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.handlers import (
    BYPASS_REMOTE_PREFILL_ANNOTATION,
    DecodeWorkerHandler,
    PrefillWorkerHandler,
    _validate_disagg_parallel_sampling,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
]

_PROMPT = {"prompt_token_ids": [1, 2, 3]}


class _Sentinel(Exception):
    """Stops handler execution at the validation call site."""


def _request(n: int, **overrides):
    return {
        "token_ids": [1, 2, 3],
        "sampling_options": {"n": n, "temperature": 1.0},
        **overrides,
    }


def _context():
    return SimpleNamespace(
        id=lambda: "req-1",
        trace_headers=lambda: [],
        notify_first_token=lambda: None,
    )


def _handler(cls, mode: DisaggregationMode):
    handler = object.__new__(cls)
    handler.config = SimpleNamespace(
        disaggregation_mode=mode,
        enable_rl=False,
    )
    handler.default_sampling_params = {}
    handler.model_max_len = None
    handler._multimodal_request_processor = SimpleNamespace(
        prepare_input=AsyncMock(
            side_effect=lambda request, request_id, context, mode: SimpleNamespace(
                request=request,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                pre_rendered_prompt=None,
                custom_data=None,
            )
        )
    )
    handler._build_prompt_from_request = lambda *args, **kwargs: dict(_PROMPT)
    handler._custom_encoder = None
    return handler


async def _exhaust(generator):
    return [chunk async for chunk in generator]


def _install_spy(monkeypatch):
    calls = []

    def spy(sampling_params, *, enabled):
        calls.append((sampling_params.n, enabled))
        raise _Sentinel

    monkeypatch.setattr(handlers, "_validate_disagg_parallel_sampling", spy)
    return calls


def test_helper_allows_single_sample_when_enabled():
    _validate_disagg_parallel_sampling(SamplingParams(n=1), enabled=True)


@pytest.mark.parametrize("n", [2, 3])
def test_helper_rejects_parallel_samples_when_enabled(n):
    with pytest.raises(InvalidArgument, match="supports only n=1"):
        _validate_disagg_parallel_sampling(SamplingParams(n=n), enabled=True)


def test_helper_ignores_parallel_samples_when_disabled():
    _validate_disagg_parallel_sampling(SamplingParams(n=2), enabled=False)


async def test_prefill_rejects_n_greater_1_before_engine_call():
    handler = _handler(PrefillWorkerHandler, DisaggregationMode.PREFILL)
    handler.engine_client = SimpleNamespace(generate=AsyncMock())

    with pytest.raises(InvalidArgument, match="supports only n=1"):
        await _exhaust(handler._generate_token_mode(_request(2), _context(), "req-1"))

    handler.engine_client.generate.assert_not_called()


async def test_decode_rejects_n_greater_1_before_engine_call():
    handler = _handler(DecodeWorkerHandler, DisaggregationMode.DECODE)
    handler.engine_client = SimpleNamespace(generate=AsyncMock())

    with pytest.raises(InvalidArgument, match="supports only n=1"):
        await _exhaust(handler._generate_token_mode(_request(2), _context(), "req-1"))

    handler.engine_client.generate.assert_not_called()


async def test_decode_validation_disabled_when_aggregated(monkeypatch):
    handler = _handler(DecodeWorkerHandler, DisaggregationMode.AGGREGATED)
    calls = _install_spy(monkeypatch)

    with pytest.raises(_Sentinel):
        await _exhaust(handler._generate_token_mode(_request(2), _context(), "req-1"))

    assert calls == [(2, False)]


async def test_prefill_validation_skips_local_probe(monkeypatch):
    handler = _handler(PrefillWorkerHandler, DisaggregationMode.PREFILL)
    calls = _install_spy(monkeypatch)

    with pytest.raises(_Sentinel):
        await _exhaust(
            handler._generate_token_mode(
                _request(2, **{HEALTH_CHECK_KEY: True}),
                _context(),
                "req-1",
            )
        )

    assert calls == [(2, False)]


async def test_decode_validation_skipped_after_conditional_bypass(monkeypatch):
    handler = _handler(DecodeWorkerHandler, DisaggregationMode.DECODE)
    calls = _install_spy(monkeypatch)

    with pytest.raises(_Sentinel):
        await _exhaust(
            handler._generate_token_mode(
                _request(2, annotations=[BYPASS_REMOTE_PREFILL_ANNOTATION]),
                _context(),
                "req-1",
            )
        )

    assert calls == [(2, False)]
