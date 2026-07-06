# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DYN_SKIP_FRONTEND_TOKENIZE behaviour in the SGLang engine.

Tests cover:
- _get_input_param returns {"prompt": text} when token_ids absent + prompt_text set.
- _get_input_param falls through to input_param_manager on the normal token path.
- from_args forces skip_tokenizer_init=False when DYN_SKIP_FRONTEND_TOKENIZE=1.

No GPU or SGLang engine initialisation is required.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler():
    """Return a BaseWorkerHandler instance with engine and tokenizer stubbed out."""
    from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

    handler = BaseWorkerHandler.__new__(BaseWorkerHandler)
    handler.use_sglang_tokenizer = False
    handler.input_param_manager = MagicMock()
    handler.input_param_manager.get_input_param.return_value = [1, 2, 3]
    return handler


# ---------------------------------------------------------------------------
# _get_input_param
# ---------------------------------------------------------------------------


def test_get_input_param_returns_prompt_text_dict_when_tokenization_skipped():
    """{"prompt": text} is returned when token_ids absent and prompt_text set."""
    handler = _make_handler()
    request = {"prompt_text": "Hello from engine tokenizer", "sampling_options": {}}
    result = handler._get_input_param(request)
    assert result == {"prompt": "Hello from engine tokenizer"}
    handler.input_param_manager.get_input_param.assert_not_called()


def test_get_input_param_ignores_prompt_text_when_token_ids_present():
    """Falls through to input_param_manager when token_ids is non-empty."""
    handler = _make_handler()
    handler.input_param_manager.get_input_param.return_value = [1, 2, 3]
    request = {
        "token_ids": [1, 2, 3],
        "prompt_text": "should be ignored",
    }
    result = handler._get_input_param(request)
    assert result != {"prompt": "should be ignored"}
    handler.input_param_manager.get_input_param.assert_called_once()


def test_get_input_param_fallthrough_when_no_prompt_text():
    """Falls through to input_param_manager when prompt_text is absent."""
    handler = _make_handler()
    handler.input_param_manager.get_input_param.return_value = [10, 20]
    request = {"token_ids": [10, 20]}
    handler._get_input_param(request)
    handler.input_param_manager.get_input_param.assert_called_once()


def test_get_input_param_fallthrough_when_token_ids_and_prompt_text_both_empty():
    """Both absent → falls through to input_param_manager (empty TokensPrompt path)."""
    handler = _make_handler()
    request = {}
    handler._get_input_param(request)
    handler.input_param_manager.get_input_param.assert_called_once()


# ---------------------------------------------------------------------------
# from_args: skip_tokenizer_init enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_args_forces_skip_tokenizer_init_false_when_flag_set(monkeypatch):
    """DYN_SKIP_FRONTEND_TOKENIZE=1 must force server_args.skip_tokenizer_init=False."""
    monkeypatch.setenv("DYN_SKIP_FRONTEND_TOKENIZE", "1")

    from dynamo.common.backend.engine import DisaggregationMode

    server_args = SimpleNamespace(
        skip_tokenizer_init=True,  # default SGLang behaviour
        model_path="test-model",
        served_model_name=None,
        enable_custom_logit_processor=False,
    )
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False,
        namespace="test",
        component="backend",
        endpoint="generate",
        # extra fields WorkerConfig.from_runtime_config reads
        enable_prefix_caching=False,
        session_control=False,
        component_metrics_dp_ranks=[],
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=dynamo_args,
        serving_mode=DisaggregationMode.AGGREGATED,
    )

    with patch(
        "dynamo.sglang.llm_engine.parse_args", new=AsyncMock(return_value=config)
    ):
        with patch("dynamo.sglang.llm_engine.WorkerConfig") as mock_wc:
            mock_wc.from_runtime_config.return_value = MagicMock()
            from dynamo.sglang.llm_engine import SglangLLMEngine

            await SglangLLMEngine.from_args([])

    assert (
        server_args.skip_tokenizer_init is False
    ), "skip_tokenizer_init must be False when DYN_SKIP_FRONTEND_TOKENIZE=1"


@pytest.mark.asyncio
async def test_from_args_leaves_skip_tokenizer_init_unchanged_when_flag_absent(
    monkeypatch,
):
    """Without the flag, skip_tokenizer_init is not modified."""
    monkeypatch.delenv("DYN_SKIP_FRONTEND_TOKENIZE", raising=False)

    from dynamo.common.backend.engine import DisaggregationMode

    server_args = SimpleNamespace(
        skip_tokenizer_init=True,
        model_path="test-model",
        served_model_name=None,
        enable_custom_logit_processor=False,
    )
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False,
        namespace="test",
        component="backend",
        endpoint="generate",
        enable_prefix_caching=False,
        session_control=False,
        component_metrics_dp_ranks=[],
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=dynamo_args,
        serving_mode=DisaggregationMode.AGGREGATED,
    )

    with patch(
        "dynamo.sglang.llm_engine.parse_args", new=AsyncMock(return_value=config)
    ):
        with patch("dynamo.sglang.llm_engine.WorkerConfig") as mock_wc:
            mock_wc.from_runtime_config.return_value = MagicMock()
            from dynamo.sglang.llm_engine import SglangLLMEngine

            await SglangLLMEngine.from_args([])

    assert (
        server_args.skip_tokenizer_init is True
    ), "skip_tokenizer_init must not be changed when DYN_SKIP_FRONTEND_TOKENIZE is absent"


@pytest.mark.asyncio
async def test_from_args_forces_skip_tokenizer_init_false_for_prefill_worker(
    monkeypatch,
):
    """Prefill workers also receive raw prompt text and need the tokenizer."""
    monkeypatch.setenv("DYN_SKIP_FRONTEND_TOKENIZE", "1")

    from dynamo.common.backend.engine import DisaggregationMode

    server_args = SimpleNamespace(
        skip_tokenizer_init=True,
        model_path="test-model",
        served_model_name=None,
        enable_custom_logit_processor=False,
    )
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False,
        namespace="test",
        component="backend",
        endpoint="generate",
        enable_prefix_caching=False,
        session_control=False,
        component_metrics_dp_ranks=[],
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=dynamo_args,
        serving_mode=DisaggregationMode.PREFILL,
    )

    with patch(
        "dynamo.sglang.llm_engine.parse_args", new=AsyncMock(return_value=config)
    ):
        with patch("dynamo.sglang.llm_engine.WorkerConfig") as mock_wc:
            mock_wc.from_runtime_config.return_value = MagicMock()
            from dynamo.sglang.llm_engine import SglangLLMEngine

            await SglangLLMEngine.from_args([])

    assert server_args.skip_tokenizer_init is False, (
        "Prefill workers receive raw prompt text and must tokenize it; "
        "skip_tokenizer_init must be False"
    )
