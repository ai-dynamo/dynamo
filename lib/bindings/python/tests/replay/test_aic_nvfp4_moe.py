# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AIC perf-model resilience on NVFP4 MoE configs.

Verifies that None-returning op.query() results and None _nextn attributes
do not cause TypeError crashes in _predict_context_latency and
_predict_generation_latency.  Exercises the fix for GitHub #9398.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from dynamo.llm import MockEngineArgs

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _nvfp4_moe_engine_args():
    """Minimal MockEngineArgs carrying AIC config for an NVFP4 MoE model."""
    payload = {
        "block_size": 512,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 65536,
        "num_gpu_blocks": 100000,
        "speedup_ratio": 1.0,
        "aic_backend": "vllm",
        "aic_system": "h200_sxm",
        "aic_backend_version": "0.12.0",
        "aic_tp_size": 1,
        "aic_model_path": "Qwen/Qwen3-32B",
    }
    return MockEngineArgs.from_json(json.dumps(payload))


class _NoneQueryOp:
    """Fake op whose .query() returns None — simulating NVFP4 MoE behavior."""

    _name: str = "nvfp4_moe_dummy_op"

    def query(self, database, x, batch_size, beam_width, s, model_name, **kwargs):
        return None


class _ValidQueryOp:
    """Fake op whose .query() returns a numeric result."""

    _name: str = "valid_op"

    def query(self, database, x, batch_size, beam_width, s, model_name, **kwargs):
        return 0.5


def _make_fake_session(context_ops, generation_ops, nextn=None):
    """Patch AicSession internals to carry ops with configurable .query() / _nextn."""
    with patch("dynamo._internal.aic._load_aiconfigurator") as mock_load:
        mock_aic = MagicMock()
        mock_load.return_value = mock_aic

        # Stub out database / backend / model so AicSession.__init__ completes
        mock_db = MagicMock()
        mock_backend = MagicMock()
        mock_model = MagicMock()
        mock_model.model_name = "Qwen/Qwen3-32B"
        mock_model.context_ops = context_ops
        mock_model.generation_ops = generation_ops
        if nextn is not None:
            mock_model._nextn = nextn
        else:
            del mock_model._nextn  # attribute not set at all

        # Replace attribute-style mock setup with dict-subscript-style to match
        # what AicSession.__init__ actually calls (aic["get_model"](), not
        # aic.get_model()).  Attribute-style caused MagicMock["get_model"] to
        # return an unrelated auto-mock instead of mock_model, so self._model in
        # AicSession received a bare MagicMock with an empty context_ops list.
        mock_load.return_value = {
            "get_database": MagicMock(return_value=mock_db),
            "get_model": MagicMock(return_value=mock_model),
            "get_backend": MagicMock(return_value=mock_backend),
            "config": MagicMock(),
            "InferenceSession": MagicMock(return_value=MagicMock()),
        }

        session = _AicSessionUnderTest(
            backend_name="vllm",
            system="h200_sxm",
            model_path="Qwen/Qwen3-32B",
            tp_size=1,
        )
        return session


class _AicSessionUnderTest:
    """Exposes _predict_context_latency / _predict_generation_latency for testing."""

    def __init__(self, backend_name, system, model_path, tp_size):
        # normal init path but we already patched _load_aiconfigurator above
        from dynamo._internal.aic import AicSession

        self._session = AicSession(
            backend_name=backend_name,
            system=system,
            model_path=model_path,
            tp_size=tp_size,
        )

    def predict_context(self, batch_size, effective_isl, prefix):
        return self._session._predict_context_latency(batch_size, effective_isl, prefix)

    def predict_generation(self, batch_size, isl, osl):
        return self._session._predict_generation_latency(batch_size, isl, osl)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_predict_context_latency_tolerates_none_query_result():
    """All ops returning None from query() yields 0.0, not TypeError."""
    context_ops = [_NoneQueryOp(), _NoneQueryOp()]
    generation_ops = []
    session = _make_fake_session(context_ops, generation_ops)

    result = session.predict_context(batch_size=8, effective_isl=512, prefix=0)

    assert isinstance(result, float)
    assert result == 0.0


def test_predict_context_latency_tolerates_mixed_query_result():
    """A mix of None and valid op.query() results returns sum of valids."""
    context_ops = [_NoneQueryOp(), _ValidQueryOp(), _NoneQueryOp()]
    generation_ops = []
    session = _make_fake_session(context_ops, generation_ops)

    result = session.predict_context(batch_size=8, effective_isl=512, prefix=0)

    # _ValidQueryOp().query() returns 0.5
    assert isinstance(result, float)
    assert result == 0.5


def test_predict_generation_latency_tolerates_none_nextn():
    """When _nextn is None (missing attribute), no TypeError is raised."""
    context_ops = []
    generation_ops = [_ValidQueryOp()]
    # _nextn is not set on the model → getattr returns None
    session = _make_fake_session(context_ops, generation_ops, nextn=None)

    result = session.predict_generation(batch_size=8, isl=512, osl=128)

    # effective_batch_size = 8 * ((None if None else 0) + 1) = 8
    # _ValidQueryOp returns 0.5 per call, one call per step
    assert isinstance(result, float)
    assert result >= 0.0


def test_predict_generation_latency_tolerates_all_none_query_results():
    """Generation loop with all-None op.query() returns 0.0, not TypeError."""
    context_ops = []
    generation_ops = [_NoneQueryOp(), _NoneQueryOp()]
    session = _make_fake_session(context_ops, generation_ops, nextn=0)

    result = session.predict_generation(batch_size=8, isl=512, osl=128)

    assert isinstance(result, float)
    assert result == 0.0
