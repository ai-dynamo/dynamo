# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for autodeploy backend support in TRTLLM."""

import contextlib
from unittest import mock

import pydantic
import pytest
from tensorrt_llm._torch.auto_deploy import LlmArgs as ADLlmArgs

from dynamo.trtllm.engine import Backend, TensorRTLLMEngine, get_llm_engine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    # NOTE: these tests do not actually require a GPU, but the workflow validation
    # `.github/workflows/container-validation-backends.yml` does not make use of
    # the `gpu_0` marker.
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]
_PYTORCH_LLM_CLS_NAME = "dynamo.trtllm.engine.LLM"
_AUTODEPLOY_LLM_CLS_NAME = "tensorrt_llm._torch.auto_deploy.LLM"


class TestTensorRTLLMEngine:
    @pytest.mark.parametrize("backend", ["foo", "bar", "cpp"])
    def test_raises_on_unsupported_backends(self, backend):
        with pytest.raises(ValueError, match="Unsupported backend"):
            TensorRTLLMEngine(engine_args={"backend": backend})

    @pytest.mark.parametrize(
        "backend, expected_cls_name",
        [
            ("pytorch", _PYTORCH_LLM_CLS_NAME),
            ("_autodeploy", _AUTODEPLOY_LLM_CLS_NAME),
        ],
    )
    @pytest.mark.asyncio
    async def test_picks_expected_llm_cls(self, backend, expected_cls_name):
        with mock.patch(expected_cls_name) as mocked_cls:
            engine = TensorRTLLMEngine(engine_args={"backend": backend})
            await engine.initialize()

        mocked_cls.assert_called_once()

    @pytest.mark.parametrize(
        "engine_args, is_forbidden",
        [
            ({"build_config": {}}, True),
            ({"tensor_parallel_size": 7}, True),
            ({"pipeline_parallel_size": 3}, True),
            ({"context_parallel_size": 3}, True),
            ({"moe_cluster_parallel_size": 3}, True),
            ({"moe_tensor_parallel_size": 3}, True),
            ({"moe_expert_parallel_size": 3}, True),
            ({"enable_attention_dp": True}, True),
            # Default value is an empty dict.
            ({"cp_config": {"foo", "bar"}}, True),
            ({"scheduler_config": {}}, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_unsupported_args_get_pruned_for_autodeploy(
        self, engine_args, is_forbidden
    ):
        engine_args["backend"] = Backend.AUTODEPLOY
        # This allows us to catch cases where a field being pruned away is now supported by
        # AutoDeploy when bumping TRT-LLM.
        with pytest.raises(
            pydantic.ValidationError
        ) if is_forbidden else contextlib.nullcontext():
            ADLlmArgs(model="foo", **engine_args)

        engine = TensorRTLLMEngine(engine_args=engine_args)
        # This should no longer throw an error since the pruning should have kicked in.
        ADLlmArgs(model="foo", **engine.engine_args)


@pytest.mark.parametrize("backend", ["pytorch", "_autodeploy"])
@pytest.mark.asyncio
async def test_get_llm_engine_forwards_backend(backend):
    engine_args = {"foo": mock.Mock(), "backend": backend}
    with mock.patch(
        "dynamo.trtllm.engine.TensorRTLLMEngine", return_value=mock.AsyncMock()
    ) as mocked_engine:
        async with get_llm_engine(engine_args=engine_args):
            pass

    mocked_engine.assert_called_once_with(engine_args, None)
