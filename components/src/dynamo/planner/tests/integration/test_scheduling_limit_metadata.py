# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An independent scheduling ceiling survives native registration into Planner."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.mdc import MdcEntry, worker_info_from_mdc
from dynamo.planner.core.base import build_worker_capabilities
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.environment.state import DeploymentState
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.capacity import publish_vllm_scheduling_budget

pytestmark = [
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(15),
]


@pytest.mark.asyncio
async def test_native_scheduling_limit_reaches_planner(tmp_path, monkeypatch):
    monkeypatch.setenv("DYN_FILE_KV", str(tmp_path))
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    try:
        endpoint = runtime.endpoint("test.schedulebudget.generate")
        await asyncio.wait_for(endpoint.register_endpoint_instance(), timeout=5)
        config = ModelRuntimeConfig()
        config.max_num_batched_tokens = 256
        config.max_num_seqs = 8
        publish_vllm_scheduling_budget(
            config,
            SimpleNamespace(max_num_batched_tokens=256, max_num_scheduled_tokens=64),
        )
        # Tensor registration exercises native metadata transport without weights.
        await asyncio.wait_for(
            register_model(
                ModelInput.Tensor,
                ModelType.TensorBased,
                endpoint,
                "schedule-budget",
                runtime_config=config,
                worker_type=WorkerType.Aggregated,
            ),
            timeout=5,
        )

        cards = [
            json.loads(path.read_text())["card_json"]
            for path in (tmp_path / "v1/mdc").rglob("*")
            if path.is_file()
        ]
        assert len(cards) == 1
        assert cards[0]["runtime_config"]["max_num_batched_tokens"] == 256
        assert (
            cards[0]["runtime_config"]["runtime_data"]["max_num_scheduled_tokens"] == 64
        )
        state = DeploymentState()
        state.decode.info = worker_info_from_mdc(
            MdcEntry(card_json=cards[0]),
            SubComponentType.DECODE,
            backend="vllm",
        )
        capabilities = build_worker_capabilities(state).decode
        planner_config = PlannerConfig.model_construct(backend="vllm", mode="agg")
        model = PlannerEnginePerfModel(
            worker_type="aggregated", config=planner_config, capabilities=capabilities
        )
        for counter, tokens in enumerate([8, 16, 32, 48, 64], start=1):
            model.add_observations(
                {
                    ("worker", 0): ForwardPassMetrics(
                        worker_id="worker",
                        counter_id=counter,
                        wall_time=0.001 + tokens * 0.00001,
                        scheduled_requests=ScheduledRequestMetrics(
                            num_prefill_requests=1, sum_prefill_tokens=tokens
                        ),
                    )
                }
            )
        assert model.has_sufficient_data()
        core = model._engine_model
        chunks = []
        estimate_forward = core._estimate_forward_pass_seconds

        def observe(metrics_by_rank):
            chunks.extend(
                item.scheduled_requests.sum_prefill_tokens for item in metrics_by_rank
            )
            return estimate_forward(metrics_by_rank)

        monkeypatch.setattr(core, "_estimate_forward_pass_seconds", observe)
        estimate = model.estimate_queued_prefill_time(
            [
                ForwardPassMetrics(
                    queued_requests=QueuedRequestMetrics(
                        num_prefill_requests=1, sum_prefill_tokens=192
                    ),
                )
            ],
            max_num_batched_tokens=256,
            add_next_request=False,
        )

        assert estimate is not None and estimate > 0
        assert chunks == [64]  # Three equal full chunks are priced as a group.
        assert core._limits.max_num_batched_tokens == 256
        assert core._limits.scheduling_token_budget == 64
        assert model._build_options()["max_num_tokens"] == 256
    finally:
        runtime.shutdown()
