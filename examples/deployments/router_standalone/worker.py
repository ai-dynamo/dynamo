# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import uuid
from typing import AsyncGenerator, Optional

import zmq
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_events import KVEventsConfig
from vllm.inputs.data import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = logging.getLogger(__name__)


class MetricsPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(self, port: int) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        logger.info(f"ZMQ publisher initialized on port {port}")

    def record(
        self,
        scheduler_stats: SchedulerStats,
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        # Send metrics over ZMQ
        metrics_data = {
            "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
            "kv_cache_usage": scheduler_stats.kv_cache_usage,
        }

        self.socket.send_json(metrics_data)

    def log_engine_initialized(self) -> None:
        pass


class LoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(self, port: int) -> None:
        self.port = port

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return MetricsPublisher(port=self.port)


class VllmWorkers:
    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        block_size: int = 64,
        base_kv_events_port: int = 5557,
        base_metrics_port: int = 5657,
        num_workers: int = 1,
    ):
        os.environ["VLLM_NO_USAGE_STATS"] = "1"

        self.num_workers = num_workers
        self.llms: list[AsyncLLM] = []

        for worker_id in range(num_workers):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
            zmq_port = base_kv_events_port + worker_id
            metrics_port = base_metrics_port + worker_id

            model_config = ModelConfig(
                model=model,
                enforce_eager=True,
            )

            cache_config = CacheConfig(
                block_size=block_size,
                enable_prefix_caching=True,
            )

            kv_events_config = KVEventsConfig(
                enable_kv_cache_events=True,
                publisher="zmq",
                endpoint=f"tcp://*:{zmq_port}",
            )

            scheduler_config = SchedulerConfig(
                scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler"
            )

            observability_config = ObservabilityConfig()

            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                kv_events_config=kv_events_config,
                scheduler_config=scheduler_config,
                observability_config=observability_config,
            )

            self.llms.append(
                AsyncLLM.from_vllm_config(
                    vllm_config=vllm_config,
                    stat_loggers=[LoggerFactory(port=metrics_port)],
                )
            )

    async def direct(
        self, prompt: TokensPrompt, worker_id: int, sampling_params: SamplingParams
    ) -> AsyncGenerator[RequestOutput, None]:
        outputs = self.llms[worker_id].generate(
            prompt,
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
        )
        async for output in outputs:
            yield output
