# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import asyncio
import json
import logging
import os
import uuid
from typing import AsyncGenerator, Optional

import zmq

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
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
        self, scheduler_stats: SchedulerStats, iteration_stats: Optional[IterationStats]
    ):
        # Send metrics over ZMQ
        metrics_data = {
            "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
            "gpu_cache_usage": scheduler_stats.gpu_cache_usage,
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

            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                kv_events_config=kv_events_config,
                scheduler_config=scheduler_config,
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


# only for testing
async def main():
    from dynamo._core import RadixTree, ZmqKvEventListener

    """Main routine to run VLLM workers and receive their messages."""
    # Initialize workers to start publishing messages
    workers = VllmWorkers(num_workers=2)

    # Set up ZMQ subscribers
    context = zmq.Context()

    # Subscriber for kv_events (port 5557)
    kv_events_socket = context.socket(zmq.SUB)
    kv_events_socket.connect("tcp://localhost:5557")
    kv_events_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
    kv_events_socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout for non-blocking

    # Subscriber for metrics (port 5657)
    metrics_socket = context.socket(zmq.SUB)
    metrics_socket.connect("tcp://localhost:5657")
    metrics_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
    metrics_socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout for non-blocking

    logger.info(
        "ZMQ subscribers connected to ports 5557 (kv_events) and 5657 (metrics)"
    )

    # Start a dummy generation to trigger message publishing
    dummy_prompt = TokensPrompt(prompt_token_ids=[10, 11, 12, 13, 14, 15, 16])
    dummy_sampling_params = SamplingParams(max_tokens=100, ignore_eos=True)
    generation_task = asyncio.create_task(
        consume_generation(workers, dummy_prompt, dummy_sampling_params)
    )

    zmq_listener = ZmqKvEventListener("tcp://localhost:5557", "", 64)
    radix_tree = RadixTree()

    # Main message receiving loop
    try:
        while True:
            # Try to receive kv_events
            try:
                # Convert to JSON format
                json_events = await zmq_listener.get_events()

                for json_event in json_events:
                    event = json.loads(json_event)
                    print(f"[KV_EVENTS] {event}")
                    radix_tree.apply_event(0, json.dumps(event).encode("utf-8"))

            except zmq.Again:
                # No message available
                pass
            except Exception as e:
                logger.error(f"Error receiving kv_events: {e}")

            # Try to receive metrics
            try:
                metrics_message = metrics_socket.recv_json(zmq.NOBLOCK)
                print(f"[METRICS] {metrics_message}")
            except zmq.Again:
                # No message available
                pass
            except Exception as e:
                logger.error(f"Error receiving metrics: {e}")

            # Wait 0.1 seconds before next poll
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        generation_task.cancel()
        kv_events_socket.close()
        metrics_socket.close()
        context.term()


async def consume_generation(
    workers: VllmWorkers, prompt: TokensPrompt, sampling_params: SamplingParams
):
    """Helper function to consume generation output to trigger publishing."""
    try:
        async for output in workers.direct(
            prompt, worker_id=0, sampling_params=sampling_params
        ):
            # Just consume the output to keep generation running
            pass
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
