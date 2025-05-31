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
import argparse
from argparse import Namespace
import logging
import signal
import os
import json
import sys

from sglang_router import Router
from sglang_router_rs import PolicyType

from components.worker import SGLangWorker
from utils.protocol import PreprocessedRequest

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service, depends
from dynamo.sdk.lib.config import ServiceConfig

cur = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(cur)))
from llm.utils.check_worker import check_required_workers

logger = logging.getLogger(__name__)


def parse_args(service_name, prefix) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers required before proceeding",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model that is being served",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model that is being served",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model that is being served",
    )        
    parser.add_argument(
        "--policy",
        type=str,
        default="cache_aware",
        choices=["random", "round_robin", "cache_aware"],
        help="Load balancing policy to use",
    )
    parser.add_argument(
        "--worker-startup-timeout-secs",
        type=int,
        default=300,
        help="Timeout in seconds for worker startup",
    )
    parser.add_argument(
        "--worker-startup-check-interval",
        type=int,
        default=10,
        help="Interval in seconds between checks for worker startup",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.5,
        help="Cache threshold (0.0-1.0) for cache-aware routing",
    )
    parser.add_argument(
        "--balance-abs-threshold",
        type=int,
        default=32,
        help="Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold. Otherwise, use cache aware",
    )
    parser.add_argument(
        "--balance-rel-threshold",
        type=float,
        default=1.0001,
        help="Load balancing is triggered when (max_load - min_load) > abs_threshold AND max_load > min_load * rel_threshold. Otherwise, use cache aware",
    )
    parser.add_argument(
        "--eviction-interval",
        type=int,
        default=60,
        help="Interval in seconds between cache eviction operations",
    )
    parser.add_argument(
        "--max-tree-size",
        type=int,
        default=2**24,
        help="Maximum size of the approximation tree for cache-aware routing",
    )
    parser.add_argument(
        "--max-payload-size",
        type=int,
        default=4 * 1024 * 1024,
        help="Maximum payload size in bytes",
    )    

    config = ServiceConfig.get_instance()
    config_args = config.as_args(service_name, prefix=prefix)
    args = parser.parse_args(config_args)
    return args


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SGLangRouter:

    worker = depends(SGLangWorker)

    def __init__(self):
        self.args = parse_args(self.__class__.__name__, "")
        logger.info("SGLangRouter initialized")

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        logger.info("Registering LLM for discovery")
        comp_ns, comp_name = SGLangRouter.dynamo_address()
        endpoint = runtime.namespace(comp_ns).component(comp_name).endpoint("chat/completions")
        await register_llm(
            ModelType.Backend,
            endpoint,
            self.args.model_path,
            self.args.served_model_name,
        )

        comp_ns, comp_name = SGLangWorker.dynamo_address()
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        await check_required_workers(self.worker_client, self.args.min_workers)

        worker_ids = self.worker_client.endpoint_ids()
        logger.info(f"worker_ids: {str(worker_ids)}")

        self.router = Router(
            worker_urls=[str(worker_id) for worker_id in worker_ids],
            policy=self.policy_from_str(self.args.policy),
            worker_startup_timeout_secs=self.args.worker_startup_timeout_secs,
            worker_startup_check_interval=self.args.worker_startup_check_interval,
            cache_threshold=self.args.cache_threshold,
            balance_abs_threshold=self.args.balance_abs_threshold,
            balance_rel_threshold=self.args.balance_rel_threshold,
            eviction_interval_secs=self.args.eviction_interval,
            max_tree_size=self.args.max_tree_size,
            max_payload_size=self.args.max_payload_size,
        )

    def policy_from_str(self, policy_str: str) -> PolicyType:
        """Convert policy string to PolicyType enum."""
        policy_map = {
            "random": PolicyType.Random,
            "round_robin": PolicyType.RoundRobin,
            "cache_aware": PolicyType.CacheAware,
        }
        return policy_map[policy_str]


    @dynamo_endpoint(name="chat/completions")
    async def generate(self, request: PreprocessedRequest):
        token_ids_str = " ".join(str(x) for x in request.token_ids)
        body = {"prompt": token_ids_str}
        best_worker_id = self.router.select_generate_worker(json.dumps(body).encode('utf-8'), "/v1/chat/completions")
        logger.info(f"best_worker_id: {str(best_worker_id)}")   
        if best_worker_id:
            engine_generator = await self.worker_client.direct(request.model_dump_json(), int(best_worker_id))
        else:
            engine_generator = await self.worker_client.generate(request.model_dump_json())
        async for resp in engine_generator:
            yield resp.data()
