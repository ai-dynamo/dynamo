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
import logging
import os
import json
from huggingface_hub import hf_hub_download
from huggingface_hub_utils import HfHubHTTPError

from common.base_engine import BaseTensorrtLLMEngine
from common.parser import parse_tensorrt_llm_args
from common.protocol import TRTLLMWorkerRequest
from common.utils import ServerType
from components.prefill_worker import TensorRTLLMPrefillWorker

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMWorker(BaseTensorrtLLMEngine):
    prefill_worker = depends(TensorRTLLMPrefillWorker)

    def __init__(self):
        logger.info("Initializing TensorRT-LLM Worker")
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        args, engine_config = parse_tensorrt_llm_args(config_args)
        self.served_model_name = args.served_model_name
        worker_id = dynamo_context["endpoints"][0].lease_id()
        namespace, _ = TensorRTLLMWorker.dynamo_address()  # type: ignore
        self._min_prefill_workers = args.min_prefill_workers
        super().__init__(
            namespace_str=namespace,
            component_str=class_name,
            worker_id=worker_id,
            engine_config=engine_config,
            remote_prefill=args.remote_prefill,
            min_workers=args.min_workers,
            disagg_config_file=args.llmapi_disaggregated_config,
            block_size=args.block_size,
            router=args.router,
            server_type=ServerType.GEN,
        )

    @async_on_start
    async def async_init(self):
        self._init_engine()

        runtime = dynamo_context["runtime"]
        logger.info("Registering LLM for discovery")
        comp_ns, comp_name = TensorRTLLMWorker.dynamo_address()  # type: ignore
        endpoint = runtime.namespace(comp_ns).component(comp_name).endpoint("generate")

        try:
            # The model identifier from the config, e.g., a Hugging Face repo ID or a local path.
            model_identifier = self._engine_config.model_path or self._engine_config.model_name

            try:
                # By "downloading" the file, we get the exact path to it in the central cache.
                logger.info(f"Checking for chat_template.jinja in repo: {model_identifier}")
                chat_template_path = hf_hub_download(
                    repo_id=model_identifier,
                    filename="chat_template.jinja",
                    local_files_only=False,  # Ensure it downloads if not present
                )

                # If we found it, get the path to tokenizer_config.json and patch it in-place.
                tokenizer_config_path = hf_hub_download(
                    repo_id=model_identifier,
                    filename="tokenizer_config.json",
                    local_files_only=False,
                )
                logger.info(f"Found chat_template.jinja, patching: {tokenizer_config_path}")

                with open(chat_template_path, "r", encoding="utf-8") as f:
                    chat_template = f.read()

                with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                    tokenizer_config = json.load(f)

                # Only inject if it's not already present or is null/empty.
                if not tokenizer_config.get("chat_template"):
                    tokenizer_config["chat_template"] = chat_template
                    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                        json.dump(tokenizer_config, f, indent=2)
                    logger.info("Successfully patched chat_template into tokenizer_config.json in the HF cache.")
                else:
                    logger.info("chat_template already present in cached tokenizer_config.json, skipping patch.")

            except HfHubHTTPError as e:
                # This is an expected and non-fatal error if the model is correctly configured without a separate .jinja file.
                if e.response.status_code == 404:
                    logger.info("No chat_template.jinja found in repo, proceeding without patching.")
                else:
                    logger.warning(f"A non-404 HTTP error occurred when trying to download chat template, skipping patch: {e}")
            except Exception as e:
                logger.warning(f"An unexpected error occurred during chat_template patching, skipping: {e}")


            # Register the model. The frontend (dynamo-run) will now use the patched file from the cache.
            await register_llm(
                ModelType.Backend,
                endpoint,
                model_identifier,
                self.served_model_name,
                kv_cache_block_size=self._kv_block_size,
            )
            logger.info("Successfully registered LLM for discovery")
        except Exception as e:
            logger.error(f"Failed to register LLM for discovery: {e}")
            raise

        if self._remote_prefill:
            runtime = dynamo_context["runtime"]
            comp_ns, comp_name = TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore
            self._prefill_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("generate")
                .client()
            )
            while len(self._prefill_client.instance_ids()) < self._min_prefill_workers:
                logger.info(
                    f"Waiting for prefill workers to be ready.\n"
                    f" Current: {len(self._prefill_client.instance_ids())},"
                    f" Required: {self._min_prefill_workers}"
                )
                await asyncio.sleep(30)

        if self._kv_metrics_publisher is not None:
            task = asyncio.create_task(self.create_metrics_publisher_endpoint())
            task.add_done_callback(
                lambda _: logger.info("metrics publisher endpoint created")
            )

        logger.info("TensorRT-LLM Worker initialized")

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self._kv_metrics_publisher.create_endpoint(component)

    @endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response
        
