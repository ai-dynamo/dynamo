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
import logging
from dataclasses import dataclass
from typing import Any, Optional

from common.protocol import DisaggregatedTypeConverter, TRTLLMWorkerRequest
from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from tensorrt_llm.serve.openai_protocol import (
    DisaggregatedParams as OAIDisaggregatedParams,
)

from dynamo.llm import get_tensorrtllm_engine, get_tensorrtllm_publisher
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024


def parse_endpoint(endpoint: str) -> tuple[str, str, str]:
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        raise ValueError(
            f"Invalid endpoint format: '{endpoint}'. "
            "Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )

    return (endpoint_parts[0], endpoint_parts[1], endpoint_parts[2])


@dataclass
class BaseEngineConfig:
    """Base engine configuration"""

    namespace: str
    component: str
    endpoint: str
    model_path: str
    served_model_name: Optional[str] = None
    kv_block_size: int = 32
    extra_engine_args: str = ""
    publish_events_and_metrics: bool = False
    disaggregation_mode: str = "prefill_and_decode"
    remote_prefill_endpoint: Optional[str] = None
    lease_id: int = 0

    def __str__(self) -> str:
        return (
            f"Config(namespace={self.namespace}, "
            f"component={self.component}, "
            f"endpoint={self.endpoint}, "
            f"model_path={self.model_path}, "
            f"served_model_name={self.served_model_name}, "
            f"kv_block_size={self.kv_block_size}, "
            f"extra_engine_args={self.extra_engine_args}, "
            f"publish_events_and_metrics={self.publish_events_and_metrics}, "
            f"disaggregation_mode={self.disaggregation_mode}, "
            f"remote_prefill_endpoint={self.remote_prefill_endpoint}, "
            f"lease_id={self.lease_id})"
        )


class BaseTensorrtLLMEngine:
    def __init__(
        self,
        config: BaseEngineConfig,
    ):
        self._config = config
        self._prefill_client = None
        self._llm_engine = None
        self._llm_engine_context = None
        self._llm_publisher = None
        self._llm_publisher_context = None
        self._runtime = None
        self._first_generation = True
        # Initialize default sampling params
        self.default_sampling_params = SamplingParams()

    async def initialize(self, runtime: DistributedRuntime):
        """Initialize the engine and prefill client if needed"""
        self._runtime = runtime

        # Convert model path to Path object if it's a local path, otherwise keep as string
        model_path = str(self._config.model_path)

        # Initialize the LLM engine
        engine_args: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": 1,
            "backend": "pytorch",
            "skip_tokenizer_init": True,
        }

        if self._config.extra_engine_args:
            # TODO: Support extra engine args from json file as well.
            engine_args = update_llm_args_with_extra_options(
                engine_args, self._config.extra_engine_args
            )
        # Update the model path in the config to the model path used by the engine.
        self._config.model_path = str(engine_args["model"])
        if not self._config.model_path:
            raise ValueError(
                "Model specification is required. Present neither in the config nor in the extra engine args."
            )

        # Populate default sampling params from the model
        tokenizer = tokenizer_factory(self._config.model_path)
        self.default_sampling_params = SamplingParams()
        self.default_sampling_params._setup(tokenizer)
        self.default_sampling_params.stop = None

        if self._config.publish_events_and_metrics:
            # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
            kv_cache_config: dict[str, Any] | Any = None
            if "kv_cache_config" not in engine_args:
                kv_cache_config = {}
                kv_cache_config[
                    "event_buffer_max_size"
                ] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
            else:
                kv_cache_config = engine_args["kv_cache_config"]
                if (
                    hasattr(kv_cache_config, "event_buffer_max_size")
                    and not kv_cache_config.event_buffer_max_size
                ):
                    kv_cache_config.event_buffer_max_size = (
                        DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
                    )
                elif (
                    isinstance(kv_cache_config, dict)
                    and "event_buffer_max_size" not in kv_cache_config
                ):
                    kv_cache_config[
                        "event_buffer_max_size"
                    ] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
                engine_args["kv_cache_config"] = kv_cache_config

            # Enable iter perf stats by default if we are publishing events and metrics.
            if not engine_args.get("enable_iter_perf_stats"):
                engine_args["enable_iter_perf_stats"] = True

            # Only pytorch backend is supported for now to publish events and metrics.
            if engine_args.get("backend") != "pytorch":
                logging.error(
                    "Only pytorch backend is supported for now to publish events and metrics."
                )
                raise RuntimeError(
                    "Only pytorch backend is supported for now to publish events and metrics. Hence, KV router is not supported."
                )

        logging.info(f"TRTLLM engine args: {engine_args}")

        # Get the engine using the asynccontextmanager
        self._llm_engine_context = get_tensorrtllm_engine(engine_args)
        if self._llm_engine_context is not None:
            self._llm_engine = await self._llm_engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to create LLM engine context")

        if (
            self._config.publish_events_and_metrics
            and self._config.disaggregation_mode != "prefill"
        ):
            kv_listener = runtime.namespace(self._config.namespace).component(
                self._config.component
            )
            self._llm_publisher_context = get_tensorrtllm_publisher(
                kv_listener,
                self._llm_engine,
                kv_listener,
                self._config.lease_id,
                self._config.kv_block_size,
            )
            if self._llm_publisher_context is not None:
                self._llm_publisher = await self._llm_publisher_context.__aenter__()
            else:
                raise RuntimeError("Failed to create LLM publisher context")

        # Initialize prefill client if in decode mode
        if self._config.disaggregation_mode == "decode":
            if self._config.remote_prefill_endpoint is None:
                raise ValueError("remote_prefill_endpoint is required for decode mode")
            logging.info(
                f"Initializing remote prefill client for endpoint: {self._config.remote_prefill_endpoint}"
            )
            (
                parsed_namespace,
                parsed_component_name,
                parsed_endpoint_name,
            ) = parse_endpoint(self._config.remote_prefill_endpoint)
            if self._runtime is not None:
                self._prefill_client = (
                    await self._runtime.namespace(parsed_namespace)
                    .component(parsed_component_name)
                    .endpoint(parsed_endpoint_name)
                    .client()
                )
            else:
                raise RuntimeError("Runtime not initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self._llm_publisher_context:
            try:
                await self._llm_publisher_context.__aexit__(None, None, None)
            except Exception as e:
                logging.error(f"Error during publisher cleanup: {e}")
            finally:
                self._llm_publisher = None
                self._llm_publisher_context = None

        if self._llm_engine_context:
            try:
                await self._llm_engine_context.__aexit__(None, None, None)
            except Exception as e:
                logging.error(f"Error during engine cleanup: {e}")
            finally:
                self._llm_engine = None
                self._llm_engine_context = None

        self._prefill_client = None

    async def remote_prefill(self, request: TRTLLMWorkerRequest):
        """
        Send a prefill request to the remote prefill worker.

        Args:
            request: The original request to be sent for prefill

        Returns:
            The response from the remote prefill worker

        Raises:
            ValueError: If prefill client is not initialized or multiple responses received
        """
        prefill_request = request.model_copy(deep=True)
        # TRTLLM requires max_tokens to be set for prefill requests.
        prefill_request.stop_conditions.max_tokens = 1
        prefill_request.disaggregated_params = OAIDisaggregatedParams(
            request_type="context_only"
        )

        if self._prefill_client is None:
            raise ValueError("Prefill client not initialized")
        try:
            # TODO: Use smart KV router to determine which prefill worker to use. This would also require supporting publishing events for prefill workers.
            remote_prefill_responses = [
                remote_prefill_response
                async for remote_prefill_response in await self._prefill_client.round_robin(
                    prefill_request.model_dump_json()
                )
            ]
        except Exception as e:
            raise ValueError(f"Error in remote prefill: {e}")

        if len(remote_prefill_responses) > 1:
            raise ValueError(
                "Prefill worker returned more than one response. This is currently not supported in remote prefill mode."
            )

        if len(remote_prefill_responses) == 0:
            raise ValueError("No response received from remote prefill worker")

        remote_prefill_response = remote_prefill_responses[0]
        return remote_prefill_response

    async def generate(self, request: TRTLLMWorkerRequest):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        if self._llm_publisher:
            publishers_error = self._llm_publisher.check_error_queue()
            if publishers_error:
                raise publishers_error

        inputs = request.token_ids

        # Decode the disaggregated params from the request
        disaggregated_params = DisaggregatedTypeConverter.to_llm_disaggregated_params(
            request.disaggregated_params
        )
        num_output_tokens_so_far = 0

        if self._config.disaggregation_mode == "decode":
            # Run prefill/context phase remotely if disaggregation mode is decode.
            try:
                prefill_result = await self.remote_prefill(request)
            except Exception as e:
                raise ValueError(f"Error in remote prefill: {e}")

            remote_prefill_response = prefill_result.data()
            if (
                remote_prefill_response["finish_reason"] == "stop"
                or remote_prefill_response["finish_reason"] == "error"
            ):
                yield remote_prefill_response
                return
            num_output_tokens_so_far = len(remote_prefill_response["token_ids"])

            # Decode the disaggregated params from the remote prefill response
            # Decode the disaggregated params from the remote prefill response
            disaggregated_params = (
                DisaggregatedTypeConverter.to_llm_disaggregated_params(
                    OAIDisaggregatedParams(
                        **remote_prefill_response["disaggregated_params"]
                    )
                )
            )

            # Send the first token response to the client
            first_token_response = remote_prefill_response
            first_token_response.pop("disaggregated_params")
            yield first_token_response

            # Set the disaggregated params to generation_only for the rest of the generation
            disaggregated_params.request_type = "generation_only"

        sampling_params = self.default_sampling_params
        for key, value in request.sampling_options.model_dump().items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request.stop_conditions.max_tokens
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        ignore_eos = request.stop_conditions.ignore_eos
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        # TODO: Disable streaming for context only requests when adding disagg support
        async for res in self._llm_engine.llm.generate_async(
            inputs=inputs,
            sampling_params=sampling_params,
            disaggregated_params=disaggregated_params,
            streaming=(self._config.disaggregation_mode != "prefill"),
        ):
            # TRTLLM engine needs to start generating tokens first before stats
            # can be retrieved.
            if self._first_generation and self._llm_publisher:
                self._llm_publisher.start()
                self._first_generation = False

            if res.finished and self._config.disaggregation_mode != "prefill":
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            if self._config.disaggregation_mode == "prefill":
                # Return the disaggregated params only when operating in prefill mode.
                out[
                    "disaggregated_params"
                ] = DisaggregatedTypeConverter.to_oai_disaggregated_params(
                    output.disaggregated_params
                ).model_dump()

            yield out
            num_output_tokens_so_far = next_total_toks
