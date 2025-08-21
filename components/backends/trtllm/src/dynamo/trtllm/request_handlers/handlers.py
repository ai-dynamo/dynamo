# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

import torch

import dynamo.nixl_connect as nixl_connect
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
    HandlerBase,
    RequestHandlerConfig,
)
from dynamo.trtllm.utils.encode_utils import EncodeUtils

configure_dynamo_logging()


class RequestHandlerFactory:
    def __init__(self):
        self.handlers = {
            "prefill": PrefillHandler,
            "decode": DecodeHandler,
            "encode": EncodeHandler,
            "prefill_and_decode": AggregatedHandler,
        }

    def _validate_config(self, config: RequestHandlerConfig):
        if config.disaggregation_mode.value not in self.handlers:
            raise ValueError(
                f"Invalid disaggregation_mode '{config.disaggregation_mode.value}'"
            )

        if not config.next_client:
            if (
                config.disaggregation_mode == DisaggregationMode.PREFILL
                and config.disaggregation_strategy
                == DisaggregationStrategy.PREFILL_FIRST
            ):
                raise ValueError(
                    "Next client is required for the main worker when disaggregation_mode='prefill' and disaggregation_strategy='prefill_first'."
                )
            if (
                config.disaggregation_mode == DisaggregationMode.DECODE
                and config.disaggregation_strategy
                == DisaggregationStrategy.DECODE_FIRST
            ):
                raise ValueError(
                    "Next client is required for the decode worker when disaggregation_mode='decode' and disaggregation_strategy='decode_first'."
                )

    def get_request_handler(self, config: RequestHandlerConfig) -> HandlerBase:
        self._validate_config(config)
        return self.handlers[config.disaggregation_mode.value](config)


def get_request_handler(config: RequestHandlerConfig) -> HandlerBase:
    return RequestHandlerFactory().get_request_handler(config)


class AggregatedHandler(HandlerBase):
    """
    Handler for the aggregated mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def generate(self, request: dict):
        # Implement all steps locally.
        async for res in self.generate_locally(request):
            yield res


class EncodeHandler(HandlerBase):
    """
    Handler for the encode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)
        self.encodings = None
        self.auxiliary_data = {}

    async def generate(self, request: dict):
        if self.connector:
            # Load embeddings first to get the actual shape
            messages = request.get("messages", [])
            _, _, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                messages
            )
            if embedding_paths:
                loaded_data = self.multimodal_processor.load_tensor_from_path_or_url(
                    embedding_paths[0]
                )

                # Handle both tensor and dictionary formats
                if isinstance(loaded_data, dict):
                    # Dictionary format (e.g., maverick_mm_embed_seashore_v3.pt)
                    self.encodings = loaded_data.get("mm_embeddings")
                    if self.encodings is None:
                        yield {
                            "error": "Dictionary embeddings missing 'mm_embeddings' key"
                        }
                        return

                    # Store auxiliary data for later transmission
                    self.auxiliary_data = {
                        k: v for k, v in loaded_data.items() if k != "mm_embeddings"
                    }
                else:
                    # Tensor format (e.g., llava_next_mm_embed_seashore.pt)
                    self.encodings = loaded_data
                    self.auxiliary_data = {}
            else:
                # Placeholder for TRTLLM Encoder to be called
                # TRTLLM Encoder will return a memory handler on the the encoder GPU with the encodings
                logging.warning(
                    "No embedding paths found, NIXL transfer for image urls not supported by TRTLLM Encoder yet"
                )
                yield {"error": "No embedding paths found"}
                return

            # Create readable operation with main embeddings tensor (works for both formats)
            descriptor = nixl_connect.Descriptor(self.encodings)
            with self.connector.create_readable(descriptor) as readable_op:
                # Get the metadata for the readable operation
                op_metadata = readable_op.metadata()

                # Send back shape info, readable metadata, and serialized auxiliary data
                response = {
                    "nixl_readable_metadata": op_metadata.model_dump(),
                    "embeddings_shape": list(self.encodings.shape),
                    "embeddings_dtype": str(self.encodings.dtype),
                    "auxiliary_data": EncodeUtils.serialize_tensor_dict(
                        self.auxiliary_data
                    ),  # Serialize tensors for JSON
                }
                yield response

                # Wait for the prefill worker to complete the read operation
                logging.debug(
                    "EncodeHandler waiting for PrefillHandler to read embeddings..."
                )
                await readable_op.wait_for_completion()
                logging.debug("EncodeHandler completed readable operation.")
            return

        if not request.get("streaming", False):
            yield request
            return

        yield request


class PrefillHandler(HandlerBase):
    """
    Handler for the prefill mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_encode_with_nixl(self, request: dict):
        # 2. Get response with shape info and readable metadata
        encode_response = None
        async for res in await self.encode_client.round_robin(request):
            encode_response = res.data()
            break

        if not encode_response:
            raise RuntimeError("Did not receive a response from the encode worker.")

        if "error" in encode_response:
            raise RuntimeError(f"EncodeHandler error: {encode_response['error']}")

        # 3. Extract dynamic shape, metadata, and auxiliary data
        embeddings_shape = encode_response["embeddings_shape"]
        embeddings_dtype_str = encode_response["embeddings_dtype"]
        auxiliary_data = encode_response.get("auxiliary_data", {})
        readable_metadata = nixl_connect.RdmaMetadata.model_validate(
            encode_response["nixl_readable_metadata"]
        )

        # 4. Dynamically allocate tensor with correct shape and dtype
        # Convert dtype string back to torch dtype
        embeddings_dtype = EncodeUtils.get_torch_dtype_from_string(embeddings_dtype_str)

        encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)

        # 5. Create descriptor for our allocated tensor
        descriptor = nixl_connect.Descriptor(encodings_tensor)

        # 6. Create read operation to read from EncodeHandler
        read_op = await self.connector.begin_read(readable_metadata, descriptor)
        with read_op:
            # 7. Wait for the read operation to complete
            await read_op.wait_for_completion()
            logging.debug(
                f"PrefillHandler successfully read embeddings: {encodings_tensor.shape}"
            )

        # 8. Reconstruct original format and return
        if auxiliary_data:
            # Deserialize auxiliary tensors and reconstruct dictionary format
            deserialized_auxiliary = EncodeUtils.deserialize_tensor_dict(auxiliary_data)
            result = {"mm_embeddings": encodings_tensor}
            result.update(deserialized_auxiliary)
            return result
        else:
            # Return just the tensor
            return encodings_tensor

    async def remote_decode(self, request: dict):
        async for res in await self.next_client.round_robin(request):
            yield res.data()

    async def generate(self, request: dict):
        logging.debug(f"PrefillHandler.generate received request: {request}")
        embeddings_tensor = None

        if self.multimodal_processor:
            _, _, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                request.get("messages", [])
            )
            # This check will be removed once TRTLLM Encoder is integrated.
            if embedding_paths:
                # If an encoder is configured and the request needs encoding, call it.
                if self.encode_client and self.connector:
                    logging.debug(
                        "PrefillHandler calling Encode Worker via remote_encode_with_nixl"
                    )
                    embeddings_tensor = await self.remote_encode_with_nixl(request)

        # Request is ready for prefill.
        # Generate the prefill response locally
        prefill_request = copy.deepcopy(request)
        prefill_response = None
        response_count = 0
        async for res in self.generate_locally(prefill_request, embeddings_tensor):
            prefill_response = res
            response_count += 1
            if response_count > 1:
                raise ValueError("Prefill response should be generated only once.")

        if (
            self.disaggregation_strategy == DisaggregationStrategy.PREFILL_FIRST
            and not self.check_error(prefill_response)
        ):
            # If operating under prefill_first strategy, the prefill handler needs to trigger
            # the decode handler.
            if prefill_response is not None:
                request["disaggregated_params"] = prefill_response[
                    "disaggregated_params"
                ]
            async for res in self.remote_decode(request):
                yield res
        else:
            # Return response to the decode handler.
            yield prefill_response


class DecodeHandler(HandlerBase):
    """
    Handler for the decode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_prefill(self, request: dict):
        async for res in await self.next_client.round_robin(request):
            yield res

    async def generate(self, request: dict):
        if self.disaggregation_strategy == DisaggregationStrategy.DECODE_FIRST:
            prefill_response = None
            # If operating under decode_first strategy, the decode handler needs to trigger
            # the prefill handler.
            response_count = 0
            # Do not yield the prefill response directly.
            # Instead, capture it and extract the state.
            async for res in self.remote_prefill(request):
                prefill_response = res
                response_count += 1
                if response_count > 1:
                    raise ValueError("Prefill response should be generated only once.")

            response_data = (
                prefill_response.data() if prefill_response is not None else None
            )
            if prefill_response is not None and self.check_error(response_data):
                yield response_data
                return

            if prefill_response is not None and response_data is not None:
                request["disaggregated_params"] = response_data["disaggregated_params"]

        async for res in self.generate_locally(request):
            yield res
