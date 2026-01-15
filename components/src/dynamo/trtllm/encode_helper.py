# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

import torch
from tensorrt_llm.inputs import default_multimodal_input_loader

import dynamo.nixl_connect as nixl_connect
from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec


class EncodeHelper:
    """Utility class for encoding and serialization operations."""

    @staticmethod
    def serialize_tensor_dict(tensor_dict: dict) -> dict:
        """Serialize a dictionary of tensors to JSON-serializable format.

        Args:
            tensor_dict: Dictionary containing tensors and other values

        Returns:
            Dictionary with tensors converted to JSON-serializable format

        Example:
            >>> tensor_dict = {"tokens": torch.tensor([1, 2, 3], dtype=torch.int64)}
            >>> serialized = EncodeHelper.serialize_tensor_dict(tensor_dict)
            >>> # Result: {"tokens": {"data": [1, 2, 3], "shape": [3], "dtype": "torch.int64"}}
        """
        serialized = {}
        for key, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor):
                serialized[key] = {
                    "data": tensor.tolist(),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            else:
                # Non-tensor values pass through unchanged
                serialized[key] = tensor
        return serialized

    @staticmethod
    def deserialize_tensor_dict(serialized_dict: dict) -> dict:
        """Deserialize a dictionary back to tensors.

        Args:
            serialized_dict: Dictionary with serialized tensor data

        Returns:
            Dictionary with tensors reconstructed from serialized format

        Example:
            >>> serialized = {"tokens": {"data": [1, 2, 3], "shape": [3], "dtype": "torch.int64"}}
            >>> tensors = EncodeHelper.deserialize_tensor_dict(serialized)
            >>> # Result: {"tokens": tensor([1, 2, 3], dtype=torch.int64)}
        """
        deserialized = {}

        for key, value in serialized_dict.items():
            if (
                isinstance(value, dict)
                and "data" in value
                and "shape" in value
                and "dtype" in value
            ):
                # Reconstruct tensor from serialized format
                dtype = EncodeHelper.get_torch_dtype_from_string(value["dtype"])
                tensor = torch.tensor(value["data"], dtype=dtype)
                deserialized[key] = tensor
            else:
                # Non-tensor values pass through unchanged
                deserialized[key] = value
        return deserialized

    @staticmethod
    def get_torch_dtype_from_string(dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch.dtype object.

        Args:
            dtype_str: String representation of torch dtype (e.g., "torch.float32")

        Returns:
            Corresponding torch.dtype object

        Example:
            >>> dtype = EncodeHelper.get_torch_dtype_from_string("torch.bfloat16")
            >>> # Result: torch.bfloat16
        """
        dtype_map = {
            # Floating point types
            "torch.float64": torch.float64,
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            # FP8 types
            "torch.float8_e4m3fn": torch.float8_e4m3fn,
            "torch.float8_e4m3fnuz": torch.float8_e4m3fnuz,
            "torch.float8_e5m2": torch.float8_e5m2,
            "torch.float8_e5m2fnuz": torch.float8_e5m2fnuz,
            "torch.float8_e8m0fnu": torch.float8_e8m0fnu,
            # Signed integer types
            "torch.int64": torch.int64,
            "torch.int32": torch.int32,
            "torch.int16": torch.int16,
            "torch.int8": torch.int8,
            # Unsigned integer types
            "torch.uint64": torch.uint64,
            "torch.uint32": torch.uint32,
            "torch.uint16": torch.uint16,
            "torch.uint8": torch.uint8,
            # Complex types
            "torch.complex128": torch.complex128,
            "torch.complex64": torch.complex64,
            # Quantized types
            "torch.qint8": torch.qint8,
            "torch.quint8": torch.quint8,
            "torch.qint32": torch.qint32,
            "torch.quint4x2": torch.quint4x2,
            # Boolean type
            "torch.bool": torch.bool,
        }
        return dtype_map.get(dtype_str, torch.float32)

    @staticmethod
    async def read_embeddings_from_encode_response(
        encode_response: Dict[str, Any], connector: nixl_connect.Connector
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Read embeddings from encode worker response using NIXL and reconstruct original format.

        Args:
            encode_response: Response from encode worker containing metadata and NIXL info
            connector: NIXL connector for reading operations

        Returns:
            Either a single tensor or dictionary containing mm_embeddings and auxiliary data

        Raises:
            RuntimeError: If there's an error in the encode response or NIXL operations
        """
        if nixl_connect is None:
            raise RuntimeError("Dynamo NIXL Connect library is not available.")

        if "error" in encode_response:
            raise RuntimeError(f"EncodeHandler error: {encode_response['error']}")

        # Extract dynamic shape, metadata, and auxiliary data
        embeddings_shape = encode_response["embeddings_shape"]
        embeddings_dtype_str = encode_response["embeddings_dtype"]
        auxiliary_data = encode_response.get("auxiliary_data", {})
        readable_metadata = nixl_connect.RdmaMetadata.model_validate(
            encode_response["nixl_readable_metadata"]
        )

        # Dynamically allocate tensor with correct shape and dtype
        embeddings_dtype = EncodeHelper.get_torch_dtype_from_string(
            embeddings_dtype_str
        )
        encodings_tensor = torch.zeros(*embeddings_shape, dtype=embeddings_dtype)

        # Create descriptor for our allocated tensor
        descriptor = nixl_connect.Descriptor(encodings_tensor)

        # Create read operation to read from EncodeHandler
        read_op = await connector.begin_read(readable_metadata, descriptor)
        with read_op:
            # Wait for the read operation to complete
            await read_op.wait_for_completion()
            logging.debug(
                f"Successfully read embeddings via NIXL: {encodings_tensor.shape}"
            )

        # Reconstruct original format and return
        if auxiliary_data:
            # Deserialize auxiliary tensors and reconstruct dictionary format
            deserialized_auxiliary = EncodeHelper.deserialize_tensor_dict(
                auxiliary_data
            )
            result = {"mm_embeddings": encodings_tensor}
            result.update(deserialized_auxiliary)
            return result
        else:
            # Return just the tensor
            return encodings_tensor

    @staticmethod
    async def process_encode_request(
        request: Dict[str, Any],
        multimodal_processor,
        connector: Optional[nixl_connect.Connector],
        tokenizer=None,
        model_dir=None,
        model_type=None,
        engine=None,
    ):
        """
        Process an ENCODE-mode request.

        Supported flows:
        - Embedding-path flow (NIXL transfer):
          - Extract `embedding_paths` from OpenAI messages (URLs ending in .pt/.pth/.bin)
          - Load embeddings tensor (or dict containing mm_embeddings + auxiliary tensors)
          - Create a NIXL readable op and yield metadata to the prefill worker, then wait
            for read completion.
        - Full EPD flow (MultimodalEncoder):
          - Extract `image_urls` + `text_prompt`
          - Build encoder inputs via TRT-LLM's `default_multimodal_input_loader`
          - Run `MultimodalEncoder.generate(...)` to obtain `disaggregated_params`
          - Encode the params for network transfer and also return the processed prompt
            + token ids for downstream prefill/decode consistency.

        Args:
            request: Request containing OpenAI-format multimodal messages
            multimodal_processor: Multimodal processor used to extract prompt/media and load embeddings
            connector: NIXL connector (required only for embedding_paths flow)

        Yields:
            Response dictionary:
            - For NIXL transfer: nixl_readable_metadata + shape/dtype (+ auxiliary_data)
            - For full EPD: ep_disaggregated_params (+ processed_prompt + prompt_token_ids)
        """
        # Load embeddings first to get the actual shape
        # Extract messages from extra_args (set by Rust preprocessor for multimodal) or fall back to direct field
        if multimodal_processor is None:
            yield {"error": "No multimodal_processor configured on encode worker"}
            return

        messages = request.get("extra_args", {}).get(
            "messages", request.get("messages", [])
        )
        (
            text_prompt,
            image_urls,
            embedding_paths,
        ) = multimodal_processor.extract_prompt_and_media(messages)

        if embedding_paths:
            if connector is None:
                yield {"error": "NIXL connector is required for embedding_paths encode"}
                return
            # Load the embeddings data
            logging.info(f"EncodeHelper: loading embeddings from {embedding_paths[0]}")
            loaded_data = multimodal_processor.load_tensor_from_path_or_url(
                embedding_paths[0]
            )
            # Handle both tensor and dictionary formats
            if isinstance(loaded_data, dict):
                # Dictionary format (e.g., maverick_mm_embed_seashore_v3.pt)
                encodings = loaded_data.get("mm_embeddings")
                if encodings is None:
                    yield {"error": "Dictionary embeddings missing 'mm_embeddings' key"}
                    return

                # Store auxiliary data for later transmission
                auxiliary_data = {
                    k: v for k, v in loaded_data.items() if k != "mm_embeddings"
                }
            else:
                # Tensor format (e.g., llava_next_mm_embed_seashore.pt)
                encodings = loaded_data
                auxiliary_data = {}
            # Create readable operation with main embeddings tensor (works for both formats)
            descriptor = nixl_connect.Descriptor(encodings)
            with await connector.create_readable(descriptor) as readable_op:
                # Get the metadata for the readable operation
                op_metadata = readable_op.metadata()

                # Send back shape info, readable metadata, and serialized auxiliary data
                response = {
                    "nixl_readable_metadata": op_metadata.model_dump(),
                    "embeddings_shape": list(encodings.shape),
                    "embeddings_dtype": str(encodings.dtype),
                    "auxiliary_data": EncodeHelper.serialize_tensor_dict(
                        auxiliary_data
                    ),
                }
                yield response

                # Wait for the prefill worker to complete the read operation
                logging.debug(
                    "EncodeHelper waiting for PrefillHandler to read embeddings..."
                )
                await readable_op.wait_for_completion()
                logging.debug("EncodeHelper completed readable operation.")
        elif image_urls and text_prompt:
            # Use trtllm MultimodalEncoder to generate embeddings
            # NOTE: `default_multimodal_input_loader` needs `model_dir` / `model_type` to
            # construct the correct multimodal input format and apply model-specific
            # preprocessing. These are derived from Dynamo's multimodal processor config.
            if model_dir is None or model_type is None:
                yield {
                    "error": "model_dir and model_type are required for full EPD encode"
                }
                return
            inputs = default_multimodal_input_loader(
                tokenizer=tokenizer,
                model_dir=model_dir,
                model_type=model_type,
                modality="image",
                prompts=[text_prompt],
                media=image_urls[0],
            )
            # Log encoder input
            input_keys = (
                list(inputs[0].keys())
                if inputs and isinstance(inputs[0], dict)
                else "N/A"
            )
            logging.info(f"[EPD ENCODE] MultimodalEncoder input keys: {input_keys}")

            # engine.llm is the MultimodalEncoder instance
            # MultimodalEncoder.generate() returns a list of GenerationResult objects
            if engine is None:
                yield {"error": "No engine configured on encode worker for full EPD"}
                return

            # NOTE: MultimodalEncoder.generate() is synchronous. Run it off-thread to avoid
            # blocking the encode worker's event loop under concurrency.
            encoder_outputs = await asyncio.to_thread(
                lambda: list(engine.llm.generate(inputs))
            )
            if not encoder_outputs:
                logging.error("ENCODE WORKER: encoder_outputs is empty")
                yield {"ep_disaggregated_params": None}
                return
            ep_disaggregated_params = encoder_outputs[0].disaggregated_params
            if ep_disaggregated_params is None:
                logging.error(
                    "ENCODE WORKER: encoder_outputs[0].disaggregated_params is None"
                )
                yield {"ep_disaggregated_params": None}
                return
            # Log encoder output
            has_handles = (
                hasattr(ep_disaggregated_params, "multimodal_embedding_handles")
                and ep_disaggregated_params.multimodal_embedding_handles
            )
            num_handles = (
                len(ep_disaggregated_params.multimodal_embedding_handles)
                if has_handles
                else 0
            )
            logging.info(
                f"[EPD ENCODE] MultimodalEncoder output: multimodal_embedding_handles={num_handles}"
            )

            if not has_handles:
                logging.warning(
                    "ENCODE WORKER: ep_disaggregated_params has no multimodal_embedding_handles"
                )
            # Prepare for Network Transfer
            encoded_params = DisaggregatedParamsCodec.encode(ep_disaggregated_params)
            params_dict = asdict(encoded_params)
            # Also send the processed prompt (which includes <image> tokens)
            # default_multimodal_input_loader returns a list, get the first element
            processed_prompt = None
            prompt_token_ids = None
            if isinstance(inputs, list) and len(inputs) > 0:
                first_input = inputs[0]
                if isinstance(first_input, dict):
                    processed_prompt = first_input.get("prompt")
                else:
                    processed_prompt = getattr(first_input, "prompt", None)
                # Tokenize the processed prompt for prefill worker
                if processed_prompt and tokenizer is not None:
                    # NOTE: processed_prompt already contains template/placeholder tokens
                    # (e.g. <image>, [INST], etc.). Adding special tokens here can change
                    # token alignment across EPD stages (prefill/decode), so we explicitly
                    # avoid adding them.
                    prompt_token_ids = tokenizer.encode(
                        processed_prompt, add_special_tokens=False
                    )
            logging.debug(
                "ENCODE WORKER: Extracted processed_prompt (len=%s)",
                len(processed_prompt) if processed_prompt is not None else None,
            )
            yield {
                "ep_disaggregated_params": params_dict,
                "processed_prompt": processed_prompt,  # Prompt with <image> tokens
                "prompt_token_ids": prompt_token_ids,  # Token IDs for consistency
            }
            return
        else:
            # Tensor format (e.g., llava_next_mm_embed_seashore.pt)
            encodings = loaded_data
            auxiliary_data = {}

        # Create readable operation with main embeddings tensor (works for both formats)
        descriptor = nixl_connect.Descriptor(encodings)
        with await connector.create_readable(descriptor) as readable_op:
            # Get the metadata for the readable operation
            op_metadata = readable_op.metadata()

            # Send back shape info, readable metadata, and serialized auxiliary data
            response = {
                "nixl_readable_metadata": op_metadata.model_dump(),
                "embeddings_shape": list(encodings.shape),
                "embeddings_dtype": str(encodings.dtype),
                "auxiliary_data": EncodeHelper.serialize_tensor_dict(auxiliary_data),
            }
            return
