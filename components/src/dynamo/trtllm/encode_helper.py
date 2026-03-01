# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

import torch
from tensorrt_llm.inputs import (
    MultimodalDataTracker,
    add_multimodal_placeholders,
    apply_chat_template as trtllm_apply_chat_template,
)

import dynamo.nixl_connect as nixl_connect
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec


class EncodeHelper:
    """Utility class for encoding and serialization operations."""

    # Shared ImageLoader for full EPD flow (async image loading)
    _image_loader: Optional[ImageLoader] = None
    _image_loader_lock = threading.Lock()

    # Cached AutoProcessor for apply_chat_template (loaded once, shared)
    _processor = None
    _processor_lock = threading.Lock()

    @classmethod
    def _get_image_loader(cls) -> ImageLoader:
        if cls._image_loader is None:
            with cls._image_loader_lock:
                if cls._image_loader is None:
                    cls._image_loader = ImageLoader()
        return cls._image_loader

    @classmethod
    def _get_processor(cls, model_dir: str):
        """Lazily load and cache an AutoProcessor for chat-template application."""
        if cls._processor is None:
            with cls._processor_lock:
                if cls._processor is None:
                    from transformers import AutoProcessor

                    cls._processor = AutoProcessor.from_pretrained(
                        model_dir, use_fast=True, trust_remote_code=True
                    )
        return cls._processor

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

    # =========================================================================
    # ENCODE REQUEST PROCESSING
    # =========================================================================
    #
    # Two supported flows:
    #
    # 1. EMBEDDING-PATH FLOW (Pre-computed embeddings via NIXL)
    #    - User sends URL ending in .pt/.pth/.bin
    #    - Encode worker loads tensor, creates NIXL readable op
    #    - Prefill worker reads embeddings via RDMA
    #    - Use case: Customer has pre-computed embeddings from custom encoder
    #
    # 2. FULL EPD FLOW (Image URLs via MultimodalEncoder)
    #    - User sends image URL (http/https/base64)
    #    - Encode worker runs TRT-LLM's MultimodalEncoder.generate()
    #    - Returns disaggregated_params to prefill worker
    #    - Use case: Standard VLM inference with TRT-LLM's encoder
    #
    # =========================================================================

    @staticmethod
    async def _process_embedding_path_flow(
        embedding_paths: list,
        multimodal_processor,
        connector: nixl_connect.Connector,
    ):
        """
        Process pre-computed embeddings via NIXL transfer.

        Loads embeddings from a file path/URL and creates a NIXL readable operation
        for the prefill worker to read via RDMA.

        Args:
            embedding_paths: List of paths to embedding files (.pt/.pth/.bin)
            multimodal_processor: Processor to load embeddings
            connector: NIXL connector for RDMA transfer

        Yields:
            Response with NIXL metadata, shape, dtype, and auxiliary data
        """
        logging.info(f"EncodeHelper: loading embeddings from {embedding_paths[0]}")
        loaded_data = multimodal_processor.load_tensor_from_path_or_url(
            embedding_paths[0]
        )

        # Handle both tensor and dictionary formats
        if isinstance(loaded_data, dict):
            # Dictionary format: contains 'mm_embeddings' key plus auxiliary data
            encodings = loaded_data.get("mm_embeddings")
            if encodings is None:
                yield {"error": "Dictionary embeddings missing 'mm_embeddings' key"}
                return
            auxiliary_data = {
                k: v for k, v in loaded_data.items() if k != "mm_embeddings"
            }
        else:
            # Tensor format: raw embeddings tensor
            encodings = loaded_data
            auxiliary_data = {}

        # Create NIXL readable operation for prefill worker to read
        descriptor = nixl_connect.Descriptor(encodings)
        with await connector.create_readable(descriptor) as readable_op:
            op_metadata = readable_op.metadata()
            response = {
                "nixl_readable_metadata": op_metadata.model_dump(),
                "embeddings_shape": list(encodings.shape),
                "embeddings_dtype": str(encodings.dtype),
                "auxiliary_data": EncodeHelper.serialize_tensor_dict(auxiliary_data),
            }
            yield response

            # Wait for prefill worker to complete the read
            logging.debug(
                "EncodeHelper waiting for PrefillHandler to read embeddings..."
            )
            await readable_op.wait_for_completion()
            logging.debug("EncodeHelper completed readable operation.")

    @classmethod
    def _build_vlm_prompt(cls, text_prompt, num_images, model_type, tokenizer, model_dir):
        """Build a prompt with model-specific vision placeholders and chat template.

        Reuses TRT-LLM's ``MultimodalDataTracker`` and ``add_multimodal_placeholders``
        for inserting the correct model-specific image placeholders (e.g.
        ``<|vision_start|><|image_pad|><|vision_end|>`` for Qwen), then
        ``apply_chat_template`` for wrapping with the model's chat format.

        This mirrors the prompt-building logic inside TRT-LLM's
        ``default_multimodal_input_loader`` without the expensive media-loading
        part, keeping the TTFT improvement from PR #5910 intact.

        Returns:
            The prompt string with vision placeholders and chat wrapping,
            or *None* on failure (caller should fall back to raw text).
        """
        try:
            # 1. Count model-specific image placeholders via TRT-LLM's tracker
            mm_data_tracker = MultimodalDataTracker(model_type)
            for _ in range(num_images):
                mm_data_tracker.add_data("image", None)
            mm_placeholder_counts = mm_data_tracker.placeholder_counts()

            # 2. Insert placeholders into text (position is model-specific)
            content = text_prompt
            if mm_placeholder_counts:
                content = add_multimodal_placeholders(
                    model_type, text_prompt, mm_placeholder_counts
                )

            # 3. Apply chat template via TRT-LLM (handles edge-case models)
            processor = cls._get_processor(model_dir)
            prompt = trtllm_apply_chat_template(
                model_type=model_type,
                tokenizer=tokenizer,
                processor=processor,
                conversation=[{"role": "user", "content": content}],
                add_generation_prompt=True,
                mm_placeholder_counts=[mm_placeholder_counts],
            )
            return prompt
        except Exception as e:
            logging.warning(
                "_build_vlm_prompt failed (%s), falling back to raw text_prompt", e
            )
            return None

    @staticmethod
    async def _process_full_epd_flow(
        text_prompt: str,
        image_urls: list,
        tokenizer,
        model_dir: str,
        model_type: str,
        engine,
    ):
        """
        Process image URLs via TRT-LLM's MultimodalEncoder (full EPD flow).

        Runs MultimodalEncoder.generate() to produce disaggregated_params
        containing multimodal embedding handles for the prefill worker.

        Args:
            text_prompt: Text portion of the prompt
            image_urls: List of image URLs to process
            tokenizer: Tokenizer for decoding prompt_token_ids from encoder output
            model_dir: Path to model directory
            model_type: Model type string (TRT-LLM model identifier)
            engine: TensorRTLLMEngine with MultimodalEncoder

        Yields:
            Response with ep_disaggregated_params, processed_prompt, and prompt_token_ids
        """
        # Load images with shared ImageLoader (async, same as multimodal_processor PD flow).
        image_items = [{"Url": u} for u in image_urls]
        image_loader = EncodeHelper._get_image_loader()
        pil_images = await image_loader.load_image_batch(image_items)
        if not pil_images:
            logging.error("ENCODE WORKER: no images loaded from image_urls")
            yield {"ep_disaggregated_params": None}
            return

        # Build prompt with model-specific vision placeholders and chat template.
        # Uses TRT-LLM's MultimodalDataTracker + add_multimodal_placeholders +
        # apply_chat_template — the same logic as default_multimodal_input_loader
        # but without the expensive media loading (images are loaded above).
        prompt = EncodeHelper._build_vlm_prompt(
            text_prompt, len(pil_images), model_type, tokenizer, model_dir
        )
        if prompt is None:
            prompt = text_prompt

        # IMPORTANT: We pass "prompt" (not "prompt_token_ids") so that TRT-LLM's
        # _preprocess takes the correct multimodal branch which runs the model's
        # input processor.
        processed_mm_data = {"image": pil_images}
        inputs = [{"prompt": prompt, "multi_modal_data": processed_mm_data}]

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

        if ep_disaggregated_params.multimodal_embedding_handles is None:
            logging.warning(
                "ENCODE WORKER: ep_disaggregated_params.multimodal_embedding_handles is None"
            )

        # Prepare for network transfer
        encoded_params = DisaggregatedParamsCodec.encode(ep_disaggregated_params)
        params_dict = asdict(encoded_params)

        # IMPORTANT: Use the *unexpanded* prompt (from _build_vlm_prompt) as
        # processed_prompt — NOT the decoded encoder output tokens.
        #
        # The encoder's input processor expands each <|image_pad|> placeholder
        # into ~N tokens (one per visual patch, e.g. 1225 for 512x512).  If we
        # decode those expanded tokens back to text, we get ~N <|image_pad|>
        # occurrences.  On the prefill side, TRT-LLM's get_prompt_token_ids
        # re-tokenizes the prompt and expects exactly 1 <|image_pad|> per image
        # (matching 1 mm_handle), then expands it itself.  Sending the expanded
        # text causes the "Number of images must match number of mm_handles"
        # assertion failure.
        #
        # The unexpanded `prompt` has exactly 1 <|image_pad|> per image, which
        # is what get_prompt_token_ids expects.
        first_output = encoder_outputs[0]
        prompt_token_ids = (
            list(first_output.prompt_token_ids)
            if first_output.prompt_token_ids
            else None
        )
        processed_prompt = prompt  # unexpanded — 1 placeholder per image
        logging.debug(
            "ENCODE WORKER: processed_prompt (unexpanded, len=%s)",
            len(processed_prompt) if processed_prompt is not None else None,
        )

        yield {
            "ep_disaggregated_params": params_dict,
            "processed_prompt": processed_prompt,
            "prompt_token_ids": prompt_token_ids,
        }

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
        Process an ENCODE-mode request. Dispatches to the appropriate flow.

        Args:
            request: Request containing OpenAI-format multimodal messages
            multimodal_processor: Processor to extract prompt/media and load embeddings
            connector: NIXL connector (required only for embedding_paths flow)
            tokenizer: Tokenizer for the model
            model_dir: Path to model directory
            model_type: Model type string
            engine: TensorRTLLMEngine instance

        Yields:
            Response dictionary based on the flow:
            - Embedding-path flow: nixl_readable_metadata + shape/dtype + auxiliary_data
            - Full EPD flow: ep_disaggregated_params + processed_prompt + prompt_token_ids
        """
        if multimodal_processor is None:
            yield {"error": "No multimodal_processor configured on encode worker"}
            return

        # Extract messages and determine which flow to use
        messages = request.get("extra_args", {}).get(
            "messages", request.get("messages", [])
        )
        (
            text_prompt,
            image_urls,
            embedding_paths,
        ) = multimodal_processor.extract_prompt_and_media(messages)

        # Flow 1: Embedding-path flow (pre-computed embeddings via NIXL)
        if embedding_paths:
            if connector is None:
                yield {"error": "NIXL connector is required for embedding_paths encode"}
                return
            async for response in EncodeHelper._process_embedding_path_flow(
                embedding_paths, multimodal_processor, connector
            ):
                yield response

        # Flow 2: Full EPD flow (image URLs via MultimodalEncoder)
        elif image_urls and text_prompt:
            if model_dir is None or model_type is None:
                yield {
                    "error": "model_dir and model_type are required for full EPD encode"
                }
                return
            if engine is None:
                yield {"error": "No engine configured on encode worker for full EPD"}
                return
            async for response in EncodeHelper._process_full_epd_flow(
                text_prompt, image_urls, tokenizer, model_dir, model_type, engine
            ):
                yield response

        # No valid multimodal content found
        else:
            yield {"error": "No embedding_paths or image_urls found in request"}
