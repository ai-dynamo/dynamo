# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

import torch

import dynamo.nixl_connect as nixl_connect
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec


class EncodeHelper:
    """Utility class for encoding and serialization operations."""

    # Shared ImageLoader for full EPD flow (async image loading)
    _image_loader: Optional[ImageLoader] = None
    _image_loader_lock = threading.Lock()

    @classmethod
    def _get_image_loader(cls) -> ImageLoader:
        if cls._image_loader is None:
            with cls._image_loader_lock:
                if cls._image_loader is None:
                    cls._image_loader = ImageLoader()
        return cls._image_loader

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

    @staticmethod
    def _build_prompt_from_messages(messages, tokenizer):
        """Build a prompt string from OpenAI-format messages using the model's
        HuggingFace chat template.

        VLMs such as Qwen2-VL embed vision-placeholder tokens (e.g.
        ``<|vision_start|><|image_pad|><|vision_end|>``) inside their chat
        template.  The Rust frontend may use a *different* template (e.g.
        LLaVA's ``[INST]...[/INST]``) that omits these placeholders, so
        decoding the Rust ``token_ids`` back to text produces a prompt without
        image placeholders.  To fix this we re-apply the model's own HF chat
        template to the original messages, which inserts the correct
        placeholders for every model.

        The OpenAI ``image_url`` content items are converted to the generic
        ``{"type": "image"}`` form expected by HF chat templates.

        Returns:
            The prompt string with vision placeholders, or *None* if the
            tokenizer does not support ``apply_chat_template``.
        """
        # Resolve the underlying HF tokenizer (TransformersTokenizer wraps it)
        hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        apply_fn = getattr(hf_tokenizer, "apply_chat_template", None)
        if apply_fn is None:
            return None

        # Convert OpenAI format → HF chat-template format
        converted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # String content (no images)
            if isinstance(content, str):
                converted_messages.append({"role": role, "content": content})
                continue

            # List of content items (may include images)
            converted_content = []
            for item in content:
                if isinstance(item, str):
                    converted_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    ctype = item.get("type", "")
                    if ctype == "image_url":
                        # HF templates expect {"type": "image"}, not the
                        # OpenAI {"type": "image_url", "image_url": {...}}
                        converted_content.append({"type": "image"})
                    elif ctype == "text":
                        converted_content.append(item)
                    else:
                        converted_content.append(item)
            converted_messages.append({"role": role, "content": converted_content})

        try:
            prompt = apply_fn(
                converted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception as e:
            logging.warning(
                "ENCODE WORKER: apply_chat_template failed (%s), "
                "falling back to token_ids decode",
                e,
            )
            return None

    @staticmethod
    async def _process_full_epd_flow(
        prompt_token_ids_from_request: list,
        image_urls: list,
        tokenizer,
        model_dir: str,
        model_type: str,
        engine,
        messages=None,
    ):
        """
        Process image URLs via TRT-LLM's MultimodalEncoder (full EPD flow).

        Runs MultimodalEncoder.generate() to produce disaggregated_params
        containing multimodal embedding handles for the prefill worker.

        Args:
            prompt_token_ids_from_request: token IDs from the request (Rust preprocessor)
            image_urls: List of image URLs to process
            tokenizer: Tokenizer for decoding prompt_token_ids_from_request
            model_dir: Path to model directory (unused; kept for API compatibility)
            model_type: Model type string (unused; kept for API compatibility)
            engine: TensorRTLLMEngine with MultimodalEncoder
            messages: Original OpenAI-format messages (used to build prompt
                with the model's HF chat template, which inserts correct
                vision placeholders for VLMs like Qwen2-VL)

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

        # Build the text prompt.  We prefer using the model's HF chat template
        # applied to the original messages because:
        # 1. The Rust frontend may use a different chat template that omits
        #    vision-placeholder tokens (e.g. <|vision_start|><|image_pad|>
        #    <|vision_end|> for Qwen2-VL).
        # 2. Decoding the Rust token_ids back to text then produces a prompt
        #    WITHOUT image placeholders → the VLM's HF processor cannot
        #    expand image tokens → multimodal_lengths stays None → assertion
        #    failure in EncoderSampler.
        # By re-applying the model's own template we guarantee the correct
        # placeholders are present for every VLM.
        prompt_text = None
        if messages and tokenizer is not None:
            prompt_text = EncodeHelper._build_prompt_from_messages(
                messages, tokenizer
            )
            if prompt_text:
                logging.debug(
                    "ENCODE WORKER: built prompt from messages via HF chat template "
                    "(first 300 chars): %r",
                    prompt_text[:300],
                )

        # Fallback: decode token_ids from the Rust preprocessor.
        # This works for models whose Rust template already includes vision
        # placeholders (e.g. LLaVA).
        if prompt_text is None and tokenizer is not None:
            prompt_text = tokenizer.decode(
                prompt_token_ids_from_request, skip_special_tokens=False
            )
            logging.debug(
                "ENCODE WORKER: using decoded token_ids as prompt "
                "(first 300 chars): %r",
                prompt_text[:300] if prompt_text else None,
            )

        if prompt_text is None:
            raise RuntimeError(
                "Cannot run full EPD encode flow without a tokenizer to "
                "build prompt text"
            )

        processed_mm_data = {"image": pil_images}
        # IMPORTANT: Do NOT include prompt_token_ids in the input dict.
        # TRT-LLM's _preprocess dispatches on an if/elif chain where
        # "prompt_token_ids" is checked BEFORE "prompt". If both are present,
        # TRT-LLM takes the prompt_token_ids branch and skips multimodal
        # processing entirely (no input_processor, no multimodal hashing,
        # no multimodal_lengths → assertion failure in EncoderSampler).
        # By providing only "prompt" + "multi_modal_data", we ensure TRT-LLM
        # takes the correct multimodal branch that runs the input processor.
        input_dict = {
            "prompt": prompt_text,
            "multi_modal_data": processed_mm_data,
            "mm_processor_kwargs": {},
        }
        inputs = [input_dict]

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

        # Return the prompt text (with vision placeholders) so the prefill
        # worker can pass it to TRT-LLM's _preprocess.  The `is_mm_disagg`
        # branch in TRT-LLM calls `get_prompt_token_ids(inputs, mm_handles)`
        # which expects the prompt to contain the correct number of image
        # placeholders matching mm_handles.  We MUST use `prompt_text` (built
        # from the model's HF chat template) rather than decoding the Rust
        # token_ids — the Rust template may omit vision placeholders.
        processed_prompt = prompt_text

        logging.debug(
            "ENCODE WORKER: Extracted processed_prompt (len=%s)",
            len(processed_prompt) if processed_prompt is not None else None,
        )

        yield {
            "ep_disaggregated_params": params_dict,
            "processed_prompt": processed_prompt,
            "prompt_token_ids": prompt_token_ids_from_request,
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
            _,
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
        elif image_urls and request.get("token_ids"):
            if model_dir is None or model_type is None:
                yield {
                    "error": "model_dir and model_type are required for full EPD encode"
                }
                return
            if engine is None:
                yield {"error": "No engine configured on encode worker for full EPD"}
                return
            # Use token_ids from request (Rust preprocessor already applied
            # chat template and tokenized; token_ids then include image placeholder tokens
            # if the model's tokenizer_config chat template emits them).
            # We also pass the original messages so that _process_full_epd_flow
            # can re-apply the model's HF chat template to get correct vision
            # placeholders (the Rust template may differ from the model's).
            token_ids = request.get("token_ids")
            async for response in EncodeHelper._process_full_epd_flow(
                token_ids,
                image_urls,
                tokenizer,
                model_dir,
                model_type,
                engine,
                messages=messages,
            ):
                yield response

        # No valid multimodal content found
        else:
            yield {
                "error": "No embedding_paths or image_urls found in request, or image_urls without text_prompt or token_ids"
            }
