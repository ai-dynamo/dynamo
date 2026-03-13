# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
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
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], list]:
        """
        Read embeddings from encode worker response using NIXL and reconstruct original format.

        Args:
            encode_response: Response from encode worker containing metadata and NIXL info
            connector: NIXL connector for reading operations

        Returns:
            - List[torch.Tensor]: When multiple plain-tensor embeddings were sent
            - torch.Tensor: When a single plain-tensor embedding was sent
            - dict: When a dict-format embedding (with auxiliary data) was sent

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
        per_embedding_shapes = encode_response.get("per_embedding_shapes")
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
            # Dict format: reconstruct dictionary with mm_embeddings + auxiliary data
            deserialized_auxiliary = EncodeHelper.deserialize_tensor_dict(
                auxiliary_data
            )
            result = {"mm_embeddings": encodings_tensor}
            result.update(deserialized_auxiliary)
            return result
        elif per_embedding_shapes and len(per_embedding_shapes) > 1:
            # Multiple plain tensors were concatenated for transfer — split back
            split_sizes = [
                int(torch.prod(torch.tensor(shape[:-1])).item())
                for shape in per_embedding_shapes
            ]
            parts = torch.split(encodings_tensor, split_sizes, dim=0)
            return [p.reshape(shape) for p, shape in zip(parts, per_embedding_shapes)]
        else:
            # Single plain tensor
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

        Loads embeddings from one or more file paths/URLs and creates a NIXL readable
        operation for the prefill worker to read via RDMA.

        For multiple plain-tensor embeddings, all tensors are concatenated along dim=0
        for a single NIXL transfer. The per-embedding shapes are stored in the response
        so the reader can split them back into a list.

        For dict-format embeddings (containing auxiliary data), only a single path is
        supported. If multiple dict-format paths are provided, only the first is used.

        Args:
            embedding_paths: List of paths to embedding files (.pt/.pth/.bin)
            multimodal_processor: Processor to load embeddings
            connector: NIXL connector for RDMA transfer

        Yields:
            Response with NIXL metadata, shape, dtype, auxiliary data, and
            per_embedding_shapes for multi-embedding reconstruction
        """
        logging.info(f"EncodeHelper: loading embeddings from {embedding_paths}")

        # Load all embedding files
        loaded_tensors = []
        auxiliary_data_list = []
        has_dict_format = False

        for path in embedding_paths:
            loaded_data = multimodal_processor.load_tensor_from_path_or_url(path)
            if isinstance(loaded_data, dict):
                has_dict_format = True
                tensor = loaded_data.get("mm_embeddings")
                if tensor is None:
                    yield {
                        "error": f"Dictionary embeddings missing 'mm_embeddings' key in {path}"
                    }
                    return
                auxiliary_data_list.append(
                    {k: v for k, v in loaded_data.items() if k != "mm_embeddings"}
                )
            else:
                tensor = loaded_data
                auxiliary_data_list.append({})
            loaded_tensors.append(tensor)

        if has_dict_format and len(embedding_paths) > 1:
            logging.warning(
                "Multiple embedding paths with dict format are not supported. "
                "Only the first embedding will be used."
            )
            loaded_tensors = [loaded_tensors[0]]
            auxiliary_data_list = [auxiliary_data_list[0]]

        per_embedding_shapes = [list(t.shape) for t in loaded_tensors]

        if has_dict_format:
            # Dict format: single embedding with auxiliary data (original behavior)
            encodings = loaded_tensors[0]
            auxiliary_data = auxiliary_data_list[0]
        elif len(loaded_tensors) == 1:
            # Single plain tensor (original behavior)
            encodings = loaded_tensors[0]
            auxiliary_data = {}
        else:
            # Multiple plain tensors: flatten each to [rows, hidden_dim] and concatenate
            # so they can be transferred in a single NIXL operation
            flat_tensors = [t.reshape(-1, t.shape[-1]) for t in loaded_tensors]
            encodings = torch.cat(flat_tensors, dim=0)
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
                "per_embedding_shapes": per_embedding_shapes,
            }
            yield response

            # Wait for prefill worker to complete the read
            logging.debug(
                "EncodeHelper waiting for PrefillHandler to read embeddings..."
            )
            await readable_op.wait_for_completion()
            logging.debug("EncodeHelper completed readable operation.")

    @staticmethod
    async def _process_full_epd_flow(
        prompt_token_ids_from_request: list,
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
            prompt_token_ids_from_request: token IDs from the request (Rust preprocessor)
            image_urls: List of image URLs to process
            tokenizer: Tokenizer for decoding prompt_token_ids_from_request
            model_dir: Path to model directory (unused; kept for API compatibility)
            model_type: Model type string (unused; kept for API compatibility)
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

        processed_mm_data = {"image": pil_images}
        inputs = [
            {
                "prompt_token_ids": prompt_token_ids_from_request,
                "multi_modal_data": processed_mm_data,
                "mm_processor_kwargs": {},
            }
        ]

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

        # Extract processed prompt (includes <image> tokens) for prefill/decode consistency.
        # NOTE: processed_prompt will contain template/placeholder tokens
        # (e.g. <image>, [INST], etc.). Adding special tokens here can change
        # token alignment across EPD stages (prefill/decode), so we explicitly
        # avoid adding them.
        processed_prompt = None
        if tokenizer is not None:
            processed_prompt = tokenizer.decode(
                prompt_token_ids_from_request, skip_special_tokens=False
            )

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
        multimodal_processor: Any,
        connector: Optional[nixl_connect.Connector],
        tokenizer: Any = None,
        model_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        engine: Any = None,
    ) -> AsyncGenerator[dict, None]:
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
            if engine is None or not engine.encoder_available:
                yield {
                    "error": (
                        "MultimodalEncoder is not available on this encode worker. "
                        "The model architecture may not support standalone encoder "
                        "in TRT-LLM. Use the embedding-path flow or run without "
                        "disaggregated encode mode."
                    )
                }
                return
            # Use token_ids from request (Rust preprocessor already applied
            # chat template and tokenized; token_ids then include image placeholder tokens
            # if the model's tokenizer_config chat template emits them).
            token_ids = request.get("token_ids")
            async for response in EncodeHelper._process_full_epd_flow(
                token_ids,  # type: ignore
                image_urls,
                tokenizer,
                model_dir,
                model_type,
                engine,
            ):
                yield response

        # No valid multimodal content found
        else:
            yield {
                "error": "No embedding_paths or image_urls found in request, or image_urls without text_prompt or token_ids"
            }
