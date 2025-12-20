# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, Union

import torch

import dynamo.nixl_connect as nixl_connect


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
    async def process_embedding_request(
        request: Dict[str, Any],
        multimodal_processor,
        connector: nixl_connect.Connector,
    ):
        """
        Process embedding request by loading embeddings and creating NIXL readable operation.

        Args:
            request: Request containing messages with embedding paths
            multimodal_processor: Multimodal processor for loading embeddings
            connector: NIXL connector for creating readable operations

        Yields:
            Response dictionary with NIXL metadata and embeddings info, or error response
        """
        # Load embeddings first to get the actual shape
        # Extract messages from extra_args (set by Rust preprocessor for multimodal) or fall back to direct field
        messages = request.get("extra_args", {}).get(
            "messages", request.get("messages", [])
        )
        _, _, embedding_paths = multimodal_processor.extract_prompt_and_media(messages)

        if not embedding_paths:
            # Placeholder for TRTLLM Encoder to be called
            # TRTLLM Encoder will return a memory handler on the encoder GPU with the encodings
            logging.warning(
                "No embedding paths found, NIXL transfer for image urls not supported by TRTLLM Encoder yet"
            )
            yield {"error": "No embedding paths found"}
            return

        # Load the embeddings data
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
        with connector.create_readable(descriptor) as readable_op:
            # Get the metadata for the readable operation
            op_metadata = readable_op.metadata()

            # Send back shape info, readable metadata, and serialized auxiliary data
            response = {
                "nixl_readable_metadata": op_metadata.model_dump(),
                "embeddings_shape": list(encodings.shape),
                "embeddings_dtype": str(encodings.dtype),
                "auxiliary_data": EncodeHelper.serialize_tensor_dict(auxiliary_data),
            }
            yield response

            # Wait for the prefill worker to complete the read operation
            logging.debug(
                "EncodeHelper waiting for PrefillHandler to read embeddings..."
            )
            await readable_op.wait_for_completion()
            logging.debug("EncodeHelper completed readable operation.")
