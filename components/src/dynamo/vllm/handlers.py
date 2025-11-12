# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import logging
import os
import zlib
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Final

import torch
from PIL import Image
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

import dynamo.nixl_connect as connect
from dynamo.llm import ZmqKvEventPublisher
from dynamo.runtime.logging import configure_dynamo_logging

from .engine_monitor import VllmEngineMonitor
from .multimodal_utils.image_loader import ImageLoader

# For constructing RdmaMetadata from Decoded variant
try:
    from dynamo.nixl_connect import OperationKind, RdmaMetadata, SerializedDescriptor
except ImportError:
    # If nixl_connect not available, will fail at runtime when Decoded variant encountered
    RdmaMetadata = None
    SerializedDescriptor = None
    OperationKind = None

# Multimodal data dictionary keys
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def build_sampling_params(
    request: Dict[str, Any], default_sampling_params: Dict[str, Any]
) -> SamplingParams:
    """
    Build SamplingParams from a PreprocessedRequest.

    Args:
        request: The PreprocessedRequest dict with 'sampling_options' and 'stop_conditions'
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = False

    # Apply sampling_options
    for key, value in request["sampling_options"].items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Apply stop_conditions
    for key, value in request["stop_conditions"].items():
        if value is not None and hasattr(sampling_params, key):
            # Do not add stop key to sampling params - dynamo handles stop conditions directly
            if key == "stop":
                continue
            setattr(sampling_params, key, value)

    return sampling_params


class BaseWorkerHandler(ABC):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, runtime, component, engine, default_sampling_params):
        self.runtime = runtime
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers: list[ZmqKvEventPublisher] | None = None
        self.engine_monitor = VllmEngineMonitor(runtime, engine)
        self.image_loader = ImageLoader()
        self._connector = None  # Lazy-initialized on first Decoded variant

    @abstractmethod
    async def generate(self, request, context) -> AsyncGenerator[dict, None]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill):
        """Background task that monitors for context cancellation and aborts the request."""
        try:
            await context.async_killed_or_stopped()
            # If we reach here, the context was stopped or killed
            await self.engine_client.abort(request_id)
            logger.debug(
                f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
            )
        except asyncio.CancelledError:
            # Task was cancelled, normal cleanup if not aborted
            pass
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(self, context, request_id, is_prefill=False):
        """Context manager that creates and automatically cleans up an abort monitoring task."""
        task = asyncio.create_task(self._monitor_abort(context, request_id, is_prefill))
        try:
            yield task
        finally:
            # Cancel the abort monitoring task when exiting the context
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Override in subclasses if cleanup is needed."""
        pass

    async def _ensure_connector_initialized(self):
        """
        Lazy initialization of NIXL connector.
        Only called when Decoded variant is encountered.
        """
        if self._connector is None:
            logger.info("Initializing NIXL connector for decoded media support")
            self._connector = connect.Connector()
            await self._connector.initialize()
            logger.info("NIXL connector initialized")

    async def _read_decoded_image_via_nixl(
        self, decoded_meta: Dict[str, Any]
    ) -> Image.Image:
        """
        Read decoded image data via NIXL RDMA.

        Args:
            decoded_meta: Dictionary containing:
                - nixl_metadata: Base64-encoded NIXL agent metadata
                - nixl_descriptor: {addr, size, mem_type, device_id}
                - shape: [height, width, channels]
                - dtype: Data type (e.g., "UINT8")
                - metadata: Optional image metadata (format, color_type, etc.)

        Returns:
            PIL.Image object
        """
        # Ensure connector is initialized
        await self._ensure_connector_initialized()

        # Extract and validate required fields
        if (
            "nixl_metadata" not in decoded_meta
            or "shape" not in decoded_meta
            or "nixl_descriptor" not in decoded_meta
        ):
            raise ValueError(
                f"Decoded variant missing required fields. Got keys: {decoded_meta.keys()}"
            )

        nixl_metadata_str = decoded_meta["nixl_metadata"]
        nixl_descriptor = decoded_meta["nixl_descriptor"]
        shape = decoded_meta["shape"]
        dtype_str = decoded_meta.get("dtype", "UINT8")

        # Frontend only sends UINT8 for images currently
        if dtype_str != "UINT8":
            raise ValueError(
                f"Unsupported dtype: {dtype_str} (only UINT8 supported for images)"
            )

        # Create empty tensor to receive RDMA data
        # Shape from frontend is [height, width, channels]
        tensor = torch.empty(shape, dtype=torch.uint8, device="cpu")
        local_descriptor = connect.Descriptor(tensor)

        # Construct RdmaMetadata object from decoded_meta
        # Frontend sends nixl_descriptor with {addr, size, mem_type, device_id}
        # Need to convert to SerializedDescriptor format
        mem_type = nixl_descriptor.get("mem_type", "Dram")
        device_str = (
            "cpu"
            if mem_type == "Dram"
            else f"cuda:{nixl_descriptor.get('device_id', 0)}"
        )

        serialized_desc = SerializedDescriptor(
            device=device_str, ptr=nixl_descriptor["addr"], size=nixl_descriptor["size"]
        )

        # Fix nixl_metadata format issue:
        # Backend expects: "b64:<zlib_compressed_base64>"
        # Frontend sends: "b64:<uncompressed_base64>" (PR #3988 bug)
        # Workaround: Compress if not already compressed
        if nixl_metadata_str.startswith("b64:"):
            # Decode to check if compressed
            try:
                decoded_bytes = base64.b64decode(nixl_metadata_str[4:])
                # Try to decompress - if it works, already compressed
                try:
                    zlib.decompress(decoded_bytes)
                    # Already compressed, use as-is
                    final_nixl_metadata = nixl_metadata_str
                except zlib.error:
                    # Not compressed, need to compress
                    compressed = zlib.compress(decoded_bytes, level=6)
                    reencoded = base64.b64encode(compressed).decode("utf-8")
                    final_nixl_metadata = f"b64:{reencoded}"
                    logger.debug("Compressed uncompressed NIXL metadata from frontend")
            except Exception as e:
                raise ValueError(f"Failed to decode nixl_metadata: {e}")
        else:
            final_nixl_metadata = nixl_metadata_str

        rdma_metadata = RdmaMetadata(
            descriptors=[serialized_desc],
            nixl_metadata=final_nixl_metadata,
            notification_key=f"decoded-image-{decoded_meta.get('shape', 'unknown')}",
            operation_kind=int(OperationKind.READ),
        )

        # Read via NIXL RDMA
        read_op = await self._connector.begin_read(rdma_metadata, local_descriptor)
        await read_op.wait_for_completion()
        logger.debug(f"Loaded image via NIXL RDMA: shape={shape}")

        # Convert tensor to PIL.Image
        # Tensor shape is [H, W, C], dtype is uint8
        # PIL.Image.fromarray expects numpy array
        numpy_array = tensor.numpy()

        # Determine PIL mode based on number of channels (common cases)
        # Frontend sends 3D array [H, W, C]
        num_channels = shape[2]
        if num_channels == 3:
            mode = "RGB"  # Most common
        elif num_channels == 4:
            mode = "RGBA"
        elif num_channels == 1:
            mode = "L"  # Grayscale
            numpy_array = numpy_array.squeeze(-1)
        else:
            raise ValueError(
                f"Unsupported channel count: {num_channels} (expected 1, 3, or 4)"
            )

        pil_image = Image.fromarray(numpy_array, mode=mode)
        return pil_image

    async def _extract_multimodal_data(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """
        Extract and decode multimodal data from PreprocessedRequest.

        Supports two variants:
        1. Url: Frontend passes URL, backend decodes (fallback, slower)
        2. Decoded: Frontend decoded, NIXL RDMA transfer (optimal, faster)
        """
        if "multi_modal_data" not in request or request["multi_modal_data"] is None:
            return None

        mm_map = request["multi_modal_data"]
        vllm_mm_data = {}

        # Process image_url entries
        images = []
        for item in mm_map.get(IMAGE_URL_KEY, []):
            if isinstance(item, dict) and DECODED_VARIANT_KEY in item:
                # Fast path: Frontend decoded, NIXL RDMA transfer (PR #3988)
                decoded_meta = item[DECODED_VARIANT_KEY]
                image = await self._read_decoded_image_via_nixl(decoded_meta)
                images.append(image)
                logger.info(
                    f"✓ Using DECODED path: Loaded image via NIXL RDMA "
                    f"(shape={decoded_meta.get('shape')}, dtype={decoded_meta.get('dtype')})"
                )
            elif isinstance(item, dict) and URL_VARIANT_KEY in item:
                # Fallback path: Decode URL in Python backend (current behavior)
                url = item[URL_VARIANT_KEY]
                image = await self.image_loader.load_image(url)
                images.append(image)
                logger.info(
                    f"⊙ Using URL path: Loaded image from URL (type={url.split(':')[0]})"
                )

        if images:
            # vLLM expects single image or list
            vllm_mm_data["image"] = images[0] if len(images) == 1 else images
            logger.debug(f"Extracted {len(images)} image(s) for multimodal processing")

        # Handle video_url entries (future expansion)
        if VIDEO_URL_KEY in mm_map:
            logger.warning("Video multimodal data not yet supported in standard worker")

        return vllm_mm_data if vllm_mm_data else None

    async def generate_tokens(
        self, prompt, sampling_params, request_id, data_parallel_rank=None
    ):
        try:
            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                data_parallel_rank=data_parallel_rank,
            )

            num_output_tokens_so_far = 0
            try:
                async for res in gen:
                    # res is vllm's RequestOutput

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
                    yield out
                    num_output_tokens_so_far = next_total_toks
            except asyncio.CancelledError:
                # raise EngineShGeneratorExit when engine exits so that frontend can migrate the request
                raise GeneratorExit(
                    "Decode engine was shut down during token generation"
                ) from None

        except EngineDeadError as e:
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        component,
        engine,
        default_sampling_params,
    ):
        super().__init__(runtime, component, engine, default_sampling_params)

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation
        request_id = context.id()
        logger.debug(f"Decode Request ID: {request_id}")

        # Extract and decode multimodal data if present
        multi_modal_data = await self._extract_multimodal_data(request)

        prompt = TokensPrompt(
            prompt_token_ids=request["token_ids"], multi_modal_data=multi_modal_data
        )

        # Build sampling params from request
        sampling_params = build_sampling_params(request, self.default_sampling_params)

        # Extract disaggregated_params from request (set by prefill router in Rust frontend)
        disaggregated_params = request.get("disaggregated_params")
        if disaggregated_params:
            # Prefill was performed - use the disaggregated params
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = disaggregated_params.get(
                "kv_transfer_params"
            )
            logger.debug(
                f"Using disaggregated params from prefill for request {request_id}"
            )

        dp_rank = request.get("dp_rank", None)

        async with self._abort_monitor(context, request_id):
            try:
                async for tok in self.generate_tokens(
                    prompt, sampling_params, request_id, data_parallel_rank=dp_rank
                ):
                    yield tok
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, runtime, component, engine, default_sampling_params):
        super().__init__(runtime, component, engine, default_sampling_params)

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation with decode phase
        request_id = context.id()
        logger.debug(f"Prefill Request ID: {request_id}")

        # Extract and decode multimodal data if present
        multi_modal_data = await self._extract_multimodal_data(request)

        token_ids = request["token_ids"]
        prompt = TokensPrompt(
            prompt_token_ids=token_ids, multi_modal_data=multi_modal_data
        )

        # Build sampling params from request using shared utility
        sampling_params = build_sampling_params(request, self.default_sampling_params)

        # Configure for prefill-only mode with remote decode
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        # Override for prefill: only generate 1 token
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        dp_rank = request.get("dp_rank", None)

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self.engine_client.generate(
                    prompt, sampling_params, request_id, data_parallel_rank=dp_rank
                )
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)

            try:
                async for res in gen:
                    logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                    token_ids = res.outputs[0].token_ids if res.outputs else []

                    output: Dict[str, Any] = {
                        "token_ids": list(token_ids),
                        "disaggregated_params": (
                            {"kv_transfer_params": res.kv_transfer_params}
                            if res.kv_transfer_params
                            else None
                        ),
                    }

                    yield output
            except asyncio.CancelledError:
                # raise the error because we cannot migrate prefill requests
                raise GeneratorExit(
                    "Prefill engine was shut down during token generation"
                ) from None
