# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modality-specific output formatters for vLLM-Omni.

Extracted from OmniHandler and AudioGenerationHandler so that any consumer
(aggregated handler, disaggregated router, test harness) can format engine
output without creating an engine or loading model weights.
"""

import asyncio
import base64
import logging
import tempfile
import time
import uuid
from io import BytesIO
from typing import Any, Dict, Optional

from dynamo.common.utils.engine_response import normalize_finish_reason

logger = logging.getLogger(__name__)


class TextFormatter:
    """Formats LLM text output as OpenAI chat completion chunks."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def format(
        self,
        request_output: Any,
        request_id: str,
        *,
        previous_text: str = "",
    ) -> Dict[str, Any] | None:
        if not request_output.outputs:
            return _error_chunk(request_id, self._model_name, "No outputs from engine")

        output = request_output.outputs[0]
        delta_text = output.text[len(previous_text) :]

        chunk: Dict[str, Any] = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta_text},
                    "finish_reason": (
                        normalize_finish_reason(output.finish_reason)
                        if output.finish_reason
                        else None
                    ),
                }
            ],
        }

        if output.finish_reason:
            chunk["usage"] = _build_completion_usage(request_output)

        return chunk


class DiffusionFormatter:
    """Formats diffusion output (images/video frames) for the frontend.

    Handles both image and video — routes by request_type since vllm-omni
    reports final_output_type="image" for all diffusion outputs.
    """

    def __init__(
        self,
        model_name: str,
        media_fs: Any,
        media_http_url: Optional[str],
        default_fps: int = 16,
    ) -> None:
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._default_fps = default_fps

    async def format(
        self, stage_output: Any, request_id: str, *, request_type: Any, **ctx: Any
    ) -> Dict[str, Any] | None:
        images = (
            stage_output.images if hasattr(stage_output, "images") else stage_output
        )
        if not images:
            return None
        from dynamo.common.utils.output_modalities import RequestType

        if request_type == RequestType.VIDEO_GENERATION:
            return await self._encode_video(
                images, request_id, fps=ctx.get("fps", self._default_fps)
            )
        return await self._encode_image(
            images,
            request_id,
            request_type=request_type,
            response_format=ctx.get("response_format"),
        )

    async def _encode_video(
        self, images: list, request_id: str, fps: int
    ) -> Dict[str, Any] | None:
        from diffusers.utils.export_utils import export_to_video

        from dynamo.common.protocols.video_protocol import NvVideosResponse, VideoData
        from dynamo.common.storage import upload_to_fs
        from dynamo.common.utils.video_utils import normalize_video_frames

        if not images:
            return None
        try:
            start_time = time.time()
            frame_list = normalize_video_frames(images)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                await asyncio.to_thread(export_to_video, frame_list, tmp.name, fps)
                video_bytes = tmp.read()
            video_url = await upload_to_fs(
                self._media_fs,
                f"videos/{request_id}.mp4",
                video_bytes,
                self._media_http_url,
            )
            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[VideoData(url=video_url)],
                inference_time_s=time.time() - start_time,
            ).model_dump()
        except Exception as e:
            logger.error("Failed to encode video for request %s: %s", request_id, e)
            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="failed",
                progress=0,
                created=int(time.time()),
                data=[],
                error=str(e),
            ).model_dump()

    async def _encode_image(
        self,
        images: list,
        request_id: str,
        *,
        request_type: Any,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        from dynamo.common.protocols.image_protocol import ImageData, NvImagesResponse
        from dynamo.common.utils.output_modalities import RequestType

        if not images:
            return _error_chunk(request_id, self._model_name, "No images generated")

        data_urls = await self._prepare_images(images, request_id, response_format)

        if request_type == RequestType.CHAT_COMPLETION:
            return {
                "id": request_id,
                "created": int(time.time()),
                "object": "chat.completion.chunk",
                "model": self._model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": [
                                {"type": "image_url", "image_url": {"url": u}}
                                for u in data_urls
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        if request_type == RequestType.IMAGE_GENERATION:
            image_data_list = []
            for data_url in data_urls:
                if response_format == "url":
                    image_data_list.append(ImageData(url=data_url))
                elif response_format == "b64_json" or response_format is None:
                    b64 = (
                        data_url.split(",", 1)[1]
                        if data_url.startswith("data:")
                        else data_url
                    )
                    image_data_list.append(ImageData(b64_json=b64))
                else:
                    raise ValueError(f"Invalid response format: {response_format}")
            return NvImagesResponse(
                created=int(time.time()), data=image_data_list
            ).model_dump()

        return None

    async def _prepare_images(
        self, images: list, request_id: str, response_format: Optional[str] = None
    ) -> list:
        from dynamo.common.storage import upload_to_fs

        outlist = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            if response_format == "url":
                url = await upload_to_fs(
                    self._media_fs,
                    f"images/{request_id}/{uuid.uuid4()}.png",
                    image_bytes,
                    self._media_http_url,
                )
                outlist.append(url)
            elif response_format == "b64_json" or response_format is None:
                outlist.append(
                    f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                )
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        return outlist


def _error_chunk(
    request_id: str, model_name: str, error_message: str
) -> Dict[str, Any]:
    """Error response in OpenAI chat.completion.chunk format."""
    return {
        "id": request_id,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": f"Error: {error_message}"},
                "finish_reason": "error",
            }
        ],
    }


def _build_completion_usage(request_output: Any) -> Dict[str, Any]:
    """Build completion usage stats from a vLLM RequestOutput."""
    prompt_tokens = (
        len(request_output.prompt_token_ids)
        if getattr(request_output, "prompt_token_ids", None)
        else None
    )
    completion_tokens = len(request_output.outputs[0].token_ids)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": (
            {"cached_tokens": num_cached}
            if (num_cached := getattr(request_output, "num_cached_tokens", None))
            else None
        ),
    }
