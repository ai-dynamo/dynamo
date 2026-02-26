# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, Optional, Union

from diffusers.utils import export_to_video
from fsspec.implementations.dirfs import DirFileSystem
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.protocols.image_protocol import (
    ImageData,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.common.storage import upload_to_fs
from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import (
    compute_num_frames,
    normalize_video_frames,
    parse_size,
)
from dynamo.vllm.omni.base_handler import BaseOmniHandler

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_FPS = 16


@dataclass
class EngineInputs:
    """Parsed engine inputs ready for AsyncOmni.generate().

    Attributes:
        prompt: OmniTextPrompt dict for the engine.
        sampling_params_list: Per-stage sampling parameters, or None for defaults.
        request_type: The resolved request type (may differ from the initial parse
            when a chat completion request carries video params).
        fps: Frames per second, only meaningful for video requests.
        response_format: Desired response format (e.g. "url" or "b64_json" for
            image requests). None means use the default for the request type.
    """

    prompt: OmniTextPrompt
    sampling_params_list: list | None = None
    request_type: RequestType = RequestType.CHAT_COMPLETION
    fps: int = 0
    response_format: str | None = None


class OmniHandler(BaseOmniHandler):
    """Unified handler for multi-stage pipelines using vLLM-Omni.

    Handles text-to-text, text-to-image, and text-to-video generation.
    """

    def __init__(
        self,
        runtime,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
        media_output_fs: Optional[DirFileSystem] = None,
        media_output_http_url: Optional[str] = None,
    ):
        """Initialize the unified Omni handler.

        Args:
            runtime: Dynamo distributed runtime.
            component: Dynamo component handle.
            config: Parsed Config object from args.py.
            default_sampling_params: Default sampling parameters dict.
            shutdown_event: Optional asyncio event for graceful shutdown.
            media_output_fs: Filesystem for storing generated images/videos.
            media_output_http_url: Base URL for rewriting media paths in responses.
        """
        super().__init__(
            runtime=runtime,
            config=config,
            default_sampling_params=default_sampling_params,
            shutdown_event=shutdown_event,
        )
        self.media_output_fs = media_output_fs
        self.media_output_http_url = media_output_http_url

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict, None]:
        """Generate outputs via the unified OpenAI mode.

        Args:
            request: Raw request dictionary from the Rust frontend.
            context: Dynamo context for request tracking.

        Yields:
            Response dictionaries.
        """
        request_id = context.id()
        logger.debug(f"Omni Request ID: {request_id}")

        async for chunk in self._generate_openai_mode(request, context, request_id):
            yield chunk

    async def _generate_openai_mode(
        self, request: Dict[str, Any], context, request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Single generation path for all request protocols and output modalities."""

        parsed_request, request_type = parse_request_type(
            request, self.config.output_modalities
        )
        inputs = self.build_engine_inputs(parsed_request, request_type)

        generate_kwargs: Dict[str, Any] = {
            "prompt": inputs.prompt,
            "request_id": request_id,
        }
        if inputs.sampling_params_list is not None:
            generate_kwargs["sampling_params_list"] = inputs.sampling_params_list

        previous_text = ""

        async with self._abort_monitor(context, request_id):
            try:
                async for stage_output in self.engine_client.generate(
                    **generate_kwargs,
                ):
                    if (
                        stage_output.final_output_type == "text"
                        and stage_output.request_output
                    ):
                        chunk = self._format_text_chunk(
                            stage_output.request_output,
                            request_id,
                            previous_text,
                        )
                        if chunk:
                            output = stage_output.request_output.outputs[0]
                            previous_text = output.text
                            yield chunk

                    elif (
                        stage_output.final_output_type == "image"
                        and stage_output.images
                    ):
                        # vllm-omni uses final_output_type="image" for both
                        # image and video diffusion outputs. Use the parsed
                        # request type to route to the correct formatter.
                        if inputs.request_type == RequestType.VIDEO_GENERATION:
                            chunk = await self._format_video_chunk(
                                stage_output.images,
                                request_id,
                                fps=inputs.fps,
                            )
                        else:
                            chunk = await self._format_image_chunk(
                                stage_output.images,
                                request_id,
                                response_format=inputs.response_format,
                                request_type=inputs.request_type,
                            )
                        if chunk:
                            yield chunk

            except GeneratorExit:
                logger.info(f"Request {request_id} aborted due to shutdown")
                raise
            except Exception as e:
                logger.error(f"Error during generation for request {request_id}: {e}")
                yield self._error_chunk(request_id, str(e))

    def build_engine_inputs(
        self,
        parsed_request: Union[
            NvCreateImageRequest, NvCreateVideoRequest, Dict[str, Any]
        ],
        request_type: RequestType,
    ) -> EngineInputs:
        """Convert a parsed request into AsyncOmni engine inputs.

        Args:
            parsed_request: Output from parse_request_type -- a Pydantic model
                for image/video requests, or a raw dict for chat completions.
            request_type: The RequestType determined by parse_request_type.

        Returns:
            EngineInputs ready for engine_client.generate().
        """
        if request_type == RequestType.CHAT_COMPLETION:
            return self._engine_inputs_from_chat(parsed_request)
        elif request_type == RequestType.IMAGE_GENERATION:
            return self._engine_inputs_from_image(parsed_request)
        elif request_type == RequestType.VIDEO_GENERATION:
            return self._engine_inputs_from_video(parsed_request)

        elif request_type == RequestType.AUDIO_GENERATION:
            raise NotImplementedError("Audio generation is not yet supported")

        raise ValueError(f"Unknown request type: {request_type}")

    def _engine_inputs_from_chat(self, request: Dict[str, Any]) -> EngineInputs:
        """Build engine inputs from a chat completions request dict."""

        # Chat completions request does not support extra_body passthrough
        # So, we can't extract any diffusion related params from the raw_request
        # It falls back to default sampling params
        text_prompt = self._extract_text_prompt(request)
        if text_prompt is None:
            raise ValueError("No user message found in chat completion request")

        prompt = OmniTextPrompt(prompt=text_prompt)

        sampling_params_list = None

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            request_type=RequestType.CHAT_COMPLETION,
            fps=0,
        )

    def _engine_inputs_from_image(self, req: NvCreateImageRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateImageRequest."""
        width, height = parse_size(req.size, default_w=1024, default_h=1024)
        nvext = req.nvext

        prompt = OmniTextPrompt(
            prompt=req.prompt,
            negative_prompt=nvext.negative_prompt
            if nvext and nvext.negative_prompt
            else None,
        )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
        )
        if req.n is not None:
            sp.num_outputs_per_prompt = req.n
        if nvext:
            if nvext.num_inference_steps is not None:
                sp.num_inference_steps = nvext.num_inference_steps
            if nvext.guidance_scale is not None:
                sp.guidance_scale = nvext.guidance_scale
            if nvext.seed is not None:
                sp.seed = nvext.seed

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=[sp],
            request_type=RequestType.IMAGE_GENERATION,
            response_format=req.response_format,
        )

    def _engine_inputs_from_video(self, req: NvCreateVideoRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateVideoRequest."""
        width, height = parse_size(req.size)
        nvext = req.nvext

        nvext_fps = nvext.fps if nvext else None
        nvext_num_frames = nvext.num_frames if nvext else None

        num_frames = compute_num_frames(
            num_frames=nvext_num_frames,
            seconds=req.seconds,
            fps=nvext_fps,
            default_fps=DEFAULT_VIDEO_FPS,
        )
        fps = nvext_fps if nvext_fps is not None else DEFAULT_VIDEO_FPS

        prompt = OmniTextPrompt(
            prompt=req.prompt,
            negative_prompt=nvext.negative_prompt
            if nvext and nvext.negative_prompt
            else None,
        )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
        )
        if nvext:
            if nvext.num_inference_steps is not None:
                sp.num_inference_steps = nvext.num_inference_steps
            if nvext.guidance_scale is not None:
                sp.guidance_scale = nvext.guidance_scale
            if nvext.seed is not None:
                sp.seed = nvext.seed
        if fps is not None:
            sp.fps = fps

        logger.info(
            f"Video diffusion request: prompt='{req.prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, fps={fps}"
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=[sp],
            request_type=RequestType.VIDEO_GENERATION,
            fps=fps,
        )

    async def _prepare_image_output(
        self, images: list, request_id: str, response_format: str | None = None
    ) -> list:
        """Prepare image output for response.

        Args:
            images: List of PIL Image objects.
            request_id: Unique request identifier.
            response_format: Response format ("url" or "b64_json").

        Returns:
            List of image URLs or base64 data-URL strings.
        """
        outlist = []

        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            if response_format == "url":
                storage_path = f"images/{request_id}/{uuid.uuid4()}.png"
                url = await upload_to_fs(
                    self.media_output_fs,
                    storage_path,
                    image_bytes,
                    self.media_output_http_url,
                )
                outlist.append(url)
            elif response_format == "b64_json" or response_format is None:
                img_base64 = base64.b64encode(image_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{img_base64}"
                outlist.append(data_url)
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        return outlist

    async def _format_image_chunk(
        self,
        images: list,
        request_id: str,
        response_format: str | None = None,
        request_type: RequestType = RequestType.IMAGE_GENERATION,
    ) -> Dict[str, Any] | None:
        """Format image output for the appropriate endpoint response.

        Args:
            images: List of PIL Image objects generated by AsyncOmni engine.
            request_id: Unique request identifier.
            response_format: Response format (url, b64_json, None).
            request_type: Request type (chat completion, image generation).

        Returns:
            Formatted response dict, or None if no images generated.
        """
        if not images:
            return self._error_chunk(request_id, "No images generated")

        data_urls = await self._prepare_image_output(
            images, request_id, response_format
        )

        if request_type == RequestType.CHAT_COMPLETION:
            chunk = {
                "id": request_id,
                "created": int(time.time()),
                "object": "chat.completion.chunk",
                "model": self.config.served_model_name or self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}}
                                for data_url in data_urls
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            return chunk
        elif request_type == RequestType.IMAGE_GENERATION:
            image_data_list = []
            for data_url in data_urls:
                if response_format == "url":
                    image_data_list.append(ImageData(url=data_url))
                elif response_format == "b64_json" or response_format is None:
                    if data_url.startswith("data:image"):
                        _, b64_part = data_url.split(",", 1)
                        image_data_list.append(ImageData(b64_json=b64_part))
                    else:
                        image_data_list.append(ImageData(b64_json=data_url))
                else:
                    raise ValueError(f"Invalid response format: {response_format}")

            output = NvImagesResponse(created=int(time.time()), data=image_data_list)
            return output.model_dump()

    async def _format_video_chunk(
        self,
        images: list,
        request_id: str,
        fps: int,
    ) -> Dict[str, Any] | None:
        """Convert diffusion output frames to MP4 and return as NvVideosResponse.

        Args:
            images: List of PIL Image frames from the diffusion stage.
            request_id: Unique request identifier.
            fps: Frames per second for the output video.

        Returns:
            ``NvVideosResponse.model_dump()`` dict, or ``None`` if no frames.
        """
        if not images:
            return None

        try:
            start_time = time.time()

            frame_list = normalize_video_frames(images)

            logger.info(
                f"Encoding {len(frame_list)} frames to MP4 for request {request_id} "
                f"(fps={fps})"
            )

            # Encode frames to MP4 via temp file, then read bytes for upload
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                await asyncio.to_thread(export_to_video, frame_list, tmp.name, fps)
                video_bytes = tmp.read()

            # Upload via filesystem
            storage_path = f"videos/{request_id}.mp4"
            video_url = await upload_to_fs(
                self.media_output_fs,
                storage_path,
                video_bytes,
                self.media_output_http_url,
            )

            logger.info(f"Video uploaded to {video_url} for request {request_id}")

            inference_time = time.time() - start_time

            response = NvVideosResponse(
                id=request_id,
                object="video",
                model=self.config.served_model_name or self.config.model,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[VideoData(url=video_url)],
                inference_time_s=inference_time,
            )
            return response.model_dump()

        except Exception as e:
            logger.error(f"Failed to encode video for request {request_id}: {e}")
            error_response = NvVideosResponse(
                id=request_id,
                object="video",
                model=self.config.served_model_name or self.config.model,
                status="failed",
                progress=0,
                created=int(time.time()),
                data=[],
                error=str(e),
            )
            return error_response.model_dump()

    def _format_text_chunk(
        self,
        request_output,
        request_id: str,
        previous_text: str,
    ) -> Dict[str, Any] | None:
        """Format text output as OpenAI chat completion chunk."""
        if not request_output.outputs:
            return self._error_chunk(request_id, "No outputs from engine")

        output = request_output.outputs[0]

        # Calculate delta text (new text since last chunk)
        delta_text = output.text[len(previous_text) :]

        chunk = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self.config.served_model_name or self.config.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta_text,
                    },
                    "finish_reason": self._normalize_finish_reason(output.finish_reason)
                    if output.finish_reason
                    else None,
                }
            ],
        }

        # Add usage on final chunk
        if output.finish_reason:
            chunk["usage"] = self._build_completion_usage(request_output)

        return chunk
