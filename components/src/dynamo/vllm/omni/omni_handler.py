# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Union

from pydantic import BaseModel
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.protocols.image_protocol import ImageNvExt, NvCreateImageRequest
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import (
    compute_num_frames,
    encode_to_mp4,
    frames_to_numpy,
    parse_video_size,
)
from dynamo.vllm.omni.base_handler import BaseOmniHandler

logger = logging.getLogger(__name__)

# Default values for video generation parameters
DEFAULT_VIDEO_FPS = 16
DEFAULT_VIDEO_OUTPUT_DIR = "/tmp/dynamo_videos"  # noqa: S108


@dataclass
class EngineInputs:
    """Parsed engine inputs ready for AsyncOmni.generate().

    Attributes:
        prompt: OmniTextPrompt dict for the engine.
        sampling_params_list: Per-stage sampling parameters, or None for defaults.
        request_type: The resolved request type (may differ from the initial parse
            when a chat completion request carries video params).
        fps: Frames per second, only meaningful for video requests.
    """

    prompt: OmniTextPrompt
    sampling_params_list: list | None = None
    request_type: RequestType = RequestType.CHAT_COMPLETION
    fps: int = 0


class OmniHandler(BaseOmniHandler):
    """Unified handler for multi-stage pipelines using vLLM-Omni.

    Handles text-to-text, text-to-image, and text-to-video generation.
    """

    def __init__(
        self,
        runtime,
        component,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
    ):
        """Initialize the unified Omni handler.

        Args:
            runtime: Dynamo distributed runtime.
            component: Dynamo component handle.
            config: Parsed Config object from args.py.
            default_sampling_params: Default sampling parameters dict.
            shutdown_event: Optional asyncio event for graceful shutdown.
        """
        super().__init__(
            runtime=runtime,
            component=component,
            config=config,
            default_sampling_params=default_sampling_params,
            shutdown_event=shutdown_event,
        )

        # Video output configuration (from CLI args, with safe defaults)
        self.output_dir = getattr(config, "video_output_dir", DEFAULT_VIDEO_OUTPUT_DIR)
        self.default_fps = getattr(config, "default_video_fps", DEFAULT_VIDEO_FPS)

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
                        if inputs.request_type == RequestType.VIDEO_GENERATION:
                            chunk = await self._format_video_chunk(
                                stage_output.images, request_id, inputs.fps
                            )
                        else:
                            chunk = self._format_image_chunk(
                                stage_output.images, request_id
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
        parsed_request: Union[BaseModel, Dict[str, Any]],
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
            return self._engine_inputs_from_chat(parsed_request)  # type: ignore[arg-type]

        if request_type == RequestType.IMAGE_GENERATION:
            return self._engine_inputs_from_image(parsed_request)  # type: ignore[arg-type]

        if request_type == RequestType.VIDEO_GENERATION:
            return self._engine_inputs_from_video(parsed_request)  # type: ignore[arg-type]

        if request_type == RequestType.AUDIO_GENERATION:
            raise NotImplementedError("Audio generation is not yet supported")

        raise ValueError(f"Unknown request type: {request_type}")

    def _engine_inputs_from_chat(self, request: Dict[str, Any]) -> EngineInputs:
        """Build engine inputs from a chat completions request dict.

        Chat completions can carry diffusion parameters in extra_body.
        If num_frames > 1 the request is promoted to VIDEO_GENERATION so the
        output formatter knows to encode frames as MP4.

        When no diffusion overrides are present, sampling_params_list is left
        as None so AsyncOmni falls back to the YAML stage defaults.
        """
        text_prompt = self._extract_text_prompt(request)
        extra_body = self._extract_extra_body(request)
        negative_prompt = extra_body.get("negative_prompt", "")

        prompt = OmniTextPrompt(prompt=text_prompt, negative_prompt=negative_prompt)

        # Build diffusion sampling params only if the caller provides overrides.
        sp = OmniDiffusionSamplingParams()
        has_overrides = False
        for field in (
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_2",
            "true_cfg_scale",
            "seed",
            "num_outputs_per_prompt",
            "fps",
            "boundary_ratio",
            "flow_shift",
        ):
            if field in extra_body:
                setattr(sp, field, extra_body[field])
                has_overrides = True

        sampling_params_list = [sp] if has_overrides else None

        requested_num_frames = extra_body.get("num_frames", 1)
        is_video = requested_num_frames is not None and requested_num_frames > 1
        fps = extra_body.get("fps", self.default_fps)

        resolved_type = (
            RequestType.VIDEO_GENERATION if is_video else RequestType.CHAT_COMPLETION
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            request_type=resolved_type,
            fps=fps,
        )

    def _engine_inputs_from_image(self, req: NvCreateImageRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateImageRequest.

        Mirrors the parsing logic in vllm-omni api_server.py generate_images().
        The nvext block carries diffusion-specific parameters while n and size
        are top-level fields.
        """
        nvext = req.nvext or ImageNvExt()

        prompt = OmniTextPrompt(
            prompt=req.prompt,
            negative_prompt=nvext.negative_prompt or "",
        )

        sp = OmniDiffusionSamplingParams()
        has_overrides = False

        if req.n is not None:
            sp.num_outputs_per_prompt = req.n
            has_overrides = True

        if req.size is not None:
            parts = req.size.split("x")
            if len(parts) == 2:
                sp.width = int(parts[0])
                sp.height = int(parts[1])
                has_overrides = True

        if nvext.num_inference_steps is not None:
            sp.num_inference_steps = nvext.num_inference_steps
            has_overrides = True
        if nvext.guidance_scale is not None:
            sp.guidance_scale = nvext.guidance_scale
            has_overrides = True
        if nvext.seed is not None:
            sp.seed = nvext.seed
            has_overrides = True

        sampling_params_list = [sp] if has_overrides else None

        logger.info(
            f"Image generation request: prompt='{req.prompt[:50]}...', "
            f"size={req.size or 'default'}, n={req.n or 1}"
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            request_type=RequestType.IMAGE_GENERATION,
        )

    def _engine_inputs_from_video(self, req: NvCreateVideoRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateVideoRequest."""
        width, height = parse_video_size(req.size)
        num_frames = compute_num_frames(
            num_frames=req.num_frames,
            seconds=req.seconds,
            fps=req.fps,
            default_fps=self.default_fps,
        )
        fps = req.fps if req.fps is not None else self.default_fps

        prompt = OmniTextPrompt(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
        )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
        )
        if req.num_inference_steps is not None:
            sp.num_inference_steps = req.num_inference_steps
        if req.guidance_scale is not None:
            sp.guidance_scale = req.guidance_scale
        if req.seed is not None:
            sp.seed = req.seed
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

    def _format_image_chunk(
        self,
        images: list,
        request_id: str,
    ) -> Dict[str, Any] | None:
        """Format image output as OpenAI chat completion chunk with base64 data URLs."""
        from io import BytesIO

        if not images:
            return self._error_chunk(request_id, "No images generated")

        # Convert images to base64 data URLs
        data_urls = []
        for idx, img in enumerate(images):
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_base64}"
            data_urls.append(data_url)
            logger.info(f"Generated image {idx} for request {request_id}")

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

            # Convert PIL images to numpy array
            frames = frames_to_numpy(images)

            logger.info(
                f"Encoding {len(frames)} frames to MP4 for request {request_id} "
                f"(shape={frames.shape}, fps={fps})"
            )

            # Run encoding in thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            video_path = await loop.run_in_executor(
                None,
                encode_to_mp4,
                frames,
                self.output_dir,
                request_id,
                fps,
            )

            logger.info(f"Video saved to {video_path} for request {request_id}")

            inference_time = time.time() - start_time

            response = NvVideosResponse(
                id=request_id,
                object="video",
                model=self.config.served_model_name or self.config.model,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[VideoData(url=video_path)],
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
