# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import logging
import time
from typing import Any, AsyncGenerator, Dict

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)
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


class OmniHandler(BaseOmniHandler):
    """Unified handler for multi-stage pipelines using vLLM-Omni.

    Handles text-to-text, text-to-image, and text-to-video generation
    """

    DIFFUSION_PARAM_FIELDS: tuple[str, ...] = (
        # Dimensions
        "height",
        "width",
        "num_frames",
        # Scheduler / inference
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        # Control
        "seed",
        "num_outputs_per_prompt",
        "fps",
        # Advanced (model-specific, but user-configurable)
        "boundary_ratio",
        "flow_shift",
    )

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
            request: Request dictionary (chat completions or NvCreateVideoRequest).
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

        # Need a unified way to parse different request protocols and build engine inputs.
        # Right now we have image and text via chat completions and video via NvCreateVideoRequest.
        # So, messages is for text/image and prompt is for video.
        if "messages" in request:
            (
                prompt,
                sampling_params_list,
                is_video_request,
                fps,
            ) = self._build_inputs_from_chat(request)
        else:
            (
                prompt,
                sampling_params_list,
                is_video_request,
                fps,
            ) = self._build_inputs_from_video_request(request)

        previous_text = ""

        generate_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "request_id": request_id,
        }
        if sampling_params_list is not None:
            generate_kwargs["sampling_params_list"] = sampling_params_list

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
                        if is_video_request:
                            chunk = await self._format_video_chunk(
                                stage_output.images, request_id, fps
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

    def _build_inputs_from_chat(
        self, request: Dict[str, Any]
    ) -> tuple[OmniTextPrompt, list | None, bool, int]:
        """Build engine inputs from a chat completions request.

        Returns:
            (prompt, sampling_params_list, is_video_request, fps)
        """
        text_prompt = self._extract_text_prompt(request)
        extra_body = self._extract_extra_body(request)
        negative_prompt = extra_body.get("negative_prompt", "")

        prompt = OmniTextPrompt(prompt=text_prompt, negative_prompt=negative_prompt)

        sampling_params_list = None
        if self._has_diffusion_params(extra_body):
            sampling_params_list = [self._build_diffusion_sampling_params(extra_body)]

        requested_num_frames = extra_body.get("num_frames", 1)
        is_video_request = requested_num_frames is not None and requested_num_frames > 1
        fps = extra_body.get("fps", self.default_fps)

        return prompt, sampling_params_list, is_video_request, fps

    def _build_inputs_from_video_request(
        self, request: Dict[str, Any]
    ) -> tuple[OmniTextPrompt, list, bool, int]:
        """Build engine inputs from an NvCreateVideoRequest dict.

        Flattens the request fields into a plain dict and delegates to
        ``_build_diffusion_sampling_params`` so that both code-paths share
        the same parameter mapping logic.

        Returns:
            (prompt, sampling_params_list, is_video_request, fps)
        """
        req = NvCreateVideoRequest(**request)

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

        # Flatten into a params dict for the unified builder
        diffusion_kwargs: Dict[str, Any] = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }
        if req.num_inference_steps is not None:
            diffusion_kwargs["num_inference_steps"] = req.num_inference_steps
        if req.guidance_scale is not None:
            diffusion_kwargs["guidance_scale"] = req.guidance_scale
        if req.seed is not None:
            diffusion_kwargs["seed"] = req.seed
        if fps is not None:
            diffusion_kwargs["fps"] = fps

        diffusion_params = self._build_diffusion_sampling_params(diffusion_kwargs)

        logger.info(
            f"Video diffusion request: prompt='{req.prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, fps={fps}"
        )

        return prompt, [diffusion_params], True, fps

    @classmethod
    def _has_diffusion_params(cls, params: Dict[str, Any]) -> bool:
        """Check if *params* contains any user-facing diffusion parameters."""
        return bool(set(cls.DIFFUSION_PARAM_FIELDS) & params.keys())

    @classmethod
    def _build_diffusion_sampling_params(
        cls,
        params: Dict[str, Any],
    ) -> OmniDiffusionSamplingParams:
        """Build ``OmniDiffusionSamplingParams`` from a flat parameter dict.
        Args:
            params: Flat dict of user-facing diffusion parameters.

        Returns:
            Configured ``OmniDiffusionSamplingParams`` instance.
        """
        sp = OmniDiffusionSamplingParams()
        for key in cls.DIFFUSION_PARAM_FIELDS:
            if key in params:
                setattr(sp, key, params[key])
        return sp

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
