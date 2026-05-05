# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional, Union, cast

import PIL.Image
from fsspec.implementations.dirfs import DirFileSystem
from vllm.sampling_params import SamplingParams
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo._core import Context
from dynamo.common.multimodal import ImageLoader
from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.protocols.image_protocol import ImageNvExt, NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.common.utils.video_utils import compute_num_frames, parse_size
from dynamo.llm.exceptions import EngineShutdown
from dynamo.vllm.omni.audio_handler import AudioGenerationHandler
from dynamo.vllm.omni.base_handler import BaseOmniHandler
from dynamo.vllm.omni.output_formatter import OutputFormatter

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
        output_format: The output format to use for the response.
            None means use the default for the request type.
    """

    prompt: Union[OmniTextPrompt, Dict[str, Any]]
    sampling_params_list: list | None = None
    request_type: RequestType = RequestType.CHAT_COMPLETION
    fps: int = 0
    speed: float = 1.0
    response_format: str | None = None
    output_format: str | None = None


class OmniHandler(BaseOmniHandler):
    """Unified handler for multi-stage pipelines using vLLM-Omni.

    Handles text-to-text, text-to-image, text-to-video, and text-to-audio generation.
    Audio/TTS logic is delegated to AudioGenerationHandler via composition.
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
        self._image_loader = ImageLoader()

        self.output_formatter = OutputFormatter(
            model_name=config.served_model_name or config.model,
            media_fs=media_output_fs,
            media_http_url=media_output_http_url,
            default_fps=getattr(config, "default_video_fps", 16),
        )

        # Audio/TTS handler — composition, not inheritance.
        self.audio = AudioGenerationHandler(
            config=config,
            engine_client=self.engine_client,
            media_output_fs=media_output_fs,
            media_output_http_url=media_output_http_url,
        )

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate outputs via the unified OpenAI mode.

        Args:
            request: Raw request dictionary from the Rust frontend.
            context: Dynamo context for request tracking.

        Yields:
            Response dictionaries.
        """
        request_id = context.id()
        assert request_id is not None, "Request ID is required"
        logger.debug(f"Omni Request ID: {request_id}")

        async for chunk in self._generate_openai_mode(request, context, request_id):
            yield chunk

    async def _generate_openai_mode(
        self, request: Dict[str, Any], context: Context, request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Single generation path for all request protocols and output modalities."""

        parsed_request_raw, request_type = parse_request_type(
            request, self.config.output_modalities
        )
        parsed_request = cast(
            Union[NvCreateImageRequest, NvCreateVideoRequest, Dict[str, Any]],
            parsed_request_raw,
        )

        # Pre-load input image for I2V requests (async I/O before sync build)
        image = None
        if (
            request_type == RequestType.VIDEO_GENERATION
            and isinstance(parsed_request, NvCreateVideoRequest)
            and parsed_request.input_reference
        ):
            try:
                image = await self._image_loader.load_image(
                    parsed_request.input_reference
                )
            except Exception as e:
                logger.warning("Failed to load I2V input_reference: %s", e)
                yield {
                    "id": request_id,
                    "object": "video",
                    "model": self.config.model,
                    "status": "failed",
                    "error": f"Failed to load input_reference: {e}",
                }
                return

        try:
            inputs = await self.build_engine_inputs(
                parsed_request, request_type, image=image
            )
        except (ValueError, NotImplementedError) as e:
            logger.error(f"Invalid request {request_id}: {e}")
            yield self._error_chunk(request_id, str(e), request_type)
            return

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
                    chunk = await self.output_formatter.format(
                        stage_output,
                        request_id,
                        request_type=inputs.request_type,
                        fps=inputs.fps,
                        response_format=inputs.response_format,
                        output_format=inputs.output_format,
                        previous_text=previous_text,
                        speed=inputs.speed,
                    )
                    if chunk:
                        # Track text state for streaming delta
                        if (
                            stage_output.final_output_type == "text"
                            and stage_output.request_output
                        ):
                            previous_text = stage_output.request_output.outputs[0].text
                        yield chunk

            except EngineShutdown:
                logger.info(f"Request {request_id} aborted due to shutdown")
                raise
            except Exception as e:
                logger.error(f"Error during generation for request {request_id}: {e}")
                yield self._error_chunk(request_id, str(e), inputs.request_type)

    async def build_engine_inputs(
        self,
        parsed_request: Union[
            NvCreateImageRequest,
            NvCreateVideoRequest,
            NvCreateAudioSpeechRequest,
            Dict[str, Any],
        ],
        request_type: RequestType,
        image: PIL.Image.Image | None = None,
    ) -> EngineInputs:
        """Convert a parsed request into AsyncOmni engine inputs.

        Args:
            parsed_request: Output from parse_request_type -- a Pydantic model
                for image/video/audio requests, or a raw dict for chat completions.
            request_type: The RequestType determined by parse_request_type.
            image: Pre-loaded PIL Image for I2V requests (from input_reference).

        Returns:
            EngineInputs ready for engine_client.generate().
        """
        if request_type == RequestType.CHAT_COMPLETION:
            assert isinstance(parsed_request, dict)
            return self._engine_inputs_from_chat(parsed_request)
        elif request_type == RequestType.IMAGE_GENERATION:
            assert isinstance(parsed_request, NvCreateImageRequest)
            return self._engine_inputs_from_image(parsed_request)
        elif request_type == RequestType.VIDEO_GENERATION:
            assert isinstance(parsed_request, NvCreateVideoRequest)
            return self._engine_inputs_from_video(parsed_request, image=image)
        elif request_type == RequestType.AUDIO_GENERATION:
            assert isinstance(parsed_request, NvCreateAudioSpeechRequest)
            return await self.audio.build_engine_inputs(parsed_request)

        raise ValueError(f"Unknown request type: {request_type}")

    def _engine_inputs_from_chat(self, request: Dict[str, Any]) -> EngineInputs:
        """Build engine inputs from a chat completions request dict."""

        text_prompt = self._extract_text_prompt(request)
        if text_prompt is None:
            raise ValueError("No user message found in chat completion request")

        prompt = OmniTextPrompt(prompt=text_prompt)

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=None,
            request_type=RequestType.CHAT_COMPLETION,
            fps=0,
        )

    @staticmethod
    def _update_if_not_none(object: Any, key: str, val: Any) -> None:
        if val is not None:
            setattr(object, key, val)

    def _build_sampling_params_list(
        self, diffusion_sp: OmniDiffusionSamplingParams
    ) -> list:
        # This is in sync with how vllm-omni builds sampling params currently.
        defaults = list(self.engine_client.default_sampling_params_list or [])
        result = []
        for i, default in enumerate(defaults):
            stage_type = self.engine_client.engine.get_stage_metadata(i).get(
                "stage_type", "llm"
            )
            if stage_type == "diffusion":
                result.append(diffusion_sp)
            else:
                result.append(
                    default.clone() if hasattr(default, "clone") else SamplingParams()
                )
        return result if result else [diffusion_sp]

    def _engine_inputs_from_image(self, req: NvCreateImageRequest) -> EngineInputs:
        """Build engine inputs from an NvCreateImageRequest."""
        width, height = parse_size(req.size, default_w=1024, default_h=1024)
        nvext = req.nvext or ImageNvExt()

        prompt = OmniTextPrompt(prompt=req.prompt)
        if nvext and nvext.negative_prompt is not None:
            prompt.negative_prompt = nvext.negative_prompt

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
        )

        # TODO: Apply LoRA Request params here and move to shared utilities for disaggregated stages to use as well.

        self._update_if_not_none(sp, "num_outputs_per_prompt", req.n)

        self._update_if_not_none(sp, "num_inference_steps", nvext.num_inference_steps)
        self._update_if_not_none(sp, "guidance_scale", nvext.guidance_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        sp.seed = (
            nvext.seed if nvext.seed is not None else random.randint(0, 2**32 - 1)
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=self._build_sampling_params_list(sp),
            request_type=RequestType.IMAGE_GENERATION,
            response_format=req.response_format,
        )

    def _engine_inputs_from_video(
        self,
        req: NvCreateVideoRequest,
        image: PIL.Image.Image | None = None,
    ) -> EngineInputs:
        """Build engine inputs from an NvCreateVideoRequest.

        Args:
            req: Parsed video generation request.
            image: Pre-loaded PIL Image for I2V. When provided, the image is
                attached to the prompt via ``multi_modal_data`` so vllm-omni's
                I2V pipeline pre-process can use it.
        """
        width, height = parse_size(req.size)
        nvext = req.nvext or VideoNvExt()

        num_frames = compute_num_frames(
            num_frames=nvext.num_frames,
            seconds=req.seconds,
            fps=nvext.fps,
            default_fps=DEFAULT_VIDEO_FPS,
        )
        fps = nvext.fps if nvext.fps is not None else DEFAULT_VIDEO_FPS

        prompt = OmniTextPrompt(prompt=req.prompt)
        if nvext.negative_prompt is not None:
            prompt.negative_prompt = nvext.negative_prompt

        if image is not None:
            prompt["multi_modal_data"] = {"image": image}
            logger.info(
                "I2V: attached image (%dx%d) to multi_modal_data",
                image.size[0],
                image.size[1],
            )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self._update_if_not_none(sp, "num_inference_steps", nvext.num_inference_steps)
        self._update_if_not_none(sp, "guidance_scale", nvext.guidance_scale)
        sp.seed = (
            nvext.seed if nvext.seed is not None else random.randint(0, 2**32 - 1)
        )
        self._update_if_not_none(sp, "boundary_ratio", nvext.boundary_ratio)
        self._update_if_not_none(sp, "guidance_scale_2", nvext.guidance_scale_2)
        self._update_if_not_none(sp, "fps", fps)

        logger.info(
            f"Video diffusion request: prompt='{req.prompt[:50]}...', "
            f"size={width}x{height}, frames={num_frames}, fps={fps}"
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=self._build_sampling_params_list(sp),
            request_type=RequestType.VIDEO_GENERATION,
            fps=fps,
        )
