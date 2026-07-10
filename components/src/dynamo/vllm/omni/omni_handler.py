# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, Optional, Union, cast

import PIL.Image
from fsspec.implementations.dirfs import DirFileSystem
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from dynamo._core import Context
from dynamo.common.multimodal import ImageLoader
from dynamo.common.protocols.audio_protocol import NvCreateAudioSpeechRequest
from dynamo.common.protocols.image_protocol import ImageNvExt, NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
from dynamo.common.rl import RLAdminValidationError, require_lora_load_request
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.common.utils.video_utils import compute_num_frames, parse_size
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.vllm.handlers import LoRAInfo, get_lora_manager
from dynamo.vllm.omni.audio_handler import AudioGenerationHandler
from dynamo.vllm.omni.base_handler import BaseOmniHandler
from dynamo.vllm.omni.output_formatter import OutputFormatter
from dynamo.vllm.omni.utils import (
    build_image_generation_prompt,
    image_generation_negative_prompt_from_request,
    image_generation_sampling_overrides,
    image_generation_size_from_request,
)

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
    lora_request: LoRARequest | None = None


class OmniHandler(BaseOmniHandler):
    """Unified handler for multi-stage pipelines using vLLM-Omni.

    Handles text-to-image, text-to-video, image-to-video, and text-to-audio generation.
    Audio/TTS logic is delegated to AudioGenerationHandler via composition.
    """

    @staticmethod
    def _apply_lora_to_sampling_params(
        sampling_params_list: list | None,
        lora_request: LoRARequest | None,
    ) -> None:
        """Attach LoRA to diffusion sampling params in-place.

        AsyncOmni diffusion stages consume LoRA from OmniDiffusionSamplingParams.
        The top-level generate(lora_request=...) argument is not sufficient for
        diffusion-only paths.
        """
        if lora_request is None or sampling_params_list is None:
            return

        for sp in sampling_params_list:
            if isinstance(sp, OmniDiffusionSamplingParams):
                sp.lora_request = lora_request

    def _resolve_and_apply_lora(
        self,
        model_name: str | None,
        sampling_params_list: list | None,
    ) -> LoRARequest | None:
        lora_request = super()._resolve_lora_request(model_name)
        self._apply_lora_to_sampling_params(sampling_params_list, lora_request)
        return lora_request

    @staticmethod
    def _extract_lora_name_from_request(request: Any) -> str | None:
        """Best-effort LoRA name extraction for admin unload compatibility.

        Accepts multiple request shapes used by compatibility aliases and
        engine-update forwarding layers.
        """
        if not isinstance(request, dict):
            return None

        # Canonical body shape.
        lora_name = request.get("lora_name")
        if isinstance(lora_name, str) and lora_name:
            return lora_name

        # Compatibility shapes that may appear in alias forwarding.
        for key in ("name", "adapter_name", "model"):
            value = request.get(key)
            if isinstance(value, str) and value:
                return value

        return None

    @staticmethod
    def _local_path_from_uri(uri: str) -> str:
        if uri.startswith("file://"):
            return uri[len("file://") :]
        return uri

    def __init__(
        self,
        runtime,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
        media_output_fs: Optional[DirFileSystem] = None,
        media_output_http_url: Optional[str] = None,
        generate_endpoint=None,
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
        self.generate_endpoint = generate_endpoint

        # Keep parity with BaseWorkerHandler LoRA resolver contract.
        self._served_model_name = config.served_model_name or config.model
        self.engine_args = SimpleNamespace(model=config.model)

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

    def _lora_enabled(self) -> bool:
        return bool(getattr(self.config, "enable_lora", False))

    async def load_lora(self, request=None):
        try:
            lora_name, lora_uri = require_lora_load_request(request)
        except RLAdminValidationError as e:
            yield {"status": "error", "message": str(e)}
            return
        if lora_name in (self.config.served_model_name, self.config.model):
            yield {
                "status": "error",
                "message": (
                    "LoRA name must not match base model names "
                    f"('{self.config.served_model_name}' or '{self.config.model}')."
                ),
                "lora_name": lora_name,
            }
            return

        lock = self._get_lora_lock(lora_name)
        async with lock:
            if lora_name in self.loaded_loras:
                lora_id = self.loaded_loras[lora_name].id
                yield {
                    "status": "success",
                    "message": f"LoRA adapter '{lora_name}' already loaded",
                    "lora_name": lora_name,
                    "lora_id": lora_id,
                }
                return

            try:
                if lora_uri.startswith("file://"):
                    lora_path = self._local_path_from_uri(lora_uri)
                    if not os.path.exists(lora_path):
                        yield {
                            "status": "error",
                            "message": f"Local LoRA path does not exist: {lora_path}",
                        }
                        return
                else:
                    lora_manager = get_lora_manager()
                    if lora_manager is None:
                        yield {
                            "status": "error",
                            "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true for URI-based LoRA loading.",
                        }
                        return
                    download_result = await lora_manager.download_lora(lora_uri)
                    if download_result.get("status") != "success":
                        yield {
                            "status": "error",
                            "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                        }
                        return
                    lora_path = download_result["local_path"]

                lora_id = lora_name_to_id(lora_name)
                add_ok = await self.engine_client.add_lora(
                    LoRARequest(
                        lora_name=lora_name,
                        lora_int_id=lora_id,
                        lora_path=lora_path,
                    )
                )
                if not add_ok:
                    yield {
                        "status": "error",
                        "message": (
                            "Engine rejected LoRA adapter. "
                            "Adapter may be incompatible with this base model."
                        ),
                        "lora_name": lora_name,
                    }
                    return
                self.loaded_loras[lora_name] = LoRAInfo(id=lora_id, path=lora_path)
                logger.info("LoRA '%s' loaded and available for use", lora_name)

                if self.generate_endpoint is not None:
                    try:
                        runtime_config = ModelRuntimeConfig()
                        model_type = get_output_modalities(
                            self.config.output_modalities,
                            self.config.model,
                        )
                        if model_type is None:
                            model_type = ModelType.Images

                        await register_model(
                            model_input=ModelInput.Text,
                            model_type=model_type,
                            endpoint=self.generate_endpoint,
                            model_path=self.config.model,
                            kv_cache_block_size=self.config.engine_args.block_size,
                            runtime_config=runtime_config,
                            user_data={"lora_adapter": True, "lora_id": lora_id},
                            lora_name=lora_name,
                            base_model_path=self.config.model,
                            worker_type=WorkerType.Aggregated,
                            needs=[],
                        )
                        logger.info(
                            "Registered LoRA '%s' on endpoint %s",
                            lora_name,
                            self.generate_endpoint,
                        )
                    except Exception as reg_err:
                        logger.exception(
                            "Failed to register LoRA '%s' in discovery; rolling back",
                            lora_name,
                        )
                        try:
                            await self.engine_client.remove_lora(lora_id)
                            self.loaded_loras.pop(lora_name, None)
                        except Exception:
                            logger.exception(
                                "Failed to rollback LoRA '%s' after discovery registration failure",
                                lora_name,
                            )
                        yield {
                            "status": "error",
                            "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {reg_err!s}",
                            "lora_name": lora_name,
                        }
                        return

                yield {
                    "status": "success",
                    "message": f"LoRA adapter '{lora_name}' loaded successfully",
                    "lora_name": lora_name,
                    "lora_id": lora_id,
                }
            except Exception as e:
                logger.exception("Failed to load LoRA adapter: %s", e)
                yield {"status": "error", "message": str(e)}
            finally:
                self._lora_state.cleanup_lock_if_not_loaded(lora_name, lock)

    async def unload_lora(self, request=None):
        lora_name = self._extract_lora_name_from_request(request)
        if not lora_name:
            yield {"status": "error", "message": "'lora_name' is required in request"}
            return

        lock = self._get_lora_lock(lora_name)
        async with lock:
            lora = self.loaded_loras.get(lora_name)
            if lora is None:
                yield {
                    "status": "error",
                    "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.loaded_loras.keys())}",
                }
                return

            try:
                lora_id = lora.id
                try:
                    await self.engine_client.remove_lora(lora_id)
                except Exception as remove_err:
                    logger.exception(
                        "Failed to remove LoRA '%s' from engine",
                        lora_name,
                    )
                    yield {
                        "status": "error",
                        "message": f"Failed to unload LoRA '{lora_name}' from engine: {remove_err!s}",
                        "lora_name": lora_name,
                    }
                    return

                self.loaded_loras.pop(lora_name, None)

                if self.generate_endpoint is not None:
                    try:
                        await unregister_model(
                            endpoint=self.generate_endpoint,
                            lora_name=lora_name,
                        )
                    except Exception as unreg_err:
                        logger.exception(
                            "Failed to unregister LoRA '%s' from discovery after engine removal; rolling back",
                            lora_name,
                        )
                        try:
                            await self.engine_client.add_lora(
                                LoRARequest(
                                    lora_name=lora_name,
                                    lora_int_id=lora_id,
                                    lora_path=lora.path,
                                )
                            )
                            self.loaded_loras[lora_name] = lora
                        except Exception as rollback_err:
                            logger.exception(
                                "Failed to rollback LoRA '%s' in engine after discovery unregistration failure",
                                lora_name,
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"Failed to unregister LoRA '{lora_name}' from discovery registry: {unreg_err!s}. "
                                    f"Rollback also failed: {rollback_err!s}"
                                ),
                                "lora_name": lora_name,
                            }
                            return

                        yield {
                            "status": "error",
                            "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {unreg_err!s}",
                            "lora_name": lora_name,
                        }
                        return

                logger.info("LoRA '%s' unloaded", lora_name)

                yield {
                    "status": "success",
                    "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                    "lora_name": lora_name,
                    "lora_id": lora_id,
                }
                    "lora_id": lora_id,
                }
            except Exception as e:
                logger.exception("Failed to unload LoRA adapter: %s", e)
                yield {"status": "error", "message": str(e)}
            finally:
                self._lora_state.cleanup_lock_if_not_loaded(lora_name, lock)

    async def list_loras(self, request=None):
        async for response in super().list_loras(request):
            yield response

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
        # Keep top-level LoRA only for paths that do not carry stage params.
        if inputs.lora_request is not None and inputs.sampling_params_list is None:
            generate_kwargs["lora_request"] = inputs.lora_request

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

        output_modalities = {
            str(modality).lower() for modality in (self.config.output_modalities or [])
        }
        if "image" in output_modalities:
            width, height = image_generation_size_from_request(request)
            prompt = build_image_generation_prompt(
                text_prompt,
                height,
                width,
                negative_prompt=image_generation_negative_prompt_from_request(request),
                multi_modal_data=request.get("multi_modal_data"),
            )
            sp = OmniDiffusionSamplingParams(height=height, width=width)
            for arg, value in image_generation_sampling_overrides(
                request, height, width
            ).items():
                if hasattr(sp, arg):
                    setattr(sp, arg, value)
            sampling_params_list = self._build_sampling_params_list(sp)
        else:
            prompt = OmniTextPrompt(prompt=text_prompt)
            sampling_params_list = None

        lora_request = self._resolve_and_apply_lora(
            request.get("model"),
            sampling_params_list,
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            request_type=RequestType.CHAT_COMPLETION,
            fps=0,
            lora_request=lora_request,
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
            metadata = self.engine_client.engine.get_stage_metadata(i)
            stage_type = getattr(metadata, "stage_type", "llm")
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

        prompt = build_image_generation_prompt(
            req.prompt,
            height,
            width,
            negative_prompt=nvext.negative_prompt,
        )

        sp = OmniDiffusionSamplingParams(
            height=height,
            width=width,
        )
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

        sampling_params_list = self._build_sampling_params_list(sp)
        lora_request = self._resolve_and_apply_lora(req.model, sampling_params_list)

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            request_type=RequestType.IMAGE_GENERATION,
            response_format=req.response_format,
            lora_request=lora_request,
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

        sampling_params_list = self._build_sampling_params_list(sp)
        lora_request = self._resolve_and_apply_lora(req.model, sampling_params_list)

        logger.info(
            "Video diffusion request: prompt='%s...', size=%sx%s, frames=%s, fps=%s",
            req.prompt[:50],
            width,
            height,
            num_frames,
            fps,
        )

        return EngineInputs(
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            request_type=RequestType.VIDEO_GENERATION,
            fps=fps,
            lora_request=lora_request,
        )
