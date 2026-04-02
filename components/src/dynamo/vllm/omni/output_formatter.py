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


class AudioFormatter:
    """Formats audio multimodal_output → NvAudioSpeechResponse."""

    def __init__(
        self, model_name: str, media_fs: Any, media_http_url: Optional[str]
    ) -> None:
        from dynamo.common.protocols.audio_protocol import AudioData

        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._AudioData = AudioData

    async def format(
        self, stage_output: Any, request_id: str, **ctx: Any
    ) -> Dict[str, Any] | None:
        mm_output = (
            stage_output.multimodal_output
            if hasattr(stage_output, "multimodal_output")
            else stage_output
        )
        if not mm_output:
            return self._error_response(request_id, "No audio generated")

        response_format = ctx.get("response_format")
        speed = ctx.get("speed", 1.0)

        try:
            start_time = time.time()
            audio_np, sample_rate = self._extract_audio_tensor(mm_output)

            encode_fmt = (
                "wav"
                if response_format in (None, "url", "b64_json")
                else response_format
            )
            assert encode_fmt is not None
            audio_bytes, media_type = await asyncio.to_thread(
                self._encode_audio, audio_np, sample_rate, encode_fmt, speed
            )

            logger.info(
                "Audio encoded for request %s: %d samples, sr=%d, %d bytes %s",
                request_id,
                len(audio_np),
                sample_rate,
                len(audio_bytes),
                encode_fmt,
            )

            if response_format == "url":
                from dynamo.common.storage import upload_to_fs

                ext = encode_fmt if encode_fmt != "opus" else "ogg"
                url = await upload_to_fs(
                    self._media_fs,
                    f"audios/{request_id}/{uuid.uuid4()}.{ext}",
                    audio_bytes,
                    self._media_http_url,
                )
                audio_data_obj = self._AudioData(url=url)
            else:
                audio_data_obj = self._AudioData(
                    b64_json=base64.b64encode(audio_bytes).decode()
                )

            from dynamo.common.protocols.audio_protocol import NvAudioSpeechResponse

            return NvAudioSpeechResponse(
                id=request_id,
                object="audio.speech",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[audio_data_obj],
                inference_time_s=time.time() - start_time,
            ).model_dump()

        except Exception as e:
            logger.error("Failed to process audio for request %s: %s", request_id, e)
            return self._error_response(request_id, str(e))

    def _extract_audio_tensor(self, mm_output: Dict[str, Any]) -> tuple:
        import numpy as np
        import torch

        audio_key = "audio" if "audio" in mm_output else "model_outputs"
        audio_val = mm_output.get(audio_key)
        if audio_val is None:
            raise ValueError(
                f"No audio data in multimodal_output. Keys: {list(mm_output.keys())}"
            )

        if isinstance(audio_val, list):
            audio_val = torch.cat(audio_val, dim=-1)

        if hasattr(audio_val, "float"):
            audio_np = audio_val.float().detach().cpu().numpy()
        elif isinstance(audio_val, np.ndarray):
            audio_np = audio_val.astype(np.float32)
        else:
            audio_np = np.array(audio_val, dtype=np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        sr_raw = mm_output.get("sr", 24000)
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1] if sr_raw else 24000
        sample_rate = sr_raw.item() if hasattr(sr_raw, "item") else int(sr_raw)

        return audio_np, sample_rate

    def _encode_audio(
        self, audio_np: Any, sample_rate: int, fmt: str = "wav", speed: float = 1.0
    ) -> tuple:
        import soundfile as sf

        if speed != 1.0:
            try:
                import librosa

                audio_np = librosa.effects.time_stretch(y=audio_np, rate=speed)
            except ImportError:
                logger.warning("librosa not installed, ignoring speed adjustment")

        fmt = (fmt or "wav").lower()
        format_map = {
            "wav": ("WAV", "audio/wav", {}),
            "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3": ("MP3", "audio/mpeg", {}),
            "aac": ("AAC", "audio/aac", {}),
            "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
        }

        if fmt not in format_map:
            logger.warning("Unsupported format '%s', defaulting to wav", fmt)
            fmt = "wav"

        sf_format, media_type, kwargs = format_map[fmt]

        buf = BytesIO()
        sf.write(buf, audio_np, sample_rate, format=sf_format, **kwargs)
        return buf.getvalue(), media_type

    def _error_response(self, request_id: str, error: str) -> Dict[str, Any]:
        from dynamo.common.protocols.audio_protocol import NvAudioSpeechResponse

        return NvAudioSpeechResponse(
            id=request_id,
            model=self._model_name,
            status="failed",
            created=int(time.time()),
            error=error,
        ).model_dump()


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


class OutputFormatter:
    """Dispatches raw engine output to modality-specific formatters.

    Shared by OmniHandler (aggregated) and any future disaggregated router.
    """

    def __init__(
        self,
        model_name: str,
        media_fs: Any = None,
        media_http_url: Optional[str] = None,
        default_fps: int = 16,
    ) -> None:
        self._formatters: Dict[str, Any] = {
            "text": TextFormatter(model_name),
            "image": DiffusionFormatter(
                model_name, media_fs, media_http_url, default_fps
            ),
            "audio": AudioFormatter(model_name, media_fs, media_http_url),
        }

    async def format(
        self,
        stage_output: Any,
        request_id: str,
        *,
        request_type: Any = None,
        **ctx: Any,
    ) -> Dict[str, Any] | None:
        fmt_type = getattr(stage_output, "final_output_type", None)
        formatter = self._formatters.get(fmt_type) if fmt_type else None
        if formatter is None:
            return None

        # TextFormatter is sync and takes request_output, not stage_output.
        if fmt_type == "text":
            ro = getattr(stage_output, "request_output", None)
            if not ro:
                return None
            return formatter.format(
                ro, request_id, previous_text=ctx.get("previous_text", "")
            )

        return await formatter.format(
            stage_output, request_id, request_type=request_type, **ctx
        )
