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
import time
import uuid
from collections.abc import Mapping
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

try:
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
except (ImportError, OSError):
    mux_video_audio_bytes = None  # type: ignore[assignment]

from dynamo.common.protocols.audio_protocol import AudioData, NvAudioSpeechResponse
from dynamo.common.protocols.image_protocol import ImageData, NvImagesResponse
from dynamo.common.protocols.video_protocol import NvVideosResponse, VideoData
from dynamo.common.storage import upload_to_fs
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.output_modalities import RequestType
from dynamo.common.utils.video_utils import (
    encode_to_video_bytes,
    frames_to_numpy,
    normalize_video_frames,
)
from dynamo.vllm.handlers import build_prompt_tokens_details
from dynamo.vllm.omni.utils import is_empty_payload

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_SAMPLE_RATE = 24000


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
        model_config: Any = None,
    ) -> None:
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._default_fps = default_fps
        self._default_audio_sample_rate = (
            self._extract_audio_sample_rate_from_config(model_config)
            or DEFAULT_AUDIO_SAMPLE_RATE
        )

    async def format(
        self, stage_output: Any, request_id: str, *, request_type: Any, **ctx: Any
    ) -> Dict[str, Any] | None:
        images = (
            stage_output.images if hasattr(stage_output, "images") else stage_output
        )

        if request_type == RequestType.VIDEO_GENERATION:
            return await self._encode_video(
                images,
                request_id,
                multimodal_output=self._extract_multimodal_output(stage_output),
                fps=ctx.get("fps", self._default_fps),
                response_format=ctx.get("response_format"),
                output_format=ctx.get("output_format"),
            )
        if is_empty_payload(images):
            return None
        return await self._encode_image(
            images,
            request_id,
            request_type=request_type,
            response_format=ctx.get("response_format"),
        )

    async def _encode_video(
        self,
        images: Any,
        request_id: str,
        fps: int,
        multimodal_output: dict[str, Any] | None = None,
        response_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        requested_output_format = output_format
        output_format = "mp4"
        response_format = response_format or "url"
        if response_format not in ("url", "b64_json"):
            raise ValueError(
                f"Unsupported response_format: {response_format!r}; expected 'url' or 'b64_json'"
            )
        if requested_output_format not in (None, "mp4"):
            logger.warning(
                "Ignoring unsupported video output_format hint %r; encoding MP4",
                requested_output_format,
            )
        try:
            start_time = time.time()
            multimodal_output = multimodal_output or {}
            videos = self._split_video_outputs(images, multimodal_output)
            if not videos:
                raise ValueError("No video outputs found in generation result")
            resolved_fps = self._resolve_int_metadata(
                multimodal_output, "fps", "video"
            ) or int(fps)
            audio_sample_rate = self._resolve_audio_sample_rate(multimodal_output)
            audio_outputs = self._split_audio_outputs(
                multimodal_output.get("audio"), len(videos)
            )

            data = []
            for index, (video, audio) in enumerate(
                zip(videos, audio_outputs, strict=True)
            ):
                frames_np = self._video_to_numpy_frames(video)
                if audio is None:
                    # The codec-compliant standard image retains its existing
                    # royalty-free VP9 path for silent video models.
                    video_bytes = await asyncio.to_thread(
                        encode_to_video_bytes,
                        frames_np,
                        fps=resolved_fps,
                        output_format=output_format,
                    )
                else:
                    if mux_video_audio_bytes is None:
                        raise RuntimeError(
                            "Generated audio requires PyAV with H.264 and AAC encoders; "
                            "use the video-audio codec overlay"
                        )
                    audio_np = self._audio_to_numpy(audio)
                    video_bytes = await asyncio.to_thread(
                        mux_video_audio_bytes,
                        frames_np,
                        audio_np,
                        fps=float(resolved_fps),
                        audio_sample_rate=audio_sample_rate,
                    )

                video_data = VideoData(
                    output_format=output_format,
                    fps=resolved_fps,
                    audio_sample_rate=audio_sample_rate if audio is not None else None,
                )
                if response_format == "b64_json":
                    video_data.b64_json = base64.b64encode(video_bytes).decode("utf-8")
                else:
                    filename = (
                        f"videos/{request_id}.{output_format}"
                        if len(videos) == 1
                        else f"videos/{request_id}/{index}.{output_format}"
                    )
                    video_data.url = await upload_to_fs(
                        self._media_fs,
                        filename,
                        video_bytes,
                        self._media_http_url,
                    )
                data.append(video_data)

            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=data,
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

    @staticmethod
    def _extract_multimodal_output(stage_output: Any) -> dict[str, Any]:
        multimodal_output = getattr(stage_output, "multimodal_output", None)
        if isinstance(multimodal_output, Mapping):
            return dict(multimodal_output)

        request_output = getattr(stage_output, "request_output", None)
        if isinstance(request_output, dict):
            multimodal_output = request_output.get("multimodal_output")
            if multimodal_output is None:
                multimodal_output = request_output.get("_multimodal_output")
        elif request_output is not None:
            multimodal_output = getattr(request_output, "multimodal_output", None)
            if multimodal_output is None:
                multimodal_output = getattr(request_output, "_multimodal_output", None)
        return dict(multimodal_output) if isinstance(multimodal_output, Mapping) else {}

    @staticmethod
    def _split_video_outputs(
        images: Any, multimodal_output: dict[str, Any]
    ) -> list[Any]:
        videos = images
        if is_empty_payload(videos):
            videos = multimodal_output.get("video")
        if videos is None:
            return []
        if isinstance(videos, (np.ndarray, torch.Tensor)):
            if videos.ndim == 5:
                return [videos[index] for index in range(videos.shape[0])]
            return [videos]
        if isinstance(videos, (list, tuple)):
            videos = list(videos)
            if not videos:
                return []
            first = videos[0]
            if isinstance(first, (np.ndarray, torch.Tensor)) and first.ndim == 5:
                flattened: list[Any] = []
                for batch in videos:
                    if (
                        not isinstance(batch, (np.ndarray, torch.Tensor))
                        or batch.ndim != 5
                    ):
                        raise ValueError("Video output batches must all be 5-D")
                    flattened.extend(batch[index] for index in range(batch.shape[0]))
                return flattened
            if isinstance(first, (np.ndarray, torch.Tensor)) and first.ndim == 4:
                return videos
            if isinstance(first, list):
                return videos
            return [videos]
        return [videos]

    @staticmethod
    def _video_to_numpy_frames(video: Any) -> np.ndarray:
        """Normalize one video to uint8 ``(frames, height, width, channels)``."""
        if isinstance(video, torch.Tensor):
            video = video.detach().float().cpu().numpy()

        if isinstance(video, np.ndarray):
            if video.ndim == 3:
                video = video[None, ...]
            if video.ndim != 4:
                raise ValueError(
                    f"Expected a 4-D video tensor, got shape {video.shape}"
                )
            if video.shape[-1] in (1, 3, 4):
                frames = video
            elif video.shape[0] in (1, 3, 4):
                # vLLM-Omni's channel-first 4-D convention is CFHW. Prefer
                # that interpretation when both leading dimensions are
                # channel-sized; FCHW remains unambiguous for other frame
                # counts.
                frames = video.transpose(1, 2, 3, 0)
            elif video.shape[1] in (1, 3, 4):
                frames = video.transpose(0, 2, 3, 1)
            else:
                raise ValueError(
                    "Video tensor must use CFHW, FCHW, or FHWC channel layout, "
                    f"got shape {video.shape}"
                )
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] == 4:
                frames = frames[..., :3]
            if np.issubdtype(frames.dtype, np.floating) and frames.min(initial=0.0) < 0:
                frames = (frames + 1.0) / 2.0
            return frames_to_numpy(list(np.ascontiguousarray(frames)))

        frames = normalize_video_frames(video if isinstance(video, list) else [video])
        normalized = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().float().cpu().numpy()
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                if frame.shape[-1] in (1, 3, 4):
                    pass
                elif frame.shape[0] in (1, 3, 4):
                    frame = frame.transpose(1, 2, 0)
                else:
                    raise ValueError(
                        "Video frame must use CHW or HWC channel layout, "
                        f"got shape {frame.shape}"
                    )
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if (
                    np.issubdtype(frame.dtype, np.floating)
                    and frame.min(initial=0.0) < 0
                ):
                    frame = (frame + 1.0) / 2.0
            normalized.append(frame)
        return frames_to_numpy(normalized)

    @staticmethod
    def _split_audio_outputs(audio: Any, expected_count: int) -> list[Any | None]:
        if audio is None:
            return [None] * expected_count
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            if audio.ndim > 1 and audio.shape[0] == expected_count:
                return [audio[index] for index in range(expected_count)]
            if expected_count == 1:
                return [audio]
        if isinstance(audio, (list, tuple)):
            if len(audio) == expected_count:
                return list(audio)
            if expected_count == 1:
                return [audio]
        return [audio] + [None] * (expected_count - 1)

    @staticmethod
    def _audio_to_numpy(audio: Any) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            return audio.detach().float().cpu().numpy()
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32, copy=False)
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _resolve_int_metadata(
        multimodal_output: dict[str, Any],
        key: str,
        metadata_section: str,
        metadata_key: str | None = None,
    ) -> int | None:
        value = multimodal_output.get(key)
        if value is None:
            metadata = multimodal_output.get("metadata")
            if isinstance(metadata, Mapping):
                section = metadata.get(metadata_section)
                if isinstance(section, Mapping):
                    value = section.get(metadata_key or key)
        if value is None:
            return None
        try:
            resolved = int(value.item() if hasattr(value, "item") else value)
        except (TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

    def _resolve_audio_sample_rate(self, multimodal_output: dict[str, Any]) -> int:
        for key in ("audio_sample_rate", "sample_rate", "sampling_rate", "sr"):
            sample_rate = self._coerce_positive_int(multimodal_output.get(key))
            if sample_rate is not None:
                return sample_rate

        metadata = multimodal_output.get("metadata")
        audio_metadata = (
            metadata.get("audio") if isinstance(metadata, Mapping) else None
        )
        if isinstance(audio_metadata, Mapping):
            for key in (
                "audio_sample_rate",
                "sample_rate",
                "sampling_rate",
                "sr",
            ):
                sample_rate = self._coerce_positive_int(audio_metadata.get(key))
                if sample_rate is not None:
                    return sample_rate

        return self._default_audio_sample_rate

    @classmethod
    def _extract_audio_sample_rate_from_config(
        cls, config: Any, seen: set[int] | None = None
    ) -> int | None:
        if config is None:
            return None

        seen = seen or set()
        if id(config) in seen:
            return None
        seen.add(id(config))

        if isinstance(config, Mapping):
            values = config
        else:
            values = getattr(config, "__dict__", {})
            if not isinstance(values, Mapping):
                return None

        for key in (
            "output_sampling_rate",
            "audio_sample_rate",
            "sample_rate",
            "sampling_rate",
        ):
            sample_rate = cls._coerce_positive_int(values.get(key))
            if sample_rate is not None:
                return sample_rate

        for component_name in ("vocoder", "audio_vae"):
            component = values.get(component_name)
            sample_rate = cls._extract_audio_sample_rate_from_config(component, seen)
            if sample_rate is not None:
                return sample_rate
            if isinstance(component, Mapping):
                component_config = component.get("config")
            else:
                component_values = getattr(component, "__dict__", {})
                component_config = (
                    component_values.get("config")
                    if isinstance(component_values, Mapping)
                    else None
                )
            sample_rate = cls._extract_audio_sample_rate_from_config(
                component_config, seen
            )
            if sample_rate is not None:
                return sample_rate

        return None

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            resolved = int(value.item() if hasattr(value, "item") else value)
        except (TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

    async def _encode_image(
        self,
        images: list,
        request_id: str,
        *,
        request_type: Any,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        if is_empty_payload(images):
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
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._AudioData = AudioData  # stored for use in format()

    async def format(
        self, stage_output: Any, request_id: str, **ctx: Any
    ) -> Dict[str, Any] | None:
        mm_output = (
            stage_output.multimodal_output
            if hasattr(stage_output, "multimodal_output")
            else stage_output
        )
        if is_empty_payload(mm_output):
            return self._error_response(request_id, "No audio generated")

        response_format = ctx.get("response_format")
        output_format = ctx.get("output_format")
        speed = ctx.get("speed", 1.0)

        try:
            start_time = time.time()
            audio_np, sample_rate = self._extract_audio_tensor(mm_output)

            encode_fmt = "wav" if output_format is None else output_format
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
                ext = encode_fmt if encode_fmt != "opus" else "ogg"
                url = await upload_to_fs(
                    self._media_fs,
                    f"audios/{request_id}/{uuid.uuid4()}.{ext}",
                    audio_bytes,
                    self._media_http_url,
                )
                audio_data_obj = self._AudioData(output_format=encode_fmt, url=url)
            else:
                audio_data_obj = self._AudioData(
                    output_format=encode_fmt,
                    b64_json=base64.b64encode(audio_bytes).decode(),
                )

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
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    prompt_tokens = (
        len(prompt_token_ids)
        if prompt_token_ids is not None and not is_empty_payload(prompt_token_ids)
        else None
    )
    completion_tokens = len(request_output.outputs[0].token_ids)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": build_prompt_tokens_details(
            getattr(request_output, "num_cached_tokens", None)
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
        model_config: Any = None,
    ) -> None:
        diffusion_formatter = DiffusionFormatter(
            model_name, media_fs, media_http_url, default_fps, model_config
        )
        self._formatters: Dict[str, Any] = {
            "text": TextFormatter(model_name),
            "image": diffusion_formatter,
            "video": diffusion_formatter,
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
