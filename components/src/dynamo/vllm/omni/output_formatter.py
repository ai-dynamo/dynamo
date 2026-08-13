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
import struct
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

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


@dataclass
class AudioStreamState:
    """Request-local state for de-duplicating incremental audio output."""

    emitted_chunks: int = 0
    first_chunk: bool = True


@dataclass
class AudioAggregateState:
    """Request-local raw audio accumulated for one final encode."""

    chunks: list[np.ndarray] = field(default_factory=list)
    sample_rate: int | None = None
    emitted_chunks: int = 0


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
        if is_empty_payload(images):
            return None

        if request_type == RequestType.VIDEO_GENERATION:
            return await self._encode_video(
                images,
                request_id,
                fps=ctx.get("fps", self._default_fps),
                response_format=ctx.get("response_format"),
                output_format=ctx.get("output_format"),
            )
        return await self._encode_image(
            images,
            request_id,
            request_type=request_type,
            response_format=ctx.get("response_format"),
        )

    async def _encode_video(
        self,
        images: list,
        request_id: str,
        fps: int,
        response_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        output_format = output_format or "mp4"
        response_format = response_format or "url"
        if response_format not in ("url", "b64_json"):
            raise ValueError(
                f"Unsupported response_format: {response_format!r}; expected 'url' or 'b64_json'"
            )
        if output_format != "mp4":
            raise ValueError(
                f"Unsupported output_format: {output_format!r}; only 'mp4' is supported"
            )
        try:
            start_time = time.time()
            # Encode with the in-tree VP9 (libvpx-vp9) encoder rather
            # than diffusers.export_to_video, whose imageio backend defaults to the
            # H.264 codec that the codec-compliant image no longer ships (it would
            # fail with "No valid H.264 encoder was found"). encode_to_video_bytes
            # is the same shared helper the TRT-LLM video handler uses; VP9-in-mp4
            # is valid and decodes with our VP8/VP9 allowlist.
            frames_np = frames_to_numpy(normalize_video_frames(images))
            video_bytes = await asyncio.to_thread(
                encode_to_video_bytes, frames_np, fps=fps, output_format=output_format
            )

            if response_format == "b64_json":
                video_data = VideoData(
                    output_format=output_format,
                    b64_json=base64.b64encode(video_bytes).decode("utf-8"),
                )
            else:
                video_url = await upload_to_fs(
                    self._media_fs,
                    f"videos/{request_id}.{output_format}",
                    video_bytes,
                    self._media_http_url,
                )
                video_data = VideoData(output_format=output_format, url=video_url)

            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[video_data],
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
        stream_state = ctx.get("audio_stream_state")
        aggregate_state = ctx.get("audio_aggregate_state")

        try:
            start_time = time.time()
            audio_np, sample_rate = self._extract_audio_tensor(
                mm_output,
                chunk_state=(
                    stream_state if stream_state is not None else aggregate_state
                ),
            )
            if audio_np.size == 0:
                return None

            if aggregate_state is not None:
                self._append_audio_chunk(aggregate_state, audio_np, sample_rate)
                return None

            encode_fmt = ("wav" if output_format is None else output_format).lower()
            assert encode_fmt is not None
            if stream_state is not None:
                audio_bytes, media_type = await asyncio.to_thread(
                    self._encode_audio_chunk,
                    audio_np,
                    sample_rate,
                    encode_fmt,
                    stream_state,
                )
            else:
                audio_bytes, media_type = await asyncio.to_thread(
                    self._encode_audio, audio_np, sample_rate, encode_fmt, speed
                )

            logger.info(
                "Audio encoded for request %s: %d samples, sr=%d, %d bytes %s",
                request_id,
                audio_np.shape[-1],
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

    async def finish_aggregate(
        self, request_id: str, aggregate_state: AudioAggregateState, **ctx: Any
    ) -> Dict[str, Any]:
        """Encode all buffered raw chunks as one complete audio file."""
        if not aggregate_state.chunks or aggregate_state.sample_rate is None:
            return self._error_response(request_id, "No audio generated")

        audio_np = np.concatenate(aggregate_state.chunks, axis=-1)
        return await self.format(
            {"audio": audio_np, "sr": aggregate_state.sample_rate},
            request_id,
            **ctx,
        )

    def _append_audio_chunk(
        self,
        state: AudioAggregateState,
        audio_np: np.ndarray,
        sample_rate: int,
    ) -> None:
        if audio_np.ndim == 3:
            if audio_np.shape[0] != 1:
                raise ValueError(
                    f"Expected one audio batch, got shape {audio_np.shape}"
                )
            audio_np = audio_np[0]
        if audio_np.ndim == 1:
            num_channels = 1
        elif audio_np.ndim == 2 and audio_np.shape[0] in (1, 2):
            num_channels = int(audio_np.shape[0])
        else:
            raise ValueError(
                f"Expected mono or stereo audio, got shape {audio_np.shape}"
            )

        if state.sample_rate is not None and state.sample_rate != sample_rate:
            raise ValueError(
                f"Audio sample rate changed from {state.sample_rate} to {sample_rate}"
            )
        if state.chunks:
            first_chunk = state.chunks[0]
            first_num_channels = 1 if first_chunk.ndim == 1 else first_chunk.shape[0]
            if first_num_channels != num_channels:
                raise ValueError("Audio channel count changed while generating")

        state.sample_rate = sample_rate
        state.chunks.append(audio_np)

    def _extract_audio_tensor(
        self,
        mm_output: Dict[str, Any],
        *,
        chunk_state: AudioStreamState | AudioAggregateState | None = None,
    ) -> tuple[np.ndarray, int]:
        audio_key = "audio" if "audio" in mm_output else "model_outputs"
        audio_val = mm_output.get(audio_key)
        if audio_val is None:
            raise ValueError(
                f"No audio data in multimodal_output. Keys: {list(mm_output.keys())}"
            )

        if isinstance(audio_val, list):
            if chunk_state is not None:
                new_audio = audio_val[chunk_state.emitted_chunks :]
                chunk_state.emitted_chunks = len(audio_val)
                audio_val = new_audio
                if not audio_val:
                    return np.empty(0, dtype=np.float32), self._sample_rate(mm_output)
            audio_val = torch.cat(audio_val, dim=-1)

        if hasattr(audio_val, "float"):
            audio_np = audio_val.float().detach().cpu().numpy()
        elif isinstance(audio_val, np.ndarray):
            audio_np = audio_val.astype(np.float32)
        else:
            audio_np = np.array(audio_val, dtype=np.float32)

        return audio_np, self._sample_rate(mm_output)

    @staticmethod
    def _sample_rate(mm_output: Dict[str, Any]) -> int:
        sr_raw = mm_output.get("sr", 24000)
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1] if sr_raw else 24000
        return sr_raw.item() if hasattr(sr_raw, "item") else int(sr_raw)

    def _encode_audio_chunk(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        fmt: str,
        stream_state: AudioStreamState,
    ) -> tuple[bytes, str]:
        _, num_channels = self._normalize_audio_layout(audio_np)
        pcm_bytes, _ = self._encode_audio(audio_np, sample_rate, "pcm")
        if fmt == "wav" and stream_state.first_chunk:
            pcm_bytes = self._wav_stream_header(sample_rate, num_channels) + pcm_bytes
        stream_state.first_chunk = False
        return pcm_bytes, "audio/wav" if fmt == "wav" else "audio/pcm"

    @staticmethod
    def _normalize_audio_layout(audio_np: np.ndarray) -> tuple[np.ndarray, int]:
        """Convert vLLM-Omni channel-first audio to soundfile's layout."""
        if audio_np.ndim == 3:
            if audio_np.shape[0] != 1:
                raise ValueError(
                    f"Expected one audio batch, got shape {audio_np.shape}"
                )
            audio_np = audio_np[0]

        if audio_np.ndim == 1:
            return audio_np, 1

        if audio_np.ndim == 2:
            num_channels = int(audio_np.shape[0])
            if num_channels not in (1, 2):
                raise ValueError(
                    f"Expected mono or stereo audio, got shape {audio_np.shape}"
                )
            return audio_np.T, num_channels

        raise ValueError(f"Unexpected audio shape {audio_np.shape}")

    @staticmethod
    def _wav_stream_header(
        sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16
    ) -> bytes:
        """Build a PCM WAV header whose payload length is not known yet."""
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        placeholder_size = 0xFFFFFFFF

        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            placeholder_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            placeholder_size,
        )

    def _encode_audio(
        self, audio_np: Any, sample_rate: int, fmt: str = "wav", speed: float = 1.0
    ) -> tuple:
        audio_np, _ = self._normalize_audio_layout(audio_np)
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

    async def finish_audio(
        self,
        request_id: str,
        aggregate_state: AudioAggregateState,
        **ctx: Any,
    ) -> Dict[str, Any]:
        return await self._formatters["audio"].finish_aggregate(
            request_id, aggregate_state, **ctx
        )
