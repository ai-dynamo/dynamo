# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side combined HF processor for frontend-decoded multimodal media.

When ``--frontend-decoding`` is on, Dynamo's Rust frontend decodes images and
video frames and ships the raw RGB tensors to this worker over NIXL RDMA.
SGLang's own video ingestion only accepts a decodable source (URL/bytes),
never pre-decoded frames, so for any request that
carries decoded video we run the model's full HF processor here and hand
SGLang a single precomputed ``processor_output`` dict plus the expanded
``input_ids``.

Contract (verified against SGLang v0.5.13.post1):
  - ``engine.tokenizer_manager.processor`` is the raw HF AutoProcessor.
  - The processor is called once per request over all decoded images+videos so
    interleaved placeholders expand correctly: one combined dict, passed once.
  - SGLang uses the top-level ``input_ids`` verbatim for the preprocessed path
    (``base_processor.process_and_combine_mm_data`` -> ``input_ids =
    base_output.input_ids``) and builds mm items from the dict by attr name,
    so we pass the processor's expanded ``input_ids`` and the dict together.
  - Field choice: a video-only dict goes under ``video_data``; a dict that also
    carries image tensors goes under ``image_data`` (SGLang classifies items by
    attr name, not by which field carried them).
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

import dynamo.nixl_connect as nixl_connect
from dynamo.common.utils.media_nixl import read_decoded_media_via_nixl
from dynamo.common.utils.runtime import run_async

logger = logging.getLogger(__name__)

# Rust externally-tagged DecodedMediaMetadata::Video variant key.
_VIDEO_METADATA_VARIANT = "Video"


def _normalize_video_metadata(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    """Map the Rust ``VideoMetadata`` serde variant to HF video metadata.

    The frontend ships ``{"Video": {source_fps, source_duration,
    source_total_frames, frames_indices, sampled_timestamps}}``. With
    ``do_sample_frames=False`` the Qwen3-VL processor's timestamp expansion
    reads only ``fps`` and ``frames_indices`` (source-frame positions);
    ``total_num_frames`` is carried for correctness.
    """
    if not raw:
        raise ValueError(
            "Decoded video is missing metadata required by the HF processor"
        )
    meta = raw.get(_VIDEO_METADATA_VARIANT, raw)
    return {
        "fps": meta["source_fps"],
        "duration": meta["source_duration"],
        "total_num_frames": meta["source_total_frames"],
        "frames_indices": np.asarray(meta["frames_indices"], dtype=np.int64),
        "video_backend": "dynamo",
    }


class DecodedMmProcessor:
    """Runs the model's HF processor on NIXL-transferred decoded media."""

    # Processor-output keys we forward per modality. Anything else the HF
    # processor emits (attention_mask, mm_token_type_ids, ...) is ignored by
    # SGLang's collect_mm_items_from_processor_output, so forwarding the whole
    # dict is safe; we keep an explicit allow-list for clarity and to drop the
    # unexpanded image/video placeholder bookkeeping.
    _IMAGE_KEYS: Tuple[str, ...] = ("pixel_values", "image_grid_thw", "image_sizes")
    _VIDEO_KEYS: Tuple[str, ...] = (
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    )

    def __init__(self, processor: Any) -> None:
        """Args:
        processor: ``engine.tokenizer_manager.processor`` (raw HF AutoProcessor).
        """
        self._processor = processor
        # Lazy NIXL connector for reading decoded media from frontend memory.
        self._nixl_connector = nixl_connect.Connector()
        run_async(self._nixl_connector.initialize)

    async def _read_image(self, decoded_metadata: Dict[str, Any]) -> np.ndarray:
        # Returns CPU [H, W, 3] uint8 RGB.
        return await read_decoded_media_via_nixl(self._nixl_connector, decoded_metadata)

    async def _read_video(
        self, decoded_metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Returns ([T, H, W, 3] uint8 RGB, HF video metadata).
        frames, raw_meta = await read_decoded_media_via_nixl(
            self._nixl_connector, decoded_metadata, return_metadata=True
        )
        return np.ascontiguousarray(frames), _normalize_video_metadata(raw_meta)

    def _run_processor(
        self,
        text: str,
        images: List[np.ndarray],
        videos: List[np.ndarray],
        video_metadata: List[Dict[str, Any]],
    ) -> Any:
        """Single combined HF processor call. Runs in a worker thread."""
        kwargs: Dict[str, Any] = {}
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
            # Frames are already sampled by the frontend decoder; trust them
            # and supply source metadata for timestamp-token expansion.
            kwargs["video_metadata"] = video_metadata
            kwargs["do_sample_frames"] = False
        return self._processor(
            text=[text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

    @staticmethod
    def _assemble_dict(
        processor_output: Any, has_image: bool, has_video: bool
    ) -> Dict[str, Any]:
        combined: Dict[str, Any] = {"format": "processor_output"}
        get = processor_output.get
        if has_image:
            for key in DecodedMmProcessor._IMAGE_KEYS:
                value = get(key)
                if value is not None:
                    combined[key] = value
        if has_video:
            for key in DecodedMmProcessor._VIDEO_KEYS:
                value = get(key)
                if value is not None:
                    combined[key] = value
        return combined

    async def build(
        self,
        formatted_prompt: str,
        image_items: List[Dict[str, Any]],
        video_items: List[Dict[str, Any]],
    ) -> Tuple[List[int], Dict[str, Any], str]:
        """Read decoded media, run the HF processor, assemble the SGLang input.

        Returns ``(input_ids, processor_output_dict, field)`` where ``field`` is
        ``"image_data"`` or ``"video_data"`` — the async_generate kwarg the dict
        must be passed under.
        """
        # Schedule all reads as tasks; on the first failure cancel and drain the
        # rest so a failed request can't leave large RDMA transfers running on
        # the shared NIXL connector (and no coroutine is left un-awaited).
        num_images = len(image_items)
        coros = [self._read_image(item) for item in image_items]
        coros += [self._read_video(item) for item in video_items]
        tasks = [asyncio.ensure_future(c) for c in coros]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        images: List[np.ndarray] = list(results[:num_images])
        video_results = results[num_images:]
        videos = [frames for frames, _ in video_results]
        video_metadata = [meta for _, meta in video_results]

        processor_output = await asyncio.to_thread(
            self._run_processor, formatted_prompt, images, videos, video_metadata
        )

        # One request, one prompt -> batch dim of 1; [0] takes that row.
        assert (
            len(processor_output["input_ids"]) == 1
        ), "DecodedMmProcessor expects a single-request (batch=1) processor output"
        input_ids = processor_output["input_ids"][0].tolist()
        combined = self._assemble_dict(
            processor_output, has_image=bool(images), has_video=bool(videos)
        )
        # Image-bearing dicts ride image_data; video-only rides video_data. The
        # combined dict is classified by attr name regardless, but a video-only
        # dict under image_data would leave video_data empty and skip SGLang's
        # video metadata path, so route by presence of images.
        field = "image_data" if images else "video_data"
        logger.debug(
            "DecodedMmProcessor: %d image(s), %d video(s) -> %d input_ids via %s",
            len(images),
            len(videos),
            len(input_ids),
            field,
        )
        return input_ids, combined, field
