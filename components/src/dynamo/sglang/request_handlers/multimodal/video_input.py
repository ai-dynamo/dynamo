# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang adapter for videos decoded and sampled by the Dynamo frontend."""

from typing import Any, Dict

import numpy as np
import torch
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


class FrontendDecodedVideo(np.ndarray, VideoDecoderWrapper):
    def __new__(
        cls, video_frames: Any, video_metadata: Dict[str, Any]
    ) -> "FrontendDecodedVideo":
        video = np.ascontiguousarray(video_frames).view(cls)
        duration = float(video_metadata.get("duration") or 0)
        source_fps = float(video_metadata.get("fps") or 0)
        frame_indices = video_metadata.get("frames_indices")

        video._source_fps = source_fps
        video._duration = duration
        video._frame_indices = (
            [int(index) for index in frame_indices]
            if frame_indices is not None
            else None
        )
        video._total_num_frames = int(
            video_metadata.get("total_num_frames") or len(video_frames)
        )

        # SGLang does not yet expose a model-independent pre-sampled-video
        # contract. Preserve the sampled span as an effective FPS so its
        # processor does not collapse a sparse frontend sample to fewer frames.
        effective_fps = len(video_frames) / duration if duration > 0 else source_fps
        if (
            source_fps > 0
            and frame_indices is not None
            and len(frame_indices) == len(video_frames)
            and len(frame_indices) > 1
        ):
            span_frames = float(frame_indices[-1]) - float(frame_indices[0])
            if span_frames > 0:
                effective_fps = (len(video_frames) - 1) * source_fps / span_frames

        video._avg_fps = effective_fps
        if video._avg_fps <= 0:
            raise ValueError("Frontend-decoded video metadata must contain a valid fps")
        return video

    def __init__(self, video_frames: Any, video_metadata: Dict[str, Any]):
        pass

    def __array_finalize__(self, source: Any) -> None:
        if source is not None:
            self._avg_fps = getattr(source, "_avg_fps", 0.0)
            self._source_fps = getattr(source, "_source_fps", 0.0)
            self._duration = getattr(source, "_duration", 0.0)
            self._frame_indices = getattr(source, "_frame_indices", None)
            self._total_num_frames = getattr(source, "_total_num_frames", 0)

    @property
    def avg_fps(self) -> float:
        return self._avg_fps

    def get_frames_as_tensor(self, indices: list[int]):
        return torch.from_numpy(np.asarray(self)[indices])

    def get_frames_at(self, indices: list[int]):
        return np.asarray(self)[indices]

    def as_processor_input(self) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Return sampled frames with the source timeline used by the frontend."""
        if self._source_fps <= 0:
            raise ValueError("Frontend-decoded video metadata must contain a valid fps")
        if self._frame_indices is None or len(self._frame_indices) != len(self):
            raise ValueError(
                "Frontend-decoded video frame indices must match the frame count"
            )

        frames = torch.from_numpy(np.asarray(self)).permute(0, 3, 1, 2).contiguous()
        return frames, {
            "fps": self._source_fps,
            "duration": self._duration,
            "total_num_frames": self._total_num_frames,
            "frames_indices": list(self._frame_indices),
            "video_backend": "dynamo_frontend",
        }

    def close(self) -> None:
        pass


def as_sglang_video(frames: Any, metadata: Dict[str, Any]) -> FrontendDecodedVideo:
    """Expose transferred frames through SGLang's predecoded-video contract."""
    return FrontendDecodedVideo(frames, metadata)
