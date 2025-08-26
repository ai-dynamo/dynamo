# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from io import BytesIO
from typing import Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


async def open_video_container(video_content_stream: BytesIO, video_url: str) -> av.container.InputContainer:
    """
    Open a video container from a BytesIO stream using PyAV.
    
    Args:
        video_content_stream: BytesIO stream containing video data
        video_url: Original video URL for error reporting
        
    Returns:
        Opened PyAV container
        
    Raises:
        ValueError: If video format is invalid or corrupted
    """
    def open_video_container_sync():
        try:
            return av.open(video_content_stream, mode="r")
        except av.FFmpegError as ave:
            logger.error(
                f"PyAV error opening video stream from {video_url}: {ave}"
            )
            raise ValueError(
                f"Invalid video format or corrupted data from {video_url}."
            ) from ave
        except Exception as e:
            logger.error(
                f"Unexpected error opening video stream from {video_url} with PyAV: {e}"
            )
            raise ValueError(
                f"Unexpected error opening video from {video_url}."
            ) from e

    return await asyncio.to_thread(open_video_container_sync)


def get_video_metadata(container: av.container.InputContainer) -> Tuple[int, float]:
    """
    Extract metadata from video container.
    
    Args:
        container: Opened PyAV container
        
    Returns:
        Tuple of (total_frames, duration_in_seconds)
    """
    if not container or not container.streams.video:
        return 0, 0.0
        
    stream_info = container.streams.video[0]
    total_frames = stream_info.frames
    
    # Duration can be useful for streams where total_frames is 0
    if stream_info.duration and stream_info.time_base:
        duration_sec = float(stream_info.duration * stream_info.time_base)
    else:
        duration_sec = 0.0
        
    return total_frames, duration_sec


def calculate_frame_sampling_indices(
    total_frames: int, 
    num_frames_to_sample: int, 
    duration_sec: float = 0,
    video_url: str = ""
) -> np.ndarray:
    """
    Calculate frame indices to sample from a video.
    
    Args:
        total_frames: Total number of frames in the video
        num_frames_to_sample: Number of frames to sample
        duration_sec: Duration of video in seconds (for logging)
        video_url: Video URL for logging purposes
        
    Returns:
        Array of frame indices to sample
        
    Raises:
        ValueError: If video has 0 frames and 0 duration
    """
    if total_frames == 0 and duration_sec == 0:
        logger.error(f"Video file '{video_url}' has 0 frames and 0 duration.")
        raise ValueError(f"Video {video_url} has 0 frames and 0 duration.")
        
    if total_frames == 0 and duration_sec > 0:
        logger.warning(
            f"Video {video_url} reports 0 frames but has duration {duration_sec:.2f}s. "
            "Frame sampling may be based on requested count directly."
        )

    logger.debug(
        f"Video {video_url} has {total_frames} frames (duration: {duration_sec:.2f}s). "
        f"Sampling {num_frames_to_sample} frames."
    )
    
    indices: np.ndarray
    if total_frames > 0:
        if total_frames < num_frames_to_sample:
            logger.warning(
                f"Video frames ({total_frames}) < samples ({num_frames_to_sample}). "
                f"Using all {total_frames} available frames."
            )
            indices = np.arange(0, total_frames).astype(int)
        else:
            indices = np.linspace(
                0, total_frames - 1, num_frames_to_sample, dtype=int
            )
    else:  # total_frames is 0 (likely a stream), sample by count.
        logger.warning(
            f"Video {video_url} frame count is 0. Attempting to sample {num_frames_to_sample} "
            "frames by index. This might fail if stream is too short."
        )
        indices = np.arange(0, num_frames_to_sample).astype(int)

    # Ensure indices are unique, especially after linspace for small numbers.
    indices = np.unique(indices)

    # Safety checks for edge cases
    if len(indices) == 0 and total_frames > 0:
        # If unique resulted in empty but there are frames, sample at least one
        actual_samples = min(num_frames_to_sample, total_frames)
        indices = np.arange(0, actual_samples).astype(int)
    elif len(indices) == 0 and total_frames == 0:
        # If indices is empty and total_frames is 0, let downstream handle this case
        pass

    logger.info(f"Selected frame indices for {video_url}: {indices.tolist()}")
    return indices


def resize_video_frames(
    frames_tensor: torch.Tensor, 
    target_height: int, 
    target_width: int
) -> torch.Tensor:
    """
    Resize video frames using PyTorch interpolation.
    
    Args:
        frames_tensor: Input tensor with shape (T, H, W, C)
        target_height: Target frame height
        target_width: Target frame width
        
    Returns:
        Resized tensor with shape (T, target_height, target_width, C)
    """
    # Permute to (T, C, H, W) for interpolate
    frames_tensor_chw = frames_tensor.permute(0, 3, 1, 2).float()

    # Resize
    resized_frames_tensor_chw = F.interpolate(
        frames_tensor_chw,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Permute back to (T, H_new, W_new, C)
    resized_frames_tensor_hwc = resized_frames_tensor_chw.permute(0, 2, 3, 1)
    
    logger.debug(f"Resized frames to shape: {resized_frames_tensor_hwc.shape}")
    return resized_frames_tensor_hwc


def prepare_tensor_for_rdma(
    frames_tensor: torch.Tensor, 
    request_id: str
) -> torch.Tensor:
    """
    Prepare video frames tensor for RDMA transfer.
    
    Args:
        frames_tensor: Input frames tensor
        request_id: Request ID for logging
        
    Returns:
        Tensor prepared for RDMA (CPU, uint8, contiguous)
    """
    # Ensure the tensor is contiguous, on CPU and uint8 for the NIXL buffer.
    tensor_for_descriptor = frames_tensor.to(
        device="cpu", dtype=torch.uint8
    ).contiguous()

    logger.info(
        f"Req {request_id}: Preparing raw frames tensor (shape: {tensor_for_descriptor.shape}, "
        f"dtype: {tensor_for_descriptor.dtype}, device: {tensor_for_descriptor.device}, "
        f"contiguous: {tensor_for_descriptor.is_contiguous()}) for RDMA."
    )
    
    return tensor_for_descriptor
