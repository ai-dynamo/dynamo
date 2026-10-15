# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni video frame conversion to the canonical encoder format."""

import numpy as np
import torch

from dynamo.common.utils.video_utils import ensure_uint8_rgb


def to_canonical(images: list) -> np.ndarray:
    """Convert ``stage_output.images`` to canonical ``(T, H, W, 3) uint8``.

    ``stage_output.images`` is a list holding a single full-video array of shape
    ``(1, T, H, W, C)`` or ``(T, H, W, C)`` (``np.ndarray`` or ``torch.Tensor``).
    """
    array = images[0] if len(images) == 1 else images
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 5:
        array = array[0]
    return ensure_uint8_rgb(array)
