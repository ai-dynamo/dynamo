# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang video frame conversion to the canonical encoder format."""

import numpy as np

from dynamo.common.utils.video_utils import ensure_uint8_rgb, pil_frames_to_array


def to_canonical(frames: list) -> np.ndarray:
    """Convert ``DiffGenerator`` frames to canonical ``(T, H, W, 3) uint8``.

    ``frames`` is a list of per-frame ``PIL.Image`` or ``np.ndarray`` images.
    """
    return ensure_uint8_rgb(pil_frames_to_array(frames))
