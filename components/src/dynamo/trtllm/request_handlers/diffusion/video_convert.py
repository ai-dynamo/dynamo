# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM video frame conversion to the canonical encoder format."""

import numpy as np

from dynamo.common.utils.video_utils import ensure_uint8_rgb


def to_canonical(video) -> np.ndarray:
    """Convert ``VisualGenOutput.video`` to canonical ``(T, H, W, 3) uint8``.

    ``VisualGenOutput.video`` is a ``torch.Tensor`` of shape ``(1, T, H, W, C)``
    (uint8) since TRT-LLM rc9. The batch dim is squeezed and the tensor moved to
    host memory before canonicalizing.
    """
    assert (
        video.ndim == 5 and video.shape[0] == 1
    ), f"Expected video shape (1, T, H, W, C), got {tuple(video.shape)}"
    return ensure_uint8_rgb(video[0].cpu().numpy())
