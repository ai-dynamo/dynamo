# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM's image conversion, shared by the vLLM worker and DynamoMediaConnector."""

from PIL import Image
from vllm.multimodal.image import convert_image_mode, normalize_image


def vllm_rgb(
    image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Apply EXIF orientation, then composite transparency, as vLLM's loader does."""
    return convert_image_mode(normalize_image(image), "RGB", background)
