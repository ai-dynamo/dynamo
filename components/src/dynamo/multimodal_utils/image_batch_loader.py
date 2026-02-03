# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utility for batch loading images from multimodal data items."""

import asyncio
import logging
from typing import Any, Dict, Final, List

from dynamo.multimodal_utils.image_loader import ImageLoader

logger = logging.getLogger(__name__)

# Constants for multimodal data variants
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


async def load_image_batch(
    image_mm_items: List[Dict[str, Any]], image_loader: ImageLoader
) -> List[Any]:
    """
    Load a batch of images from multimodal data items.

    Reuses backend-agnostic ImageLoader for concurrent image loading with caching.

    Args:
        image_mm_items: List of multimodal data items for images
            Format: [{"Url": "https://..."}, ...] or [{"Decoded": <rdma_handle>}, ...]
        image_loader: ImageLoader instance to use for loading

    Returns:
        List of PIL Images in RGB format

    Raises:
        Exception: If any image fails to load (aggregates all errors)
    """
    image_futures = []
    for item in image_mm_items:
        if isinstance(item, dict) and URL_VARIANT_KEY in item:
            url = item[URL_VARIANT_KEY]
            image_futures.append(image_loader.load_image(url))
            logger.debug(f"Preparing to load image from URL: {url[:80]}...")
        elif isinstance(item, dict) and DECODED_VARIANT_KEY in item:
            # Frontend-decoded images via NIXL (future support)
            logger.warning(
                "Frontend-decoded multimodal data not yet supported. "
                "Use backend decoding (default) for now."
            )
            # TODO: Add support for NIXL-transferred tensors when --frontend-decoding is enabled
            continue

    if not image_futures:
        return []

    # Load all images concurrently
    results = await asyncio.gather(*image_futures, return_exceptions=True)
    loaded_images = []
    collective_exceptions = ""

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            url = image_mm_items[i].get(URL_VARIANT_KEY, "unknown")
            logger.error(f"Failed to load image from {url[:80]}...: {result}")
            collective_exceptions += (
                f"Failed to load image from {url[:80]}...: {result}\n"
            )
            continue

        # Keep as PIL Image
        loaded_images.append(result)

    if collective_exceptions:
        raise Exception(collective_exceptions)

    return loaded_images
