# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base class for TensorRT-LLM diffusion request handlers.

Holds the constructor setup and size parsing that the image and video
handlers have in common. Each handler keeps its own dimension validation.
"""

import asyncio
import logging
from abc import abstractmethod
from typing import Callable, Optional

from fsspec.implementations.dirfs import DirFileSystem

from dynamo.trtllm.configs.diffusion_config import DiffusionConfig
from dynamo.trtllm.engines.diffusion_engine import DiffusionEngine
from dynamo.trtllm.request_handlers.base_generative_handler import BaseGenerativeHandler

logger = logging.getLogger(__name__)


class DiffusionHandlerBase(BaseGenerativeHandler):
    """Common setup and size parsing for diffusion handlers.

    Subclasses implement generate() and _validate_dimensions().
    """

    def __init__(
        self,
        engine: DiffusionEngine,
        config: DiffusionConfig,
        get_fs: Callable[[str], DirFileSystem],
    ):
        """Initialize the shared handler state.

        Args:
            engine: The DiffusionEngine instance.
            config: Diffusion generation configuration.
            get_fs: Builds the media output filesystem from a URL. Subclasses
                pass the get_fs from their own module, so patching it there
                still takes effect.
        """
        self.engine = engine
        self.config = config
        if not config.media_output_fs_url:
            raise ValueError(
                "media_output_fs_url must be set; use --media-output-fs-url or DYN_MEDIA_OUTPUT_FS_URL."
            )
        self.media_output_fs = get_fs(config.media_output_fs_url)
        self.media_output_http_url = config.media_output_http_url
        # Serialize pipeline access — the diffusion pipeline is not thread-safe
        # (mutable instance state, unprotected CUDA graph cache).
        # asyncio.Lock suspends waiting coroutines cooperatively so the event
        # loop stays free for health checks and signal handling.
        self._generate_lock = asyncio.Lock()

    def _parse_size(self, size: Optional[str]) -> tuple[int, int]:
        """Parse 'WxH' string to (width, height) tuple.

        The API accepts size as a string (e.g., "832x480") to match the format
        used by OpenAI's image generation API (/v1/images/generations).
        This method converts that string to a (width, height) tuple for the engine.

        Args:
            size: Size string in 'WxH' format (e.g., '832x480').

        Returns:
            Tuple of (width, height).

        Raises:
            ValueError: If dimensions exceed configured max_width/max_height.
        """
        if not size:
            width, height = self.config.default_width, self.config.default_height
        else:
            try:
                w, h = size.split("x")
                width, height = int(w), int(h)
            except (ValueError, AttributeError):
                logger.warning(f"Invalid size format: {size}, using defaults")
                width, height = self.config.default_width, self.config.default_height

        # Validate dimensions to prevent OOM
        self._validate_dimensions(width, height)
        return width, height

    @abstractmethod
    def _validate_dimensions(self, width: int, height: int) -> None:
        """Raise ValueError if (width, height) is not allowed by this handler."""
