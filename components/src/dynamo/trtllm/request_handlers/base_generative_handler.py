# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base generative handler for all Dynamo handlers.

This module defines the minimal interface that all generative handlers must implement.
It provides a common base class for LLM, video diffusion, and image diffusion handlers.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from dynamo._core import Context


class BaseGenerativeHandler(ABC):
    """Minimal base class for all generative handlers (LLM, video, image).

    All handlers in the Dynamo system should inherit from this class and
    implement the generate() method. This ensures a consistent interface
    for the endpoint serving infrastructure.
    """

    @abstractmethod
    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate response from request.

        This is the main entry point called by Dynamo's endpoint.serve_endpoint().
        Subclasses implement the specific generation logic for their modality.

        Args:
            request: Request dictionary with generation parameters.
            context: Dynamo context for request tracking and cancellation.

        Yields:
            Response dictionaries. For streaming outputs, multiple dicts may be
            yielded. For non-streaming outputs (like video), a single dict is yielded.

        Raises:
            NotImplementedError: If called on BaseGenerativeHandler directly.
        """
        raise NotImplementedError
        # Note: This yield is needed to make this an async generator
        yield {}  # pragma: no cover
