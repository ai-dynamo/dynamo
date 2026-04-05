# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for disaggregated omni stage workers and connectors.
"""

from typing import Any, AsyncGenerator, Protocol, runtime_checkable


@runtime_checkable
class StageEngine(Protocol):
    """Any engine that can generate outputs for a single pipeline stage.

    Matches AsyncOmni.generate() signature — the only vllm_omni engine
    with a consistent async generator interface for both LLM and diffusion.
    """

    def generate(
        self,
        prompt: Any,
        request_id: str = "",
        *,
        sampling_params_list: Any = None,
    ) -> AsyncGenerator[Any, None]:
        ...


@runtime_checkable
class StageConnector(Protocol):
    """Inter-stage transport owned by the router (e.g. SharedMemoryConnector)."""

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, Any]:
        """Write payload to transport. Returns (ok, serialized_size, metadata)."""
        ...

    def cleanup(self, request_id: str) -> None:
        """Release transport resources for this request."""
        ...
