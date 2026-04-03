# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for disaggregated omni stage workers and connectors.
"""

import dataclasses
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


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


class StageOutput(BaseModel):
    """Validated output dict from a stage worker.

    Unknown keys are silently dropped (extra="ignore") to prevent arbitrary
    stage output from accumulating across stages. Only protocol fields pass through.
    finished/error are consumed by the router and not forwarded to subsequent stages.
    """

    model_config = ConfigDict(extra="ignore")
    # TODO: Fix shm_meta thing later. This should be removed
    shm_meta: dict | None = None
    connector_meta: dict | None = None
    original_prompt: dict | None = None
    stage_connector_refs: dict | None = None
    finished: bool | None = None
    error: str | None = None

    def to_next_stage_request(self, request_id: str) -> dict:
        """Build the request dict for the next stage: only inter-stage protocol fields."""
        fields = self.model_dump(
            include={
                "shm_meta",
                "connector_meta",
                "original_prompt",
                "stage_connector_refs",
            },
            exclude_none=True,
        )
        fields["request_id"] = request_id
        return fields


@dataclasses.dataclass
class OmniInterStageRequest:
    """Protocol message passed between stage workers via the router.

    The router passes this opaquely without inspecting stage_connector_refs.
    Workers accumulate connector refs as the pipeline progresses, allowing
    any stage to reconstruct stage_list for N-stage processor functions.

    JSON-serializable: original_prompt is a TypedDict (dict subclass) with
    no tensors. Tensors (token_ids, images) travel via the connector payload.
    """

    request_id: str

    # OmniPromptType | list | None — typed as Any to avoid importing vllm_omni at
    # module level. Set once by the router at pipeline start, never modified by workers.
    original_prompt: Any

    # Grows as the pipeline progresses: {} → {0: ref0} → {0: ref0, 1: ref1} → ...
    stage_connector_refs: dict[int, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "original_prompt": self.original_prompt,
            "stage_connector_refs": self.stage_connector_refs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OmniInterStageRequest":
        return cls(
            request_id=d["request_id"],
            original_prompt=d["original_prompt"],
            # JSON serializes dict keys as strings — convert back to int
            stage_connector_refs={
                int(k): v for k, v in d.get("stage_connector_refs", {}).items()
            },
        )
