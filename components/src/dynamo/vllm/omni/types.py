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


class StageOutput(BaseModel):
    """Validated output dict from a stage worker.

    Unknown keys are silently dropped (extra="ignore") to prevent arbitrary
    stage output from accumulating across stages. Only protocol fields pass through.
    finished/error are consumed by the router and not forwarded to subsequent stages.
    """

    model_config = ConfigDict(extra="ignore")
    # TODO: shm_meta should move to a YAML-configured connector edge (issue-005)
    shm_meta: dict | None = None
    original_prompt: dict | None = None
    stage_connector_refs: dict | None = None
    finished: bool | None = None
    error: str | None = None

    def to_next_stage_request(self, request_id: str) -> dict:
        """Build the request dict for the next stage: only inter-stage protocol fields.

        shm_meta is intentionally excluded — it is final-stage → router only.
        """
        fields = self.model_dump(
            include={"original_prompt", "stage_connector_refs"},
            exclude_none=True,
        )
        fields["request_id"] = request_id
        return fields


class StageRequest(BaseModel):
    """Validated request dict received by a stage worker from the router.

    extra="ignore" handles all three request shapes:
      Stage 0:   {request_id, engine_inputs, original_prompt, stage_connector_refs: {}}
      Stage N>0: {request_id, original_prompt, stage_connector_refs: {0: ref0, ...}}
      Direct:    raw frontend request (no router, single-stage deployment)
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str | None = None
    engine_inputs: Any = None
    original_prompt: dict | None = None
    # String keys from JSON — caller must normalize with _int_keyed().
    stage_connector_refs: dict | None = None


def _int_keyed(d: dict | None) -> dict[int, Any]:
    """Normalize JSON-deserialized string keys back to int for stage_connector_refs."""
    if not d:
        return {}
    return {int(k): v for k, v in d.items()}


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
            stage_connector_refs=_int_keyed(d.get("stage_connector_refs")),
        )
