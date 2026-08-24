# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Router configuration owned by the Dynamo AISimulate adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrefillLoadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "aic"] = "none"


class RouterConfig(BaseModel):
    """Concrete Router input for ``aisimulate predict --stack dynamo``."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["round_robin", "kv_router"] = "round_robin"
    prefill_load_model: PrefillLoadModel = Field(default_factory=PrefillLoadModel)
    overlap_score_credit: float | None = Field(default=None, ge=0.0)
    prefill_load_scale: float | None = Field(default=None, ge=0.0)
    temperature: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_policy(self) -> RouterConfig:
        kv_fields = (
            self.overlap_score_credit,
            self.prefill_load_scale,
            self.temperature,
        )
        if self.policy == "round_robin" and (
            self.prefill_load_model.type != "none"
            or any(value is not None for value in kv_fields)
        ):
            raise ValueError(
                "round_robin requires prefill_load_model.type='none' and "
                "rejects KV-router knobs"
            )
        return self
