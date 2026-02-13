# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Backend(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM = "trtllm"
    AUTO = "auto"


class SearchStrategy(str, Enum):
    """Controls profiling thoroughness."""

    RAPID = "rapid"
    THOROUGH = "thorough"


class ModelCacheSpec(BaseModel):
    """PVC-based model cache configuration."""

    pvc_name: str = Field(
        ...,
        description="Name of the PVC containing the model weights.",
    )
    model_path_in_pvc: str = Field(
        ...,
        description="Path to the model checkpoint directory within the PVC "
        "(e.g. 'deepseek-r1' or 'models/Llama-3.1-405B-FP8').",
    )
    pvc_mount_path: str = Field(
        default="/opt/model-cache",
        description="Mount path for the PVC inside the container.",
    )


class ModelSpec(BaseModel):
    model_name: str = Field(
        ...,
        description="Model name or identifier (e.g. 'meta-llama/Llama-3.1-405B'). "
        "Can be a HuggingFace ID or a private model name. Always required.",
    )
    model_cache: Optional[ModelCacheSpec] = Field(
        default=None,
        description="Optional PVC model cache configuration. "
        "When provided, weights are loaded from the PVC instead of downloading from HF.",
    )


class BackendSpec(BaseModel):
    backend: Backend = Field(
        default=Backend.VLLM,
        description="Inference backend to use.",
    )
    dynamo_image: str = Field(
        description="Full K8s dynamo image reference (e.g. "
        "'nvcr.io/nvidia/dynamo-runtime:latest'). "
    )


class HardwareSpec(BaseModel):
    gpu_sku: str = Field(
        ...,
        description="GPU SKU identifier (e.g. 'H100_SXM', 'A100_80GB').",
    )
    vram_mb: float = Field(
        ...,
        gt=0,
        description="VRAM per GPU in MiB.",
    )
    total_gpus: int = Field(
        ...,
        gt=0,
        description="Total number of GPUs available in the cluster.",
    )
    num_gpus_per_node: int = Field(
        ...,
        gt=0,
        description="Number of GPUs per node.",
    )


class WorkloadSpec(BaseModel):
    isl: int = Field(
        default=3000,
        description="Target input sequence length (tokens).",
    )
    osl: int = Field(
        default=500,
        description="Target output sequence length (tokens).",
    )
    concurrency: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target concurrency level. "
        "Required (or request_rate) when the planner is disabled. "
        "Will be ignored if the planner is enabled.",
    )
    request_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target request rate (req/s). "
        "Required (or concurrency) when the planner is disabled. "
        "Will be ignored if the planner is enabled.",
    )


class SLASpec(BaseModel):
    ttft: Optional[float] = Field(
        default=None,
        description="Target Time-To-First-Token in milliseconds.",
    )
    itl: Optional[float] = Field(
        default=None,
        description="Target Inter-Token Latency in milliseconds.",
    )
    e2e_latency: Optional[float] = Field(
        default=None,
        description="Target end-to-end request latency in milliseconds. "
        "Alternative to specifying ttft + itl.",
    )

    @model_validator(mode="after")
    def _validate_sla_option(self) -> "SLASpec":
        has_ttft_itl = self.ttft is not None and self.itl is not None
        has_e2e = self.e2e_latency is not None

        if not has_ttft_itl and not has_e2e:
            raise ValueError("SLA must specify either (ttft and itl) or e2e_latency.")
        if has_ttft_itl and has_e2e:
            raise ValueError(
                "SLA must specify either (ttft and itl) or e2e_latency, not both."
            )
        if (self.ttft is not None) != (self.itl is not None):
            raise ValueError("ttft and itl must both be provided together.")
        return self


class MockerSpec(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Whether to deploy mocker workers instead of real workers.",
    )


class PlannerPreDeploymentSweepMode(str, Enum):
    """Pre-deployment sweeping thoroughness for planner profiling."""

    NONE = "none"
    RAPID = "rapid"
    THOROUGH = "thorough"


class PlannerSpec(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Whether to deploy planner workers instead of real workers.",
    )
    planner_pre_deployment_sweeping: Optional[PlannerPreDeploymentSweepMode] = Field(
        default=None,
        description="Pre-deployment sweeping mode for planner in-depth profiling. "
        "None means no pre-deployment sweep, which means only load-based scaling in planner is possible."
        "Rapid means using AIC to simulate the engine performance, requires AIC supporting model x backend x hardware combination."
        "Thorough means using real GPUs to measure the engine performance, will take several hours."
        "If mocker is enabled, pre-deployment sweep is required.",
    )
    planner_args_list: Optional[list[str]] = Field(
        default=None,
        description="List of planner arguments.",
    )


class FeaturesSpec(BaseModel):
    """Feature toggles (planner, mocker)."""

    planner: PlannerSpec = Field(
        default_factory=PlannerSpec,
        description="Planner configuration. Disabled by default.",
    )
    mocker: MockerSpec = Field(
        default_factory=MockerSpec,
        description="Mocker configuration. Disabled by default.",
    )

    @model_validator(mode="after")
    def _mocker_requires_planner_pre_deployment_sweeping(self) -> "FeaturesSpec":
        if self.mocker.enabled and (
            self.planner.planner_pre_deployment_sweeping is None
            or self.planner.planner_pre_deployment_sweeping
            == PlannerPreDeploymentSweepMode.NONE
        ):
            raise ValueError(
                "Mocker requires planner pre-deployment sweeping to be enabled "
                "(set to 'rapid' or 'thorough') because it relies on in-depth "
                "profiling data produced during the sweep."
            )
        return self


class AugmentedDGDR(BaseModel):
    """User input DGDR + operator-augmented fields.

    This is the canonical request schema for the Dynamo Graph Deployment
    Recommendation (DGDR) profiler.  Users fill in model, backend, workload,
    sla, and feature preferences; the operator auto-fills hardware details.
    """

    model: ModelSpec = Field(
        ...,
        description="Model specification (HF ID or PVC checkpoint path).",
    )
    backend: BackendSpec = Field(
        ...,
        description="Backend and container image configuration.",
    )
    hardware: HardwareSpec = Field(
        ...,
        description="Hardware description (auto-filled by operator).",
    )
    workload: WorkloadSpec = Field(
        default_factory=WorkloadSpec,
        description="Target workload shape (ISL / OSL).",
    )
    sla: SLASpec = Field(
        ...,
        description="Service-level agreement targets.",
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.RAPID,
        description="Overall search strategy: rapid or thorough.",
    )
    features: FeaturesSpec = Field(
        default_factory=FeaturesSpec,
        description="Feature toggles (planner, mocker, etc.).",
    )
    auto_apply: bool = Field(
        default=True,
        description="If true, automatically create and apply the generated DGD after profiling. "
        "If false, store results for user inspection via GUI.",
    )

    @model_validator(mode="after")
    def _target_load_required_when_planner_disabled(self) -> "AugmentedDGDR":
        has_load = (
            self.workload.concurrency is not None
            or self.workload.request_rate is not None
        )
        if not self.features.planner.enabled and not has_load:
            raise ValueError(
                "When the planner is disabled, either 'workload.concurrency' or "
                "'workload.request_rate' must be provided."
            )
        return self

    model_config = {"extra": "forbid"}
