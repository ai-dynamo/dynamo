// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Worker-confirmed isolated scheduling for zero-decode candidate scoring.
pub const SGLANG_SYSTEMONE_SERIAL_V1: &str = "sglang_systemone_serial_v1";

/// Worker-confirmed raw requested-ID scoring through the Python vLLM worker.
pub const VLLM_CAUSAL_DECISION_V1: &str = "vllm_causal_decision_v1";
pub const VLLM_D3_DECISION_V1: &str = "vllm_d3_decision_v1";
pub const VLLM_CLEF_FLASH_DECISION_V1: &str = "vllm_clef_flash_joint_decision_v1";
pub const VLLM_CLEF_OMNI_DECISION_V1: &str = "vllm_clef_omni_joint_decision_v1";
pub const VLLM_LIQUID_DECISION_V1: &str = "vllm_liquid_decision_v1";
pub const VLLM_DIFFUSION_DECISION_V1: &str = "vllm_diffusion_canvas_decision_v1";

pub(crate) const VLLM_NATIVE_DECISION_MAX_CANDIDATES: &str = "vllm_native_decision_max_candidates";
pub(crate) const VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1: &str =
    "vllm_native_decision_qualification_failure_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenBackend {
    Sglang,
    VllmCausal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeExecutorKind {
    Head,
    Read,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecisionProfileKind {
    Token(TokenBackend),
    Native(NativeExecutorKind),
}

const SGLANG_DECISION_LIMITS: (usize, usize) = (128, 255);
const VLLM_DECISION_LIMITS: (usize, usize) = (64, 26);
const VLLM_D3_DECISION_LIMITS: (usize, usize) = (64, 255);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum DecisionExecutionProfile {
    #[serde(rename = "sglang_systemone_serial_v1")]
    SglangSerialV1,
    #[serde(rename = "vllm_causal_decision_v1")]
    VllmCausalV1,
    #[serde(rename = "vllm_d3_decision_v1")]
    VllmD3V1,
    #[serde(rename = "vllm_clef_flash_joint_decision_v1")]
    VllmClefFlashV1,
    #[serde(rename = "vllm_clef_omni_joint_decision_v1")]
    VllmClefOmniV1,
    #[serde(rename = "vllm_liquid_decision_v1")]
    VllmLiquidV1,
    #[serde(rename = "vllm_diffusion_canvas_decision_v1")]
    VllmDiffusionV1,
}

impl DecisionExecutionProfile {
    pub(crate) fn decision_capability(self) -> &'static str {
        match self {
            Self::SglangSerialV1 => SGLANG_SYSTEMONE_SERIAL_V1,
            Self::VllmCausalV1 => VLLM_CAUSAL_DECISION_V1,
            Self::VllmD3V1 => VLLM_D3_DECISION_V1,
            Self::VllmClefFlashV1 => VLLM_CLEF_FLASH_DECISION_V1,
            Self::VllmClefOmniV1 => VLLM_CLEF_OMNI_DECISION_V1,
            Self::VllmLiquidV1 => VLLM_LIQUID_DECISION_V1,
            Self::VllmDiffusionV1 => VLLM_DIFFUSION_DECISION_V1,
        }
    }

    pub(crate) fn kind(self) -> DecisionProfileKind {
        match self {
            Self::SglangSerialV1 => DecisionProfileKind::Token(TokenBackend::Sglang),
            Self::VllmCausalV1 => DecisionProfileKind::Token(TokenBackend::VllmCausal),
            Self::VllmD3V1 | Self::VllmClefFlashV1 | Self::VllmClefOmniV1 => {
                DecisionProfileKind::Native(NativeExecutorKind::Head)
            }
            Self::VllmLiquidV1 | Self::VllmDiffusionV1 => {
                DecisionProfileKind::Native(NativeExecutorKind::Read)
            }
        }
    }

    pub(crate) fn limits(self) -> (usize, usize) {
        match self {
            Self::SglangSerialV1 => SGLANG_DECISION_LIMITS,
            Self::VllmD3V1 => VLLM_D3_DECISION_LIMITS,
            Self::VllmCausalV1
            | Self::VllmClefFlashV1
            | Self::VllmClefOmniV1
            | Self::VllmLiquidV1
            | Self::VllmDiffusionV1 => VLLM_DECISION_LIMITS,
        }
    }
}
