// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-native scoring transports used by the reusable decision API core.

mod native;
mod pooling;
mod vllm;

pub use native::{build_native_score_request, parse_candidate_scores};
pub(crate) use pooling::{
    CHUNKED_CONTEXT_END_SENTINEL, CHUNKED_POOLING_SYSTEM_PROMPT, POOLING_SYSTEM_PROMPT,
    PoolingDecisionConfig, build_pooling_request, extract_marker_logits,
    format_chunked_question_text, format_joint_user_prompt, format_pooling_state_text,
    reduce_head_logits,
};
pub(crate) use vllm::VllmScoreCollector;
pub use vllm::causal_scoring_request;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScoringBackend {
    Sglang,
    Vllm,
    VllmPooling,
}

impl ScoringBackend {
    pub(crate) fn format(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
            Self::VllmPooling => {
                unreachable!("pooling backend dispatches via in-process pooling engine")
            }
        }
    }

    pub(crate) fn output_tokens(self) -> usize {
        match self {
            Self::Sglang | Self::VllmPooling => 0,
            Self::Vllm => 1,
        }
    }

    pub(crate) fn max_candidates(self) -> usize {
        match self {
            Self::Sglang | Self::VllmPooling => 255,
            // The qualified causal profile uses A-Z labels, not the engine's 128-ID limit.
            Self::Vllm => 26,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SystemOneError {
    #[error("backend candidate scores are invalid: {0}")]
    CandidateScores(String),
}
