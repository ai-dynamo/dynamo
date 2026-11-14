// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang transport used by the reusable decision API core.

pub(crate) mod head;
mod native;
pub mod vllm;

pub use native::{build_native_score_request, parse_candidate_scores};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScoringBackend {
    Sglang,
    Vllm,
}

impl ScoringBackend {
    pub(crate) fn format(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }

    pub(crate) fn output_tokens(self) -> usize {
        usize::from(self == Self::Vllm)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SystemOneError {
    #[error("backend candidate scores are invalid: {0}")]
    CandidateScores(String),
}
