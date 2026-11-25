// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-native scoring transports used by the reusable decision API core.

mod native;
mod vllm;
mod worker;

pub use native::{build_native_score_request, parse_candidate_scores};
pub(crate) use vllm::VllmScoreCollector;
pub use vllm::causal_scoring_request;
pub(crate) use worker::{WORKER_CANONICAL_V1, WorkerDecisionCapabilities, WorkerDecisionResult};

/// Backend adapter for label scoring orchestrated by Dynamo's frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScoringBackend {
    Sglang,
    Vllm,
}

impl ScoringBackend {
    pub(crate) fn decision_capabilities(self) -> dynamo_decisions::Capabilities {
        dynamo_decisions::Capabilities {
            max_questions: dynamo_decisions::ParseOptions::default().max_questions,
            max_candidates: self.max_candidates(),
            supports_predicate: true,
            supports_choice: true,
            supports_score: true,
            measured_cache_reads: true,
            prompt_format_version: 1,
        }
    }

    pub(crate) fn format(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }

    pub(crate) fn output_tokens(self) -> usize {
        match self {
            Self::Sglang => 0,
            Self::Vllm => 1,
        }
    }

    pub(crate) fn max_candidates(self) -> usize {
        match self {
            Self::Sglang => 255,
            // The qualified vLLM profile uses A-Z labels, not the engine's 128-ID limit.
            Self::Vllm => 26,
        }
    }
}

/// Selects who owns decision prompt preparation and score interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecisionExecution {
    /// Dynamo renders and tokenizes prompts, then reduces backend label scores.
    DynamoFrontend(ScoringBackend),
    /// The worker accepts canonical requests and returns outcomes and usage
    /// through version 1 of the canonical decision worker contract.
    WorkerCanonicalV1,
}

impl DecisionExecution {
    pub(crate) fn scoring_backend(self) -> Option<ScoringBackend> {
        match self {
            Self::DynamoFrontend(backend) => Some(backend),
            Self::WorkerCanonicalV1 => None,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SystemOneError {
    #[error("backend candidate scores are invalid: {0}")]
    CandidateScores(String),
}
