// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend scoring transports used by the reusable decision API core.

mod native;
pub mod vllm;

pub use native::{build_native_score_request, parse_candidate_scores};

/// Frontend-owned dispatch marker; public HTTP and gRPC metadata strip this key.
pub(crate) const DECISION_DISPATCH_METADATA_KEY: &str = "dynamo.internal.decision";

pub(crate) const MAX_DECISION_INPUT_TOKENS: usize = 1_048_576;
pub(crate) const MAX_DECISION_ENCODING_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SystemOneError {
    #[error("backend candidate scores are invalid: {0}")]
    CandidateScores(String),
}
