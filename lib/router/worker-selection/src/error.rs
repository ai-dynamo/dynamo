// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The error a filter, scorer, or picker reports; the kernel wraps it in its own.

#[derive(Debug, thiserror::Error)]
pub enum WorkerSelectionPolicyError {
    #[error("worker selection policy failed: {0}")]
    Failed(String),

    #[error("scorer {scorer_index} produced a non-finite cost for candidate row {row}")]
    NonFiniteCost { scorer_index: usize, row: usize },

    #[error(
        "picker returned candidate row {row}, but the candidate table contains {candidate_count} rows"
    )]
    InvalidPickerRow { row: usize, candidate_count: usize },
}

impl WorkerSelectionPolicyError {
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }
}
