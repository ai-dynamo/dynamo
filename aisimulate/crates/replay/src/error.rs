// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

pub type ReplayResult<T> = Result<T, ReplayError>;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReplayError {
    #[error("invalid replay spec: {0}")]
    InvalidSpec(String),

    #[error("engine error: {0}")]
    Engine(String),

    #[error("placement policy error: {0}")]
    Placement(String),

    #[error("scaling policy error: {0}")]
    Scaling(String),

    #[error("replay reached quiescence with {unfinished_requests} unfinished request(s)")]
    Deadlock { unfinished_requests: usize },

    #[error("replay invariant violated: {0}")]
    Invariant(String),
}

pub(crate) fn placement_boundary(error: anyhow::Error) -> anyhow::Error {
    ReplayError::Placement(format!("{error:#}")).into()
}

pub(crate) fn scaling_boundary(error: anyhow::Error) -> anyhow::Error {
    ReplayError::Scaling(format!("{error:#}")).into()
}

pub(crate) fn engine_boundary(error: anyhow::Error) -> anyhow::Error {
    ReplayError::Engine(format!("{error:#}")).into()
}

/// Recover a boundary-classified failure without interpreting its message.
/// Unclassified runtime failures are internal consistency errors.
pub(crate) fn runtime_error(error: anyhow::Error) -> ReplayError {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<ReplayError>())
        .cloned()
        .unwrap_or_else(|| ReplayError::Invariant(format!("{error:#}")))
}
