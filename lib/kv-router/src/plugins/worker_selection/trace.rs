// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bounded, cost-only record of one selection: which candidates were scored and at
//! what cost. Nothing else leaves the stage.

use dynamo_router_protocol::worker::WorkerWithDpRank;

/// How many candidates a selection trace retains: the `top_k` lowest-cost rows
/// plus the selected worker, or every candidate when `top_k` is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionTraceSpec {
    pub top_k: usize,
}

/// A scored candidate: the worker and its total cost after every scorer ran, in
/// the policy's cost units (scheduler-tracked KV blocks for the builtin policy).
/// Lower is better.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectionTraceRow {
    pub worker: WorkerWithDpRank,
    pub cost: f64,
}

/// Per-candidate costs captured for one worker selection.
///
/// Rows are unordered. When `rows.len() < candidate_count`, the selection was
/// truncated to the `top_k` lowest-cost candidates plus the selected worker,
/// which is always present.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionTrace {
    /// Eligible candidates the policy scored, before truncation.
    pub candidate_count: usize,
    /// Whether the host pinned the selection to one worker and rank.
    pub pinned: bool,
    /// Retained candidates, unordered.
    pub rows: Vec<SelectionTraceRow>,
}
