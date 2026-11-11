// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric snapshot values, independent of model discovery and event publication.

use serde::{Deserialize, Serialize};

/// One frontend's complete load report. Each frame replaces the previous frame
/// from the same frontend incarnation; the Relay never merges partial frames.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendLoadFrame {
    /// Stable across event-publisher reconnects.
    pub frontend_instance_id: u64,
    /// Identifies the lifetime of the counters; reconnecting does not reset it.
    pub incarnation: u64,
    /// Starts at 1 and strictly increases within one incarnation.
    pub sequence: u64,
    pub is_serving_ready: bool,
    pub models: Vec<FrontendModelLoad>,
}

/// Load for one canonical model on one frontend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendModelLoad {
    pub model: String,
    pub is_ready: bool,
    pub gauges: RequestGauges,
    pub totals: RequestTotals,
}

/// In-flight requests at publish time.
///
/// Prompt token counts are known once a request has been tokenized; a request
/// still being tokenized contributes zero input tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestGauges {
    /// Admitted requests that have not produced their first output token.
    pub requests_awaiting_first_token: u64,
    pub requests_generating: u64,
    /// Prompt tokens of `requests_awaiting_first_token`.
    pub awaiting_first_token_input_tokens: u64,
    /// Prompt tokens of every in-flight request.
    pub inflight_input_tokens: u64,
}

impl RequestGauges {
    pub fn add(&mut self, other: &Self) {
        self.requests_awaiting_first_token = self
            .requests_awaiting_first_token
            .saturating_add(other.requests_awaiting_first_token);
        self.requests_generating = self
            .requests_generating
            .saturating_add(other.requests_generating);
        self.awaiting_first_token_input_tokens = self
            .awaiting_first_token_input_tokens
            .saturating_add(other.awaiting_first_token_input_tokens);
        self.inflight_input_tokens = self
            .inflight_input_tokens
            .saturating_add(other.inflight_input_tokens);
    }
}

/// Cumulative request counters since the frontend incarnation started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestTotals {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_cancelled: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl RequestTotals {
    pub fn add(&mut self, other: &Self) {
        self.requests_started = self.requests_started.saturating_add(other.requests_started);
        self.requests_completed = self
            .requests_completed
            .saturating_add(other.requests_completed);
        self.requests_failed = self.requests_failed.saturating_add(other.requests_failed);
        self.requests_cancelled = self
            .requests_cancelled
            .saturating_add(other.requests_cancelled);
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
    }

    /// Growth since `previous`, from the same frontend incarnation. A total that
    /// went backwards (an alias unregistered, say) grows by nothing; a restart
    /// arrives as a new incarnation and starts from a fresh baseline instead.
    pub fn growth_since(&self, previous: &Self) -> Self {
        let grow = |current: u64, previous: u64| current.saturating_sub(previous);
        Self {
            requests_started: grow(self.requests_started, previous.requests_started),
            requests_completed: grow(self.requests_completed, previous.requests_completed),
            requests_failed: grow(self.requests_failed, previous.requests_failed),
            requests_cancelled: grow(self.requests_cancelled, previous.requests_cancelled),
            input_tokens: grow(self.input_tokens, previous.input_tokens),
            output_tokens: grow(self.output_tokens, previous.output_tokens),
        }
    }
}

/// Model identity and readiness supplied by the service composition.
#[derive(Clone, Debug)]
pub struct ModelLoadView {
    pub name: String,
    pub is_ready: bool,
    pub aliases: Vec<String>,
}
