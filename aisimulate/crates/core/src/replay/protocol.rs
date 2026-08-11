// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-owned values that cross workload, placement, reporting, and engine
//! adapter boundaries.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Where the prompt token identities carried by a lowered request came from.
///
/// Length-only specifications still need deterministic token identities for
/// scheduler execution. Those synthetic identities are deliberately
/// marked so a KV-aware placement adapter cannot mistake them for authored
/// prompt content.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPromptTokenSource {
    /// The workload supplied exact prompt token IDs (or an equivalent trace
    /// representation suitable for KV-aware placement).
    #[default]
    Materialized,
    /// Replay synthesized deterministic execution-only IDs from a prompt
    /// length. These IDs must not be used to model prefix-cache affinity.
    LengthOnlySynthetic,
}

/// Replay-owned correlation data retained while a request crosses placement,
/// engine, and reporting boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRequestContext {
    pub authored_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
    #[serde(default)]
    pub prompt_token_source: ReplayPromptTokenSource,
}

/// One materialized replay request before it enters a generalized engine.
#[doc(hidden)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<u32>,
    pub max_output_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    pub uuid: Option<Uuid>,
    pub dp_rank: u32,
    /// Optional rank requested by the authored ReplaySpec. The runtime-owned
    /// `dp_rank` remains the effective rank used by legacy workload DTOs;
    /// placement policies may choose whether to honor this preference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_dp_rank: Option<u32>,
    pub arrival_timestamp_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub priority: i32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub strict_priority: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_class: Option<String>,
    /// Correlation/provenance retained by Replay but ignored by the native
    /// scheduler. Legacy runtime inputs may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_context: Option<ReplayRequestContext>,
}

fn is_zero_i32(value: &i32) -> bool {
    *value == 0
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

impl DirectRequest {
    pub fn request_id(&self) -> Option<Uuid> {
        self.uuid
    }

    pub fn input_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    pub fn arrival_time_ms(&self) -> Option<f64> {
        self.arrival_timestamp_ms
    }

    pub fn preferred_dp_rank(&self) -> Option<u32> {
        self.preferred_dp_rank
    }

    pub fn router_priorities(&self) -> (f64, u32) {
        (f64::from(self.priority.max(0)), self.strict_priority)
    }

    pub fn policy_class(&self) -> Option<&str> {
        self.policy_class.as_deref()
    }

    pub fn prompt_tokens_are_placement_safe(&self) -> bool {
        self.replay_context.as_ref().is_none_or(|context| {
            context.prompt_token_source != ReplayPromptTokenSource::LengthOnlySynthetic
        })
    }
}

/// Client-visible output released at a replay pass-completion boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OutputSignal {
    pub(crate) uuid: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) token_id: Option<u32>,
    pub(crate) completed: bool,
    #[serde(default)]
    pub(crate) rejected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) handoff_delay_ms: Option<f64>,
    /// Prompt tokens served from KV cache at first admission. Present only on
    /// the request's first output signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cached_tokens: Option<usize>,
}

/// Replay-neutral forward-pass observation consumed by scaling policies.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForwardPassSnapshot {
    pub version: u32,
    pub worker_id: String,
    pub dp_rank: u32,
    pub counter_id: u64,
    pub num_prefill_requests: u32,
    pub sum_prefill_tokens: u64,
    pub var_prefill_length: f64,
    pub sum_prefill_kv_tokens: u64,
    pub num_decode_requests: u32,
    pub sum_decode_kv_tokens: u64,
    pub var_decode_kv_tokens: f64,
    pub num_queued_prefill: u32,
    pub sum_queued_prefill_tokens: u64,
    pub var_queued_prefill_length: f64,
    pub num_queued_decode: u32,
    pub sum_queued_decode_kv_tokens: u64,
    pub var_queued_decode_kv_tokens: f64,
    pub wall_time_secs: f64,
}
