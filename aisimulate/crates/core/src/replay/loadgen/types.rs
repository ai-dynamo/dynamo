// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::trace::synthesize_validated_trace_tokens;
use crate::replay::protocol::DirectRequest;

pub const OUTPUT_REPLAY_ID_ANNOTATION_KEY: &str = "output_replay_id";
pub const OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY: &str = "output_replay_consumer";

pub fn output_replay_id_annotation(replay_key: &str) -> String {
    format!("{OUTPUT_REPLAY_ID_ANNOTATION_KEY}:{replay_key}")
}

pub fn effective_replay_key(
    request_id: Option<&str>,
    session_id: Option<&str>,
    turn_index: usize,
    line_index: usize,
) -> String {
    if let Some(request_id) = request_id.map(str::trim).filter(|value| !value.is_empty()) {
        return request_id.to_string();
    }
    if let Some(session_id) = session_id.map(str::trim).filter(|value| !value.is_empty()) {
        return format!("{session_id}:{turn_index}");
    }
    format!("line:{line_index}")
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    pub block_size: usize,
    pub sessions: Vec<SessionTrace>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgenticTrace {
    pub block_size: usize,
    pub turns: Vec<AgenticTurnTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceFileFormat {
    Mooncake,
    /// Mooncake-shaped rows where follow-up turns contain new input deltas.
    /// Offline replay accumulates each generated output and the next input delta
    /// per session before computing engine block hashes. Use this only for delta
    /// traces: it expands compact session turns into cumulative prompts and can
    /// use much more memory than `Mooncake`.
    MooncakeDelta,
    /// Mooncake request/cache rows plus explicit request-level workflow
    /// dependencies. Each row dispatches after `wait_for` completions plus its
    /// authored delay/tool wait.
    AgenticMooncake,
    AppliedComputeAgentic,
    Dynamo,
}

/// One producer-neutral Mooncake request row.
///
/// This mirrors the external JSONL schema without depending on Dynamo's
/// producer-side `dynamo-data-gen` crate. Dynamo request-trace lowering lives
/// in the Mocker adapter and converts into this DTO.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MooncakeRow {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, alias = "input_tokens")]
    pub input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    pub output_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub hash_ids: Option<Vec<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "created_time"
    )]
    pub timestamp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "delay_ms")]
    pub delay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_class: Option<String>,
}

/// Harness tool span attached to an agentic Mooncake row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgenticToolEvent {
    pub tool_call_id: String,
    pub tool_class: String,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: u64,
    pub duration_ms: f64,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
}

/// One producer-neutral agentic Mooncake request row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticMooncakeRow {
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, alias = "input_tokens")]
    pub input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    pub output_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub hash_ids: Option<Vec<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "created_time"
    )]
    pub timestamp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "delay_ms")]
    pub delay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wait_for: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_reset: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_wait_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_events: Vec<AgenticToolEvent>,
}

impl AgenticMooncakeRow {
    pub(crate) fn dependency_delay_ms(&self) -> f64 {
        self.delay.unwrap_or(0.0) + self.tool_wait_ms.unwrap_or(0.0)
    }
}

impl TraceFileFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mooncake => "mooncake",
            Self::MooncakeDelta => "mooncake-delta",
            Self::AgenticMooncake => "agentic_mooncake",
            Self::AppliedComputeAgentic => "applied_compute_agentic",
            Self::Dynamo => "dynamo",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionTrace {
    pub session_id: String,
    pub first_arrival_timestamp_ms: Option<f64>,
    pub turns: Vec<TurnTrace>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnTrace {
    pub input_length: usize,
    pub max_output_tokens: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub replay_key: Option<String>,
    pub hash_ids: Vec<u32>,
    pub delay_after_previous_ms: f64,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgenticTurnTrace {
    pub request_id: String,
    pub session_id: String,
    pub input_length: usize,
    pub max_output_tokens: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub replay_key: Option<String>,
    pub hash_ids: Vec<u32>,
    pub first_ready_timestamp_ms: Option<f64>,
    pub delay_after_dependencies_ms: f64,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub wait_for: Vec<String>,
    pub prefix_reset: bool,
}

#[derive(Debug, Clone)]
pub struct LengthSpec {
    pub mean: usize,
    pub stddev: f64,
}

#[derive(Debug, Clone)]
pub enum ArrivalSpec {
    Burst,
    ConstantQps { qps: f64 },
    PoissonQps { qps: f64 },
    GammaQps { qps: f64, smoothness: f64 },
}

#[derive(Debug, Clone)]
pub enum DelaySpec {
    None,
    ConstantMs(f64),
    ExponentialMs { mean_ms: f64 },
}

#[derive(Debug, Clone)]
pub struct SyntheticTraceSpec {
    pub block_size: usize,
    pub num_sessions: usize,
    pub turns_per_session: usize,
    pub input_tokens: LengthSpec,
    pub output_tokens: LengthSpec,
    pub shared_prefix_ratio: f64,
    pub num_prefix_groups: usize,
    pub first_turn_arrivals: ArrivalSpec,
    pub inter_turn_delays: DelaySpec,
    pub seed: u64,
    pub arrival_seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum SessionPartitionSpec {
    Random { num_partitions: usize, seed: u64 },
    RoundRobin { num_partitions: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRequestHashes {
    /// Token-only hashes for complete engine blocks.
    ///
    /// These remain raw identities here; a Dynamo placement adapter converts
    /// them to `LocalBlockHash` at its API boundary.
    pub local_block_hashes: Vec<u64>,
    /// Rolling, sequence-aware identities for the same complete blocks.
    pub sequence_hashes: Vec<u64>,
}

impl ReplayRequestHashes {
    /// Materialize stable replay block identities without importing a concrete
    /// Router implementation or its wire types.
    pub fn from_tokens(tokens: &[u32], engine_block_size: u32) -> Self {
        if engine_block_size == 0 {
            return Self {
                local_block_hashes: Vec::new(),
                sequence_hashes: Vec::new(),
            };
        }

        let block_size = engine_block_size as usize;
        let local_block_hashes = tokens
            .chunks_exact(block_size)
            .map(|block| {
                let mut bytes = Vec::with_capacity(std::mem::size_of_val(block));
                for token in block {
                    bytes.extend_from_slice(&token.to_le_bytes());
                }
                xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, 1337)
            })
            .collect::<Vec<_>>();
        let mut sequence_hashes = Vec::with_capacity(local_block_hashes.len());
        for &block_hash in &local_block_hashes {
            let sequence_hash =
                sequence_hashes
                    .last()
                    .copied()
                    .map_or(block_hash, |parent: u64| {
                        let mut bytes = [0_u8; std::mem::size_of::<[u64; 2]>()];
                        bytes[..8].copy_from_slice(&parent.to_le_bytes());
                        bytes[8..].copy_from_slice(&block_hash.to_le_bytes());
                        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, 1337)
                    });
            sequence_hashes.push(sequence_hash);
        }

        Self {
            local_block_hashes,
            sequence_hashes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReadyTurn {
    pub request_uuid: Uuid,
    pub session_id: String,
    pub turn_index: usize,
    pub emit_session_metadata: bool,
    pub replay_key: Option<String>,
    pub scheduled_ready_at_ms: f64,
    pub replay_hashes: Option<ReplayRequestHashes>,
    pub request: DirectRequest,
}

/// A request whose prompt may still be represented by one hash id per trace
/// block. Offline replay keeps this compact form while an aggregated or
/// prefill router queues the request and materializes tokens only when a
/// worker admits it.
#[doc(hidden)]
#[derive(Debug)]
pub enum ReplayRequestPayload {
    Materialized(DirectRequest),
    Deferred {
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: Vec<u32>,
        trace_block_size: usize,
    },
}

impl ReplayRequestPayload {
    pub fn materialized(request: DirectRequest) -> Self {
        Self::Materialized(request)
    }

    pub fn deferred(
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: Vec<u32>,
        trace_block_size: usize,
    ) -> Self {
        debug_assert!(request_metadata.tokens.is_empty());
        Self::Deferred {
            request_metadata,
            input_length,
            hash_ids,
            trace_block_size,
        }
    }

    pub fn input_length(&self) -> usize {
        match self {
            Self::Materialized(request) => request.tokens.len(),
            Self::Deferred { input_length, .. } => *input_length,
        }
    }

    pub fn metadata(&self) -> &DirectRequest {
        match self {
            Self::Materialized(request) => request,
            Self::Deferred {
                request_metadata, ..
            } => request_metadata,
        }
    }

    pub fn metadata_mut(&mut self) -> &mut DirectRequest {
        match self {
            Self::Materialized(request) => request,
            Self::Deferred {
                request_metadata, ..
            } => request_metadata,
        }
    }

    pub fn materialized_tokens(&self) -> Option<&[u32]> {
        match self {
            Self::Materialized(request) => Some(&request.tokens),
            Self::Deferred { .. } => None,
        }
    }

    pub fn materialized_request(&self) -> Option<&DirectRequest> {
        match self {
            Self::Materialized(request) => Some(request),
            Self::Deferred { .. } => None,
        }
    }

    pub fn prompt_tokens(&self) -> Vec<u32> {
        match self {
            Self::Materialized(request) => request.tokens.clone(),
            Self::Deferred {
                input_length,
                hash_ids,
                trace_block_size,
                ..
            } => synthesize_validated_trace_tokens(*input_length, hash_ids, *trace_block_size),
        }
    }

    pub fn into_direct_request(self) -> DirectRequest {
        match self {
            Self::Materialized(request) => request,
            Self::Deferred {
                mut request_metadata,
                input_length,
                hash_ids,
                trace_block_size,
            } => {
                request_metadata.tokens =
                    synthesize_validated_trace_tokens(input_length, &hash_ids, trace_block_size);
                request_metadata
            }
        }
    }

    pub fn materialize(&mut self) -> Option<&DirectRequest> {
        if matches!(self, Self::Deferred { .. }) {
            let payload = std::mem::replace(self, Self::Materialized(DirectRequest::default()));
            *self = Self::Materialized(payload.into_direct_request());
        }
        self.materialized_request()
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CompactReadyTurn {
    pub request_uuid: Uuid,
    pub session_id: String,
    pub turn_index: usize,
    pub replay_key: Option<String>,
    pub scheduled_ready_at_ms: f64,
    pub replay_hashes: Option<ReplayRequestHashes>,
    pub emit_session_metadata: bool,
    pub request: ReplayRequestPayload,
}

impl CompactReadyTurn {
    #[doc(hidden)]
    pub fn into_ready_turn(self) -> ReadyTurn {
        ReadyTurn {
            request_uuid: self.request_uuid,
            session_id: self.session_id,
            turn_index: self.turn_index,
            emit_session_metadata: self.emit_session_metadata,
            replay_key: self.replay_key,
            scheduled_ready_at_ms: self.scheduled_ready_at_ms,
            replay_hashes: self.replay_hashes,
            request: self.request.into_direct_request(),
        }
    }
}
