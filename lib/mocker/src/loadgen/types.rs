// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, WorkerId, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use dynamo_tokens::SequenceHash;
use uuid::Uuid;

use crate::common::protocols::DirectRequest;

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

#[derive(Debug, Clone, PartialEq)]
pub enum DynamoRequestTrace {
    Standard(Trace),
    Agentic(AgenticTrace),
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
    pub hash_ids: Vec<u64>,
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
    pub hash_ids: Vec<u64>,
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
}

#[derive(Debug, Clone, Copy)]
pub enum SequenceHashMode {
    Raw,
    Cumulative,
}

#[derive(Debug, Clone, Copy)]
pub enum SessionPartitionSpec {
    Random { num_partitions: usize, seed: u64 },
    RoundRobin { num_partitions: usize },
}

#[derive(Debug, Clone)]
pub struct RouterSequence {
    pub worker_id: WorkerId,
    pub local_hashes: Vec<LocalBlockHash>,
    pub external_hashes: Vec<ExternalSequenceBlockHash>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRequestHashes {
    pub local_block_hashes: Vec<LocalBlockHash>,
    pub sequence_hashes: Vec<SequenceHash>,
}

impl ReplayRequestHashes {
    pub(crate) fn from_tokens(tokens: &[u32], engine_block_size: u32) -> Self {
        let local_block_hashes =
            compute_block_hash_for_seq(tokens, engine_block_size, BlockHashOptions::default());
        let sequence_hashes = compute_seq_hash_for_block(&local_block_hashes);

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
    pub replay_key: Option<String>,
    pub scheduled_ready_at_ms: f64,
    pub replay_hashes: Option<ReplayRequestHashes>,
    pub request: DirectRequest,
}
