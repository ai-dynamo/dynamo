// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use dynamo_data_gen::request_trace::{
    agentic::lower_agentic_mooncake_rows,
    load::{RequestTraceMode, load_request_trace_records},
    mooncake::lower_mooncake_rows,
};
use dynamo_data_gen::{AgenticMooncakeRow, MooncakeRow};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, WorkerId, XXH3_SEED, compute_seq_hash_for_block,
};
use dynamo_tokens::compute_hash_v2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use uuid::Uuid;

use super::driver::WorkloadDriver;
use super::types::{
    AgenticTrace, AgenticTurnTrace, ArrivalSpec, DelaySpec, DynamoRequestTrace, LengthSpec,
    ReplayRequestHashes, RouterSequence, SequenceHashMode, SessionPartitionSpec, SessionTrace,
    SyntheticTraceSpec, Trace, TraceFileFormat, TurnTrace, effective_replay_key,
};
use super::{SYNTHETIC_OUTPUT_SEED, planned_output_token_ids};
use crate::common::protocols::DirectRequest;

#[derive(Debug, Deserialize)]
struct RawAppliedComputeAgenticRecord {
    num_turns: usize,
    input_prompt_length: usize,
    assistant_response_length: Vec<usize>,
    tool_call_output_length: Vec<usize>,
    tool_call_latency: Vec<f64>,
    final_assistant_response_length: usize,
}

impl DynamoRequestTrace {
    pub fn from_request_trace_files(
        paths: &[PathBuf],
        expected_block_size: Option<usize>,
    ) -> Result<Self> {
        validate_trace_files(TraceFileFormat::Dynamo, paths)?;

        let loaded = load_request_trace_records(paths)?;
        match loaded.mode()? {
            RequestTraceMode::Standard => {
                let mut builder = None;
                let mut row_index = 0;
                let block_size = lower_mooncake_rows(loaded.requests, |block_size, row| {
                    let builder =
                        builder.get_or_insert_with(|| MooncakeTraceBuilder::new(block_size));
                    builder.push(row_index, row)?;
                    row_index += 1;
                    Ok(())
                })?;
                validate_dynamo_trace_block_size(expected_block_size, block_size)?;
                let builder = builder.expect("request trace lowering must emit at least one row");
                Ok(Self::Standard(builder.finish()))
            }
            RequestTraceMode::Agentic => {
                let mut builder = None;
                let mut row_index = 0;
                let block_size = lower_agentic_mooncake_rows(loaded, |block_size, row| {
                    let builder =
                        builder.get_or_insert_with(|| AgenticTraceBuilder::new(block_size));
                    builder.push(row_index, row)?;
                    row_index += 1;
                    Ok(())
                })?;
                validate_dynamo_trace_block_size(expected_block_size, block_size)?;
                let builder = builder.expect("request trace lowering must emit at least one row");
                Ok(Self::Agentic(builder.finish()?))
            }
        }
    }
}

pub fn validate_trace_files(format: TraceFileFormat, paths: &[PathBuf]) -> Result<()> {
    if paths.is_empty() {
        bail!("trace replay requires at least one trace file");
    }
    if format != TraceFileFormat::Dynamo && paths.len() != 1 {
        bail!(
            "trace_format='{}' requires exactly one trace file, got {}",
            format.as_str(),
            paths.len()
        );
    }
    Ok(())
}

fn validate_dynamo_trace_block_size(expected: Option<usize>, embedded: usize) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if expected != embedded {
        bail!(
            "trace_block_size {expected} does not match embedded Dynamo request trace block size {embedded}"
        );
    }
    Ok(())
}

fn single_turn_request_uuid(_request_ordinal: usize) -> Uuid {
    #[cfg(feature = "replay-bench")]
    {
        Uuid::from_u128(_request_ordinal as u128 + 1)
    }
    #[cfg(not(feature = "replay-bench"))]
    {
        Uuid::new_v4()
    }
}

pub(super) fn validate_synthesizable_prompt(
    input_length: usize,
    hash_ids: &[u64],
    trace_block_size: usize,
) -> Result<()> {
    if trace_block_size == 0 {
        bail!("trace_block_size must be greater than 0");
    }
    let synthesizable_capacity = hash_ids
        .len()
        .checked_mul(trace_block_size)
        .context("synthesized prompt capacity overflow")?;
    let required_hash_ids = input_length.div_ceil(trace_block_size);
    if hash_ids.len() < required_hash_ids {
        bail!(
            "input_length {} exceeds synthesized capacity {}",
            input_length,
            synthesizable_capacity
        );
    }

    Ok(())
}

pub(super) fn synthesize_trace_tokens(
    input_length: usize,
    hash_ids: &[u64],
    trace_block_size: usize,
) -> Result<Vec<u32>> {
    validate_synthesizable_prompt(input_length, hash_ids, trace_block_size)?;

    let mut tokens = Vec::with_capacity(input_length);
    for &hash_id in hash_ids {
        let token_id = hash_id as u32;
        let remaining = input_length - tokens.len();
        tokens.extend(std::iter::repeat_n(
            token_id,
            remaining.min(trace_block_size),
        ));
        if tokens.len() == input_length {
            break;
        }
    }

    if tokens.len() != input_length {
        bail!(
            "failed to synthesize {} tokens from {} hash_ids",
            input_length,
            hash_ids.len()
        );
    }

    Ok(tokens)
}

fn trace_to_replay_hashes(
    input_length: usize,
    hash_ids: &[u64],
    trace_block_size: usize,
    engine_block_size: usize,
) -> Result<ReplayRequestHashes> {
    if engine_block_size == 0 {
        bail!("engine_block_size must be greater than 0");
    }

    let tokens = synthesize_trace_tokens(input_length, hash_ids, trace_block_size)?;
    let engine_block_size =
        u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
    Ok(ReplayRequestHashes::from_tokens(&tokens, engine_block_size))
}

impl TurnTrace {
    pub fn synthesize_tokens(&self, trace_block_size: usize) -> Result<Vec<u32>> {
        synthesize_trace_tokens(self.input_length, &self.hash_ids, trace_block_size)
    }

    pub fn to_direct_request(
        &self,
        trace_block_size: usize,
        request_uuid: Uuid,
        arrival_timestamp_ms: Option<f64>,
    ) -> Result<DirectRequest> {
        let tokens = self.synthesize_tokens(trace_block_size)?;
        Ok(DirectRequest {
            tokens,
            max_output_tokens: self.max_output_tokens,
            output_token_ids: self.output_token_ids.clone(),
            uuid: Some(request_uuid),
            dp_rank: 0,
            arrival_timestamp_ms,
            priority: self.priority,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class.clone(),
        })
    }

    pub fn to_replay_hashes(
        &self,
        trace_block_size: usize,
        engine_block_size: usize,
    ) -> Result<ReplayRequestHashes> {
        trace_to_replay_hashes(
            self.input_length,
            &self.hash_ids,
            trace_block_size,
            engine_block_size,
        )
    }
}

impl AgenticTurnTrace {
    pub fn synthesize_tokens(&self, trace_block_size: usize) -> Result<Vec<u32>> {
        synthesize_trace_tokens(self.input_length, &self.hash_ids, trace_block_size)
    }

    pub fn to_replay_hashes(
        &self,
        trace_block_size: usize,
        engine_block_size: usize,
    ) -> Result<ReplayRequestHashes> {
        trace_to_replay_hashes(
            self.input_length,
            &self.hash_ids,
            trace_block_size,
            engine_block_size,
        )
    }
}

struct MooncakeTraceBuilder {
    trace_block_size: usize,
    sessions: Vec<SessionTrace>,
    session_indices: HashMap<String, usize>,
    last_timestamps: Vec<Option<f64>>,
}

impl MooncakeTraceBuilder {
    fn new(trace_block_size: usize) -> Self {
        Self {
            trace_block_size,
            sessions: Vec::new(),
            session_indices: HashMap::new(),
            last_timestamps: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    fn push(&mut self, line_idx: usize, raw: MooncakeRow) -> Result<()> {
        let request_id = raw.request_id;
        let raw_session_id = raw.session_id;
        let session_id = raw_session_id
            .clone()
            .unwrap_or_else(|| format!("request_{}", line_idx + 1));
        let hash_ids = raw
            .hash_ids
            .ok_or_else(|| anyhow!("trace line {} is missing hash_ids", line_idx + 1))?;
        // Clamp input_length to the synthesizable capacity: in the mooncake
        // trace format, input_length is the full prompt token count which may
        // exceed hash_ids.len() * block_size (cached portion only).
        let synthesizable_capacity = hash_ids
            .len()
            .checked_mul(self.trace_block_size)
            .ok_or_else(|| anyhow!("trace line {} synthesized capacity overflow", line_idx + 1))?;
        let input_length = raw
            .input_length
            .unwrap_or(synthesizable_capacity)
            .min(synthesizable_capacity);
        let output_length = raw
            .output_length
            .ok_or_else(|| anyhow!("trace line {} is missing output_length", line_idx + 1))?;
        let output_token_ids = raw.output_token_ids;
        if let Some(output_token_ids) = output_token_ids.as_ref()
            && output_token_ids.len() != output_length
        {
            bail!(
                "trace line {} output_length {} does not match output_token_ids length {}",
                line_idx + 1,
                output_length,
                output_token_ids.len()
            );
        }
        let timestamp_ms = raw.timestamp;
        let explicit_delay_ms = raw.delay;
        let priority = raw.priority.unwrap_or(0);
        let strict_priority = raw.strict_priority.unwrap_or(0);
        let policy_class = raw.policy_class.clone();

        let session_index = *self
            .session_indices
            .entry(session_id.clone())
            .or_insert_with(|| {
                let idx = self.sessions.len();
                self.sessions.push(SessionTrace {
                    session_id: session_id.clone(),
                    first_arrival_timestamp_ms: timestamp_ms,
                    turns: Vec::new(),
                });
                self.last_timestamps.push(timestamp_ms);
                idx
            });

        let session = self
            .sessions
            .get_mut(session_index)
            .expect("newly inserted session must exist");
        let turn_idx = session.turns.len();
        let replay_key = output_token_ids.as_ref().map(|_| {
            effective_replay_key(
                request_id.as_deref(),
                raw_session_id.as_deref(),
                turn_idx,
                line_idx,
            )
        });
        let delay_after_previous_ms = if turn_idx == 0 {
            let delay = explicit_delay_ms.unwrap_or(0.0);
            if delay != 0.0 {
                bail!(
                    "trace line {} sets delay on the first turn of session {}",
                    line_idx + 1,
                    session.session_id
                );
            }
            0.0
        } else if let Some(delay_ms) = explicit_delay_ms {
            delay_ms
        } else if let Some(timestamp_ms) = timestamp_ms {
            let previous_timestamp_ms = self.last_timestamps[session_index].ok_or_else(|| {
                anyhow!(
                    "trace line {} for session {} cannot infer delay without a previous timestamp",
                    line_idx + 1,
                    session.session_id
                )
            })?;
            timestamp_ms - previous_timestamp_ms
        } else {
            0.0
        };

        if !delay_after_previous_ms.is_finite() || delay_after_previous_ms < 0.0 {
            bail!(
                "trace line {} has invalid delay {}",
                line_idx + 1,
                delay_after_previous_ms
            );
        }

        if hash_ids.len() * self.trace_block_size < input_length {
            bail!(
                "trace line {} input_length {} exceeds synthesized capacity {}",
                line_idx + 1,
                input_length,
                hash_ids.len() * self.trace_block_size
            );
        }

        session.turns.push(TurnTrace {
            input_length,
            max_output_tokens: output_length,
            output_token_ids,
            replay_key,
            hash_ids,
            delay_after_previous_ms,
            priority,
            strict_priority,
            policy_class,
        });
        if let Some(timestamp_ms) = timestamp_ms {
            self.last_timestamps[session_index] = Some(timestamp_ms);
        }
        Ok(())
    }

    fn finish(self) -> Trace {
        Trace {
            block_size: self.trace_block_size,
            sessions: self.sessions,
        }
    }
}

impl Trace {
    pub fn from_mooncake(path: &Path, trace_block_size: usize) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }

        let file = File::open(path)
            .with_context(|| format!("failed to open trace file {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut builder = MooncakeTraceBuilder::new(trace_block_size);

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed to read line {} from {}",
                    line_idx + 1,
                    path.display()
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }

            let row = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse line {} from {} as JSON",
                    line_idx + 1,
                    path.display()
                )
            })?;
            builder.push(line_idx, row)?;
        }

        if builder.is_empty() {
            bail!("trace file {} did not contain any requests", path.display());
        }

        Ok(builder.finish())
    }

    pub fn from_mooncake_rows(rows: Vec<MooncakeRow>, trace_block_size: usize) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }
        let mut builder = MooncakeTraceBuilder::new(trace_block_size);
        for (line_idx, row) in rows.into_iter().enumerate() {
            builder.push(line_idx, row)?;
        }
        if builder.is_empty() {
            bail!("Mooncake rows did not contain any requests");
        }
        Ok(builder.finish())
    }

    pub fn from_applied_compute_agentic(
        path: &Path,
        trace_block_size: usize,
        shared_prefix_ratio: f64,
        num_prefix_groups: usize,
    ) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }
        if !(0.0..=1.0).contains(&shared_prefix_ratio) {
            bail!(
                "shared_prefix_ratio must be between 0.0 and 1.0, got {}",
                shared_prefix_ratio
            );
        }

        let file = File::open(path)
            .with_context(|| format!("failed to open trace file {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut sessions = Vec::new();
        let mut next_unique_hash = 1_u64;

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed to read line {} from {}",
                    line_idx + 1,
                    path.display()
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }

            let raw: RawAppliedComputeAgenticRecord =
                serde_json::from_str(&line).with_context(|| {
                    format!(
                        "failed to parse line {} from {} as JSON",
                        line_idx + 1,
                        path.display()
                    )
                })?;

            for (name, values) in [
                (
                    "assistant_response_length",
                    raw.assistant_response_length.len(),
                ),
                ("tool_call_output_length", raw.tool_call_output_length.len()),
                ("tool_call_latency", raw.tool_call_latency.len()),
            ] {
                if values != raw.num_turns {
                    bail!(
                        "trace line {} field {} length {} does not match num_turns {}",
                        line_idx + 1,
                        name,
                        values,
                        raw.num_turns
                    );
                }
            }

            if raw.input_prompt_length == 0 {
                bail!(
                    "trace line {} input_prompt_length must be positive",
                    line_idx + 1
                );
            }

            let group_id = if shared_prefix_ratio > 0.0 && num_prefix_groups > 0 {
                Some(line_idx % num_prefix_groups)
            } else {
                None
            };
            let mut current_input_length = raw.input_prompt_length;
            let mut hash_ids = Vec::new();
            let shared_initial_blocks = ((current_input_length.div_ceil(trace_block_size) as f64)
                * shared_prefix_ratio)
                .round() as usize;
            extend_applied_compute_agentic_hash_ids(
                &mut hash_ids,
                current_input_length,
                trace_block_size,
                shared_initial_blocks,
                group_id,
                &mut next_unique_hash,
            )?;

            let mut turns = Vec::with_capacity(raw.num_turns + 1);
            let mut next_turn_delay_ms = 0.0;
            for turn_idx in 0..raw.num_turns {
                let tool_call_latency = raw.tool_call_latency[turn_idx];
                if !tool_call_latency.is_finite() || tool_call_latency < 0.0 {
                    bail!(
                        "trace line {} tool_call_latency[{}] must be a finite non-negative number",
                        line_idx + 1,
                        turn_idx
                    );
                }

                turns.push(TurnTrace {
                    input_length: current_input_length,
                    max_output_tokens: raw.assistant_response_length[turn_idx],
                    hash_ids: hash_ids.clone(),
                    delay_after_previous_ms: next_turn_delay_ms,
                    ..Default::default()
                });

                current_input_length = current_input_length
                    .checked_add(raw.assistant_response_length[turn_idx])
                    .and_then(|value| value.checked_add(raw.tool_call_output_length[turn_idx]))
                    .ok_or_else(|| {
                        anyhow!(
                            "trace line {} cumulative input length overflow",
                            line_idx + 1
                        )
                    })?;
                extend_applied_compute_agentic_hash_ids(
                    &mut hash_ids,
                    current_input_length,
                    trace_block_size,
                    shared_initial_blocks,
                    group_id,
                    &mut next_unique_hash,
                )?;
                next_turn_delay_ms = tool_call_latency * 1000.0;
            }

            turns.push(TurnTrace {
                input_length: current_input_length,
                max_output_tokens: raw.final_assistant_response_length,
                hash_ids,
                delay_after_previous_ms: next_turn_delay_ms,
                ..Default::default()
            });

            sessions.push(SessionTrace {
                session_id: format!("applied_compute_agentic_session_{}", line_idx + 1),
                first_arrival_timestamp_ms: None,
                turns,
            });
        }

        if sessions.is_empty() {
            bail!("trace file {} did not contain any requests", path.display());
        }

        Ok(Self {
            block_size: trace_block_size,
            sessions,
        })
    }

    pub fn synthetic(spec: SyntheticTraceSpec) -> Result<Self> {
        if spec.block_size == 0 {
            bail!("block_size must be greater than 0");
        }
        if spec.num_sessions == 0 {
            bail!("num_sessions must be greater than 0");
        }
        if spec.turns_per_session == 0 {
            bail!("turns_per_session must be greater than 0");
        }
        if !(0.0..=1.0).contains(&spec.shared_prefix_ratio) {
            bail!(
                "shared_prefix_ratio must be between 0.0 and 1.0, got {}",
                spec.shared_prefix_ratio
            );
        }

        let mut rng = StdRng::seed_from_u64(spec.seed);
        let mut sessions = Vec::with_capacity(spec.num_sessions);
        let mut first_arrivals = Vec::with_capacity(spec.num_sessions);
        let mean_gap_ms = arrival_spec_mean_gap_ms(&spec.first_turn_arrivals)?;
        let mut next_arrival_ms = 0.0;

        for session_idx in 0..spec.num_sessions {
            if session_idx == 0 {
                first_arrivals.push(0.0);
                continue;
            }
            next_arrival_ms +=
                sample_arrival_gap_ms(&spec.first_turn_arrivals, mean_gap_ms, &mut rng)?;
            first_arrivals.push(next_arrival_ms);
        }

        let mut next_unique_hash = 1_u64;
        for (session_idx, first_arrival_timestamp_ms) in first_arrivals.into_iter().enumerate() {
            let group_id = if spec.num_prefix_groups > 0 && spec.shared_prefix_ratio > 0.0 {
                Some(rng.random_range(0..spec.num_prefix_groups) as u64)
            } else {
                None
            };
            let mut turns = Vec::with_capacity(spec.turns_per_session);
            for turn_idx in 0..spec.turns_per_session {
                let input_length = sample_length(&spec.input_tokens, 1, &mut rng);
                let max_output_tokens = sample_length(&spec.output_tokens, 1, &mut rng);
                let num_blocks = input_length.div_ceil(spec.block_size);
                let prefix_blocks =
                    ((num_blocks as f64) * spec.shared_prefix_ratio).round() as usize;
                let prefix_blocks = prefix_blocks.min(num_blocks);
                let mut hash_ids = Vec::with_capacity(num_blocks);

                for block_idx in 0..prefix_blocks {
                    if let Some(group_id) = group_id {
                        hash_ids.push(0xD00D_0000_0000_0000 | (group_id << 32) | block_idx as u64);
                    }
                }

                while hash_ids.len() < num_blocks {
                    hash_ids.push(next_unique_hash);
                    next_unique_hash = next_unique_hash
                        .checked_add(1)
                        .expect("synthetic hash id overflow");
                }

                turns.push(TurnTrace {
                    input_length,
                    max_output_tokens,
                    hash_ids,
                    delay_after_previous_ms: if turn_idx == 0 {
                        0.0
                    } else {
                        sample_delay_ms(&spec.inter_turn_delays, &mut rng)?
                    },
                    ..Default::default()
                });
            }

            sessions.push(SessionTrace {
                session_id: format!("session_{session_idx}"),
                first_arrival_timestamp_ms: Some(first_arrival_timestamp_ms),
                turns,
            });
        }

        Ok(Self {
            block_size: spec.block_size,
            sessions,
        })
    }

    pub fn validate_for_trace_mode(&self) -> Result<()> {
        self.validate(false)
    }

    pub fn validate_for_concurrency_mode(&self) -> Result<()> {
        self.validate(true)
    }

    pub fn normalize_session_starts(mut self) -> Result<Self> {
        let Some(min_timestamp_ms) = self
            .sessions
            .iter()
            .filter_map(|session| session.first_arrival_timestamp_ms)
            .min_by(|left, right| left.total_cmp(right))
        else {
            return Ok(self);
        };

        for session in &mut self.sessions {
            if let Some(timestamp_ms) = session.first_arrival_timestamp_ms.as_mut() {
                *timestamp_ms -= min_timestamp_ms;
            }
        }
        Ok(self)
    }

    pub fn speed_up_timing(mut self, ratio: f64) -> Result<Self> {
        if !ratio.is_finite() || ratio <= 0.0 {
            bail!("ratio must be a finite positive number, got {ratio}");
        }

        for session in &mut self.sessions {
            if let Some(timestamp_ms) = session.first_arrival_timestamp_ms.as_mut() {
                *timestamp_ms /= ratio;
            }
            for turn in &mut session.turns {
                turn.delay_after_previous_ms /= ratio;
            }
        }
        Ok(self)
    }

    pub fn rescale_session_start_span(mut self, duration_ms: u64) -> Result<Self> {
        let Some(min_timestamp_ms) = self
            .sessions
            .iter()
            .filter_map(|session| session.first_arrival_timestamp_ms)
            .min_by(|left, right| left.total_cmp(right))
        else {
            return Ok(self);
        };
        let Some(max_timestamp_ms) = self
            .sessions
            .iter()
            .filter_map(|session| session.first_arrival_timestamp_ms)
            .max_by(|left, right| left.total_cmp(right))
        else {
            return Ok(self);
        };

        let target_span_ms = duration_ms as f64;
        let source_span_ms = max_timestamp_ms - min_timestamp_ms;
        for session in &mut self.sessions {
            if let Some(timestamp_ms) = session.first_arrival_timestamp_ms.as_mut() {
                *timestamp_ms = if source_span_ms == 0.0 {
                    0.0
                } else {
                    (*timestamp_ms - min_timestamp_ms) * target_span_ms / source_span_ms
                };
            }
        }
        Ok(self)
    }

    pub fn rescale_ready_span(mut self, duration_ms: u64) -> Result<Self> {
        let Some(min_start_ms) = self
            .sessions
            .iter()
            .map(|session| session.first_arrival_timestamp_ms.unwrap_or(0.0))
            .min_by(|left, right| left.total_cmp(right))
        else {
            return Ok(self);
        };

        let Some(max_ready_ms) = self
            .sessions
            .iter()
            .map(|session| {
                session.first_arrival_timestamp_ms.unwrap_or(0.0)
                    + session
                        .turns
                        .iter()
                        .enumerate()
                        .filter(|(turn_idx, _)| *turn_idx > 0)
                        .map(|(_, turn)| turn.delay_after_previous_ms)
                        .sum::<f64>()
            })
            .max_by(|left, right| left.total_cmp(right))
        else {
            return Ok(self);
        };

        let ratio = duration_ms as f64 / (max_ready_ms - min_start_ms).max(1.0);
        for session in &mut self.sessions {
            if let Some(start_ms) = session.first_arrival_timestamp_ms.as_mut() {
                *start_ms = (*start_ms - min_start_ms) * ratio;
            }
            for (turn_idx, turn) in session.turns.iter_mut().enumerate() {
                if turn_idx > 0 {
                    turn.delay_after_previous_ms *= ratio;
                }
            }
        }
        Ok(self)
    }

    pub fn expand_hash_prefix_depth(mut self, factor: usize) -> Self {
        if factor <= 1 {
            return self;
        }
        for session in &mut self.sessions {
            for turn in &mut session.turns {
                turn.input_length = turn
                    .input_length
                    .checked_mul(factor)
                    .expect("input_length expansion overflow");
                turn.hash_ids = turn
                    .hash_ids
                    .iter()
                    .flat_map(|&hash_id| {
                        let base = hash_id
                            .checked_mul(factor as u64)
                            .expect("hash prefix expansion overflow");
                        (0..factor as u64).map(move |offset| base + offset)
                    })
                    .collect();
            }
        }
        self
    }

    pub fn duplicate_hash_space(mut self, copies: usize) -> Self {
        if copies <= 1 {
            return self;
        }

        let max_hash_id = self
            .sessions
            .iter()
            .flat_map(|session| session.turns.iter())
            .flat_map(|turn| turn.hash_ids.iter().copied())
            .max()
            .unwrap_or(0);
        let offset_base = max_hash_id + 1;
        let original_sessions = self.sessions.clone();
        self.sessions.clear();

        for copy_idx in 0..copies {
            let offset = offset_base * copy_idx as u64;
            for session in &original_sessions {
                let mut duplicated = session.clone();
                duplicated.session_id = format!("{}:copy_{copy_idx}", session.session_id);
                for turn in &mut duplicated.turns {
                    turn.hash_ids = turn
                        .hash_ids
                        .iter()
                        .map(|&hash_id| {
                            hash_id
                                .checked_add(offset)
                                .expect("hash duplication overflow")
                        })
                        .collect();
                }
                self.sessions.push(duplicated);
            }
        }
        self
    }

    pub fn partition_by_session(&self, spec: SessionPartitionSpec) -> Vec<Self> {
        let num_partitions = match spec {
            SessionPartitionSpec::Random { num_partitions, .. } => num_partitions,
            SessionPartitionSpec::RoundRobin { num_partitions } => num_partitions,
        }
        .max(1);
        let mut partitions = vec![
            Self {
                block_size: self.block_size,
                sessions: Vec::new(),
            };
            num_partitions
        ];

        let mut rng = match spec {
            SessionPartitionSpec::Random { seed, .. } => Some(StdRng::seed_from_u64(seed)),
            SessionPartitionSpec::RoundRobin { .. } => None,
        };

        for (session_idx, session) in self.sessions.iter().cloned().enumerate() {
            let partition_idx = match spec {
                SessionPartitionSpec::Random { .. } => rng
                    .as_mut()
                    .expect("random partitioner must exist")
                    .random_range(0..num_partitions),
                SessionPartitionSpec::RoundRobin { .. } => session_idx % num_partitions,
            };
            partitions[partition_idx].sessions.push(session);
        }

        partitions
    }

    pub fn to_single_turn_requests(&self) -> Result<Vec<DirectRequest>> {
        let mut requests = Vec::with_capacity(self.sessions.len());
        let mut output_rng = StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);
        for (request_ordinal, session) in self.sessions.iter().enumerate() {
            if session.turns.len() != 1 {
                bail!(
                    "to_single_turn_requests requires exactly one turn per session, but session {} has {} turns",
                    session.session_id,
                    session.turns.len()
                );
            }
            let request_uuid = single_turn_request_uuid(request_ordinal);
            let mut request = session.turns[0].to_direct_request(
                self.block_size,
                request_uuid,
                session.first_arrival_timestamp_ms,
            )?;
            request.output_token_ids = Some(planned_output_token_ids(
                request.output_token_ids,
                request.max_output_tokens,
                &mut output_rng,
            ));
            requests.push(request);
        }
        Ok(requests)
    }

    pub fn is_single_turn(&self) -> bool {
        self.sessions.iter().all(|session| session.turns.len() == 1)
    }

    pub fn to_router_sequences(
        &self,
        worker_id: WorkerId,
        hash_mode: SequenceHashMode,
    ) -> Result<Vec<RouterSequence>> {
        let mut sequences = Vec::new();
        for session in &self.sessions {
            for turn in &session.turns {
                let local_hashes = turn
                    .hash_ids
                    .iter()
                    .map(|&hash_id| local_block_hash_from_id(hash_id, self.block_size))
                    .collect::<Vec<_>>();
                let external_hashes = match hash_mode {
                    SequenceHashMode::Raw => local_hashes
                        .iter()
                        .map(|hash| ExternalSequenceBlockHash(hash.0))
                        .collect(),
                    SequenceHashMode::Cumulative => compute_seq_hash_for_block(&local_hashes)
                        .into_iter()
                        .map(ExternalSequenceBlockHash)
                        .collect(),
                };
                sequences.push(RouterSequence {
                    worker_id,
                    local_hashes,
                    external_hashes,
                });
            }
        }
        Ok(sequences)
    }

    pub fn into_trace_driver(self) -> Result<WorkloadDriver> {
        self.validate_for_trace_mode()?;
        let engine_block_size = self.block_size;
        WorkloadDriver::new_trace(self, engine_block_size)
    }

    pub fn into_concurrency_driver(self, max_in_flight: usize) -> Result<WorkloadDriver> {
        self.validate_for_concurrency_mode()?;
        let engine_block_size = self.block_size;
        WorkloadDriver::new_concurrency(self, engine_block_size, max_in_flight)
    }

    pub fn into_trace_driver_with_block_size(
        self,
        engine_block_size: usize,
    ) -> Result<WorkloadDriver> {
        self.validate_for_trace_mode()?;
        WorkloadDriver::new_trace(self, engine_block_size)
    }

    pub fn into_delta_accumulating_trace_driver_with_block_size(
        self,
        engine_block_size: usize,
    ) -> Result<WorkloadDriver> {
        self.validate_for_trace_mode()?;
        WorkloadDriver::new_trace_accumulating_deltas(self, engine_block_size)
    }

    pub fn into_concurrency_driver_with_block_size(
        self,
        engine_block_size: usize,
        max_in_flight: usize,
    ) -> Result<WorkloadDriver> {
        self.validate_for_concurrency_mode()?;
        WorkloadDriver::new_concurrency(self, engine_block_size, max_in_flight)
    }

    pub fn into_delta_accumulating_concurrency_driver_with_block_size(
        self,
        engine_block_size: usize,
        max_in_flight: usize,
    ) -> Result<WorkloadDriver> {
        self.validate_for_concurrency_mode()?;
        WorkloadDriver::new_concurrency_accumulating_deltas(self, engine_block_size, max_in_flight)
    }

    fn validate(&self, allow_missing_first_timestamp: bool) -> Result<()> {
        if self.block_size == 0 {
            bail!("block_size must be greater than 0");
        }
        if self.sessions.is_empty() {
            bail!("trace must contain at least one session");
        }

        for session in &self.sessions {
            if session.turns.is_empty() {
                bail!(
                    "session {} must contain at least one turn",
                    session.session_id
                );
            }
            if !allow_missing_first_timestamp {
                let timestamp_ms = session.first_arrival_timestamp_ms.ok_or_else(|| {
                    anyhow!(
                        "trace mode requires first_arrival_timestamp_ms for session {}",
                        session.session_id
                    )
                })?;
                if !timestamp_ms.is_finite() || timestamp_ms < 0.0 {
                    bail!(
                        "session {} has invalid first_arrival_timestamp_ms {}",
                        session.session_id,
                        timestamp_ms
                    );
                }
            } else if let Some(timestamp_ms) = session.first_arrival_timestamp_ms
                && (!timestamp_ms.is_finite() || timestamp_ms < 0.0)
            {
                bail!(
                    "session {} has invalid first_arrival_timestamp_ms {}",
                    session.session_id,
                    timestamp_ms
                );
            }

            for (turn_idx, turn) in session.turns.iter().enumerate() {
                if let Some(output_token_ids) = turn.output_token_ids.as_ref()
                    && output_token_ids.len() != turn.max_output_tokens
                {
                    bail!(
                        "session {} turn {} max_output_tokens {} does not match output_token_ids length {}",
                        session.session_id,
                        turn_idx,
                        turn.max_output_tokens,
                        output_token_ids.len()
                    );
                }
                if turn.input_length == 0 {
                    bail!(
                        "session {} turn {} must have a positive input_length",
                        session.session_id,
                        turn_idx
                    );
                }
                if turn.hash_ids.is_empty() {
                    bail!(
                        "session {} turn {} must contain at least one hash id",
                        session.session_id,
                        turn_idx
                    );
                }
                validate_synthesizable_prompt(turn.input_length, &turn.hash_ids, self.block_size)
                    .with_context(|| {
                    format!(
                        "session {} turn {} has invalid prompt",
                        session.session_id, turn_idx
                    )
                })?;
                if !turn.delay_after_previous_ms.is_finite() || turn.delay_after_previous_ms < 0.0 {
                    bail!(
                        "session {} turn {} has invalid delay {}",
                        session.session_id,
                        turn_idx,
                        turn.delay_after_previous_ms
                    );
                }
                if turn_idx == 0 && turn.delay_after_previous_ms != 0.0 {
                    bail!(
                        "session {} first turn must have delay_after_previous_ms == 0.0",
                        session.session_id
                    );
                }
            }
        }

        Ok(())
    }
}

struct AgenticTraceBuilder {
    trace_block_size: usize,
    turns: Vec<AgenticTurnTrace>,
    request_ids: std::collections::HashSet<String>,
}

impl AgenticTraceBuilder {
    fn new(trace_block_size: usize) -> Self {
        Self {
            trace_block_size,
            turns: Vec::new(),
            request_ids: std::collections::HashSet::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    fn push(&mut self, line_idx: usize, raw: AgenticMooncakeRow) -> Result<()> {
        if raw.request_id.trim().is_empty() {
            bail!("trace line {} has empty request_id", line_idx + 1);
        }
        if !self.request_ids.insert(raw.request_id.clone()) {
            bail!(
                "trace line {} duplicates request_id {}",
                line_idx + 1,
                raw.request_id
            );
        }

        let delay_after_dependencies_ms = raw.dependency_delay_ms();
        let hash_ids = raw
            .hash_ids
            .ok_or_else(|| anyhow!("trace line {} is missing hash_ids", line_idx + 1))?;
        let synthesizable_capacity = hash_ids
            .len()
            .checked_mul(self.trace_block_size)
            .ok_or_else(|| anyhow!("trace line {} synthesized capacity overflow", line_idx + 1))?;
        let input_length = match raw.input_length {
            Some(input_length) if input_length > synthesizable_capacity => {
                bail!(
                    "trace line {} has input_length {} but only {} tokens can be synthesized from {} hash_ids at trace_block_size {}",
                    line_idx + 1,
                    input_length,
                    synthesizable_capacity,
                    hash_ids.len(),
                    self.trace_block_size
                );
            }
            Some(input_length) => input_length,
            None => synthesizable_capacity,
        };
        let output_length = raw
            .output_length
            .ok_or_else(|| anyhow!("trace line {} is missing output_length", line_idx + 1))?;
        let output_token_ids = raw.output_token_ids;
        if let Some(output_token_ids) = output_token_ids.as_ref()
            && output_token_ids.len() != output_length
        {
            bail!(
                "trace line {} output_length {} does not match output_token_ids length {}",
                line_idx + 1,
                output_length,
                output_token_ids.len()
            );
        }
        if !delay_after_dependencies_ms.is_finite() || delay_after_dependencies_ms < 0.0 {
            bail!(
                "trace line {} has invalid dependency delay {}",
                line_idx + 1,
                delay_after_dependencies_ms
            );
        }
        if let Some(timestamp_ms) = raw.timestamp
            && (!timestamp_ms.is_finite() || timestamp_ms < 0.0)
        {
            bail!(
                "trace line {} has invalid timestamp {}",
                line_idx + 1,
                timestamp_ms
            );
        }

        let replay_key = output_token_ids.as_ref().map(|_| {
            effective_replay_key(
                Some(raw.request_id.as_str()),
                raw.session_id.as_deref(),
                0,
                line_idx,
            )
        });
        self.turns.push(AgenticTurnTrace {
            replay_key,
            request_id: raw.request_id,
            session_id: raw
                .session_id
                .unwrap_or_else(|| format!("request_{}", line_idx + 1)),
            input_length,
            max_output_tokens: output_length,
            output_token_ids,
            hash_ids,
            first_ready_timestamp_ms: raw.timestamp,
            delay_after_dependencies_ms,
            priority: raw.priority.unwrap_or(0),
            strict_priority: raw.strict_priority.unwrap_or(0),
            policy_class: raw.policy_class,
            wait_for: raw.wait_for,
            prefix_reset: raw.prefix_reset.unwrap_or(false),
        });
        Ok(())
    }

    fn finish(self) -> Result<AgenticTrace> {
        for turn in &self.turns {
            for dependency in &turn.wait_for {
                if !self.request_ids.contains(dependency) {
                    bail!(
                        "request {} waits for unknown request_id {}",
                        turn.request_id,
                        dependency
                    );
                }
                if dependency == &turn.request_id {
                    bail!("request {} cannot wait for itself", turn.request_id);
                }
            }
        }
        validate_agentic_trace_is_acyclic(&self.turns)?;

        Ok(AgenticTrace {
            block_size: self.trace_block_size,
            turns: self.turns,
        })
    }
}

impl AgenticTrace {
    pub fn from_agentic_mooncake(path: &Path, trace_block_size: usize) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }

        let file = File::open(path)
            .with_context(|| format!("failed to open trace file {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut builder = AgenticTraceBuilder::new(trace_block_size);

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed to read line {} from {}",
                    line_idx + 1,
                    path.display()
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }

            let row = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse line {} from {} as agentic Mooncake JSON",
                    line_idx + 1,
                    path.display()
                )
            })?;
            builder.push(line_idx, row)?;
        }

        if builder.is_empty() {
            bail!(
                "agentic trace file {} did not contain any requests",
                path.display()
            );
        }

        builder.finish()
    }

    pub fn from_agentic_mooncake_rows(
        rows: Vec<AgenticMooncakeRow>,
        trace_block_size: usize,
    ) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }
        let mut builder = AgenticTraceBuilder::new(trace_block_size);
        for (line_idx, row) in rows.into_iter().enumerate() {
            builder.push(line_idx, row)?;
        }
        if builder.is_empty() {
            bail!("agentic Mooncake rows did not contain any requests");
        }
        builder.finish()
    }

    pub fn normalize_starts(mut self) -> Self {
        let Some(min_timestamp_ms) = self
            .turns
            .iter()
            .filter_map(|turn| turn.first_ready_timestamp_ms)
            .min_by(|left, right| left.total_cmp(right))
        else {
            return self;
        };

        for turn in &mut self.turns {
            if let Some(timestamp_ms) = turn.first_ready_timestamp_ms.as_mut() {
                *timestamp_ms -= min_timestamp_ms;
            }
        }
        self
    }

    pub fn speed_up_timing(mut self, ratio: f64) -> Result<Self> {
        if !ratio.is_finite() || ratio <= 0.0 {
            bail!("ratio must be a finite positive number, got {ratio}");
        }

        for turn in &mut self.turns {
            if let Some(timestamp_ms) = turn.first_ready_timestamp_ms.as_mut() {
                *timestamp_ms /= ratio;
            }
            turn.delay_after_dependencies_ms /= ratio;
        }
        Ok(self)
    }

    pub fn into_trace_driver_with_block_size(
        self,
        engine_block_size: usize,
    ) -> Result<WorkloadDriver> {
        WorkloadDriver::new_agentic_trace(self, engine_block_size)
    }
}

fn validate_agentic_trace_is_acyclic(turns: &[AgenticTurnTrace]) -> Result<()> {
    let mut index_by_id = HashMap::new();
    for (idx, turn) in turns.iter().enumerate() {
        index_by_id.insert(turn.request_id.as_str(), idx);
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        Visiting,
        Done,
    }

    fn visit<'a>(
        idx: usize,
        turns: &'a [AgenticTurnTrace],
        index_by_id: &HashMap<&'a str, usize>,
        marks: &mut Vec<Option<Mark>>,
    ) -> Result<()> {
        match marks[idx] {
            Some(Mark::Done) => return Ok(()),
            Some(Mark::Visiting) => bail!("cycle detected at request {}", turns[idx].request_id),
            None => {}
        }
        marks[idx] = Some(Mark::Visiting);
        for dependency in &turns[idx].wait_for {
            let dep_idx = *index_by_id
                .get(dependency.as_str())
                .expect("dependencies were prevalidated");
            visit(dep_idx, turns, index_by_id, marks)?;
        }
        marks[idx] = Some(Mark::Done);
        Ok(())
    }

    let mut marks = vec![None; turns.len()];
    for idx in 0..turns.len() {
        visit(idx, turns, &index_by_id, &mut marks)?;
    }
    Ok(())
}

fn extend_applied_compute_agentic_hash_ids(
    hash_ids: &mut Vec<u64>,
    input_length: usize,
    trace_block_size: usize,
    shared_initial_blocks: usize,
    group_id: Option<usize>,
    next_unique_hash: &mut u64,
) -> Result<()> {
    let target_blocks = input_length.div_ceil(trace_block_size);
    while hash_ids.len() < target_blocks {
        let block_idx = hash_ids.len();
        if block_idx < shared_initial_blocks
            && let Some(group_id) = group_id
        {
            hash_ids.push(0xA63E_0000_0000_0000 | ((group_id as u64) << 32) | block_idx as u64);
            continue;
        }
        hash_ids.push(*next_unique_hash);
        *next_unique_hash = next_unique_hash
            .checked_add(1)
            .ok_or_else(|| anyhow!("synthetic hash id overflow"))?;
    }
    Ok(())
}

fn arrival_spec_mean_gap_ms(spec: &ArrivalSpec) -> Result<f64> {
    match spec {
        ArrivalSpec::Burst => Ok(0.0),
        ArrivalSpec::ConstantQps { qps }
        | ArrivalSpec::PoissonQps { qps }
        | ArrivalSpec::GammaQps { qps, .. } => {
            if !qps.is_finite() || *qps <= 0.0 {
                bail!("qps must be a finite positive number, got {qps}");
            }
            Ok(1000.0 / qps)
        }
    }
}

fn sample_arrival_gap_ms(spec: &ArrivalSpec, mean_gap_ms: f64, rng: &mut StdRng) -> Result<f64> {
    match spec {
        ArrivalSpec::Burst => Ok(0.0),
        ArrivalSpec::ConstantQps { .. } => Ok(mean_gap_ms),
        ArrivalSpec::PoissonQps { .. } => Ok(sample_exponential_ms(mean_gap_ms, rng)),
        ArrivalSpec::GammaQps { smoothness, .. } => {
            if !smoothness.is_finite() || *smoothness <= 0.0 {
                bail!("gamma smoothness must be a finite positive number, got {smoothness}");
            }
            Ok(sample_gamma_ms(*smoothness, mean_gap_ms / smoothness, rng))
        }
    }
}

fn sample_delay_ms(spec: &DelaySpec, rng: &mut StdRng) -> Result<f64> {
    match spec {
        DelaySpec::None => Ok(0.0),
        DelaySpec::ConstantMs(delay_ms) => {
            if !delay_ms.is_finite() || *delay_ms < 0.0 {
                bail!("delay must be a finite non-negative number, got {delay_ms}");
            }
            Ok(*delay_ms)
        }
        DelaySpec::ExponentialMs { mean_ms } => {
            if !mean_ms.is_finite() || *mean_ms < 0.0 {
                bail!("mean_ms must be a finite non-negative number, got {mean_ms}");
            }
            Ok(sample_exponential_ms(*mean_ms, rng))
        }
    }
}

fn sample_length(spec: &LengthSpec, min_value: usize, rng: &mut StdRng) -> usize {
    if spec.stddev == 0.0 {
        return spec.mean.max(min_value);
    }

    let stddev = spec.stddev.abs();
    let u1 = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.random::<f64>();
    let z0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    let sample = spec.mean as f64 + z0 * stddev;
    sample.round().max(min_value as f64) as usize
}

fn sample_exponential_ms(mean_ms: f64, rng: &mut StdRng) -> f64 {
    if mean_ms == 0.0 {
        return 0.0;
    }
    let u = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
    -mean_ms * u.ln()
}

fn sample_gamma_ms(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
    if scale == 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        let u = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
        return sample_gamma_ms(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let u1 = (1.0 - rng.random::<f64>()).clamp(f64::MIN_POSITIVE, 1.0);
        let u2 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.random::<f64>();
        if u < 1.0 - 0.0331 * z.powi(4) {
            return d * v * scale;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

fn local_block_hash_from_id(hash_id: u64, block_size: usize) -> LocalBlockHash {
    let tokens: Vec<u32> = (0..block_size).map(|_| hash_id as u32).collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            tokens.as_ptr() as *const u8,
            std::mem::size_of_val(tokens.as_slice()),
        )
    };
    LocalBlockHash(compute_hash_v2(bytes, XXH3_SEED))
}
