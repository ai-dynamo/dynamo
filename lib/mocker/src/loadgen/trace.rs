// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, WorkerId, XXH3_SEED, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use dynamo_tokens::compute_hash_v2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use uuid::Uuid;

use super::driver::WorkloadDriver;
use super::types::{
    ArrivalSpec, DelaySpec, LengthSpec, ReplayRequestHashes, RouterSequence, SequenceHashMode,
    SessionPartitionSpec, SessionTrace, SyntheticTraceSpec, Trace, TurnTrace,
};
use crate::common::protocols::DirectRequest;

#[derive(Debug, Deserialize)]
struct RawMooncakeRecord {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    created_time: Option<f64>,
    #[serde(default, alias = "input_tokens")]
    input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    output_length: Option<usize>,
    #[serde(default)]
    hash_ids: Option<Vec<u64>>,
    #[serde(default)]
    delay: Option<f64>,
    #[serde(default)]
    delay_ms: Option<f64>,
}

impl TurnTrace {
    fn validate_block_size_and_capacity(&self, trace_block_size: usize) -> Result<()> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }
        if self.hash_ids.len() * trace_block_size < self.input_length {
            bail!(
                "input_length {} exceeds synthesized capacity {}",
                self.input_length,
                self.hash_ids.len() * trace_block_size
            );
        }
        Ok(())
    }

    pub(crate) fn synthesize_tokens(&self, trace_block_size: usize) -> Result<Vec<u32>> {
        self.validate_block_size_and_capacity(trace_block_size)?;

        let mut tokens = Vec::with_capacity(self.input_length);
        for &hash_id in &self.hash_ids {
            let token_id = hash_id as u32;
            tokens.extend((0..trace_block_size).map(|_| token_id));
            if tokens.len() >= self.input_length {
                tokens.truncate(self.input_length);
                break;
            }
        }

        if tokens.len() != self.input_length {
            bail!(
                "failed to synthesize {} tokens from {} hash_ids",
                self.input_length,
                self.hash_ids.len()
            );
        }

        Ok(tokens)
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
            uuid: Some(request_uuid),
            dp_rank: 0,
            arrival_timestamp_ms,
        })
    }

    pub fn to_replay_hashes(
        &self,
        trace_block_size: usize,
        engine_block_size: usize,
    ) -> Result<ReplayRequestHashes> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }

        let tokens = self.synthesize_tokens(trace_block_size)?;
        let engine_block_size =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let local_block_hashes =
            compute_block_hash_for_seq(&tokens, engine_block_size, BlockHashOptions::default());
        let sequence_hashes = compute_seq_hash_for_block(&local_block_hashes);

        Ok(ReplayRequestHashes {
            local_block_hashes,
            sequence_hashes,
        })
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
        let mut sessions = Vec::new();
        let mut session_indices = HashMap::new();
        let mut last_timestamps: Vec<Option<f64>> = Vec::new();

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

            let raw: RawMooncakeRecord = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse line {} from {} as JSON",
                    line_idx + 1,
                    path.display()
                )
            })?;

            let session_id = raw
                .session_id
                .unwrap_or_else(|| format!("request_{}", line_idx + 1));
            let hash_ids = raw
                .hash_ids
                .ok_or_else(|| anyhow!("trace line {} is missing hash_ids", line_idx + 1))?;
            // Clamp input_length to the synthesizable capacity: in the mooncake
            // trace format, input_length is the full prompt token count which may
            // exceed hash_ids.len() * block_size (cached portion only).
            let synthesizable_capacity =
                hash_ids
                    .len()
                    .checked_mul(trace_block_size)
                    .ok_or_else(|| {
                        anyhow!("trace line {} synthesized capacity overflow", line_idx + 1)
                    })?;
            let input_length = raw
                .input_length
                .unwrap_or(synthesizable_capacity)
                .min(synthesizable_capacity);
            let output_length = raw
                .output_length
                .ok_or_else(|| anyhow!("trace line {} is missing output_length", line_idx + 1))?;
            let timestamp_ms = raw.timestamp.or(raw.created_time);
            let explicit_delay_ms = raw.delay.or(raw.delay_ms);

            let session_index = *session_indices
                .entry(session_id.clone())
                .or_insert_with(|| {
                    let idx = sessions.len();
                    sessions.push(SessionTrace {
                        session_id: session_id.clone(),
                        first_arrival_timestamp_ms: timestamp_ms,
                        turns: Vec::new(),
                    });
                    last_timestamps.push(timestamp_ms);
                    idx
                });

            let session = sessions
                .get_mut(session_index)
                .expect("newly inserted session must exist");
            let turn_idx = session.turns.len();
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
                let previous_timestamp_ms = last_timestamps[session_index].ok_or_else(|| {
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

            if hash_ids.len() * trace_block_size < input_length {
                bail!(
                    "trace line {} input_length {} exceeds synthesized capacity {}",
                    line_idx + 1,
                    input_length,
                    hash_ids.len() * trace_block_size
                );
            }

            session.turns.push(TurnTrace {
                input_length,
                max_output_tokens: output_length,
                hash_ids,
                delay_after_previous_ms,
            });
            if let Some(timestamp_ms) = timestamp_ms {
                last_timestamps[session_index] = Some(timestamp_ms);
            }
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
        for session in &self.sessions {
            if session.turns.len() != 1 {
                bail!(
                    "to_single_turn_requests requires exactly one turn per session, but session {} has {} turns",
                    session.session_id,
                    session.turns.len()
                );
            }
            requests.push(session.turns[0].to_direct_request(
                self.block_size,
                Uuid::new_v4(),
                session.first_arrival_timestamp_ms,
            )?);
        }
        Ok(requests)
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

    pub fn into_concurrency_driver(self) -> Result<WorkloadDriver> {
        self.validate_for_concurrency_mode()?;
        let engine_block_size = self.block_size;
        WorkloadDriver::new_concurrency(self, engine_block_size)
    }

    pub fn into_trace_driver_with_block_size(
        self,
        engine_block_size: usize,
    ) -> Result<WorkloadDriver> {
        self.validate_for_trace_mode()?;
        WorkloadDriver::new_trace(self, engine_block_size)
    }

    pub fn into_concurrency_driver_with_block_size(
        self,
        engine_block_size: usize,
    ) -> Result<WorkloadDriver> {
        self.validate_for_concurrency_mode()?;
        WorkloadDriver::new_concurrency(self, engine_block_size)
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
                if turn.hash_ids.len() * self.block_size < turn.input_length {
                    bail!(
                        "session {} turn {} input_length {} exceeds synthesized capacity {}",
                        session.session_id,
                        turn_idx,
                        turn.input_length,
                        turn.hash_ids.len() * self.block_size
                    );
                }
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
