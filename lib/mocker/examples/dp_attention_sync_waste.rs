// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline replay harness that estimates attention-DP synchronization waste.
//!
//! Every attention-DP rank of one worker runs its own attention over its own batch, then all
//! ranks meet at the shared MoE phase. A rank that finishes attention early idles until the
//! slowest rank arrives. This harness replays a trace through the offline KV router with a CPU
//! timing approximation (attention cost grows with the rank's context; MoE cost depends on the
//! whole group's batch), rebuilds per-rank work from the per-request records, and reports how
//! much GPU time that wait burns under a given routing policy.
//!
//! Usage: see `--help`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use aisimulate_core::replay::{PerRequestRecord, ReplayRequestPool};
use anyhow::{Context, Result, bail};
use dynamo_kv_router::KvRouterConfig;
use dynamo_mocker::common::perf_model::{AisCallback, PerfModel};
use dynamo_mocker::config::MockerConfig;
use dynamo_mocker::loadgen::TraceFileFormat;
use dynamo_mocker::replay::{
    OfflineDisaggReplayConfig, ReplayRouterMode, ReplayRuntimeObservers, ReplaySessionAffinity,
    SlaThresholds, simulate_trace_file_disagg_with_router_mode_and_format_and_runtime_observers,
    simulate_trace_file_with_router_mode_and_format_and_runtime_observers, with_session_affinity,
};

/// Linear CPU approximation of one forward pass on one attention-DP rank.
#[derive(Clone, Copy, Debug, serde::Serialize)]
struct Timing {
    /// Fixed attention-phase cost per pass (kernel launches, small batches).
    attention_fixed_ms: f64,
    /// Attention cost per context token on the rank (KV read bound).
    attention_ms_per_kv_token: f64,
    /// Fixed MoE-phase cost per pass (expert weight read, dispatch/combine).
    moe_fixed_ms: f64,
    /// MoE cost per token row on one rank. Expert parallelism spreads the group's tokens over
    /// its ranks, so each rank's MoE work follows the group's tokens divided by the EP size.
    moe_ms_per_rank_token: f64,
    /// Fixed prefill cost per pass.
    prefill_fixed_ms: f64,
    /// Prefill cost per uncached prompt token.
    prefill_ms_per_token: f64,
    /// Share of the prefill cost spent in the rank-local attention phase; the rest is the
    /// shared MoE phase.
    prefill_attention_fraction: f64,
    dp_size: usize,
}

impl Timing {
    fn attention_ms(&self, kv_tokens: f64) -> f64 {
        self.attention_fixed_ms + self.attention_ms_per_kv_token * kv_tokens
    }

    fn moe_ms(&self, group_tokens: f64) -> f64 {
        self.moe_fixed_ms + self.moe_ms_per_rank_token * group_tokens / self.dp_size as f64
    }
}

impl AisCallback for Timing {
    fn predict_prefill(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> Result<f64> {
        let new_tokens = batch_size as f64 * effective_isl.saturating_sub(prefix) as f64;
        Ok(self.prefill_fixed_ms + self.prefill_ms_per_token * new_tokens)
    }

    fn predict_decode(&self, batch_size: usize, isl: usize, _osl: usize) -> Result<f64> {
        // The engine hands this rank's batch only; the other ranks' batches are approximated
        // as equal when sizing the shared MoE phase.
        Ok(self.attention_ms(batch_size as f64 * isl as f64)
            + self.moe_ms(batch_size as f64 * self.dp_size as f64))
    }
}

/// One rank's modeled work in one time step, reconstructed from per-request records.
#[derive(Clone, Copy, Default)]
struct RankStep {
    /// Σ over decoding requests of their current context length (prompt + output so far).
    decode_context_tokens: f64,
    decode_requests: f64,
    /// Uncached prompt tokens prefilled on this rank during the step.
    prefill_tokens: f64,
}

/// Per worker (EP group): per DP rank: per time step.
type Lanes = BTreeMap<usize, Vec<Vec<RankStep>>>;

#[derive(Debug, Default, serde::Serialize)]
struct Waste {
    /// Σ_ranks (slowest rank's attention − this rank's attention) over decode passes, as a
    /// fraction of all modeled GPU time.
    decode_wait_fraction_of_gpu_time: f64,
    /// Same, with a rank that is prefilling running prefill passes (stalls included).
    total_wait_fraction_of_gpu_time: f64,
    modeled_gpu_seconds: f64,
    /// Share of rank-time (while the group is active) in which a rank has any work.
    busy_rank_fraction: f64,
}

#[derive(Clone, Copy)]
struct TurnPlacement {
    rank: (usize, usize),
    input_length: usize,
    output_length: usize,
}

/// Prompt KV tokens the prefill worker must send to the decode rank, modeling a decode-side
/// radix cache: a turn that stays on the rank holding the previous turn only needs the new
/// tokens; a turn that moved (or a first turn) needs its whole context.
#[derive(Debug, Default, serde::Serialize)]
struct KvTransfer {
    first_turn_tokens: u64,
    stayed_turn_tokens: u64,
    moved_turn_tokens: u64,
    /// Subset of `moved_turn_tokens` whose move stayed on the previous rank's host, reloadable
    /// from a shared host (G2) tier instead of the network.
    moved_same_host_tokens: u64,
    follow_up_turns: usize,
    moved_turns: usize,
    moved_same_host_turns: usize,
}

impl KvTransfer {
    fn total_tokens(&self) -> u64 {
        self.first_turn_tokens + self.stayed_turn_tokens + self.moved_turn_tokens
    }

    fn from_sessions(
        sessions: &BTreeMap<String, BTreeMap<usize, TurnPlacement>>,
        ranks_per_host: usize,
    ) -> Self {
        let mut transfer = Self::default();
        for turns in sessions.values() {
            for (prev, turn) in turns.values().zip(turns.values().skip(1)) {
                transfer.follow_up_turns += 1;
                if prev.rank == turn.rank {
                    transfer.stayed_turn_tokens += turn
                        .input_length
                        .saturating_sub(prev.input_length + prev.output_length)
                        as u64;
                    continue;
                }
                transfer.moved_turns += 1;
                transfer.moved_turn_tokens += turn.input_length as u64;
                if ranks_per_host > 0
                    && prev.rank.0 == turn.rank.0
                    && prev.rank.1 / ranks_per_host == turn.rank.1 / ranks_per_host
                {
                    transfer.moved_same_host_tokens += turn.input_length as u64;
                    transfer.moved_same_host_turns += 1;
                }
            }
            if let Some(first) = turns.values().next() {
                transfer.first_turn_tokens += first.input_length as u64;
            }
        }
        transfer
    }
}

fn route_for_pool(
    record: &PerRequestRecord,
    pool: ReplayRequestPool,
    fallback_idx: Option<usize>,
    dp_size: usize,
) -> Option<(usize, usize)> {
    let routed = record
        .routing_history
        .iter()
        .rev()
        .find(|r| r.pool == pool)
        .and_then(|r| Some((r.logical_worker_id?, r.dp_rank? as usize)));
    routed
        .or_else(|| fallback_idx.map(|idx| (idx / dp_size, idx % dp_size)))
        .filter(|(_, rank)| *rank < dp_size)
}

#[derive(Default)]
struct Timeline {
    decode: Lanes,
    prefill: Lanes,
    steps: usize,
    placed: usize,
    unplaced: usize,
    /// Decode placement per turn index per session.
    session_turns: BTreeMap<String, BTreeMap<usize, TurnPlacement>>,
    prefill_queue_wait_ms: Vec<f64>,
    /// Arrival → first token for every placed request, including queueing.
    ttft_ms: Vec<f64>,
}

/// Spread `[start_ms, end_ms)` over the lane's steps, calling `apply(slot, overlap_ms, mid_ms)`.
fn add_phase(
    lane: &mut [RankStep],
    step_ms: f64,
    start_ms: f64,
    end_ms: f64,
    mut apply: impl FnMut(&mut RankStep, f64, f64),
) {
    let first = (start_ms / step_ms).floor() as usize;
    let last = ((end_ms / step_ms).ceil() as usize).min(lane.len());
    for (step, slot) in lane.iter_mut().enumerate().take(last).skip(first) {
        let lo = (step as f64 * step_ms).max(start_ms);
        let hi = ((step + 1) as f64 * step_ms).min(end_ms);
        if hi > lo {
            apply(slot, hi - lo, (lo + hi) / 2.0);
        }
    }
}

/// Bucket every request's prefill and decode phase into per-rank time steps. Aggregated engines
/// prefill and decode on the same rank, so both phases land in the decode lanes.
fn build_timeline(records: &[PerRequestRecord], args: &Args) -> Timeline {
    let step_ms = args.step_ms;
    let horizon_ms = records
        .iter()
        .map(|r| r.terminal_time_ms)
        .fold(0.0, f64::max);
    let steps = (horizon_ms / step_ms).ceil() as usize + 1;
    let mut timeline = Timeline {
        steps,
        ..Timeline::default()
    };
    let (prefill_dp, decode_dp) = (args.prefill_dp(), args.decode_dp());
    let new_lane = |dp| vec![vec![RankStep::default(); steps]; dp];

    for record in records {
        let (prefill_pool, decode_pool) = if args.disagg {
            (ReplayRequestPool::Prefill, ReplayRequestPool::Decode)
        } else {
            (ReplayRequestPool::Agg, ReplayRequestPool::Agg)
        };
        let prefill_idx = if args.disagg {
            record.prefill_worker_idx
        } else {
            record.decode_worker_idx
        };
        let prefill_rank = route_for_pool(record, prefill_pool, prefill_idx, prefill_dp);
        let Some((group, rank)) =
            route_for_pool(record, decode_pool, record.decode_worker_idx, decode_dp)
        else {
            timeline.unplaced += 1;
            continue;
        };
        timeline.placed += 1;
        if let (Some(session_id), Some(turn_index)) = (&record.session_id, record.turn_index) {
            timeline
                .session_turns
                .entry(session_id.clone())
                .or_default()
                .insert(
                    turn_index,
                    TurnPlacement {
                        rank: (group, rank),
                        input_length: record.input_length,
                        output_length: record.output_length,
                    },
                );
        }
        if let Some(first_token) = record.first_token_ms {
            timeline
                .ttft_ms
                .push((first_token - record.arrival_time_ms).max(0.0));
        }
        timeline.prefill_queue_wait_ms.extend(
            record
                .routing_history
                .iter()
                .filter(|r| args.disagg && r.pool == ReplayRequestPool::Prefill)
                .filter_map(|r| r.queue_wait_ms),
        );

        let admit = record
            .prefill_admit_ms
            .or(record.first_admit_ms)
            .or(record.decode_admit_ms)
            .unwrap_or(record.arrival_time_ms);
        let first_token = record.first_token_ms.unwrap_or(record.terminal_time_ms);
        let decode_start = record
            .decode_admit_ms
            .map_or(first_token, |t| t.max(first_token));
        let last_token = record.last_token_ms.unwrap_or(record.terminal_time_ms);
        let new_tokens = record
            .input_length
            .saturating_sub(record.reused_input_tokens) as f64;

        // Prefill phase: uncached tokens spread uniformly over admit → first token.
        if let Some((prefill_group, prefill_rank)) = prefill_rank
            && first_token > admit
            && new_tokens > 0.0
        {
            let lanes = if args.disagg {
                &mut timeline.prefill
            } else {
                &mut timeline.decode
            };
            let lane = &mut lanes
                .entry(prefill_group)
                .or_insert_with(|| new_lane(prefill_dp))[prefill_rank];
            let rate = new_tokens / (first_token - admit);
            add_phase(lane, step_ms, admit, first_token, |slot, overlap_ms, _| {
                slot.prefill_tokens += rate * overlap_ms;
            });
        }
        // Decode phase: one request with growing context from first → last token.
        if last_token > decode_start && record.output_length > 1 {
            let lane = &mut timeline
                .decode
                .entry(group)
                .or_insert_with(|| new_lane(decode_dp))[rank];
            let (isl, osl) = (record.input_length as f64, record.output_length as f64);
            let duration = last_token - decode_start;
            add_phase(
                lane,
                step_ms,
                decode_start,
                last_token,
                |slot, overlap_ms, mid| {
                    let progress = ((mid - decode_start) / duration).clamp(0.0, 1.0);
                    slot.decode_requests += overlap_ms / step_ms;
                    slot.decode_context_tokens += (isl + osl * progress) * overlap_ms / step_ms;
                },
            );
        }
    }
    timeline
}

/// Σ_ranks (max attention − rank attention) / (dp × pass time) for one step.
fn wait_fraction(attention: &[f64], pass_ms: f64) -> f64 {
    let max = attention.iter().copied().fold(0.0, f64::max);
    attention.iter().map(|a| max - a).sum::<f64>() / (attention.len() as f64 * pass_ms)
}

/// Attention-sync waste over one pool's lanes with `dp` ranks per group.
fn summarize_waste(lanes: &Lanes, dp: usize, steps: usize, args: &Args) -> Waste {
    let (step_ms, timing) = (args.step_ms, &args.timing);
    let chunk = args.max_num_batched_tokens as f64;
    let prefill_ms = |tokens: f64| timing.prefill_ms_per_token * chunk.min(tokens);
    let (mut decode_wait, mut total_wait, mut rank_steps, mut busy_rank_steps) = (0.0, 0.0, 0, 0);
    for ranks in lanes.values() {
        for step in 0..steps {
            let rows: Vec<RankStep> = ranks.iter().map(|lane| lane[step]).collect();
            let group_requests: f64 = rows.iter().map(|r| r.decode_requests).sum();
            let group_prefill: f64 = rows.iter().map(|r| r.prefill_tokens).sum();
            if group_requests == 0.0 && group_prefill == 0.0 {
                continue;
            }
            rank_steps += dp;
            busy_rank_steps += rows
                .iter()
                .filter(|r| r.decode_requests > 0.0 || r.prefill_tokens > 0.0)
                .count();
            let decode: Vec<f64> = rows
                .iter()
                .map(|r| {
                    if r.decode_requests > 0.0 {
                        timing.attention_ms(r.decode_context_tokens)
                    } else {
                        0.0
                    }
                })
                .collect();
            let moe = timing.moe_ms(group_requests);
            if group_requests > 0.0 {
                let max = decode.iter().copied().fold(0.0, f64::max);
                decode_wait += wait_fraction(&decode, max + moe);
            }
            // A rank with prefill work runs prefill passes of `chunk` tokens instead.
            let total: Vec<f64> = rows
                .iter()
                .zip(&decode)
                .map(|(r, d)| {
                    if r.prefill_tokens > 0.0 {
                        (timing.prefill_attention_fraction * prefill_ms(r.prefill_tokens)).max(*d)
                    } else {
                        *d
                    }
                })
                .collect();
            let max = total.iter().copied().fold(0.0, f64::max);
            let prefill_moe = (1.0 - timing.prefill_attention_fraction) * prefill_ms(group_prefill);
            total_wait += wait_fraction(&total, max + moe + prefill_moe);
        }
    }
    let group_steps = (rank_steps / dp.max(1)).max(1) as f64;
    Waste {
        decode_wait_fraction_of_gpu_time: decode_wait / group_steps,
        total_wait_fraction_of_gpu_time: total_wait / group_steps,
        modeled_gpu_seconds: rank_steps as f64 * step_ms / 1000.0,
        busy_rank_fraction: busy_rank_steps as f64 / rank_steps.max(1) as f64,
    }
}

fn percentile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

#[derive(Debug, serde::Serialize)]
struct Summary {
    label: String,
    policy_yaml: Option<String>,
    timing: Timing,
    placed_requests: usize,
    unplaced_requests: usize,
    /// Decode pool (aggregated: whole engine).
    waste: Waste,
    /// Disaggregated prefill pool.
    prefill_pool: Option<Waste>,
    prefill_queue_wait_p95_ms: f64,
    ttft_p50_ms: f64,
    ttft_p95_ms: f64,
    session_affinity: ReplaySessionAffinity,
    decode_turn_moved_fraction: f64,
    kv_transfer: KvTransfer,
    /// Serialized transfer seconds: network (prefill deltas, first turns, cross-host moves) and
    /// host tier (same-host moves). The engine charges every transfer at `--kv-transfer-gbps`,
    /// so same-host moves are timed conservatively there.
    network_transfer_s: f64,
    host_transfer_s: f64,
    report: serde_json::Value,
}

struct Args {
    trace: PathBuf,
    policy_yaml: Option<PathBuf>,
    label: String,
    out: Option<PathBuf>,
    num_workers: usize,
    num_prefill_workers: Option<usize>,
    num_decode_workers: Option<usize>,
    dp_size: usize,
    disagg: bool,
    prefill_dp_size: usize,
    decode_dp_size: usize,
    num_gpu_blocks: usize,
    block_size: usize,
    trace_block_size: usize,
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    max_sim_time_ms: Option<f64>,
    speedup: f64,
    step_ms: f64,
    session_affinity: ReplaySessionAffinity,
    /// Decode engines keep a prefix cache and only receive the prompt KV they miss (SGLang
    /// `--disaggregation-decode-enable-radix-cache`); off = whole prompt every turn.
    decode_radix_cache: bool,
    /// KV bytes per token for the modeled prefill→decode transfer; 0 leaves it untimed.
    kv_bytes_per_token: usize,
    kv_transfer_gbps: f64,
    host_transfer_gbps: f64,
    decode_dp_ranks_per_host: usize,
    /// `mooncake-delta`: each row after a session's first is that turn's new tokens (AIPerf
    /// multi-turn exports); `mooncake`: every row is a complete, independent prompt.
    trace_format: TraceFileFormat,
    timing: Timing,
}

impl Args {
    fn prefill_dp(&self) -> usize {
        if self.disagg {
            self.prefill_dp_size
        } else {
            self.dp_size
        }
    }

    fn decode_dp(&self) -> usize {
        if self.disagg {
            self.decode_dp_size
        } else {
            self.dp_size
        }
    }
}

const USAGE: &str = "dp_attention_sync_waste --trace <trace.jsonl> [--policy-yaml <router-policy.yaml>] \
[--label NAME] [--out summary.json] [--num-workers N] [--dp-size N] \
[--disagg --prefill-dp-size N --decode-dp-size N --num-prefill-workers N --num-decode-workers N] \
[--num-gpu-blocks N] [--block-size N] [--trace-block-size N] [--max-num-seqs N] \
[--max-num-batched-tokens N] [--max-sim-time-ms MS] [--speedup X] [--step-ms MS] \
[--session-affinity on|off] [--prefill-session-affinity on|off] [--decode-session-affinity on|off] \
[--decode-radix-cache on|off] [--kv-bytes-per-token N] [--kv-transfer-gbps X] \
[--host-transfer-gbps X] [--decode-dp-ranks-per-host N] [--trace-format mooncake|mooncake-delta] \
[--attention-fixed-ms X] [--attention-ms-per-kv-token X] [--moe-fixed-ms X] \
[--moe-ms-per-rank-token X] [--prefill-fixed-ms X] [--prefill-ms-per-token X] \
[--prefill-attention-fraction X]";

fn parse_on_off(flag: &str, value: &str) -> Result<bool> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        other => bail!("{flag} expects on|off, got {other}"),
    }
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        trace: PathBuf::new(),
        policy_yaml: None,
        label: "default".to_string(),
        out: None,
        num_workers: 4,
        num_prefill_workers: None,
        num_decode_workers: None,
        dp_size: 4,
        disagg: false,
        prefill_dp_size: 2,
        decode_dp_size: 16,
        num_gpu_blocks: 65536,
        block_size: 64,
        trace_block_size: 512,
        max_num_seqs: 128,
        max_num_batched_tokens: 8192,
        max_sim_time_ms: None,
        speedup: 1.0,
        step_ms: 50.0,
        session_affinity: ReplaySessionAffinity::default(),
        decode_radix_cache: true,
        kv_bytes_per_token: 0,
        kv_transfer_gbps: 50.0,
        host_transfer_gbps: 225.0,
        decode_dp_ranks_per_host: 0,
        trace_format: TraceFileFormat::MooncakeDelta,
        timing: Timing {
            attention_fixed_ms: 1.5,
            attention_ms_per_kv_token: 1.25e-5,
            moe_fixed_ms: 6.0,
            moe_ms_per_rank_token: 0.32,
            prefill_fixed_ms: 2.0,
            prefill_ms_per_token: 0.05,
            prefill_attention_fraction: 0.5,
            dp_size: 4,
        },
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        let timing = &mut args.timing;
        match flag.as_str() {
            "--trace" => args.trace = value()?.into(),
            "--policy-yaml" => args.policy_yaml = Some(value()?.into()),
            "--label" => args.label = value()?,
            "--out" => args.out = Some(value()?.into()),
            "--num-workers" => args.num_workers = value()?.parse()?,
            "--num-prefill-workers" => args.num_prefill_workers = Some(value()?.parse()?),
            "--num-decode-workers" => args.num_decode_workers = Some(value()?.parse()?),
            "--dp-size" => args.dp_size = value()?.parse()?,
            "--disagg" => args.disagg = true,
            "--prefill-dp-size" => args.prefill_dp_size = value()?.parse()?,
            "--decode-dp-size" => args.decode_dp_size = value()?.parse()?,
            "--num-gpu-blocks" => args.num_gpu_blocks = value()?.parse()?,
            "--block-size" => args.block_size = value()?.parse()?,
            "--trace-block-size" => args.trace_block_size = value()?.parse()?,
            "--max-num-seqs" => args.max_num_seqs = value()?.parse()?,
            "--max-num-batched-tokens" => args.max_num_batched_tokens = value()?.parse()?,
            "--max-sim-time-ms" => args.max_sim_time_ms = Some(value()?.parse()?),
            "--speedup" => args.speedup = value()?.parse()?,
            "--step-ms" => args.step_ms = value()?.parse()?,
            "--session-affinity" => {
                let on = parse_on_off(&flag, &value()?)?;
                args.session_affinity = ReplaySessionAffinity {
                    aggregated: on,
                    prefill: on,
                    decode: on,
                };
            }
            "--prefill-session-affinity" => {
                args.session_affinity.prefill = parse_on_off(&flag, &value()?)?
            }
            "--decode-session-affinity" => {
                args.session_affinity.decode = parse_on_off(&flag, &value()?)?
            }
            "--decode-radix-cache" => args.decode_radix_cache = parse_on_off(&flag, &value()?)?,
            "--kv-bytes-per-token" => args.kv_bytes_per_token = value()?.parse()?,
            "--kv-transfer-gbps" => args.kv_transfer_gbps = value()?.parse()?,
            "--host-transfer-gbps" => args.host_transfer_gbps = value()?.parse()?,
            "--decode-dp-ranks-per-host" => args.decode_dp_ranks_per_host = value()?.parse()?,
            "--trace-format" => {
                args.trace_format = match value()?.as_str() {
                    "mooncake" => TraceFileFormat::Mooncake,
                    "mooncake-delta" => TraceFileFormat::MooncakeDelta,
                    other => bail!("--trace-format expects mooncake|mooncake-delta, got {other}"),
                }
            }
            "--attention-fixed-ms" => timing.attention_fixed_ms = value()?.parse()?,
            "--attention-ms-per-kv-token" => timing.attention_ms_per_kv_token = value()?.parse()?,
            "--moe-fixed-ms" => timing.moe_fixed_ms = value()?.parse()?,
            "--moe-ms-per-rank-token" => timing.moe_ms_per_rank_token = value()?.parse()?,
            "--prefill-fixed-ms" => timing.prefill_fixed_ms = value()?.parse()?,
            "--prefill-ms-per-token" => timing.prefill_ms_per_token = value()?.parse()?,
            "--prefill-attention-fraction" => {
                timing.prefill_attention_fraction = value()?.parse()?
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}\n{USAGE}"),
        }
    }
    anyhow::ensure!(!args.trace.as_os_str().is_empty(), "--trace is required");
    args.timing.dp_size = args.decode_dp();
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let engine_config = |dp_size: usize, worker_type: &str| -> Result<MockerConfig> {
        let mut engine = serde_json::json!({
            "backend": "vllm",
            "worker_type": worker_type,
            "block_size": args.block_size,
            "num_gpu_blocks": args.num_gpu_blocks,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": worker_type != "decode" || args.decode_radix_cache,
            "enable_chunked_prefill": true,
        });
        if args.kv_bytes_per_token > 0 {
            engine["kv_transfer_bytes_per_token"] = args.kv_bytes_per_token.into();
            engine["kv_transfer_bandwidth"] = args.kv_transfer_gbps.into();
            engine["kv_transfer_timing_mode"] = if args.decode_radix_cache {
                "destination_missing"
            } else {
                "full_prompt"
            }
            .into();
        }
        let mut mocker = MockerConfig::from_value(serde_json::json!({
            "engine": engine,
            "dp_size": dp_size,
        }))?;
        mocker.perf_model = Arc::new(PerfModel::Ais {
            callback: Arc::new(args.timing),
        });
        Ok(mocker)
    };
    let router = KvRouterConfig {
        router_policy_config: args.policy_yaml.as_ref().map(|p| p.display().to_string()),
        ..KvRouterConfig::default()
    };

    let report = with_session_affinity(args.session_affinity, || {
        if args.disagg {
            simulate_trace_file_disagg_with_router_mode_and_format_and_runtime_observers(
                OfflineDisaggReplayConfig {
                    prefill_args: engine_config(args.prefill_dp_size, "prefill")?,
                    decode_args: engine_config(args.decode_dp_size, "decode")?,
                    num_prefill_workers: args.num_prefill_workers.unwrap_or(args.num_workers),
                    num_decode_workers: args.num_decode_workers.unwrap_or(args.num_workers),
                },
                Some(router),
                None,
                &args.trace,
                args.trace_block_size,
                args.speedup,
                ReplayRouterMode::KvRouter,
                args.trace_format,
                0.0,
                0,
                true,
                args.max_sim_time_ms,
                None,
                SlaThresholds::default(),
                ReplayRuntimeObservers::default(),
            )
        } else {
            simulate_trace_file_with_router_mode_and_format_and_runtime_observers(
                engine_config(args.dp_size, "aggregated")?,
                Some(router),
                None,
                &args.trace,
                args.trace_block_size,
                args.num_workers,
                args.speedup,
                ReplayRouterMode::KvRouter,
                args.trace_format,
                0.0,
                0,
                true,
                args.max_sim_time_ms,
                None,
                SlaThresholds::default(),
                ReplayRuntimeObservers::default(),
            )
        }
    })?;

    let timeline = build_timeline(&report.per_request, &args);
    let kv_transfer =
        KvTransfer::from_sessions(&timeline.session_turns, args.decode_dp_ranks_per_host);
    let transfer_s =
        |tokens: u64, gbps: f64| tokens as f64 * args.kv_bytes_per_token as f64 / (gbps * 1e9);
    let summary = Summary {
        label: args.label.clone(),
        policy_yaml: args.policy_yaml.as_ref().map(|p| p.display().to_string()),
        timing: args.timing,
        placed_requests: timeline.placed,
        unplaced_requests: timeline.unplaced,
        waste: summarize_waste(&timeline.decode, args.decode_dp(), timeline.steps, &args),
        prefill_pool: args
            .disagg
            .then(|| summarize_waste(&timeline.prefill, args.prefill_dp(), timeline.steps, &args)),
        prefill_queue_wait_p95_ms: percentile(timeline.prefill_queue_wait_ms, 0.95),
        ttft_p50_ms: percentile(timeline.ttft_ms.clone(), 0.5),
        ttft_p95_ms: percentile(timeline.ttft_ms, 0.95),
        session_affinity: args.session_affinity,
        decode_turn_moved_fraction: kv_transfer.moved_turns as f64
            / kv_transfer.follow_up_turns.max(1) as f64,
        network_transfer_s: transfer_s(
            kv_transfer.total_tokens() - kv_transfer.moved_same_host_tokens,
            args.kv_transfer_gbps,
        ),
        host_transfer_s: transfer_s(kv_transfer.moved_same_host_tokens, args.host_transfer_gbps),
        kv_transfer,
        report: serde_json::to_value(&report)?,
    };
    if let Some(out) = &args.out {
        std::fs::write(out, serde_json::to_string_pretty(&summary)?)?;
    }
    println!(
        "{}: decode_wait={:.2}% total_wait={:.2}% of {:.0} gpu-s | ttft p50={:.0}ms p95={:.0}ms | \
         turns moved={:.1}% (same host {}) | kv transfer network={:.0}s host_tier={:.0}s | placed={} unplaced={}",
        summary.label,
        summary.waste.decode_wait_fraction_of_gpu_time * 100.0,
        summary.waste.total_wait_fraction_of_gpu_time * 100.0,
        summary.waste.modeled_gpu_seconds,
        summary.ttft_p50_ms,
        summary.ttft_p95_ms,
        summary.decode_turn_moved_fraction * 100.0,
        summary.kv_transfer.moved_same_host_turns,
        summary.network_transfer_s,
        summary.host_transfer_s,
        summary.placed_requests,
        summary.unplaced_requests,
    );
    if let Some(prefill) = &summary.prefill_pool {
        println!(
            "{} prefill pool: total_wait={:.2}% of {:.0} gpu-s busy_rank_fraction={:.3} queue_wait p95={:.0}ms",
            summary.label,
            prefill.total_wait_fraction_of_gpu_time * 100.0,
            prefill.modeled_gpu_seconds,
            prefill.busy_rank_fraction,
            summary.prefill_queue_wait_p95_ms,
        );
    }
    Ok(())
}
