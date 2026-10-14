// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline replay harness that estimates attention-DP synchronization waste.
//!
//! Every attention-DP rank of one worker runs its own attention over its own
//! batch, then all ranks meet at the shared MoE phase. A rank that finishes
//! attention early idles until the slowest rank arrives. This harness replays a
//! trace through the offline KV router with a CPU timing approximation
//! (attention cost grows with the rank's active KV tokens; MoE cost depends on
//! the whole group's batch and is split-invariant) and samples per-rank state
//! to report how much GPU time that wait burns under a given routing policy.
//!
//! Usage: see `--help`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_kv_router::KvRouterConfig;
use dynamo_mocker::common::perf_model::{AisCallback, PerfModel};
use dynamo_mocker::config::MockerConfig;
use dynamo_mocker::loadgen::TraceFileFormat;
use dynamo_mocker::replay::{
    ReplayRouterMode, ReplayRuntimeObservers, ReplayTelemetryObserver, ReplayTelemetryOptions,
    ReplayTelemetrySnapshot, SlaThresholds,
    simulate_trace_file_with_router_mode_and_format_and_runtime_observers,
};

/// Linear CPU approximation of one forward pass on one attention-DP rank.
#[derive(Clone, Copy, Debug, serde::Serialize)]
struct Timing {
    /// Fixed attention-phase cost per pass (kernel launches, small batches).
    attention_fixed_ms: f64,
    /// Attention cost per active KV token on the rank (KV read bound).
    attention_ms_per_kv_token: f64,
    /// Fixed MoE-phase cost per pass (expert weight read, dispatch/combine).
    moe_fixed_ms: f64,
    /// MoE cost per token row across the whole EP group.
    moe_ms_per_group_token: f64,
    /// Fixed prefill cost per pass.
    prefill_fixed_ms: f64,
    /// Prefill cost per uncached prompt token.
    prefill_ms_per_token: f64,
    /// Share of the prefill cost spent in the attention phase (rank-local); the rest is
    /// the shared MoE phase.
    prefill_attention_fraction: f64,
    dp_size: usize,
}

impl Timing {
    fn attention_ms(&self, active_kv_tokens: f64) -> f64 {
        self.attention_fixed_ms + self.attention_ms_per_kv_token * active_kv_tokens
    }

    fn moe_ms(&self, group_tokens: f64) -> f64 {
        self.moe_fixed_ms + self.moe_ms_per_group_token * group_tokens
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
        // The engine hands this rank's batch only; the other ranks' batches are
        // approximated as equal when sizing the shared MoE phase.
        let active_kv_tokens = batch_size as f64 * isl as f64;
        Ok(self.attention_ms(active_kv_tokens)
            + self.moe_ms(batch_size as f64 * self.dp_size as f64))
    }
}

#[derive(Default, Debug, serde::Serialize)]
struct GroupStats {
    samples: u64,
    /// Sum over samples of Σ_ranks (slowest attention − this rank's attention), ms.
    attention_wait_ms: f64,
    /// Sum over samples of dp_size × modeled pass time, ms.
    group_pass_ms: f64,
    /// Sum over samples of the slowest rank's attention time, ms.
    max_attention_ms: f64,
    request_span_sum: f64,
    kv_token_span_sum: f64,
    request_cv_sum: f64,
    kv_token_cv_sum: f64,
    running_requests_sum: f64,
}

struct WasteObserver {
    timing: Timing,
    block_size: f64,
    sample_interval_ms: f64,
    groups: BTreeMap<usize, GroupStats>,
}

impl ReplayTelemetryObserver for WasteObserver {
    fn on_sample(&mut self, snapshot: ReplayTelemetrySnapshot) -> Result<()> {
        let mut by_worker: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
        for row in &snapshot.decode_scheduler_metrics {
            by_worker.entry(row.worker_id).or_default().push((
                row.running_requests as f64,
                row.active_blocks as f64 * self.block_size,
            ));
        }
        for (worker_id, ranks) in by_worker {
            let stats = self.groups.entry(worker_id).or_default();
            let attention: Vec<f64> = ranks
                .iter()
                .map(|(_, kv_tokens)| self.timing.attention_ms(*kv_tokens))
                .collect();
            let max_attention = attention.iter().copied().fold(0.0, f64::max);
            let group_requests: f64 = ranks.iter().map(|(requests, _)| requests).sum();
            let pass = max_attention + self.timing.moe_ms(group_requests);
            stats.samples += 1;
            stats.attention_wait_ms += attention.iter().map(|a| max_attention - a).sum::<f64>();
            stats.group_pass_ms += pass * ranks.len() as f64;
            stats.max_attention_ms += max_attention;
            stats.running_requests_sum += group_requests;
            let (req_span, req_cv) = span_and_cv(ranks.iter().map(|(r, _)| *r));
            let (kv_span, kv_cv) = span_and_cv(ranks.iter().map(|(_, k)| *k));
            stats.request_span_sum += req_span;
            stats.request_cv_sum += req_cv;
            stats.kv_token_span_sum += kv_span;
            stats.kv_token_cv_sum += kv_cv;
        }
        let _ = self.sample_interval_ms;
        Ok(())
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

#[derive(Debug, Default, serde::Serialize)]
struct ReconstructedWaste {
    step_ms: f64,
    steps: usize,
    placed_requests: usize,
    unplaced_requests: usize,
    /// Σ_ranks (slowest rank's attention − this rank's attention) over all steps, decode passes
    /// only, as a fraction of all modeled GPU time and in GPU-seconds.
    decode_wait_fraction_of_gpu_time: f64,
    decode_wait_gpu_seconds: f64,
    /// Same, counting a rank that is prefilling as running prefill passes.
    total_wait_fraction_of_gpu_time: f64,
    total_wait_gpu_seconds: f64,
    modeled_gpu_seconds: f64,
    /// Mean over steps and groups of max−min Σ-context across the group's ranks.
    mean_context_span_per_group: f64,
    mean_context_cv_per_group: f64,
    mean_decode_requests_per_rank: f64,
}

fn request_rank(
    record: &aisimulate_core::replay::PerRequestRecord,
    dp_size: usize,
) -> Option<(usize, usize)> {
    if let Some(route) = record.routing_history.last()
        && let (Some(worker), Some(rank)) = (route.logical_worker_id, route.dp_rank)
    {
        return Some((worker, rank as usize));
    }
    record
        .decode_worker_idx
        .map(|idx| (idx / dp_size, idx % dp_size))
}

fn reconstruct_waste(
    records: &[aisimulate_core::replay::PerRequestRecord],
    args: &Args,
) -> ReconstructedWaste {
    let step_ms = args.sample_interval_ms;
    let dp = args.dp_size;
    let horizon_ms = records
        .iter()
        .map(|r| r.terminal_time_ms)
        .fold(0.0, f64::max);
    let steps = (horizon_ms / step_ms).ceil() as usize + 1;
    let mut lanes: BTreeMap<usize, Vec<Vec<RankStep>>> = BTreeMap::new();
    let mut placed = 0;
    let mut unplaced = 0;

    for record in records {
        let Some((group, rank)) = request_rank(record, dp) else {
            unplaced += 1;
            continue;
        };
        if rank >= dp {
            unplaced += 1;
            continue;
        }
        placed += 1;
        let lane = lanes
            .entry(group)
            .or_insert_with(|| vec![vec![RankStep::default(); steps]; dp]);
        let lane = &mut lane[rank];
        let admit = record
            .first_admit_ms
            .or(record.decode_admit_ms)
            .unwrap_or(record.arrival_time_ms);
        let first_token = record.first_token_ms.unwrap_or(record.terminal_time_ms);
        let last_token = record.last_token_ms.unwrap_or(record.terminal_time_ms);
        let isl = record.input_length as f64;
        let osl = record.output_length.max(1) as f64;
        let new_tokens = record
            .input_length
            .saturating_sub(record.reused_input_tokens) as f64;

        // Prefill phase: uncached tokens spread uniformly over admit → first token.
        if first_token > admit && new_tokens > 0.0 {
            let rate = new_tokens / (first_token - admit);
            add_phase(lane, step_ms, admit, first_token, |slot, overlap_ms, _| {
                slot.prefill_tokens += rate * overlap_ms;
            });
        }
        // Decode phase: one request with growing context from first → last token.
        if last_token > first_token && record.output_length > 1 {
            let duration = last_token - first_token;
            add_phase(
                lane,
                step_ms,
                first_token,
                last_token,
                |slot, overlap_ms, mid| {
                    let progress = ((mid - first_token) / duration).clamp(0.0, 1.0);
                    slot.decode_requests += overlap_ms / step_ms;
                    slot.decode_context_tokens += (isl + osl * progress) * overlap_ms / step_ms;
                },
            );
        }
    }

    let timing = &args.timing;
    let chunk = args.max_num_batched_tokens as f64;
    let prefill_attention_ms =
        |tokens: f64| timing.prefill_attention_fraction * timing.prefill_ms_per_token * tokens;
    let mut decode_wait_ms = 0.0;
    let mut total_wait_ms = 0.0;
    let mut gpu_ms = 0.0;
    let mut ctx_span_sum = 0.0;
    let mut ctx_cv_sum = 0.0;
    let mut ctx_samples = 0u64;
    let mut decode_requests_sum = 0.0;
    let mut rank_steps = 0u64;
    for ranks in lanes.values() {
        for step in 0..steps {
            let rows: Vec<RankStep> = ranks.iter().map(|lane| lane[step]).collect();
            let group_decode_requests: f64 = rows.iter().map(|r| r.decode_requests).sum();
            let group_prefill_tokens: f64 = rows.iter().map(|r| r.prefill_tokens).sum();
            if group_decode_requests == 0.0 && group_prefill_tokens == 0.0 {
                continue;
            }
            // Decode-only view: every rank runs decode passes over its own Σ-context.
            let decode_attention: Vec<f64> = rows
                .iter()
                .map(|r| {
                    if r.decode_requests > 0.0 {
                        timing.attention_ms(r.decode_context_tokens)
                    } else {
                        0.0
                    }
                })
                .collect();
            let decode_max = decode_attention.iter().copied().fold(0.0, f64::max);
            let decode_moe = timing.moe_ms(group_decode_requests);
            let decode_pass = decode_max + decode_moe;
            // A rank with prefill work runs prefill passes of `chunk` tokens instead; the
            // fraction of the step it spends prefilling scales its share.
            let total_attention: Vec<f64> = rows
                .iter()
                .zip(&decode_attention)
                .map(|(r, decode)| {
                    if r.prefill_tokens > 0.0 {
                        prefill_attention_ms(chunk.min(r.prefill_tokens)).max(*decode)
                    } else {
                        *decode
                    }
                })
                .collect();
            let total_max = total_attention.iter().copied().fold(0.0, f64::max);
            let prefill_moe = (1.0 - timing.prefill_attention_fraction)
                * timing.prefill_ms_per_token
                * chunk.min(group_prefill_tokens);
            let total_pass = total_max + decode_moe + prefill_moe;
            if decode_pass > 0.0 {
                let frac = decode_attention.iter().map(|a| decode_max - a).sum::<f64>()
                    / (dp as f64 * decode_pass);
                decode_wait_ms += frac * dp as f64 * step_ms;
            }
            if total_pass > 0.0 {
                let frac = total_attention.iter().map(|a| total_max - a).sum::<f64>()
                    / (dp as f64 * total_pass);
                total_wait_ms += frac * dp as f64 * step_ms;
            }
            gpu_ms += dp as f64 * step_ms;
            if group_decode_requests > 0.0 {
                let (span, cv) = span_and_cv(rows.iter().map(|r| r.decode_context_tokens));
                ctx_span_sum += span;
                ctx_cv_sum += cv;
                ctx_samples += 1;
            }
            decode_requests_sum += group_decode_requests;
            rank_steps += dp as u64;
        }
    }
    let n = ctx_samples.max(1) as f64;
    ReconstructedWaste {
        step_ms,
        steps,
        placed_requests: placed,
        unplaced_requests: unplaced,
        decode_wait_fraction_of_gpu_time: if gpu_ms > 0.0 {
            decode_wait_ms / gpu_ms
        } else {
            0.0
        },
        decode_wait_gpu_seconds: decode_wait_ms / 1000.0,
        total_wait_fraction_of_gpu_time: if gpu_ms > 0.0 {
            total_wait_ms / gpu_ms
        } else {
            0.0
        },
        total_wait_gpu_seconds: total_wait_ms / 1000.0,
        modeled_gpu_seconds: gpu_ms / 1000.0,
        mean_context_span_per_group: ctx_span_sum / n,
        mean_context_cv_per_group: ctx_cv_sum / n,
        mean_decode_requests_per_rank: decode_requests_sum / rank_steps.max(1) as f64,
    }
}

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

fn span_and_cv(values: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let n = values.clone().count() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = values.clone().sum::<f64>() / n;
    let min = values.clone().fold(f64::INFINITY, f64::min);
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    let var = values.map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let cv = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
    (max - min, cv)
}

#[derive(Debug, serde::Serialize)]
struct Summary {
    label: String,
    policy_yaml: Option<String>,
    decode_active_request_weight: f64,
    timing: Timing,
    samples: u64,
    /// Per-rank timeline reconstructed from per-request records (primary metric).
    waste: ReconstructedWaste,
    /// Engine-sampled view (unique KV blocks, running requests); secondary.
    attention_wait_fraction_of_gpu_time_sampled: f64,
    mean_request_span_per_group: f64,
    mean_request_cv_per_group: f64,
    mean_kv_token_span_per_group: f64,
    mean_kv_token_cv_per_group: f64,
    mean_running_requests_per_group: f64,
    per_group: BTreeMap<usize, GroupStats>,
    report: serde_json::Value,
}

struct Args {
    trace: PathBuf,
    policy_yaml: Option<PathBuf>,
    label: String,
    out: Option<PathBuf>,
    num_workers: usize,
    dp_size: usize,
    num_gpu_blocks: usize,
    block_size: usize,
    trace_block_size: usize,
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    max_sim_time_ms: Option<f64>,
    speedup: f64,
    sample_interval_ms: f64,
    decode_active_request_weight: f64,
    timing: Timing,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        trace: PathBuf::new(),
        policy_yaml: None,
        label: "default".to_string(),
        out: None,
        num_workers: 4,
        dp_size: 4,
        num_gpu_blocks: 65536,
        block_size: 64,
        trace_block_size: 512,
        max_num_seqs: 128,
        max_num_batched_tokens: 8192,
        max_sim_time_ms: None,
        speedup: 1.0,
        sample_interval_ms: 50.0,
        decode_active_request_weight: 0.0,
        timing: Timing {
            attention_fixed_ms: 1.5,
            attention_ms_per_kv_token: 1.25e-5,
            moe_fixed_ms: 6.0,
            moe_ms_per_group_token: 0.02,
            prefill_fixed_ms: 2.0,
            prefill_ms_per_token: 0.05,
            prefill_attention_fraction: 0.5,
            dp_size: 4,
        },
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--trace" => args.trace = value()?.into(),
            "--policy-yaml" => args.policy_yaml = Some(value()?.into()),
            "--label" => args.label = value()?,
            "--out" => args.out = Some(value()?.into()),
            "--num-workers" => args.num_workers = value()?.parse()?,
            "--dp-size" => args.dp_size = value()?.parse()?,
            "--num-gpu-blocks" => args.num_gpu_blocks = value()?.parse()?,
            "--block-size" => args.block_size = value()?.parse()?,
            "--trace-block-size" => args.trace_block_size = value()?.parse()?,
            "--max-num-seqs" => args.max_num_seqs = value()?.parse()?,
            "--max-num-batched-tokens" => args.max_num_batched_tokens = value()?.parse()?,
            "--max-sim-time-ms" => args.max_sim_time_ms = Some(value()?.parse()?),
            "--speedup" => args.speedup = value()?.parse()?,
            "--sample-interval-ms" => args.sample_interval_ms = value()?.parse()?,
            "--decode-active-request-weight" => {
                args.decode_active_request_weight = value()?.parse()?
            }
            "--attention-fixed-ms" => args.timing.attention_fixed_ms = value()?.parse()?,
            "--attention-ms-per-kv-token" => {
                args.timing.attention_ms_per_kv_token = value()?.parse()?
            }
            "--moe-fixed-ms" => args.timing.moe_fixed_ms = value()?.parse()?,
            "--moe-ms-per-group-token" => args.timing.moe_ms_per_group_token = value()?.parse()?,
            "--prefill-fixed-ms" => args.timing.prefill_fixed_ms = value()?.parse()?,
            "--prefill-ms-per-token" => args.timing.prefill_ms_per_token = value()?.parse()?,
            "--prefill-attention-fraction" => {
                args.timing.prefill_attention_fraction = value()?.parse()?
            }
            "--help" | "-h" => {
                eprintln!(
                    "dp_attention_sync_waste --trace <mooncake.jsonl> [--policy-yaml <router-policy.yaml>] \
                     [--label <name>] [--out <summary.json>] [--num-workers N] [--dp-size N] \
                     [--num-gpu-blocks N] [--block-size N] [--trace-block-size N] [--max-num-seqs N] \
                     [--max-num-batched-tokens N] [--max-sim-time-ms MS] [--speedup X] \
                     [--sample-interval-ms MS] [--decode-active-request-weight W] [timing flags]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    anyhow::ensure!(!args.trace.as_os_str().is_empty(), "--trace is required");
    args.timing.dp_size = args.dp_size;
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut mocker = MockerConfig::from_value(serde_json::json!({
        "engine": {
            "backend": "vllm",
            "block_size": args.block_size,
            "num_gpu_blocks": args.num_gpu_blocks,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": true,
            "enable_chunked_prefill": true,
        },
        "dp_size": args.dp_size,
    }))?;
    mocker.perf_model = Arc::new(PerfModel::Ais {
        callback: Arc::new(args.timing),
    });

    let mut router = KvRouterConfig {
        decode_active_request_weight: args.decode_active_request_weight,
        ..KvRouterConfig::default()
    };
    if let Some(path) = &args.policy_yaml {
        router.router_policy_config = Some(path.display().to_string());
    }

    let observer = WasteObserver {
        timing: args.timing,
        block_size: args.block_size as f64,
        sample_interval_ms: args.sample_interval_ms,
        groups: BTreeMap::new(),
    };
    let shared = Arc::new(parking_lot::Mutex::new(Some(observer)));
    struct Forward(Arc<parking_lot::Mutex<Option<WasteObserver>>>);
    impl ReplayTelemetryObserver for Forward {
        fn on_sample(&mut self, snapshot: ReplayTelemetrySnapshot) -> Result<()> {
            self.0
                .lock()
                .as_mut()
                .context("observer taken")?
                .on_sample(snapshot)
        }
    }

    let report = simulate_trace_file_with_router_mode_and_format_and_runtime_observers(
        mocker,
        Some(router),
        None,
        &args.trace,
        args.trace_block_size,
        args.num_workers,
        args.speedup,
        ReplayRouterMode::KvRouter,
        TraceFileFormat::Mooncake,
        0.0,
        0,
        true,
        args.max_sim_time_ms,
        None,
        SlaThresholds::default(),
        ReplayRuntimeObservers {
            scaling_policy: None,
            telemetry: Some(ReplayTelemetryOptions {
                sample_interval_ms: args.sample_interval_ms,
                observer: Box::new(Forward(Arc::clone(&shared))),
            }),
        },
    )?;

    let observer = shared.lock().take().context("observer missing")?;
    let groups = observer.groups;
    let total_wait: f64 = groups.values().map(|g| g.attention_wait_ms).sum();
    let total_pass: f64 = groups.values().map(|g| g.group_pass_ms).sum();
    let total_max_attention: f64 = groups
        .values()
        .map(|g| g.max_attention_ms * args.dp_size as f64)
        .sum();
    let samples: u64 = groups.values().map(|g| g.samples).sum();
    let n = samples.max(1) as f64;
    let waste = reconstruct_waste(&report.per_request, &args);
    let _ = total_max_attention;
    let summary = Summary {
        label: args.label,
        policy_yaml: args.policy_yaml.map(|p| p.display().to_string()),
        decode_active_request_weight: args.decode_active_request_weight,
        timing: args.timing,
        samples,
        waste,
        attention_wait_fraction_of_gpu_time_sampled: if total_pass > 0.0 {
            total_wait / total_pass
        } else {
            0.0
        },
        mean_request_span_per_group: groups.values().map(|g| g.request_span_sum).sum::<f64>() / n,
        mean_request_cv_per_group: groups.values().map(|g| g.request_cv_sum).sum::<f64>() / n,
        mean_kv_token_span_per_group: groups.values().map(|g| g.kv_token_span_sum).sum::<f64>() / n,
        mean_kv_token_cv_per_group: groups.values().map(|g| g.kv_token_cv_sum).sum::<f64>() / n,
        mean_running_requests_per_group: groups
            .values()
            .map(|g| g.running_requests_sum)
            .sum::<f64>()
            / n,
        per_group: groups,
        report: serde_json::to_value(&report)?,
    };
    let json = serde_json::to_string_pretty(&summary)?;
    if let Some(out) = &args.out {
        std::fs::write(out, &json)?;
    }
    println!(
        "{}: decode_wait/gpu={:.4} ({:.1} gpu-s) total_wait/gpu={:.4} ({:.1} gpu-s) of {:.0} gpu-s | ctx_span={:.0} ctx_cv={:.3} dec_req/rank={:.1} placed={} unplaced={} | sampled wait/gpu={:.4} req_span={:.2} req_cv={:.3} kv_span={:.0} kv_cv={:.3} running/group={:.1} samples={}",
        summary.label,
        summary.waste.decode_wait_fraction_of_gpu_time,
        summary.waste.decode_wait_gpu_seconds,
        summary.waste.total_wait_fraction_of_gpu_time,
        summary.waste.total_wait_gpu_seconds,
        summary.waste.modeled_gpu_seconds,
        summary.waste.mean_context_span_per_group,
        summary.waste.mean_context_cv_per_group,
        summary.waste.mean_decode_requests_per_rank,
        summary.waste.placed_requests,
        summary.waste.unplaced_requests,
        summary.attention_wait_fraction_of_gpu_time_sampled,
        summary.mean_request_span_per_group,
        summary.mean_request_cv_per_group,
        summary.mean_kv_token_span_per_group,
        summary.mean_kv_token_cv_per_group,
        summary.mean_running_requests_per_group,
        summary.samples
    );
    Ok(())
}
