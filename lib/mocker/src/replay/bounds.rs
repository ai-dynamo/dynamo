// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp;
use std::time::Duration;

use anyhow::{Result, bail};
use dynamo_tokens::Token;
use serde::Serialize;

use super::{
    ReplayPrefillLoadEstimator, ReplayRouterMode, TraceSimulationReport, normalize_trace_requests,
};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use dynamo_kv_router::config::KvRouterConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReplayUpperBoundSource {
    RoundRobin,
    KvRouter,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayBoundsReport {
    pub lower_bound_duration_ms: f64,
    pub upper_bound_duration_ms: f64,
    pub upper_bound_source: ReplayUpperBoundSource,
    pub optimistic_prefix_cache_reused_ratio: f64,
    pub round_robin_report: TraceSimulationReport,
    pub kv_router_report: Option<TraceSimulationReport>,
}

#[allow(clippy::too_many_arguments)]
pub fn estimate_request_bounds_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
) -> Result<ReplayBoundsReport> {
    let args = args.normalized()?;
    if requests.is_empty() {
        bail!("bounds estimation requires at least one request");
    }
    if num_workers == 0 {
        bail!("bounds estimation requires num_workers >= 1");
    }

    let normalized_requests = normalize_requests_for_bounds(requests, arrival_speedup_ratio)?;
    let lower_bound_duration_ms =
        optimistic_lower_bound_ms(&args, &normalized_requests, num_workers);
    let optimistic_prefix_cache_reused_ratio =
        optimistic_prefix_cache_reused_ratio(&normalized_requests, args.block_size);

    let round_robin_report = super::entrypoints::simulate_trace_requests_with_router_mode(
        args.clone(),
        router_config.clone(),
        prefill_load_estimator.clone(),
        normalized_requests.clone(),
        num_workers,
        1.0,
        ReplayRouterMode::RoundRobin,
    )?;

    let kv_router_report = if num_workers > 1 {
        Some(
            super::entrypoints::simulate_trace_requests_with_router_mode(
                args.clone(),
                router_config,
                prefill_load_estimator,
                normalized_requests,
                num_workers,
                1.0,
                ReplayRouterMode::KvRouter,
            )?,
        )
    } else {
        None
    };

    let (upper_bound_duration_ms, upper_bound_source) = match &kv_router_report {
        Some(report)
            if report.throughput.duration_ms < round_robin_report.throughput.duration_ms =>
        {
            (
                report.throughput.duration_ms,
                ReplayUpperBoundSource::KvRouter,
            )
        }
        _ => (
            round_robin_report.throughput.duration_ms,
            ReplayUpperBoundSource::RoundRobin,
        ),
    };

    Ok(ReplayBoundsReport {
        lower_bound_duration_ms,
        upper_bound_duration_ms,
        upper_bound_source,
        optimistic_prefix_cache_reused_ratio,
        round_robin_report,
        kv_router_report,
    })
}

fn normalize_requests_for_bounds(
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> Result<Vec<DirectRequest>> {
    if requests
        .iter()
        .any(|request| request.arrival_timestamp_ms.is_some())
    {
        Ok(normalize_trace_requests(requests, arrival_speedup_ratio)?
            .into_iter()
            .collect())
    } else {
        Ok(requests
            .into_iter()
            .map(|mut request| {
                request.arrival_timestamp_ms = Some(0.0);
                request
            })
            .collect())
    }
}

fn optimistic_lower_bound_ms(
    args: &MockEngineArgs,
    requests: &[DirectRequest],
    num_workers: usize,
) -> f64 {
    let mut max_completion_lb_ms = 0.0_f64;

    for request in requests {
        let arrival_ms = request.arrival_timestamp_ms.unwrap_or(0.0);
        let service_lb_ms = optimistic_decode_step_ms(
            args,
            request.tokens.len(),
            args.num_gpu_blocks * args.block_size,
        ) * request.max_output_tokens as f64;
        max_completion_lb_ms = max_completion_lb_ms.max(arrival_ms + service_lb_ms);
    }

    let _ = num_workers;
    max_completion_lb_ms
}

fn optimistic_prefix_cache_reused_ratio(requests: &[DirectRequest], block_size: usize) -> f64 {
    let total_input_tokens: usize = requests.iter().map(|request| request.tokens.len()).sum();
    if total_input_tokens == 0 {
        return 0.0;
    }

    let total_reused_tokens: usize = best_prefix_tokens_by_request(requests, block_size)
        .into_iter()
        .sum();
    total_reused_tokens as f64 / total_input_tokens as f64
}

fn best_prefix_tokens_by_request(requests: &[DirectRequest], block_size: usize) -> Vec<usize> {
    let mut best_prefixes = vec![0; requests.len()];

    for left_idx in 0..requests.len() {
        for right_idx in (left_idx + 1)..requests.len() {
            let common_prefix = common_prefix_tokens(
                &requests[left_idx].tokens,
                &requests[right_idx].tokens,
                block_size,
            );
            best_prefixes[left_idx] = best_prefixes[left_idx].max(common_prefix);
            best_prefixes[right_idx] = best_prefixes[right_idx].max(common_prefix);
        }
    }

    best_prefixes
}

fn common_prefix_tokens(left: &[Token], right: &[Token], block_size: usize) -> usize {
    let num_blocks = cmp::min(left.len(), right.len()) / block_size;
    let mut matched_blocks = 0usize;

    while matched_blocks < num_blocks {
        let start = matched_blocks * block_size;
        let end = start + block_size;
        if left[start..end] != right[start..end] {
            break;
        }
        matched_blocks += 1;
    }

    matched_blocks * block_size
}

fn optimistic_decode_step_ms(
    args: &MockEngineArgs,
    _context_length: usize,
    total_kv_tokens: usize,
) -> f64 {
    let decode_ms = args
        .perf_model
        .predict_decode_time(1, 1, 1, total_kv_tokens.max(1));
    let unscaled = Duration::from_secs_f64(decode_ms.max(0.0) / 1000.0);
    let effective_ratio = args.speedup_ratio * args.decode_speedup_ratio;
    if effective_ratio <= 0.0 || unscaled <= Duration::ZERO {
        return unscaled.as_secs_f64() * 1000.0;
    }
    (Duration::from_secs_f64(unscaled.as_secs_f64() / effective_ratio)).as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::MockEngineArgsBuilder;

    fn request(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        arrival_timestamp_ms: Option<f64>,
    ) -> DirectRequest {
        DirectRequest {
            tokens,
            max_output_tokens,
            uuid: None,
            dp_rank: 0,
            arrival_timestamp_ms,
        }
    }

    fn test_args() -> MockEngineArgs {
        MockEngineArgsBuilder::default()
            .block_size(4)
            .num_gpu_blocks(1024)
            .max_num_seqs(Some(64))
            .max_num_batched_tokens(Some(1024))
            .speedup_ratio(10.0)
            .build()
            .expect("failed to build args")
            .normalized()
            .expect("failed to normalize args")
    }

    #[test]
    fn bounds_report_keeps_lower_bound_below_upper_bound() {
        let report = estimate_request_bounds_with_router_mode(
            test_args(),
            None,
            None,
            vec![
                request(vec![1, 2, 3, 4, 5, 6, 7, 8], 8, None),
                request(vec![1, 2, 3, 4, 9, 10, 11, 12], 8, None),
                request(vec![1, 2, 3, 4, 13, 14, 15, 16], 8, None),
            ],
            2,
            1.0,
        )
        .expect("bounds estimation should succeed");

        assert!(
            report.lower_bound_duration_ms <= report.upper_bound_duration_ms,
            "lower bound {} exceeded upper bound {}",
            report.lower_bound_duration_ms,
            report.upper_bound_duration_ms
        );
        assert!(report.optimistic_prefix_cache_reused_ratio > 0.0);
        assert!(report.kv_router_report.is_some());
    }

    #[test]
    fn bounds_report_skips_kv_router_for_single_worker() {
        let report = estimate_request_bounds_with_router_mode(
            test_args(),
            None,
            None,
            vec![
                request(vec![1, 2, 3, 4], 4, Some(100.0)),
                request(vec![1, 2, 3, 4], 4, Some(200.0)),
            ],
            1,
            2.0,
        )
        .expect("bounds estimation should succeed");

        assert!(report.kv_router_report.is_none());
        assert_eq!(
            report.upper_bound_source,
            ReplayUpperBoundSource::RoundRobin
        );
        assert!(report.lower_bound_duration_ms <= report.round_robin_report.throughput.duration_ms);
    }
}
