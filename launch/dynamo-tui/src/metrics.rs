// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub collected_at: Instant,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub request_rate: Option<f64>,
    pub tokens_per_sec: Option<f64>,
    pub inflight: Option<f64>,
    pub queued: Option<f64>,
    pub total_requests: f64,
    pub total_output_tokens: f64,
}

#[derive(Debug, Clone)]
pub struct PrometheusSample {
    pub collected_at: Instant,
    pub ttft_sum: f64,
    pub ttft_count: f64,
    pub tpot_sum: f64,
    pub tpot_count: f64,
    pub requests_total: f64,
    pub output_tokens_sum: f64,
    pub inflight: f64,
    pub queued: f64,
}

impl Default for PrometheusSample {
    fn default() -> Self {
        Self {
            collected_at: Instant::now(),
            ttft_sum: 0.0,
            ttft_count: 0.0,
            tpot_sum: 0.0,
            tpot_count: 0.0,
            requests_total: 0.0,
            output_tokens_sum: 0.0,
            inflight: 0.0,
            queued: 0.0,
        }
    }
}

pub fn process_metrics(
    text: &str,
    now: Instant,
    previous: Option<&PrometheusSample>,
) -> (MetricsSnapshot, PrometheusSample) {
    let mut sample = PrometheusSample {
        collected_at: now,
        ..Default::default()
    };

    sample.ttft_sum = sum_metric(text, "dynamo_frontend_time_to_first_token_seconds_sum");
    sample.ttft_count = sum_metric(text, "dynamo_frontend_time_to_first_token_seconds_count");
    sample.tpot_sum = sum_metric(text, "dynamo_frontend_inter_token_latency_seconds_sum");
    sample.tpot_count = sum_metric(text, "dynamo_frontend_inter_token_latency_seconds_count");
    sample.requests_total = sum_metric(text, "dynamo_frontend_requests_total");
    sample.output_tokens_sum = sum_metric(text, "dynamo_frontend_output_sequence_tokens_sum");
    sample.inflight = sum_metric(text, "dynamo_frontend_inflight_requests");
    sample.queued = sum_metric(text, "dynamo_frontend_queued_requests");

    let ttft_ms = average_ms(sample.ttft_sum, sample.ttft_count);
    let tpot_ms = average_ms(sample.tpot_sum, sample.tpot_count);

    let (request_rate, tokens_per_sec) = previous
        .and_then(|prev| rate_metrics(&sample, prev))
        .unwrap_or((None, None));

    (
        MetricsSnapshot {
            collected_at: now,
            ttft_ms,
            tpot_ms,
            request_rate,
            tokens_per_sec,
            inflight: value_or_none(sample.inflight),
            queued: value_or_none(sample.queued),
            total_requests: sample.requests_total,
            total_output_tokens: sample.output_tokens_sum,
        },
        sample,
    )
}

fn sum_metric(text: &str, metric: &str) -> f64 {
    text.lines()
        .filter(|line| line.starts_with(metric))
        .filter_map(parse_sample_value)
        .sum()
}

fn parse_sample_value(line: &str) -> Option<f64> {
    line.split_whitespace()
        .last()
        .and_then(|value| value.parse::<f64>().ok())
}

fn average_ms(sum: f64, count: f64) -> Option<f64> {
    if count <= f64::EPSILON {
        None
    } else {
        Some((sum / count) * 1_000.0)
    }
}

fn rate_metrics(
    current: &PrometheusSample,
    previous: &PrometheusSample,
) -> Option<(Option<f64>, Option<f64>)> {
    let elapsed = current
        .collected_at
        .checked_duration_since(previous.collected_at)?;
    let secs = elapsed.as_secs_f64();
    if secs <= f64::EPSILON {
        return Some((None, None));
    }

    let request_delta = current.requests_total - previous.requests_total;
    let token_delta = current.output_tokens_sum - previous.output_tokens_sum;

    let request_rate = if request_delta.is_sign_negative() {
        None
    } else {
        Some(request_delta.max(0.0) / secs)
    };
    let tokens_per_sec = if token_delta.is_sign_negative() {
        None
    } else {
        Some(token_delta.max(0.0) / secs)
    };

    Some((request_rate, tokens_per_sec))
}

fn value_or_none(value: f64) -> Option<f64> {
    if value.abs() <= f64::EPSILON {
        None
    } else {
        Some(value)
    }
}
