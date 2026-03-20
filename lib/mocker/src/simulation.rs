// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, WorkerType};

#[derive(Debug, Clone)]
pub struct TraceSimulationReport {
    pub request_counts: TraceRequestCounts,
    pub throughput: TraceThroughputStats,
    pub prefix_cache_reused_ratio: f64,
    pub latency: TraceLatencyStats,
}

#[derive(Debug, Clone)]
pub struct TraceRequestCounts {
    pub num_requests: usize,
    pub completed_requests: usize,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct TraceThroughputStats {
    pub duration_ms: f64,
    pub wall_time_ms: f64,
    pub request_throughput_rps: f64,
    pub input_throughput_tok_s: f64,
    pub output_throughput_tok_s: f64,
    pub total_throughput_tok_s: f64,
}

#[derive(Debug, Clone)]
pub struct TraceDistributionStats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

#[derive(Debug, Clone)]
pub struct TraceLatencyStats {
    pub mean_queue_ms: f64,
    pub ttft: TraceDistributionStats,
    pub tpot: TraceDistributionStats,
    pub itl: TraceInterTokenLatencyStats,
    pub e2e: TraceDistributionStats,
}

#[derive(Debug, Clone)]
pub struct TraceInterTokenLatencyStats {
    pub distribution: TraceDistributionStats,
    pub max_ms: f64,
}

impl TraceSimulationReport {
    pub fn with_wall_time_ms(mut self, wall_time_ms: f64) -> Self {
        self.throughput.wall_time_ms = wall_time_ms;
        self
    }
}

impl Serialize for TraceSimulationReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(29))?;
        map.serialize_entry("num_requests", &self.request_counts.num_requests)?;
        map.serialize_entry(
            "completed_requests",
            &self.request_counts.completed_requests,
        )?;
        map.serialize_entry(
            "total_input_tokens",
            &self.request_counts.total_input_tokens,
        )?;
        map.serialize_entry(
            "total_output_tokens",
            &self.request_counts.total_output_tokens,
        )?;
        map.serialize_entry("duration_ms", &self.throughput.duration_ms)?;
        map.serialize_entry("wall_time_ms", &self.throughput.wall_time_ms)?;
        map.serialize_entry(
            "request_throughput_rps",
            &self.throughput.request_throughput_rps,
        )?;
        map.serialize_entry(
            "input_throughput_tok_s",
            &self.throughput.input_throughput_tok_s,
        )?;
        map.serialize_entry(
            "output_throughput_tok_s",
            &self.throughput.output_throughput_tok_s,
        )?;
        map.serialize_entry(
            "total_throughput_tok_s",
            &self.throughput.total_throughput_tok_s,
        )?;
        map.serialize_entry("prefix_cache_reused_ratio", &self.prefix_cache_reused_ratio)?;
        map.serialize_entry("mean_queue_ms", &self.latency.mean_queue_ms)?;
        serialize_distribution(&mut map, "ttft", &self.latency.ttft)?;
        serialize_distribution(&mut map, "tpot", &self.latency.tpot)?;
        serialize_distribution(&mut map, "itl", &self.latency.itl.distribution)?;
        map.serialize_entry("max_itl_ms", &self.latency.itl.max_ms)?;
        serialize_distribution(&mut map, "e2e_latency", &self.latency.e2e)?;
        map.end()
    }
}

fn serialize_distribution<S>(
    map: &mut S,
    prefix: &str,
    stats: &TraceDistributionStats,
) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    map.serialize_entry(&format!("mean_{prefix}_ms"), &stats.mean_ms)?;
    map.serialize_entry(&format!("median_{prefix}_ms"), &stats.median_ms)?;
    map.serialize_entry(&format!("p95_{prefix}_ms"), &stats.p95_ms)?;
    map.serialize_entry(&format!("p99_{prefix}_ms"), &stats.p99_ms)?;
    Ok(())
}

#[derive(Debug)]
struct TraceRequestStats {
    arrival_time_ms: f64,
    first_admit_ms: Option<f64>,
    token_times_ms: Vec<f64>,
    input_length: usize,
    output_length: usize,
    reused_input_tokens: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TraceRequestStatsSnapshot {
    pub arrival_time_ms: f64,
    pub first_admit_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub last_token_ms: Option<f64>,
    pub input_length: usize,
    pub output_length: usize,
    pub reused_input_tokens: usize,
}

#[derive(Debug, Default)]
pub(crate) struct TraceCollector {
    requests: Mutex<HashMap<Uuid, TraceRequestStats>>,
}

impl TraceRequestStats {
    fn first_token_ms(&self) -> Option<f64> {
        self.token_times_ms.first().copied()
    }

    fn last_token_ms(&self) -> Option<f64> {
        self.token_times_ms.last().copied()
    }

    fn mean_tpot_ms(&self) -> Option<f64> {
        let num_gaps = self.token_times_ms.len().saturating_sub(1);
        if num_gaps == 0 {
            return None;
        }

        let first_token_ms = self.first_token_ms()?;
        let last_token_ms = self.last_token_ms()?;
        Some((last_token_ms - first_token_ms).max(0.0) / num_gaps as f64)
    }

    fn itls_ms(&self) -> impl Iterator<Item = f64> + '_ {
        self.token_times_ms
            .windows(2)
            .map(|window| (window[1] - window[0]).max(0.0))
    }
}

impl TraceCollector {
    pub(crate) fn on_arrival(
        &self,
        uuid: Uuid,
        arrival_time_ms: f64,
        input_length: usize,
        output_length: usize,
    ) {
        self.requests.lock().unwrap().insert(
            uuid,
            TraceRequestStats {
                arrival_time_ms,
                first_admit_ms: None,
                token_times_ms: Vec::with_capacity(output_length),
                input_length,
                output_length,
                reused_input_tokens: 0,
            },
        );
    }

    pub(crate) fn on_admit(&self, uuid: Uuid, admit_time_ms: f64, reused_input_tokens: usize) {
        if let Some(stats) = self.requests.lock().unwrap().get_mut(&uuid) {
            stats.first_admit_ms.get_or_insert(admit_time_ms);
            stats.reused_input_tokens = stats.reused_input_tokens.max(reused_input_tokens);
        }
    }

    pub(crate) fn on_token(&self, uuid: Uuid, token_time_ms: f64) {
        if let Some(stats) = self.requests.lock().unwrap().get_mut(&uuid) {
            stats.token_times_ms.push(token_time_ms);
        }
    }

    pub(crate) fn finish(self) -> TraceSimulationReport {
        let requests = self.requests.into_inner().unwrap();
        let mut ttfts = Vec::new();
        let mut tpots = Vec::new();
        let mut itls = Vec::new();
        let mut e2e_latencies = Vec::new();
        let mut queue_latencies = Vec::new();
        let mut duration_ms = 0.0_f64;
        let mut total_input_tokens = 0usize;
        let mut total_output_tokens = 0usize;
        let mut completed_requests = 0usize;
        let mut total_reused_tokens = 0usize;

        for stats in requests.values() {
            let Some(first_admit_ms) = stats.first_admit_ms else {
                continue;
            };
            let Some(first_token_ms) = stats.first_token_ms() else {
                continue;
            };
            let Some(last_token_ms) = stats.last_token_ms() else {
                continue;
            };

            completed_requests += 1;
            total_input_tokens += stats.input_length;
            total_output_tokens += stats.output_length;
            total_reused_tokens += stats.reused_input_tokens;
            duration_ms = duration_ms.max(last_token_ms);

            let queue_ms = (first_admit_ms - stats.arrival_time_ms).max(0.0);
            let ttft_ms = (first_token_ms - stats.arrival_time_ms).max(0.0);
            let e2e_ms = (last_token_ms - stats.arrival_time_ms).max(0.0);
            queue_latencies.push(queue_ms);
            ttfts.push(ttft_ms);
            e2e_latencies.push(e2e_ms);

            if let Some(tpot_ms) = stats.mean_tpot_ms() {
                tpots.push(tpot_ms);
                itls.extend(stats.itls_ms());
            }
        }

        let duration_s = (duration_ms / 1000.0).max(1e-9);
        TraceSimulationReport {
            request_counts: TraceRequestCounts {
                num_requests: requests.len(),
                completed_requests,
                total_input_tokens,
                total_output_tokens,
            },
            throughput: TraceThroughputStats {
                duration_ms,
                wall_time_ms: 0.0,
                request_throughput_rps: completed_requests as f64 / duration_s,
                input_throughput_tok_s: total_input_tokens as f64 / duration_s,
                output_throughput_tok_s: total_output_tokens as f64 / duration_s,
                total_throughput_tok_s: (total_input_tokens + total_output_tokens) as f64
                    / duration_s,
            },
            prefix_cache_reused_ratio: if total_input_tokens == 0 {
                0.0
            } else {
                total_reused_tokens as f64 / total_input_tokens as f64
            },
            latency: TraceLatencyStats {
                mean_queue_ms: mean(&queue_latencies),
                ttft: build_distribution_stats(&ttfts),
                tpot: build_distribution_stats(&tpots),
                itl: TraceInterTokenLatencyStats {
                    distribution: build_distribution_stats(&itls),
                    max_ms: max_value(&itls),
                },
                e2e: build_distribution_stats(&e2e_latencies),
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self, uuid: Uuid) -> Option<TraceRequestStatsSnapshot> {
        self.requests
            .lock()
            .unwrap()
            .get(&uuid)
            .map(|stats| TraceRequestStatsSnapshot {
                arrival_time_ms: stats.arrival_time_ms,
                first_admit_ms: stats.first_admit_ms,
                first_token_ms: stats.first_token_ms(),
                last_token_ms: stats.last_token_ms(),
                input_length: stats.input_length,
                output_length: stats.output_length,
                reused_input_tokens: stats.reused_input_tokens,
            })
    }
}

#[derive(Debug, Deserialize)]
struct RawTraceRecord {
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
}

pub fn simulate_trace_file(
    args: MockEngineArgs,
    trace_path: &Path,
) -> Result<TraceSimulationReport> {
    if args.engine_type != EngineType::Vllm {
        bail!(
            "trace replay only supports engine_type=vllm, got {:?}",
            args.engine_type
        );
    }
    if args.worker_type != WorkerType::Aggregated {
        bail!(
            "trace replay only supports aggregated workers, got {:?}",
            args.worker_type
        );
    }
    if args.dp_size != 1 {
        bail!(
            "trace replay only supports data_parallel_size=1, got {}",
            args.dp_size
        );
    }

    let requests = load_trace_requests(trace_path, args.block_size)?;
    let started_at = Instant::now();
    let report = crate::scheduler::vllm::simulate_trace(args, requests)?;
    Ok(report.with_wall_time_ms(started_at.elapsed().as_secs_f64() * 1000.0))
}

fn load_trace_requests(trace_path: &Path, trace_block_size: usize) -> Result<Vec<DirectRequest>> {
    let file = File::open(trace_path)
        .with_context(|| format!("failed to open trace file {}", trace_path.display()))?;
    let reader = BufReader::new(file);
    let mut requests = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_idx + 1,
                trace_path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let raw: RawTraceRecord = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse line {} from {} as JSON",
                line_idx + 1,
                trace_path.display()
            )
        })?;

        let input_length = raw
            .input_length
            .ok_or_else(|| anyhow!("trace line {} is missing input_length", line_idx + 1))?;
        let output_length = raw
            .output_length
            .ok_or_else(|| anyhow!("trace line {} is missing output_length", line_idx + 1))?;
        let hash_ids = raw
            .hash_ids
            .ok_or_else(|| anyhow!("trace line {} is missing hash_ids", line_idx + 1))?;
        let arrival_timestamp_ms = raw
            .timestamp
            .or(raw.created_time)
            .ok_or_else(|| anyhow!("trace line {} is missing timestamp", line_idx + 1))?;
        let tokens = synthesize_tokens_from_hash_ids(&hash_ids, input_length, trace_block_size)
            .with_context(|| {
                format!(
                    "failed to synthesize tokens from hash_ids on line {}",
                    line_idx + 1
                )
            })?;

        requests.push(DirectRequest {
            tokens,
            max_output_tokens: output_length,
            uuid: Some(Uuid::new_v4()),
            dp_rank: 0,
            arrival_timestamp_ms: Some(arrival_timestamp_ms),
        });
    }

    if requests.is_empty() {
        bail!(
            "trace file {} did not contain any requests",
            trace_path.display()
        );
    }

    Ok(requests)
}

fn synthesize_tokens_from_hash_ids(
    hash_ids: &[u64],
    input_length: usize,
    trace_block_size: usize,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(input_length);

    for &hash_id in hash_ids {
        let token_id = u32::try_from(hash_id)
            .map_err(|_| anyhow!("hash_id {hash_id} exceeds u32::MAX for token synthesis"))?;
        // TODO: Replace this repeated-token expansion with a hash-native prompt representation.
        tokens.extend((0..trace_block_size).map(|_| token_id));
        if tokens.len() >= input_length {
            tokens.truncate(input_length);
            return Ok(tokens);
        }
    }

    bail!(
        "input_length {} exceeds synthesized capacity {} from {} hash_ids and block_size {}",
        input_length,
        hash_ids.len() * trace_block_size,
        hash_ids.len(),
        trace_block_size
    );
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_value(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn build_distribution_stats(values: &[f64]) -> TraceDistributionStats {
    TraceDistributionStats {
        mean_ms: mean(values),
        median_ms: percentile(values, 50.0),
        p95_ms: percentile(values, 95.0),
        p99_ms: percentile(values, 99.0),
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let rank = ((sorted.len() - 1) as f64 * percentile / 100.0).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(8))
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    #[test]
    fn test_replay_rejects_non_aggregated_workers() {
        let args = MockEngineArgs::builder()
            .worker_type(WorkerType::Decode)
            .build()
            .unwrap();
        let path = std::env::temp_dir().join(format!(
            "mocker_trace_{}_{}.jsonl",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            r#"{"timestamp":1.0,"input_length":4,"output_length":2,"hash_ids":[1]}"#,
        )
        .unwrap();

        let error = simulate_trace_file(args, &path).unwrap_err().to_string();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("aggregated workers"));
    }

    #[test]
    fn test_replay_report_and_json_serialization() {
        let path = std::env::temp_dir().join(format!(
            "mocker_trace_{}_{}.jsonl",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            [
                r#"{"timestamp":100.0,"input_length":4,"output_length":2,"hash_ids":[1]}"#,
                r#"{"timestamp":101.0,"input_length":4,"output_length":2,"hash_ids":[1]}"#,
                r#"{"timestamp":150.0,"input_length":4,"output_length":2,"hash_ids":[2]}"#,
            ]
            .join("\n"),
        )
        .unwrap();

        let report = simulate_trace_file(test_args(), &path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(report.request_counts.num_requests, 3);
        assert_eq!(report.request_counts.completed_requests, 3);
        assert_eq!(report.request_counts.total_input_tokens, 12);
        assert_eq!(report.request_counts.total_output_tokens, 6);
        assert!(report.throughput.duration_ms > 0.0);
        assert!(report.latency.ttft.mean_ms >= 0.0);
        assert!(report.latency.e2e.mean_ms >= report.latency.ttft.mean_ms);
        assert!(report.latency.itl.distribution.mean_ms >= 0.0);

        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["completed_requests"], 3);
        assert!(json.get("mean_ttft_ms").is_some());
        assert!(json.get("mean_e2e_latency_ms").is_some());
    }

    #[test]
    fn test_replay_itl_uses_per_token_gaps() {
        let collector = TraceCollector::default();
        let uuid = Uuid::from_u128(11);

        collector.on_arrival(uuid, 0.0, 4, 4);
        collector.on_admit(uuid, 0.0, 0);
        collector.on_token(uuid, 10.0);
        collector.on_token(uuid, 11.0);
        collector.on_token(uuid, 12.0);
        collector.on_token(uuid, 110.0);

        let report = collector.finish();

        assert!((report.latency.tpot.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert!((report.latency.itl.distribution.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert_eq!(report.latency.itl.distribution.median_ms, 1.0);
        assert_eq!(report.latency.itl.distribution.p95_ms, 98.0);
        assert_eq!(report.latency.itl.max_ms, 98.0);
    }
}
