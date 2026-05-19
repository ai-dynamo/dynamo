// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashMap;
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use std::fmt::{Display, Formatter, Result as FmtResult};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TraceSimulationReport {
    pub request_counts: TraceRequestCounts,
    pub throughput: TraceThroughputStats,
    pub prefix_cache_reused_ratio: f64,
    pub first_admission_prefix_cache_reused_ratio: f64,
    pub latency: TraceLatencyStats,
    /// Per-request records, one per admitted request. Populated by
    /// `TraceCollector::finish`. Intentionally NOT serialized into the summary
    /// JSON (see custom `Serialize` impl below) — consumers that want per-
    /// request granularity should access this field directly and serialize
    /// it themselves (e.g., the `--report-jsonl` CLI path).
    pub per_request: Vec<PerRequestRecord>,
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
    pub min_ms: f64,
    pub max_ms: f64,
    pub median_ms: f64,
    pub p75_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub std_ms: f64,
}

#[derive(Debug, Clone)]
pub struct TraceLatencyStats {
    pub ttft: TraceDistributionStats,
    pub ttst: TraceDistributionStats,
    pub tpot: TraceDistributionStats,
    pub itl: TraceInterTokenLatencyStats,
    pub e2e: TraceDistributionStats,
    pub output_token_throughput_per_user: TraceDistributionStats,
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

    pub fn processed_tokens(&self) -> usize {
        self.request_counts.total_input_tokens + self.request_counts.total_output_tokens
    }

    pub fn processed_tokens_per_s(&self) -> f64 {
        if self.throughput.wall_time_ms <= 0.0 {
            return 0.0;
        }
        self.processed_tokens() as f64 / self.throughput.wall_time_ms * 1000.0
    }

    pub fn processed_output_tokens_per_s(&self) -> f64 {
        if self.throughput.wall_time_ms <= 0.0 {
            return 0.0;
        }
        self.request_counts.total_output_tokens as f64 / self.throughput.wall_time_ms * 1000.0
    }
}

impl Display for TraceSimulationReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(
            f,
            "  completed_requests: {}",
            self.request_counts.completed_requests
        )?;
        writeln!(
            f,
            "  request_throughput_rps: {:.6}",
            self.throughput.request_throughput_rps
        )?;
        writeln!(
            f,
            "  output_throughput_tok_s: {:.6}",
            self.throughput.output_throughput_tok_s
        )?;
        writeln!(
            f,
            "  total_input_tokens: {}",
            self.request_counts.total_input_tokens
        )?;
        writeln!(
            f,
            "  total_output_tokens: {}",
            self.request_counts.total_output_tokens
        )?;
        writeln!(
            f,
            "  processed_tokens_per_s: {:.6}",
            self.processed_tokens_per_s()
        )?;
        writeln!(
            f,
            "  processed_output_tokens_per_s: {:.6}",
            self.processed_output_tokens_per_s()
        )?;
        writeln!(f, "  mean_ttft_ms: {:.6}", self.latency.ttft.mean_ms)?;
        writeln!(f, "  mean_e2e_latency_ms: {:.6}", self.latency.e2e.mean_ms)?;
        writeln!(
            f,
            "  prefix_cache_reused_ratio: {:.6}",
            self.prefix_cache_reused_ratio
        )?;
        writeln!(
            f,
            "  first_admission_prefix_cache_reused_ratio: {:.6}",
            self.first_admission_prefix_cache_reused_ratio
        )?;
        write!(f, "  wall_time_ms: {:.6}", self.throughput.wall_time_ms)
    }
}

impl Serialize for TraceSimulationReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(62))?;
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
        map.serialize_entry("processed_tokens", &self.processed_tokens())?;
        map.serialize_entry("processed_tokens_per_s", &self.processed_tokens_per_s())?;
        map.serialize_entry(
            "processed_output_tokens_per_s",
            &self.processed_output_tokens_per_s(),
        )?;
        map.serialize_entry("prefix_cache_reused_ratio", &self.prefix_cache_reused_ratio)?;
        map.serialize_entry(
            "first_admission_prefix_cache_reused_ratio",
            &self.first_admission_prefix_cache_reused_ratio,
        )?;
        serialize_distribution(&mut map, "ttft", &self.latency.ttft)?;
        serialize_distribution(&mut map, "ttst", &self.latency.ttst)?;
        serialize_distribution(&mut map, "tpot", &self.latency.tpot)?;
        serialize_distribution(&mut map, "itl", &self.latency.itl.distribution)?;
        map.serialize_entry("max_itl_ms", &self.latency.itl.max_ms)?;
        serialize_distribution(&mut map, "e2e_latency", &self.latency.e2e)?;
        serialize_rate_distribution(
            &mut map,
            "output_token_throughput_per_user",
            &self.latency.output_token_throughput_per_user,
        )?;
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
    map.serialize_entry(&format!("min_{prefix}_ms"), &stats.min_ms)?;
    map.serialize_entry(&format!("max_{prefix}_ms"), &stats.max_ms)?;
    map.serialize_entry(&format!("median_{prefix}_ms"), &stats.median_ms)?;
    map.serialize_entry(&format!("p75_{prefix}_ms"), &stats.p75_ms)?;
    map.serialize_entry(&format!("p90_{prefix}_ms"), &stats.p90_ms)?;
    map.serialize_entry(&format!("p95_{prefix}_ms"), &stats.p95_ms)?;
    map.serialize_entry(&format!("p99_{prefix}_ms"), &stats.p99_ms)?;
    map.serialize_entry(&format!("std_{prefix}_ms"), &stats.std_ms)?;
    Ok(())
}

fn serialize_rate_distribution<S>(
    map: &mut S,
    prefix: &str,
    stats: &TraceDistributionStats,
) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    map.serialize_entry(&format!("mean_{prefix}"), &stats.mean_ms)?;
    map.serialize_entry(&format!("min_{prefix}"), &stats.min_ms)?;
    map.serialize_entry(&format!("max_{prefix}"), &stats.max_ms)?;
    map.serialize_entry(&format!("median_{prefix}"), &stats.median_ms)?;
    map.serialize_entry(&format!("p75_{prefix}"), &stats.p75_ms)?;
    map.serialize_entry(&format!("p90_{prefix}"), &stats.p90_ms)?;
    map.serialize_entry(&format!("p95_{prefix}"), &stats.p95_ms)?;
    map.serialize_entry(&format!("p99_{prefix}"), &stats.p99_ms)?;
    map.serialize_entry(&format!("std_{prefix}"), &stats.std_ms)?;
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
    /// Index of the prefill worker that handled this request, if any.
    /// `None` in two situations:
    ///   - Aggregated replay (no separate prefill pool) — meaningless field.
    ///   - Offline disagg with conditional-prefill bypass — request was
    ///     routed directly to a decode worker without going through prefill.
    ///
    /// Downstream tooling derives "was_bypassed" as `prefill_worker_idx is None`
    /// in disagg mode.
    prefill_worker_idx: Option<usize>,
    /// Index of the decode worker that handled this request, if any.
    decode_worker_idx: Option<usize>,
    /// Session / turn metadata copied from the workload driver, when the
    /// trace source carries it (e.g., multi-turn Mooncake). `None` for raw
    /// single-shot request lists.
    session_id: Option<String>,
    turn_index: Option<usize>,
}

/// Flat per-request record for `--report-jsonl` emission. One JSON line per
/// request in the JSONL output; consumed by external analysis tools that want
/// per-request granularity (TTFT vs. ISL scatter, worker-residency analysis,
/// bypass classification, etc.).
#[derive(Debug, Clone, Serialize)]
pub struct PerRequestRecord {
    /// Session identifier from the trace, when present. Mirrors AIPerf's
    /// `conversation_id` field for the same purpose: bucket per-request
    /// records by multi-turn session. Placed first in the serialized output
    /// so each JSONL row leads with its session/turn identity, matching
    /// AIPerf's `profile_export.jsonl` layout.
    pub session_id: Option<String>,
    /// Zero-based turn index within `session_id`, when present.
    pub turn_index: Option<usize>,
    pub uuid: String,
    pub arrival_time_ms: f64,
    pub first_admit_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub last_token_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub ttst_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
    /// Inter-token latency for this request, in milliseconds. Matches
    /// AIPerf's `inter_token_latency` field — one scalar per request.
    pub itl_ms: Option<f64>,
    pub input_length: usize,
    pub output_length: usize,
    pub reused_input_tokens: usize,
    pub prefill_worker_idx: Option<usize>,
    pub decode_worker_idx: Option<usize>,
    first_admission_reused_input_tokens: usize,
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
    pub first_admission_reused_input_tokens: usize,
}

#[derive(Debug, Default)]
pub(crate) struct TraceCollector {
    requests: FxHashMap<Uuid, TraceRequestStats>,
    /// When `true`, `finish()` populates `TraceSimulationReport::per_request`.
    /// Default `false` to skip the ~100ms terminal pass + ~30MB allocation
    /// when the caller doesn't need per-request granularity.
    capture_per_request: bool,
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

    fn ttst_ms(&self) -> Option<f64> {
        let [first_token_ms, second_token_ms, ..] = self.token_times_ms.as_slice() else {
            return None;
        };
        Some((second_token_ms - first_token_ms).max(0.0))
    }
}

impl TraceCollector {
    /// Toggle whether `finish()` should build per-request records. Off by
    /// default; the runtimes flip it on when the caller asks for JSONL output.
    pub(crate) fn set_capture_per_request(&mut self, value: bool) {
        self.capture_per_request = value;
    }

    pub(crate) fn on_arrival(
        &mut self,
        uuid: Uuid,
        arrival_time_ms: f64,
        input_length: usize,
        output_length: usize,
    ) {
        self.requests.insert(
            uuid,
            TraceRequestStats {
                arrival_time_ms,
                first_admit_ms: None,
                token_times_ms: Vec::with_capacity(output_length),
                input_length,
                output_length,
                reused_input_tokens: 0,
                prefill_worker_idx: None,
                decode_worker_idx: None,
                session_id: None,
                turn_index: None,
                first_admission_reused_input_tokens: 0,
            },
        );
    }

    /// Attach session/turn metadata to a request. Called by the disagg/agg
    /// runtimes when the workload driver provides it (multi-turn traces).
    /// Idempotent — set-once semantics, so calling on the same uuid more than
    /// once is a no-op after the first.
    pub(crate) fn on_session_metadata(
        &mut self,
        uuid: Uuid,
        session_id: String,
        turn_index: usize,
    ) {
        if let Some(stats) = self.requests.get_mut(&uuid)
            && stats.session_id.is_none()
        {
            stats.session_id = Some(session_id);
            stats.turn_index = Some(turn_index);
        }
    }

    /// Record that `uuid` was dispatched to `worker_idx` on the prefill pool
    /// (offline disagg replay only). Idempotent — subsequent calls are no-ops
    /// once a value is set, so the first dispatch wins. Aggregated replay does
    /// not call this; for those requests `prefill_worker_idx` stays `None`.
    pub(crate) fn on_prefill_assigned(&mut self, uuid: Uuid, worker_idx: usize) {
        if let Some(stats) = self.requests.get_mut(&uuid)
            && stats.prefill_worker_idx.is_none()
        {
            stats.prefill_worker_idx = Some(worker_idx);
        }
    }

    /// Record that `uuid` was dispatched to `worker_idx` on the decode pool
    /// (offline disagg replay), or to the only pool (aggregated replay).
    /// Idempotent.
    pub(crate) fn on_decode_assigned(&mut self, uuid: Uuid, worker_idx: usize) {
        if let Some(stats) = self.requests.get_mut(&uuid)
            && stats.decode_worker_idx.is_none()
        {
            stats.decode_worker_idx = Some(worker_idx);
        }
    }

    pub(crate) fn on_admit(&mut self, uuid: Uuid, admit_time_ms: f64, reused_input_tokens: usize) {
        if let Some(stats) = self.requests.get_mut(&uuid) {
            if stats.first_admit_ms.is_none() {
                stats.first_admission_reused_input_tokens = reused_input_tokens;
                stats.first_admit_ms = Some(admit_time_ms);
            }
            stats.reused_input_tokens = stats.reused_input_tokens.max(reused_input_tokens);
        }
    }

    pub(crate) fn on_token(&mut self, uuid: Uuid, token_time_ms: f64) {
        if let Some(stats) = self.requests.get_mut(&uuid) {
            stats.token_times_ms.push(token_time_ms);
        }
    }

    /// Return (ttft_ms, mean_itl_ms) for a completed request, if available.
    pub(crate) fn request_latencies(&self, uuid: Uuid) -> Option<(f64, f64)> {
        let stats = self.requests.get(&uuid)?;
        let first_token_ms = stats.first_token_ms()?;
        let ttft_ms = (first_token_ms - stats.arrival_time_ms).max(0.0);
        let mean_itl_ms = stats.mean_tpot_ms().unwrap_or(0.0);
        Some((ttft_ms, mean_itl_ms))
    }

    pub(crate) fn finish(self) -> TraceSimulationReport {
        // Build per-request records before we move `self.requests` into the
        // summary aggregation below. Gated on `capture_per_request` — the
        // ~100ms terminal pass + ~30MB allocation only runs when a caller
        // (e.g. CLI `--report-jsonl`) asked for it. The summary report is
        // unaffected either way (custom Serialize impl skips `per_request`).
        let per_request = if self.capture_per_request {
            self.per_request_records()
        } else {
            Vec::new()
        };
        let requests = self.requests;
        let request_count = requests.len();
        let mut ttfts = Vec::with_capacity(request_count);
        let mut ttsts = Vec::with_capacity(request_count);
        let mut tpots = Vec::with_capacity(request_count);
        let mut itls = Vec::new();
        let mut e2e_latencies = Vec::with_capacity(request_count);
        let mut output_token_throughput_per_user = Vec::new();
        let mut duration_ms = 0.0_f64;
        let mut total_input_tokens = 0usize;
        let mut total_output_tokens = 0usize;
        let mut completed_requests = 0usize;
        let mut total_reused_tokens = 0usize;
        let mut total_first_admission_reused_tokens = 0usize;

        for stats in requests.values() {
            if stats.first_admit_ms.is_none() {
                continue;
            }
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
            total_first_admission_reused_tokens += stats.first_admission_reused_input_tokens;
            duration_ms = duration_ms.max(last_token_ms);

            let ttft_ms = (first_token_ms - stats.arrival_time_ms).max(0.0);
            let e2e_ms = (last_token_ms - stats.arrival_time_ms).max(0.0);
            ttfts.push(ttft_ms);
            e2e_latencies.push(e2e_ms);

            if let Some(ttst_ms) = stats.ttst_ms() {
                ttsts.push(ttst_ms);
            }

            if let Some(tpot_ms) = stats.mean_tpot_ms() {
                tpots.push(tpot_ms);
                for itl_ms in stats.itls_ms() {
                    if itl_ms > 0.0 {
                        output_token_throughput_per_user.push(1000.0 / itl_ms);
                    }
                    itls.push(itl_ms);
                }
            }
        }

        let duration_s = (duration_ms / 1000.0).max(1e-9);
        let itl_distribution = build_distribution_stats(itls);
        TraceSimulationReport {
            request_counts: TraceRequestCounts {
                num_requests: request_count,
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
            first_admission_prefix_cache_reused_ratio: if total_input_tokens == 0 {
                0.0
            } else {
                total_first_admission_reused_tokens as f64 / total_input_tokens as f64
            },
            latency: TraceLatencyStats {
                ttft: build_distribution_stats(ttfts),
                ttst: build_distribution_stats(ttsts),
                tpot: build_distribution_stats(tpots),
                itl: TraceInterTokenLatencyStats {
                    max_ms: itl_distribution.max_ms,
                    distribution: itl_distribution,
                },
                e2e: build_distribution_stats(e2e_latencies),
                output_token_throughput_per_user: build_distribution_stats(
                    output_token_throughput_per_user,
                ),
            },
            per_request,
        }
    }

    /// Flatten each retained request into a serializable `PerRequestRecord`.
    /// Used by the `--report-jsonl` CLI path to emit one JSON object per
    /// request to the JSONL file, mirroring AIPerf's per-request output shape.
    ///
    /// Only fully-completed requests (admitted, first token observed, last
    /// token observed) are emitted, so the JSONL row count matches the
    /// completed-request count in the aggregate report. Incomplete requests
    /// (e.g. truncated by a sim-time cap) appear in the summary's incomplete
    /// counters but not here.
    pub fn per_request_records(&self) -> Vec<PerRequestRecord> {
        let mut records = Vec::with_capacity(self.requests.len());
        for (uuid, stats) in &self.requests {
            let Some(first_admit_ms) = stats.first_admit_ms else {
                continue;
            };
            let Some(first_token_ms) = stats.first_token_ms() else {
                continue;
            };
            let Some(last_token_ms) = stats.last_token_ms() else {
                continue;
            };
            let ttft_ms = (first_token_ms - stats.arrival_time_ms).max(0.0);
            let e2e_latency_ms = (last_token_ms - stats.arrival_time_ms).max(0.0);
            records.push(PerRequestRecord {
                session_id: stats.session_id.clone(),
                turn_index: stats.turn_index,
                uuid: uuid.to_string(),
                arrival_time_ms: stats.arrival_time_ms,
                first_admit_ms: Some(first_admit_ms),
                first_token_ms: Some(first_token_ms),
                last_token_ms: Some(last_token_ms),
                ttft_ms: Some(ttft_ms),
                ttst_ms: stats.ttst_ms(),
                e2e_latency_ms: Some(e2e_latency_ms),
                itl_ms: stats.mean_tpot_ms(),
                input_length: stats.input_length,
                output_length: stats.output_length,
                reused_input_tokens: stats.reused_input_tokens,
                prefill_worker_idx: stats.prefill_worker_idx,
                decode_worker_idx: stats.decode_worker_idx,
            });
        }
        // Stable ordering: by arrival_time_ms (with uuid as tiebreaker) so the
        // JSONL file is reproducible across runs and matches the order
        // analysis tools usually expect.
        records.sort_by(|a, b| {
            a.arrival_time_ms
                .total_cmp(&b.arrival_time_ms)
                .then_with(|| a.uuid.cmp(&b.uuid))
        });
        records
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self, uuid: Uuid) -> Option<TraceRequestStatsSnapshot> {
        self.requests
            .get(&uuid)
            .map(|stats| TraceRequestStatsSnapshot {
                arrival_time_ms: stats.arrival_time_ms,
                first_admit_ms: stats.first_admit_ms,
                first_token_ms: stats.first_token_ms(),
                last_token_ms: stats.last_token_ms(),
                input_length: stats.input_length,
                output_length: stats.output_length,
                reused_input_tokens: stats.reused_input_tokens,
                first_admission_reused_input_tokens: stats.first_admission_reused_input_tokens,
            })
    }

    #[cfg(test)]
    pub(crate) fn snapshots(&self) -> Vec<TraceRequestStatsSnapshot> {
        self.requests
            .values()
            .map(|stats| TraceRequestStatsSnapshot {
                arrival_time_ms: stats.arrival_time_ms,
                first_admit_ms: stats.first_admit_ms,
                first_token_ms: stats.first_token_ms(),
                last_token_ms: stats.last_token_ms(),
                input_length: stats.input_length,
                output_length: stats.output_length,
                reused_input_tokens: stats.reused_input_tokens,
                first_admission_reused_input_tokens: stats.first_admission_reused_input_tokens,
            })
            .collect()
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn build_distribution_stats(mut values: Vec<f64>) -> TraceDistributionStats {
    if values.is_empty() {
        return TraceDistributionStats {
            mean_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            median_ms: 0.0,
            p75_ms: 0.0,
            p90_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            std_ms: 0.0,
        };
    }

    let min_ms = values
        .iter()
        .copied()
        .min_by(|left, right| left.total_cmp(right))
        .expect("non-empty values must have a minimum");
    let max_ms = values
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .expect("non-empty values must have a maximum");

    TraceDistributionStats {
        mean_ms: mean(&values),
        min_ms,
        max_ms,
        median_ms: percentile_in_place(&mut values, 50.0),
        p75_ms: percentile_in_place(&mut values, 75.0),
        p90_ms: percentile_in_place(&mut values, 90.0),
        p95_ms: percentile_in_place(&mut values, 95.0),
        p99_ms: percentile_in_place(&mut values, 99.0),
        std_ms: std_dev(&values),
    }
}

fn percentile_in_place(values: &mut [f64], percentile: f64) -> f64 {
    let rank = percentile_rank(values.len(), percentile);
    let (_, selected, _) = values.select_nth_unstable_by(rank, |left, right| left.total_cmp(right));
    *selected
}

fn percentile_rank(len: usize, percentile: f64) -> usize {
    let rank = ((len - 1) as f64 * percentile / 100.0).round() as usize;
    rank.min(len - 1)
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_distribution_stats_sorted(values: &[f64]) -> TraceDistributionStats {
        if values.is_empty() {
            return TraceDistributionStats {
                mean_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                median_ms: 0.0,
                p75_ms: 0.0,
                p90_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                std_ms: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        TraceDistributionStats {
            mean_ms: mean(values),
            min_ms: sorted[0],
            max_ms: *sorted.last().expect("sorted values must be non-empty"),
            median_ms: sorted[percentile_rank(sorted.len(), 50.0)],
            p75_ms: sorted[percentile_rank(sorted.len(), 75.0)],
            p90_ms: sorted[percentile_rank(sorted.len(), 90.0)],
            p95_ms: sorted[percentile_rank(sorted.len(), 95.0)],
            p99_ms: sorted[percentile_rank(sorted.len(), 99.0)],
            std_ms: std_dev(values),
        }
    }

    #[test]
    fn build_distribution_stats_matches_sorted_baseline() {
        let values = vec![
            0.0, 1.0, 1.0, 2.5, 4.0, 4.0, 7.25, 9.5, 15.0, 22.0, 22.0, 100.0,
        ];

        let expected = build_distribution_stats_sorted(&values);
        let actual = build_distribution_stats(values);

        assert_eq!(actual.mean_ms, expected.mean_ms);
        assert_eq!(actual.min_ms, expected.min_ms);
        assert_eq!(actual.max_ms, expected.max_ms);
        assert_eq!(actual.median_ms, expected.median_ms);
        assert_eq!(actual.p75_ms, expected.p75_ms);
        assert_eq!(actual.p90_ms, expected.p90_ms);
        assert_eq!(actual.p95_ms, expected.p95_ms);
        assert_eq!(actual.p99_ms, expected.p99_ms);
        assert_eq!(actual.std_ms, expected.std_ms);
    }

    /// With per-request capture on, a standard disagg-style request lifecycle
    /// (arrival → admit → prefill_assigned → decode_assigned → tokens) yields
    /// exactly one record with all fields populated correctly.
    #[test]
    fn per_request_disagg_record_populates_all_fields() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        let uuid = Uuid::from_u128(1);
        collector.on_arrival(uuid, 0.0, 100, 4);
        collector.on_admit(uuid, 5.0, 30);
        collector.on_prefill_assigned(uuid, 2);
        collector.on_decode_assigned(uuid, 7);
        collector.on_token(uuid, 50.0);
        collector.on_token(uuid, 60.0);
        collector.on_token(uuid, 75.0);
        collector.on_token(uuid, 95.0);

        let report = collector.finish();
        assert_eq!(report.per_request.len(), 1);
        let rec = &report.per_request[0];
        assert_eq!(rec.uuid, uuid.to_string());
        assert_eq!(rec.arrival_time_ms, 0.0);
        assert_eq!(rec.first_admit_ms, Some(5.0));
        assert_eq!(rec.first_token_ms, Some(50.0));
        assert_eq!(rec.last_token_ms, Some(95.0));
        assert_eq!(rec.ttft_ms, Some(50.0));
        assert_eq!(rec.ttst_ms, Some(10.0));
        assert_eq!(rec.e2e_latency_ms, Some(95.0));
        // Mean per-token gap across 4 tokens: (10 + 15 + 20) / 3 = 15.0
        assert_eq!(rec.itl_ms, Some(15.0));
        assert_eq!(rec.input_length, 100);
        assert_eq!(rec.output_length, 4);
        assert_eq!(rec.reused_input_tokens, 30);
        assert_eq!(rec.prefill_worker_idx, Some(2));
        assert_eq!(rec.decode_worker_idx, Some(7));
    }

    /// A conditional-prefill bypass is reflected by `prefill_worker_idx ==
    /// None` while `decode_worker_idx` is set. This is how downstream tooling
    /// distinguishes bypassed requests from standard disagg flow.
    #[test]
    fn per_request_bypass_leaves_prefill_worker_idx_none() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        let uuid = Uuid::from_u128(42);
        collector.on_arrival(uuid, 0.0, 100, 2);
        collector.on_admit(uuid, 5.0, 0);
        // No on_prefill_assigned call — request bypassed remote prefill.
        collector.on_decode_assigned(uuid, 1);
        collector.on_token(uuid, 30.0);
        collector.on_token(uuid, 45.0);

        let report = collector.finish();
        assert_eq!(report.per_request.len(), 1);
        let rec = &report.per_request[0];
        assert!(
            rec.prefill_worker_idx.is_none(),
            "bypassed request must have prefill_worker_idx = None"
        );
        assert_eq!(rec.decode_worker_idx, Some(1));
    }

    /// Default: capture is off, so `per_request` is empty and the ~100ms
    /// terminal pass is skipped. The summary report is otherwise identical.
    #[test]
    fn per_request_default_off() {
        let mut collector = TraceCollector::default();
        // Note: NOT calling set_capture_per_request — capture stays false.
        let uuid = Uuid::from_u128(1);
        collector.on_arrival(uuid, 0.0, 100, 2);
        collector.on_admit(uuid, 5.0, 0);
        collector.on_decode_assigned(uuid, 0);
        collector.on_token(uuid, 50.0);
        collector.on_token(uuid, 60.0);

        let report = collector.finish();
        assert!(report.per_request.is_empty());
        // Summary stats still work.
        assert_eq!(report.request_counts.completed_requests, 1);
    }

    /// Records emerge in arrival-time order, so the JSONL file produced from
    /// them is deterministic across runs (important for diff-friendly CI).
    #[test]
    fn per_request_records_are_sorted_by_arrival_time() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        // Insert out of order on purpose.
        for (uuid_n, arrival) in [(3u128, 30.0), (1, 0.0), (2, 10.0)] {
            let uuid = Uuid::from_u128(uuid_n);
            collector.on_arrival(uuid, arrival, 100, 1);
            collector.on_admit(uuid, arrival + 1.0, 0);
            collector.on_decode_assigned(uuid, 0);
            collector.on_token(uuid, arrival + 5.0);
        }
        let report = collector.finish();
        let arrivals: Vec<f64> = report
            .per_request
            .iter()
            .map(|r| r.arrival_time_ms)
            .collect();
        assert_eq!(arrivals, vec![0.0, 10.0, 30.0]);
    }

    /// Each record must round-trip cleanly to JSON — this is the format we
    /// emit to `--report-jsonl`. Guards against accidental serde regressions
    /// (e.g., adding a non-serializable field to `PerRequestRecord`).
    #[test]
    fn per_request_record_serializes_to_json_object() {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        let uuid = Uuid::from_u128(123);
        collector.on_arrival(uuid, 0.0, 50, 2);
        collector.on_admit(uuid, 1.0, 10);
        collector.on_prefill_assigned(uuid, 0);
        collector.on_decode_assigned(uuid, 1);
        collector.on_token(uuid, 20.0);
        collector.on_token(uuid, 25.0);

        let report = collector.finish();
        let line = serde_json::to_string(&report.per_request[0])
            .expect("PerRequestRecord must serialize cleanly");
        // Parse it back and spot-check a few keys to confirm shape.
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("emitted JSON must parse");
        assert!(parsed.is_object());
        assert_eq!(parsed["uuid"], uuid.to_string());
        assert_eq!(parsed["input_length"], 50);
        assert_eq!(parsed["output_length"], 2);
        assert_eq!(parsed["prefill_worker_idx"], 0);
        assert_eq!(parsed["decode_worker_idx"], 1);
        assert!(parsed["itl_ms"].is_number());
    #[test]
    fn first_admission_reuse_ignores_later_readmission_self_reuse() {
        let uuid = Uuid::from_u128(1);
        let mut collector = TraceCollector::default();
        collector.on_arrival(uuid, 0.0, 100, 1);
        collector.on_admit(uuid, 1.0, 0);
        collector.on_admit(uuid, 2.0, 80);
        collector.on_token(uuid, 3.0);

        let report = collector.finish();

        assert_eq!(report.prefix_cache_reused_ratio, 0.8);
        assert_eq!(report.first_admission_prefix_cache_reused_ratio, 0.0);
    }
}
