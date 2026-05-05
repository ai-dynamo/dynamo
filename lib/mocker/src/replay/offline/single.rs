// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::core::ReplayWorkerCore;
use super::progress::ReplayProgress;
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::WorkloadDriver;
use crate::replay::TraceCollector;
use anyhow::bail;
use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub(super) enum SingleReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

pub(super) struct SingleRuntime {
    current_time_ms: f64,
    admission: AdmissionSource,
    worker: ReplayWorkerCore,
    collector: TraceCollector,
    mode: SingleReplayMode,
    progress: ReplayProgress,
}

impl SingleRuntime {
    pub(super) fn new(
        args: MockEngineArgs,
        pending: VecDeque<DirectRequest>,
        mode: SingleReplayMode,
    ) -> Self {
        Self::new_with_source(args, AdmissionSource::Requests(pending), mode)
    }

    pub(super) fn new_workload(
        args: MockEngineArgs,
        driver: WorkloadDriver,
        mode: SingleReplayMode,
    ) -> Self {
        Self::new_with_source(args, AdmissionSource::Workload(driver), mode)
    }

    fn new_with_source(
        args: MockEngineArgs,
        admission: AdmissionSource,
        mode: SingleReplayMode,
    ) -> Self {
        let total_requests = match &admission {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        };
        Self {
            current_time_ms: 0.0,
            admission,
            worker: ReplayWorkerCore::new(args),
            collector: TraceCollector::default(),
            mode,
            progress: ReplayProgress::new(total_requests, "offline replay"),
        }
    }

    fn enqueue_trace_arrivals(&mut self) {
        let mut ready_requests = Vec::new();
        match &mut self.admission {
            AdmissionSource::Requests(pending) => loop {
                let Some(next_arrival_ms) = pending
                    .front()
                    .and_then(|request| request.arrival_timestamp_ms)
                else {
                    break;
                };
                if next_arrival_ms > self.current_time_ms {
                    break;
                }

                let request = pending
                    .pop_front()
                    .expect("front request must exist when arrival is available");
                let arrival_ms = request
                    .arrival_timestamp_ms
                    .expect("trace replay requests must have an arrival timestamp");
                ready_requests.push((request, arrival_ms));
            },
            AdmissionSource::Workload(driver) => {
                ready_requests.extend(
                    driver
                        .pop_ready(self.current_time_ms, usize::MAX)
                        .into_iter()
                        .map(|ready| (ready.request, ready.scheduled_ready_at_ms)),
                );
            }
        }

        for (request, arrival_ms) in ready_requests {
            self.record_arrival(request, arrival_ms);
        }
    }

    fn enqueue_concurrency_arrivals(&mut self, max_in_flight: usize) {
        let available = max_in_flight.saturating_sub(self.worker.num_requests());
        let mut ready_requests = Vec::new();

        match &mut self.admission {
            AdmissionSource::Requests(pending) => {
                for _ in 0..available {
                    let Some(mut request) = pending.pop_front() else {
                        break;
                    };
                    request.arrival_timestamp_ms = Some(self.current_time_ms);
                    ready_requests.push(request);
                }
            }
            AdmissionSource::Workload(driver) => {
                ready_requests.extend(
                    driver
                        .pop_ready(self.current_time_ms, available)
                        .into_iter()
                        .map(|ready| ready.request),
                );
            }
        }

        for request in ready_requests {
            self.record_arrival(request, self.current_time_ms);
        }
    }

    fn record_arrival(&mut self, request: DirectRequest, arrival_ms: f64) -> Uuid {
        let input_length = request.tokens.len();
        let output_length = request.max_output_tokens;
        let uuid = self.worker.receive(request);
        self.collector
            .on_arrival(uuid, arrival_ms, input_length, output_length);
        uuid
    }

    fn is_done(&self) -> bool {
        self.worker.is_empty()
            && match &self.admission {
                AdmissionSource::Requests(pending) => pending.is_empty(),
                AdmissionSource::Workload(driver) => driver.is_drained(),
            }
    }

    fn advance_to_next_trace_arrival(&mut self) -> anyhow::Result<()> {
        let next_arrival_ms = match &mut self.admission {
            AdmissionSource::Requests(pending) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            AdmissionSource::Workload(driver) => driver.next_ready_time_ms(),
        };
        let Some(next_arrival_ms) = next_arrival_ms else {
            bail!("trace replay reached an idle state without a pending arrival");
        };
        self.current_time_ms = next_arrival_ms;
        Ok(())
    }

    fn drive_worker(&mut self, admit_arrivals_between_steps: bool) {
        let pass = self
            .worker
            .execute_pass(&mut self.collector, self.current_time_ms);
        self.current_time_ms = pass.end_ms;
        if let AdmissionSource::Workload(driver) = &mut self.admission {
            for signal in pass.output_signals.iter().filter(|signal| signal.completed) {
                driver
                    .on_complete(signal.uuid, self.current_time_ms)
                    .expect("completed workload request must belong to a session");
            }
        }
        let completed_requests = pass
            .output_signals
            .iter()
            .filter(|signal| signal.completed)
            .count();
        for _ in 0..completed_requests {
            self.progress.inc_completed();
        }
        if admit_arrivals_between_steps {
            self.enqueue_trace_arrivals();
        }
    }

    pub(super) fn run(mut self) -> anyhow::Result<TraceCollector> {
        while !self.is_done() {
            match self.mode {
                SingleReplayMode::Trace => {
                    self.enqueue_trace_arrivals();
                    if self.worker.is_empty() {
                        self.advance_to_next_trace_arrival()?;
                        self.enqueue_trace_arrivals();
                        continue;
                    }
                    self.drive_worker(true);
                }
                SingleReplayMode::Concurrency { max_in_flight } => {
                    self.enqueue_concurrency_arrivals(max_in_flight);
                    if self.worker.is_empty() {
                        if self.is_done() {
                            break;
                        }
                        self.advance_to_next_trace_arrival()?;
                        continue;
                    }
                    self.drive_worker(false);
                }
            }
        }

        self.progress.finish();
        Ok(self.collector)
    }
}

#[cfg(test)]
mod tests {
    use super::super::entrypoints::{
        run_concurrency_workload_single_collect, run_trace_workload_single_collect,
        simulate_concurrency_single, simulate_trace_single,
    };
    use super::*;
    use crate::loadgen::{SessionTrace, Trace, TurnTrace};
    use crate::replay::{TraceRequestStatsSnapshot, TraceSimulationReport};
    use rstest::rstest;
    use std::collections::{HashMap, VecDeque};
    use uuid::Uuid;

    #[derive(Debug)]
    struct ManualReplayResult {
        report: TraceSimulationReport,
        snapshots: HashMap<Uuid, TraceRequestStatsSnapshot>,
        idle_jump_ms: f64,
        first_decode_end_ms: f64,
    }

    #[derive(Debug)]
    struct ManualConcurrencyResult {
        report: TraceSimulationReport,
        snapshots: HashMap<Uuid, TraceRequestStatsSnapshot>,
    }

    fn enqueue_trace_arrivals_manual(
        pending: &mut VecDeque<DirectRequest>,
        worker: &mut ReplayWorkerCore,
        collector: &mut TraceCollector,
        current_time_ms: f64,
    ) {
        loop {
            let Some(next_arrival_ms) = pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms)
            else {
                break;
            };
            if next_arrival_ms > current_time_ms {
                break;
            }

            let request = pending
                .pop_front()
                .expect("front request must exist when arrival is available");
            let arrival_ms = request
                .arrival_timestamp_ms
                .expect("trace replay requests must have an arrival timestamp");
            let input_length = request.tokens.len();
            let output_length = request.max_output_tokens;
            let uuid = worker.receive(request);
            collector.on_arrival(uuid, arrival_ms, input_length, output_length);
        }
    }

    fn enqueue_concurrency_arrivals_manual(
        pending: &mut VecDeque<DirectRequest>,
        worker: &mut ReplayWorkerCore,
        collector: &mut TraceCollector,
        current_time_ms: f64,
        max_in_flight: usize,
    ) {
        while worker.num_requests() < max_in_flight {
            let Some(mut request) = pending.pop_front() else {
                break;
            };

            request.arrival_timestamp_ms = Some(current_time_ms);
            let input_length = request.tokens.len();
            let output_length = request.max_output_tokens;
            let uuid = worker.receive(request);
            collector.on_arrival(uuid, current_time_ms, input_length, output_length);
        }
    }

    fn replay_args(enable_prefix_caching: bool, enable_chunked_prefill: bool) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn replay_fixture() -> Vec<DirectRequest> {
        vec![
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
            },
            DirectRequest {
                tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(101.0),
            },
            DirectRequest {
                tokens: vec![9, 9, 9, 9, 8, 8, 8, 8],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(500.0),
            },
        ]
    }

    fn multiturn_trace_fixture() -> Trace {
        Trace {
            block_size: 1,
            sessions: vec![
                SessionTrace {
                    session_id: "session-a".to_string(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![
                        TurnTrace {
                            input_length: 3,
                            max_output_tokens: 2,
                            hash_ids: vec![1, 2, 3],
                            delay_after_previous_ms: 0.0,
                        },
                        TurnTrace {
                            input_length: 5,
                            max_output_tokens: 2,
                            hash_ids: vec![4, 5, 6, 7, 8],
                            delay_after_previous_ms: 5.0,
                        },
                    ],
                },
                SessionTrace {
                    session_id: "session-b".to_string(),
                    first_arrival_timestamp_ms: Some(1.0),
                    turns: vec![TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![9, 10, 11, 12],
                        delay_after_previous_ms: 0.0,
                    }],
                },
            ],
        }
    }

    fn run_trace_manually(
        args: &MockEngineArgs,
        requests: Vec<DirectRequest>,
    ) -> ManualReplayResult {
        let mut requests = requests;
        requests.sort_by(|left, right| {
            let left_ts = left.arrival_timestamp_ms.unwrap();
            let right_ts = right.arrival_timestamp_ms.unwrap();
            left_ts.total_cmp(&right_ts)
        });

        let first_arrival_ms = requests.first().unwrap().arrival_timestamp_ms.unwrap();
        let mut pending = VecDeque::from(
            requests
                .into_iter()
                .map(|mut request| {
                    request.arrival_timestamp_ms =
                        Some(request.arrival_timestamp_ms.unwrap() - first_arrival_ms);
                    request
                })
                .collect::<Vec<_>>(),
        );

        let mut worker = ReplayWorkerCore::new(args.clone());
        let mut collector = TraceCollector::default();
        let mut current_time_ms = 0.0;
        let mut idle_jump_ms = 0.0;
        let mut first_decode_end_ms = 0.0;

        while !pending.is_empty() || !worker.is_empty() {
            enqueue_trace_arrivals_manual(
                &mut pending,
                &mut worker,
                &mut collector,
                current_time_ms,
            );

            if worker.is_empty() {
                let next_arrival_ms = pending.front().unwrap().arrival_timestamp_ms.unwrap();
                current_time_ms = next_arrival_ms;
                if idle_jump_ms == 0.0 && current_time_ms > 0.0 {
                    idle_jump_ms = current_time_ms;
                }
                enqueue_trace_arrivals_manual(
                    &mut pending,
                    &mut worker,
                    &mut collector,
                    current_time_ms,
                );
                continue;
            }

            let pass = worker.execute_pass(&mut collector, current_time_ms);
            if first_decode_end_ms == 0.0 && !pass.output_signals.is_empty() {
                first_decode_end_ms = pass.end_ms;
            }
            current_time_ms = pass.end_ms;
            enqueue_trace_arrivals_manual(
                &mut pending,
                &mut worker,
                &mut collector,
                current_time_ms,
            );
        }

        let snapshots = [
            Uuid::from_u128(11),
            Uuid::from_u128(22),
            Uuid::from_u128(33),
        ]
        .into_iter()
        .map(|uuid| (uuid, collector.snapshot(uuid).unwrap()))
        .collect();

        ManualReplayResult {
            report: collector.finish(),
            snapshots,
            idle_jump_ms,
            first_decode_end_ms,
        }
    }

    fn run_concurrency_manually(
        args: &MockEngineArgs,
        requests: Vec<DirectRequest>,
        max_in_flight: usize,
    ) -> ManualConcurrencyResult {
        let mut pending = VecDeque::from(requests);
        let mut worker = ReplayWorkerCore::new(args.clone());
        let mut collector = TraceCollector::default();
        let mut current_time_ms = 0.0;

        while !pending.is_empty() || !worker.is_empty() {
            enqueue_concurrency_arrivals_manual(
                &mut pending,
                &mut worker,
                &mut collector,
                current_time_ms,
                max_in_flight,
            );

            if worker.is_empty() {
                break;
            }

            let pass = worker.execute_pass(&mut collector, current_time_ms);
            current_time_ms = pass.end_ms;
        }

        let snapshots = [
            Uuid::from_u128(11),
            Uuid::from_u128(22),
            Uuid::from_u128(33),
        ]
        .into_iter()
        .map(|uuid| (uuid, collector.snapshot(uuid).unwrap()))
        .collect();

        ManualConcurrencyResult {
            report: collector.finish(),
            snapshots,
        }
    }

    fn assert_report_close(left: &TraceSimulationReport, right: &TraceSimulationReport) {
        let epsilon = 1e-9;
        assert_eq!(
            left.request_counts.num_requests,
            right.request_counts.num_requests
        );
        assert_eq!(
            left.request_counts.completed_requests,
            right.request_counts.completed_requests
        );
        assert_eq!(
            left.request_counts.total_input_tokens,
            right.request_counts.total_input_tokens
        );
        assert_eq!(
            left.request_counts.total_output_tokens,
            right.request_counts.total_output_tokens
        );
        assert!((left.throughput.duration_ms - right.throughput.duration_ms).abs() <= epsilon);
        assert!(
            (left.throughput.request_throughput_rps - right.throughput.request_throughput_rps)
                .abs()
                <= epsilon
        );
        assert!(
            (left.throughput.input_throughput_tok_s - right.throughput.input_throughput_tok_s)
                .abs()
                <= epsilon
        );
        assert!(
            (left.throughput.output_throughput_tok_s - right.throughput.output_throughput_tok_s)
                .abs()
                <= epsilon
        );
        assert!(
            (left.throughput.total_throughput_tok_s - right.throughput.total_throughput_tok_s)
                .abs()
                <= epsilon
        );
        assert!(
            (left.prefix_cache_reused_ratio - right.prefix_cache_reused_ratio).abs() <= epsilon
        );
        assert!((left.latency.ttft.mean_ms - right.latency.ttft.mean_ms).abs() <= epsilon);
        assert!((left.latency.ttft.min_ms - right.latency.ttft.min_ms).abs() <= epsilon);
        assert!((left.latency.ttft.max_ms - right.latency.ttft.max_ms).abs() <= epsilon);
        assert!((left.latency.ttft.median_ms - right.latency.ttft.median_ms).abs() <= epsilon);
        assert!((left.latency.ttft.p75_ms - right.latency.ttft.p75_ms).abs() <= epsilon);
        assert!((left.latency.ttft.p90_ms - right.latency.ttft.p90_ms).abs() <= epsilon);
        assert!((left.latency.ttft.p95_ms - right.latency.ttft.p95_ms).abs() <= epsilon);
        assert!((left.latency.ttft.p99_ms - right.latency.ttft.p99_ms).abs() <= epsilon);
        assert!((left.latency.ttft.std_ms - right.latency.ttft.std_ms).abs() <= epsilon);
        assert!((left.latency.ttst.mean_ms - right.latency.ttst.mean_ms).abs() <= epsilon);
        assert!((left.latency.ttst.min_ms - right.latency.ttst.min_ms).abs() <= epsilon);
        assert!((left.latency.ttst.max_ms - right.latency.ttst.max_ms).abs() <= epsilon);
        assert!((left.latency.ttst.median_ms - right.latency.ttst.median_ms).abs() <= epsilon);
        assert!((left.latency.ttst.p75_ms - right.latency.ttst.p75_ms).abs() <= epsilon);
        assert!((left.latency.ttst.p90_ms - right.latency.ttst.p90_ms).abs() <= epsilon);
        assert!((left.latency.ttst.p95_ms - right.latency.ttst.p95_ms).abs() <= epsilon);
        assert!((left.latency.ttst.p99_ms - right.latency.ttst.p99_ms).abs() <= epsilon);
        assert!((left.latency.ttst.std_ms - right.latency.ttst.std_ms).abs() <= epsilon);
        assert!((left.latency.tpot.mean_ms - right.latency.tpot.mean_ms).abs() <= epsilon);
        assert!((left.latency.tpot.min_ms - right.latency.tpot.min_ms).abs() <= epsilon);
        assert!((left.latency.tpot.max_ms - right.latency.tpot.max_ms).abs() <= epsilon);
        assert!((left.latency.tpot.median_ms - right.latency.tpot.median_ms).abs() <= epsilon);
        assert!((left.latency.tpot.p75_ms - right.latency.tpot.p75_ms).abs() <= epsilon);
        assert!((left.latency.tpot.p90_ms - right.latency.tpot.p90_ms).abs() <= epsilon);
        assert!((left.latency.tpot.p95_ms - right.latency.tpot.p95_ms).abs() <= epsilon);
        assert!((left.latency.tpot.p99_ms - right.latency.tpot.p99_ms).abs() <= epsilon);
        assert!((left.latency.tpot.std_ms - right.latency.tpot.std_ms).abs() <= epsilon);
        assert!(
            (left.latency.itl.distribution.mean_ms - right.latency.itl.distribution.mean_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.min_ms - right.latency.itl.distribution.min_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.max_ms - right.latency.itl.distribution.max_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.median_ms - right.latency.itl.distribution.median_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.p75_ms - right.latency.itl.distribution.p75_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.p90_ms - right.latency.itl.distribution.p90_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.p95_ms - right.latency.itl.distribution.p95_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.p99_ms - right.latency.itl.distribution.p99_ms).abs()
                <= epsilon
        );
        assert!(
            (left.latency.itl.distribution.std_ms - right.latency.itl.distribution.std_ms).abs()
                <= epsilon
        );
        assert!((left.latency.itl.max_ms - right.latency.itl.max_ms).abs() <= epsilon);
        assert!((left.latency.e2e.mean_ms - right.latency.e2e.mean_ms).abs() <= epsilon);
        assert!((left.latency.e2e.min_ms - right.latency.e2e.min_ms).abs() <= epsilon);
        assert!((left.latency.e2e.max_ms - right.latency.e2e.max_ms).abs() <= epsilon);
        assert!((left.latency.e2e.median_ms - right.latency.e2e.median_ms).abs() <= epsilon);
        assert!((left.latency.e2e.p75_ms - right.latency.e2e.p75_ms).abs() <= epsilon);
        assert!((left.latency.e2e.p90_ms - right.latency.e2e.p90_ms).abs() <= epsilon);
        assert!((left.latency.e2e.p95_ms - right.latency.e2e.p95_ms).abs() <= epsilon);
        assert!((left.latency.e2e.p99_ms - right.latency.e2e.p99_ms).abs() <= epsilon);
        assert!((left.latency.e2e.std_ms - right.latency.e2e.std_ms).abs() <= epsilon);
        assert!(
            (left.latency.output_token_throughput_per_user.mean_ms
                - right.latency.output_token_throughput_per_user.mean_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.min_ms
                - right.latency.output_token_throughput_per_user.min_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.max_ms
                - right.latency.output_token_throughput_per_user.max_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.median_ms
                - right.latency.output_token_throughput_per_user.median_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.p75_ms
                - right.latency.output_token_throughput_per_user.p75_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.p90_ms
                - right.latency.output_token_throughput_per_user.p90_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.p95_ms
                - right.latency.output_token_throughput_per_user.p95_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.p99_ms
                - right.latency.output_token_throughput_per_user.p99_ms)
                .abs()
                <= epsilon
        );
        assert!(
            (left.latency.output_token_throughput_per_user.std_ms
                - right.latency.output_token_throughput_per_user.std_ms)
                .abs()
                <= epsilon
        );
    }

    #[rstest]
    #[case(false, false)]
    #[case(false, true)]
    #[case(true, false)]
    #[case(true, true)]
    fn test_trace_replay_matches_manual_steps(
        #[case] enable_prefix_caching: bool,
        #[case] enable_chunked_prefill: bool,
    ) {
        let args = replay_args(enable_prefix_caching, enable_chunked_prefill);
        let manual = run_trace_manually(&args, replay_fixture());
        let replay_report = simulate_trace_single(args, replay_fixture(), 1.0).unwrap();

        let request_1 = manual.snapshots.get(&Uuid::from_u128(11)).unwrap();
        let request_2 = manual.snapshots.get(&Uuid::from_u128(22)).unwrap();
        let request_3 = manual.snapshots.get(&Uuid::from_u128(33)).unwrap();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 1.0);
        assert_eq!(request_3.arrival_time_ms, 400.0);
        assert_eq!(manual.idle_jump_ms, 400.0);
        assert_eq!(
            request_1.first_token_ms.unwrap(),
            manual.first_decode_end_ms
        );
        assert!(request_2.first_admit_ms.unwrap() >= request_2.arrival_time_ms);
        assert!(request_3.first_admit_ms.unwrap() >= request_3.arrival_time_ms);
        assert!(manual.report.latency.e2e.mean_ms >= manual.report.latency.ttft.mean_ms);

        if enable_prefix_caching {
            assert!(request_2.reused_input_tokens > 0);
            assert!(manual.report.prefix_cache_reused_ratio > 0.0);
        } else {
            assert_eq!(request_2.reused_input_tokens, 0);
            assert_eq!(manual.report.prefix_cache_reused_ratio, 0.0);
        }

        assert_report_close(&replay_report, &manual.report);
    }

    #[test]
    fn test_concurrency_replay_matches_manual_steps() {
        let args = replay_args(false, false);
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(11)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(900.0),
            },
            DirectRequest {
                tokens: vec![1, 2, 3, 4, 5, 9, 10, 11],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(22)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(1000.0),
            },
            DirectRequest {
                tokens: vec![12, 13, 14, 15, 16, 17, 18, 19],
                max_output_tokens: 2,
                uuid: Some(Uuid::from_u128(33)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
            },
        ];
        let manual = run_concurrency_manually(&args, requests.clone(), 2);
        let replay_report = simulate_concurrency_single(args, requests, 2).unwrap();

        let request_1 = manual.snapshots.get(&Uuid::from_u128(11)).unwrap();
        let request_2 = manual.snapshots.get(&Uuid::from_u128(22)).unwrap();
        let request_3 = manual.snapshots.get(&Uuid::from_u128(33)).unwrap();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, request_1.last_token_ms.unwrap());
        assert!(request_3.arrival_time_ms < request_2.last_token_ms.unwrap());
        assert_eq!(manual.report.request_counts.completed_requests, 3);
        assert_eq!(manual.report.request_counts.total_input_tokens, 24);
        assert_eq!(manual.report.request_counts.total_output_tokens, 6);

        assert_report_close(&replay_report, &manual.report);
    }

    #[test]
    fn test_trace_workload_single_unlocks_follow_up_turn_after_completion() {
        let args = replay_args(false, true);
        let collector = run_trace_workload_single_collect(args, multiturn_trace_fixture());
        let snapshots = collector.snapshots();

        let first = snapshots
            .iter()
            .find(|stats| stats.input_length == 3)
            .unwrap();
        let second = snapshots
            .iter()
            .find(|stats| stats.input_length == 5)
            .unwrap();
        let other = snapshots
            .iter()
            .find(|stats| stats.input_length == 4)
            .unwrap();

        assert_eq!(first.arrival_time_ms, 0.0);
        assert_eq!(other.arrival_time_ms, 1.0);
        assert!(second.arrival_time_ms >= first.last_token_ms.unwrap() + 5.0);
    }

    #[test]
    fn test_concurrency_workload_single_ignores_first_turn_timestamps_but_keeps_delay() {
        let args = replay_args(false, true);
        let collector = run_concurrency_workload_single_collect(args, multiturn_trace_fixture(), 1);
        let arrival_times = collector
            .snapshots()
            .into_iter()
            .map(|stats| stats.arrival_time_ms)
            .collect::<Vec<_>>();
        let report = collector.finish();

        assert!(arrival_times.contains(&0.0));
        assert!(arrival_times.iter().all(|arrival| *arrival >= 0.0));
        assert_eq!(report.request_counts.completed_requests, 3);
    }
}
