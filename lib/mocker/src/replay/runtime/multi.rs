// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::events::WorkerCompletion;
use super::normalize_trace_requests;
use super::state::OfflineWorkerState;
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::replay::{TraceCollector, TraceSimulationReport};
use anyhow::bail;
use std::collections::{BinaryHeap, VecDeque};
use uuid::Uuid;
use validator::Validate;

#[derive(Debug, Clone, Copy)]
enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

struct OfflineRuntime {
    now_ms: f64,
    next_worker_idx: usize,
    pending: VecDeque<DirectRequest>,
    workers: Vec<OfflineWorkerState>,
    collector: TraceCollector,
    completions: BinaryHeap<WorkerCompletion>,
    mode: ReplayMode,
}

impl OfflineRuntime {
    fn new(
        args: &MockEngineArgs,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: ReplayMode,
    ) -> Self {
        Self {
            now_ms: 0.0,
            next_worker_idx: 0,
            pending,
            workers: (0..num_workers)
                .map(|worker_idx| OfflineWorkerState::new(worker_idx, args.clone()))
                .collect(),
            collector: TraceCollector::default(),
            completions: BinaryHeap::new(),
            mode,
        }
    }

    fn cluster_in_flight(&self) -> usize {
        self.workers.iter().map(OfflineWorkerState::in_flight).sum()
    }

    // Record a request release on the selected worker, then enqueue it on the
    // next worker selected by deterministic round robin.
    fn assign_request(&mut self, mut request: DirectRequest, arrival_time_ms: f64) -> Uuid {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        request.uuid = Some(uuid);
        if matches!(self.mode, ReplayMode::Concurrency { .. }) {
            request.arrival_timestamp_ms = Some(arrival_time_ms);
        }

        self.collector.on_arrival(
            uuid,
            arrival_time_ms,
            request.tokens.len(),
            request.max_output_tokens,
        );

        let worker_idx = self.next_worker_idx;
        self.next_worker_idx = (self.next_worker_idx + 1) % self.workers.len();

        // TODO: If future cluster scheduling needs worker-local metrics beyond
        // these runtime-approved counters, promote them into OfflineWorkerState
        // or move to a finer event model.
        self.workers[worker_idx].receive_request(request);

        uuid
    }

    fn is_done(&self) -> bool {
        self.pending.is_empty()
            && self.completions.is_empty()
            && self.cluster_in_flight() == 0
            && self.workers.iter().all(OfflineWorkerState::is_drained)
    }

    fn next_timestamp(&self) -> Option<f64> {
        let next_completion_ms = self.completions.peek().map(|completion| completion.at_ms);
        let next_arrival_ms = match self.mode {
            ReplayMode::Trace => self
                .pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            ReplayMode::Concurrency { .. } => None,
        };

        match (next_arrival_ms, next_completion_ms) {
            (Some(arrival_ms), Some(completion_ms)) => Some(arrival_ms.min(completion_ms)),
            (Some(arrival_ms), None) => Some(arrival_ms),
            (None, Some(completion_ms)) => Some(completion_ms),
            (None, None) => None,
        }
    }

    fn apply_completed_requests(&mut self, worker_idx: usize, completed_requests: usize) {
        self.workers[worker_idx].mark_completed(completed_requests);
    }

    // Apply every worker completion at the current timestamp before releasing
    // new arrivals or topping off replay-concurrency.
    fn apply_worker_completions(&mut self) -> bool {
        let mut changed = false;
        while self
            .completions
            .peek()
            .is_some_and(|completion| completion.at_ms == self.now_ms)
        {
            let completion = self
                .completions
                .pop()
                .expect("completion must exist after peek");
            let worker = &mut self.workers[completion.worker_idx];
            worker.mark_idle();
            self.apply_completed_requests(completion.worker_idx, completion.completed_requests);
            changed = true;
        }

        changed
    }

    // Trace mode keeps arrivals in timestamp order and drains every request that
    // is visible at the current simulated time.
    fn release_trace_arrivals(&mut self) -> bool {
        let mut released_any = false;
        while self
            .pending
            .front()
            .and_then(|request| request.arrival_timestamp_ms)
            .is_some_and(|arrival_ms| arrival_ms <= self.now_ms)
        {
            let request = self
                .pending
                .pop_front()
                .expect("front request must exist when arrival is ready");
            let arrival_ms = request
                .arrival_timestamp_ms
                .expect("trace replay requests must have an arrival timestamp");
            self.assign_request(request, arrival_ms);
            released_any = true;
        }

        released_any
    }

    // Replay-concurrency uses the runtime-approved worker in-flight counters as
    // the sole source of truth for whether more requests may be released now.
    fn top_off_concurrency(&mut self, max_in_flight: usize) -> bool {
        let mut released_any = false;
        while self.cluster_in_flight() < max_in_flight {
            let Some(request) = self.pending.pop_front() else {
                break;
            };
            self.assign_request(request, self.now_ms);
            released_any = true;
        }

        released_any
    }

    // Run every idle worker until it either blocks on a future completion event
    // or becomes empty. Zero-duration completions are applied inline.
    fn drive_ready_workers(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        for worker_idx in 0..self.workers.len() {
            loop {
                if !self.workers[worker_idx].is_ready() {
                    break;
                }

                let executed = {
                    let (workers, collector) = (&mut self.workers, &mut self.collector);
                    workers[worker_idx].execute_pass(collector, self.now_ms)
                };
                changed = true;

                if executed.end_ms == self.now_ms {
                    self.apply_completed_requests(worker_idx, executed.completed_requests);
                    continue;
                }

                self.workers[worker_idx].mark_busy();
                self.completions.push(WorkerCompletion {
                    at_ms: executed.end_ms,
                    worker_idx,
                    completed_requests: executed.completed_requests,
                });
                break;
            }
        }

        Ok(changed)
    }

    fn drain_current_timestamp(&mut self) -> anyhow::Result<()> {
        loop {
            let mut changed = self.apply_worker_completions();

            changed |= match self.mode {
                ReplayMode::Trace => self.release_trace_arrivals(),
                ReplayMode::Concurrency { max_in_flight } => {
                    self.top_off_concurrency(max_in_flight)
                }
            };

            changed |= self.drive_ready_workers()?;

            if !changed {
                break;
            }
        }

        Ok(())
    }

    // Global event loop: apply completions at a timestamp, release any arrivals
    // now visible at that same timestamp, then drive newly idle workers.
    fn run(mut self) -> anyhow::Result<TraceCollector> {
        self.drain_current_timestamp()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };

            self.now_ms = next_timestamp_ms;
            self.drain_current_timestamp()?;
        }

        Ok(self.collector)
    }
}

pub(crate) fn simulate_trace_multi(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
) -> anyhow::Result<TraceSimulationReport> {
    args.validate()?;
    let pending = normalize_trace_requests(requests)?;
    let collector = OfflineRuntime::new(&args, pending, num_workers, ReplayMode::Trace).run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_multi(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> anyhow::Result<TraceSimulationReport> {
    args.validate()?;
    let pending = VecDeque::from(requests);
    let collector = OfflineRuntime::new(
        &args,
        pending,
        num_workers,
        ReplayMode::Concurrency { max_in_flight },
    )
    .run()?;
    Ok(collector.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn run_trace_multi_collect(
        args: &MockEngineArgs,
        requests: Vec<DirectRequest>,
        num_workers: usize,
    ) -> TraceCollector {
        let pending = normalize_trace_requests(requests).unwrap();
        OfflineRuntime::new(args, pending, num_workers, ReplayMode::Trace)
            .run()
            .unwrap()
    }

    fn run_concurrency_multi_collect(
        args: &MockEngineArgs,
        requests: Vec<DirectRequest>,
        max_in_flight: usize,
        num_workers: usize,
    ) -> TraceCollector {
        OfflineRuntime::new(
            args,
            VecDeque::from(requests),
            num_workers,
            ReplayMode::Concurrency { max_in_flight },
        )
        .run()
        .unwrap()
    }

    #[test]
    fn test_multi_worker_trace_round_robin_assigns_same_timestamp_requests_deterministically() {
        let args = replay_args(false, true);
        let collector = run_trace_multi_collect(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                    max_output_tokens: 4,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                },
                DirectRequest {
                    tokens: vec![7, 7, 7, 7, 8, 8, 8, 8],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(44)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(101.0),
                },
            ],
            2,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();
        let request_4 = collector.snapshot(Uuid::from_u128(44)).unwrap();
        let report = collector.finish();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, 1.0);
        assert_eq!(request_4.arrival_time_ms, 1.0);

        assert!(request_3.first_admit_ms.unwrap() >= request_1.first_token_ms.unwrap());
        assert!(request_4.first_admit_ms.unwrap() >= request_2.first_token_ms.unwrap());
        assert!(request_3.first_admit_ms.unwrap() < request_4.first_admit_ms.unwrap());

        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.request_counts.total_input_tokens, 40);
        assert_eq!(report.request_counts.total_output_tokens, 10);
    }

    #[test]
    fn test_multi_worker_concurrency_uses_worker_in_flight_for_cap_checks() {
        let args = replay_args(false, false);
        let collector = run_concurrency_multi_collect(
            &args,
            vec![
                DirectRequest {
                    tokens: vec![1, 1, 1, 1, 2, 2, 2, 2],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(11)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(900.0),
                },
                DirectRequest {
                    tokens: vec![3, 3, 3, 3, 4, 4, 4, 4],
                    max_output_tokens: 4,
                    uuid: Some(Uuid::from_u128(22)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(1000.0),
                },
                DirectRequest {
                    tokens: vec![5, 5, 5, 5, 6, 6, 6, 6],
                    max_output_tokens: 2,
                    uuid: Some(Uuid::from_u128(33)),
                    dp_rank: 0,
                    arrival_timestamp_ms: Some(100.0),
                },
            ],
            2,
            2,
        );

        let request_1 = collector.snapshot(Uuid::from_u128(11)).unwrap();
        let request_2 = collector.snapshot(Uuid::from_u128(22)).unwrap();
        let request_3 = collector.snapshot(Uuid::from_u128(33)).unwrap();
        let report = collector.finish();

        assert_eq!(request_1.arrival_time_ms, 0.0);
        assert_eq!(request_2.arrival_time_ms, 0.0);
        assert_eq!(request_3.arrival_time_ms, request_1.last_token_ms.unwrap());
        assert!(request_3.arrival_time_ms < request_2.last_token_ms.unwrap());
        assert_eq!(request_3.first_admit_ms.unwrap(), request_3.arrival_time_ms);

        assert_eq!(report.request_counts.completed_requests, 3);
        assert_eq!(report.request_counts.total_input_tokens, 24);
        assert_eq!(report.request_counts.total_output_tokens, 8);
    }
}
