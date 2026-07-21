// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use uuid::Uuid;

use super::{ReadyArrival, ReplayMode};
use crate::common::protocols::DirectRequest;
use crate::loadgen::WorkloadDriver;

enum AdmissionSource {
    /// Ordinary replay entrypoints keep their historical streaming behavior:
    /// admitted requests are popped and their token payloads can be released.
    StreamingRequests(VecDeque<DirectRequest>),
    /// Interactive sessions retain immutable trace input so many checkpoints
    /// can share the untouched future without cloning its token vectors.
    CheckpointableRequests {
        /// Immutable trace payload shared by live sessions and checkpoints.
        /// Only `next_index` is timeline-local, so a checkpoint is O(1) in
        /// the number and token volume of future requests.
        requests: Arc<[DirectRequest]>,
        next_index: usize,
        /// First validation failure in the immutable payload. ReplaySession
        /// validates this at construction, while lower-level replay callers
        /// receive the same deterministic checkpoint error without rescanning
        /// every pending token vector at every checkpoint.
        checkpoint_error: Option<Arc<str>>,
    },
    Workload(WorkloadDriver),
}

pub(in crate::replay::offline) struct AdmissionQueue {
    source: AdmissionSource,
    mode: ReplayMode,
}

#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::replay::offline) struct AdmissionQueueCheckpoint {
    requests: Arc<[DirectRequest]>,
    next_index: usize,
    mode: ReplayMode,
}

impl AdmissionQueue {
    pub(in crate::replay::offline) fn new_requests(
        source: VecDeque<DirectRequest>,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::StreamingRequests(source),
            mode,
        }
    }

    pub(in crate::replay::offline) fn new_checkpointable_requests(
        source: VecDeque<DirectRequest>,
        mode: ReplayMode,
    ) -> Self {
        let requests = source.into_iter().collect::<Vec<_>>();
        let checkpoint_error = checkpoint_validation_error(&requests);
        Self {
            source: AdmissionSource::CheckpointableRequests {
                requests: requests.into(),
                next_index: 0,
                checkpoint_error,
            },
            mode,
        }
    }

    pub(in crate::replay::offline) fn new_workload(
        driver: WorkloadDriver,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Workload(driver),
            mode,
        }
    }

    /// Snapshot raw request admissions. Workload drivers own session heaps,
    /// dependencies, and output-generation state and need their own explicit
    /// memento before they can participate in checkpointing.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::replay::offline) fn checkpoint_requests(
        &self,
    ) -> Result<AdmissionQueueCheckpoint> {
        anyhow::ensure!(
            matches!(self.mode, ReplayMode::Trace),
            "checkpoint spike supports trace-mode admissions only"
        );
        let (requests, next_index) = match &self.source {
            AdmissionSource::StreamingRequests(requests) => {
                let requests = requests.iter().cloned().collect::<Vec<_>>();
                if let Some(error) = checkpoint_validation_error(&requests) {
                    anyhow::bail!(error.to_string());
                }
                (requests.into(), 0)
            }
            AdmissionSource::CheckpointableRequests {
                requests,
                next_index,
                checkpoint_error,
            } => {
                if let Some(error) = checkpoint_error {
                    anyhow::bail!(error.to_string());
                }
                (Arc::clone(requests), *next_index)
            }
            AdmissionSource::Workload(_) => {
                anyhow::bail!("checkpoint spike does not yet support workload-driver admissions")
            }
        };
        Ok(AdmissionQueueCheckpoint {
            requests,
            next_index,
            mode: self.mode,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::replay::offline) fn restore_requests(
        checkpoint: AdmissionQueueCheckpoint,
    ) -> Self {
        Self {
            source: AdmissionSource::CheckpointableRequests {
                requests: checkpoint.requests,
                next_index: checkpoint.next_index,
                checkpoint_error: None,
            },
            mode: checkpoint.mode,
        }
    }

    pub(in crate::replay::offline) fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub(in crate::replay::offline) fn next_ready_time_ms(&mut self) -> Option<f64> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::StreamingRequests(requests)) => requests
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (
                ReplayMode::Trace,
                AdmissionSource::CheckpointableRequests {
                    requests,
                    next_index,
                    ..
                },
            ) => requests
                .get(*next_index)
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            // Concurrency: the driver owns the session cap and gates admission, so defer to
            // it directly (no in-flight clamp needed here).
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                driver.next_ready_time_ms()
            }
            (
                ReplayMode::Concurrency { .. },
                AdmissionSource::StreamingRequests(_)
                | AdmissionSource::CheckpointableRequests { .. },
            ) => None,
        }
    }

    pub(in crate::replay::offline) fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival>> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::StreamingRequests(requests)) => {
                let mut ready = Vec::new();
                while requests
                    .front()
                    .and_then(|request| request.arrival_timestamp_ms)
                    .is_some_and(|arrival_ms| arrival_ms <= now_ms)
                {
                    let request = requests
                        .pop_front()
                        .expect("front request must exist while draining admissions");
                    ready.push(ReadyArrival {
                        arrival_time_ms: request
                            .arrival_timestamp_ms
                            .expect("ready trace request must have an arrival timestamp"),
                        request,
                        replay_hashes: None,
                        session_id: None,
                        turn_index: None,
                    });
                }
                Ok(ready)
            }
            (
                ReplayMode::Trace,
                AdmissionSource::CheckpointableRequests {
                    requests,
                    next_index,
                    ..
                },
            ) => {
                let mut ready = Vec::new();
                loop {
                    let arrival_ms = requests
                        .get(*next_index)
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = requests[*next_index].clone();
                    *next_index += 1;
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms,
                        replay_hashes: None,
                        session_id: None,
                        turn_index: None,
                    });
                }
                Ok(ready)
            }
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => Ok(driver
                .pop_ready(now_ms, usize::MAX)
                .into_iter()
                .map(|ready| ReadyArrival {
                    request: ready.request,
                    arrival_time_ms: ready.scheduled_ready_at_ms,
                    replay_hashes: ready.replay_hashes,
                    session_id: Some(ready.session_id),
                    turn_index: Some(ready.turn_index),
                })
                .collect()),
            (
                ReplayMode::Concurrency { max_in_flight },
                AdmissionSource::CheckpointableRequests {
                    requests,
                    next_index,
                    ..
                },
            ) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(request) = requests.get(*next_index).cloned() else {
                        break;
                    };
                    *next_index += 1;
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms: now_ms,
                        replay_hashes: None,
                        session_id: None,
                        turn_index: None,
                    });
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
            (
                ReplayMode::Concurrency { max_in_flight },
                AdmissionSource::StreamingRequests(requests),
            ) => {
                let mut ready = Vec::new();
                let available = max_in_flight.saturating_sub(cluster_in_flight);
                for _ in 0..available {
                    let Some(request) = requests.pop_front() else {
                        break;
                    };
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms: now_ms,
                        replay_hashes: None,
                        session_id: None,
                        turn_index: None,
                    });
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                // The driver owns the session cap and only ever holds active sessions'
                // turns in its heap, so drain everything ready in heap (i.e. limit=usize MAX).
                Ok(driver
                    .pop_ready(now_ms, usize::MAX)
                    .into_iter()
                    .map(|ready| ReadyArrival {
                        request: ready.request,
                        arrival_time_ms: now_ms,
                        replay_hashes: ready.replay_hashes,
                        session_id: Some(ready.session_id),
                        turn_index: Some(ready.turn_index),
                    })
                    .collect())
            }
        }
    }

    pub(in crate::replay::offline) fn on_request_completed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
    ) -> Result<()> {
        self.on_request_terminal(uuid, now_ms, false)
    }

    pub(in crate::replay::offline) fn on_request_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        rejected: bool,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_terminal(uuid, now_ms, rejected)
    }

    pub(in crate::replay::offline) fn on_output_token(
        &mut self,
        uuid: Uuid,
        token_id: u32,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_output_token(uuid, token_id)
    }

    pub(in crate::replay::offline) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::StreamingRequests(requests) => requests.is_empty(),
            AdmissionSource::CheckpointableRequests {
                requests,
                next_index,
                ..
            } => *next_index >= requests.len(),
            AdmissionSource::Workload(driver) => driver.is_drained(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload(_))
    }

    pub(in crate::replay::offline) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::StreamingRequests(requests) => requests.len(),
            AdmissionSource::CheckpointableRequests {
                requests,
                next_index,
                ..
            } => requests.len().saturating_sub(*next_index),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        }
    }
}

fn checkpoint_validation_error(requests: &[DirectRequest]) -> Option<Arc<str>> {
    for (index, request) in requests.iter().enumerate() {
        if request.uuid.is_none() {
            return Some(
                format!(
                    "checkpoint spike requires an explicit UUID for pending trace request {index}"
                )
                .into(),
            );
        }
        let Some(output_token_ids) = request.output_token_ids.as_ref() else {
            return Some(
                format!(
                    "checkpoint spike requires planned output token IDs for pending trace request {index}"
                )
                .into(),
            );
        };
        if output_token_ids.len() != request.max_output_tokens {
            return Some(
                format!(
                    "checkpoint spike requires planned output token count to match max_output_tokens for pending trace request {index}"
                )
                .into(),
            );
        }
    }
    None
}

impl AdmissionQueueCheckpoint {
    /// Conservatively bound the physical KV footprint of the entire pending
    /// suffix. Summing each request independently deliberately double-counts
    /// shared prefixes, which makes the bound safe for proving that no future
    /// allocation can reach the eviction path.
    pub(in crate::replay::offline) fn future_kv_blocks_upper_bound(
        &self,
        block_size: usize,
    ) -> Result<usize> {
        anyhow::ensure!(block_size > 0, "checkpoint KV block size must be positive");
        self.requests[self.next_index..]
            .iter()
            .enumerate()
            .try_fold(0usize, |total, (index, request)| {
                let tokens = request
                    .tokens
                    .len()
                    .checked_add(request.max_output_tokens)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "pending trace request {index} token footprint overflowed usize"
                        )
                    })?;
                total
                    .checked_add(tokens.div_ceil(block_size))
                    .ok_or_else(|| anyhow::anyhow!("pending trace KV footprint overflowed usize"))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: u128, at_ms: f64, token: u32) -> DirectRequest {
        DirectRequest {
            tokens: vec![token; 16_384],
            max_output_tokens: 1,
            output_token_ids: Some(vec![token]),
            uuid: Some(Uuid::from_u128(id)),
            arrival_timestamp_ms: Some(at_ms),
            ..Default::default()
        }
    }

    #[test]
    fn raw_trace_checkpoints_share_the_immutable_future_payload() {
        let mut queue = AdmissionQueue::new_checkpointable_requests(
            VecDeque::from(vec![request(1, 0.0, 1), request(2, 10.0, 2)]),
            ReplayMode::Trace,
        );
        let first = queue.checkpoint_requests().unwrap();
        let second = queue.checkpoint_requests().unwrap();
        assert!(Arc::ptr_eq(&first.requests, &second.requests));
        assert_eq!(first.next_index, 0);

        let ready = queue.drain_ready(0.0, 0).unwrap();
        assert_eq!(ready.len(), 1);
        let after_admission = queue.checkpoint_requests().unwrap();
        assert!(Arc::ptr_eq(&first.requests, &after_admission.requests));
        assert_eq!(after_admission.next_index, 1);

        let restored = AdmissionQueue::restore_requests(first);
        assert_eq!(restored.total_requests(), 2);
    }

    #[test]
    fn ordinary_raw_trace_admissions_stream_consumed_payloads() {
        let mut queue = AdmissionQueue::new_requests(
            VecDeque::from(vec![request(1, 0.0, 1), request(2, 10.0, 2)]),
            ReplayMode::Trace,
        );

        assert!(matches!(
            &queue.source,
            AdmissionSource::StreamingRequests(_)
        ));
        assert_eq!(queue.drain_ready(0.0, 0).unwrap().len(), 1);
        let AdmissionSource::StreamingRequests(requests) = &queue.source else {
            panic!("ordinary replay must retain its streaming admission source");
        };
        assert_eq!(requests.len(), 1);
    }
}
