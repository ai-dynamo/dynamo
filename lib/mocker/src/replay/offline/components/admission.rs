// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::Result;
use uuid::Uuid;

use super::{ReadyArrival, ReplayMode};
use crate::common::protocols::DirectRequest;
use crate::loadgen::WorkloadDriver;

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

pub(in crate::replay::offline) struct AdmissionQueue {
    source: AdmissionSource,
    mode: ReplayMode,
}

impl AdmissionQueue {
    pub(in crate::replay::offline) fn new_requests(
        source: VecDeque<DirectRequest>,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Requests(source),
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

    pub(in crate::replay::offline) fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub(in crate::replay::offline) fn next_ready_time_ms(
        &mut self,
        cluster_in_flight: usize,
    ) -> Option<f64> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Workload(driver)) => {
                if cluster_in_flight < *max_in_flight {
                    driver.next_ready_time_ms()
                } else {
                    None
                }
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_)) => None,
        }
    }

    pub(in crate::replay::offline) fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival>> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                loop {
                    let arrival_ms = pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = pending
                        .pop_front()
                        .expect("front request must exist when arrival is ready");
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms,
                        replay_hashes: None,
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
                })
                .collect()),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(request) = pending.pop_front() else {
                        break;
                    };
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms: now_ms,
                        replay_hashes: None,
                    });
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Workload(driver)) => {
                let available = max_in_flight.saturating_sub(cluster_in_flight);
                if available == 0 {
                    return Ok(Vec::new());
                }
                Ok(driver
                    .pop_ready(now_ms, available)
                    .into_iter()
                    .map(|ready| ReadyArrival {
                        request: ready.request,
                        arrival_time_ms: now_ms,
                        replay_hashes: ready.replay_hashes,
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
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_complete(uuid, now_ms)
    }

    pub(in crate::replay::offline) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.is_empty(),
            AdmissionSource::Workload(driver) => driver.is_drained(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload(_))
    }

    pub(in crate::replay::offline) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        }
    }
}
