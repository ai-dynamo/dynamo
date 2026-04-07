// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::RouterEvent;
use uuid::Uuid;

use super::super::runtime_utils::WorkerCompletionPayload;
use crate::common::protocols::DirectRequest;
use crate::loadgen::ReplayRequestHashes;

#[derive(Debug, Clone, Copy)]
pub(in crate::replay::offline) enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::replay::offline) enum EnginePassMode {
    Visible,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerAdmission {
    pub(crate) uuid: Uuid,
    pub(crate) worker_idx: usize,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct ScheduledWorkerCompletion {
    pub(in crate::replay::offline) at_ms: f64,
    pub(in crate::replay::offline) payload: WorkerCompletionPayload,
}

#[derive(Debug, Default)]
pub(in crate::replay::offline) struct EngineEffects {
    pub(in crate::replay::offline) pass_start_kv_events: Vec<RouterEvent>,
    pub(in crate::replay::offline) immediate_completions: Vec<WorkerCompletionPayload>,
    pub(in crate::replay::offline) scheduled_completions: Vec<ScheduledWorkerCompletion>,
}

impl EngineEffects {
    pub(in crate::replay::offline) fn is_empty(&self) -> bool {
        self.pass_start_kv_events.is_empty()
            && self.immediate_completions.is_empty()
            && self.scheduled_completions.is_empty()
    }
}

#[derive(Debug, Default)]
pub(crate) struct RouterEffects {
    pub(crate) admissions: Vec<WorkerAdmission>,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct ReadyArrival {
    pub(in crate::replay::offline) request: DirectRequest,
    pub(in crate::replay::offline) arrival_time_ms: f64,
    pub(in crate::replay::offline) replay_hashes: Option<ReplayRequestHashes>,
}
