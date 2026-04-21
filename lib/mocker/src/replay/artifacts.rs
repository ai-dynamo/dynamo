// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::KvCacheEvent;
use uuid::Uuid;

use crate::common::protocols::OutputSignal;
use crate::loadgen::ReplayRequestHashes;

#[derive(Debug, Clone)]
pub struct ReplayTimedRequest {
    pub uuid: Uuid,
    pub timestamp_us: u64,
    pub scheduled_ready_at_ms: f64,
    pub input_length: usize,
    pub output_length: usize,
    pub replay_hashes: ReplayRequestHashes,
}

#[derive(Debug, Clone)]
pub struct ReplayTimedOutputSignal {
    pub signal: OutputSignal,
    pub timestamp_us: u64,
}

#[derive(Debug, Clone)]
pub struct ReplayTimedKvEvent {
    pub event: KvCacheEvent,
    pub timestamp_us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ReplayWorkerArtifacts {
    pub requests: Vec<ReplayTimedRequest>,
    pub output_signals: Vec<ReplayTimedOutputSignal>,
    pub kv_events: Vec<ReplayTimedKvEvent>,
}
