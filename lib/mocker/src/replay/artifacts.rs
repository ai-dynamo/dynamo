// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use uuid::Uuid;

use crate::common::protocols::OutputSignal;
use crate::loadgen::ReplayRequestHashes;

/// Build the minimal Native G1 artifact that exercises the producer→indexer
/// parent-ordering contract.
///
/// This is test-only plumbing shared by the mocker unit regression and
/// downstream indexer parity coverage.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn native_g1_parent_chain_artifact(block_size: usize) -> ReplayWorkerArtifacts {
    let prompt_len = block_size
        .checked_mul(3)
        .and_then(|tokens| tokens.checked_sub(2))
        .expect("ordering regression token count overflow");
    let tokens = (0..u32::try_from(prompt_len + 2)
        .expect("ordering regression token length must fit in u32"))
        .collect::<Vec<_>>();
    let kv_events = aisimulate_engine::g1_parent_chain_events(block_size)
        .into_iter()
        .enumerate()
        .map(|(ordinal, event)| ReplayTimedKvEvent {
            event: crate::engine_observations::dynamo_kv_event(event).0,
            storage_tier: StorageTier::Device,
            timestamp_us: ordinal as u64,
        })
        .collect::<Vec<_>>();
    let request_timestamp = kv_events.len() as u64;

    ReplayWorkerArtifacts {
        requests: vec![ReplayTimedRequest {
            uuid: Uuid::from_u128(1),
            timestamp_us: request_timestamp,
            scheduled_ready_at_ms: request_timestamp as f64 / 1000.0,
            input_length: tokens.len(),
            output_length: 0,
            replay_hashes: ReplayRequestHashes::from_tokens(
                &tokens,
                u32::try_from(block_size).expect("block size must fit in u32"),
            ),
        }],
        output_signals: Vec::new(),
        kv_events,
    }
}

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
    pub storage_tier: StorageTier,
    pub timestamp_us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ReplayWorkerArtifacts {
    pub requests: Vec<ReplayTimedRequest>,
    pub output_signals: Vec<ReplayTimedOutputSignal>,
    pub kv_events: Vec<ReplayTimedKvEvent>,
}
