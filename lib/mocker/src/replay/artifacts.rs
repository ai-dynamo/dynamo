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
    use crate::common::protocols::{G1Backend, KvEventPublishers};
    use crate::common::sequence::ActiveSequence;
    use crate::kv_manager::{G1Acquire, G1Manager};
    use crate::scheduler::capture_router_event_sink;

    assert!(block_size >= 2, "block size must be at least 2");
    let computed_after = block_size
        .checked_mul(3)
        .expect("ordering regression token count overflow");
    let prompt_len = computed_after - 2;
    let prompt_len_u32 =
        u32::try_from(prompt_len).expect("ordering regression prompt length must fit in u32");
    let mut tokens = (0..prompt_len_u32).collect::<Vec<_>>();
    let mut sequence = ActiveSequence::new(tokens.clone(), 2, Some(block_size), true, false);
    let owner = Uuid::from_u128(1);
    let (events, sink) = capture_router_event_sink(1);
    let publishers = KvEventPublishers::new(Some(sink), None);
    let mut manager = G1Manager::new_with_backend(3, block_size, publishers, 0, G1Backend::Native);

    let creation = sequence
        .take_creation_signal()
        .expect("three-block sequence must allocate G1 blocks");
    assert!(matches!(
        manager.process_for_request(owner, &creation, 0),
        G1Acquire::Ready(3)
    ));

    for token in [prompt_len_u32, prompt_len_u32 + 1] {
        assert!(sequence.push(token).is_none());
        tokens.push(token);
    }
    manager.finalize_computed_prefix(owner, 0, computed_after, &mut sequence);

    let kv_events = events
        .drain()
        .into_iter()
        .enumerate()
        .map(|(ordinal, event)| ReplayTimedKvEvent {
            event: event.event,
            storage_tier: event.storage_tier,
            timestamp_us: ordinal as u64,
        })
        .collect::<Vec<_>>();
    let request_timestamp = kv_events.len() as u64;

    ReplayWorkerArtifacts {
        requests: vec![ReplayTimedRequest {
            uuid: owner,
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
