// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal scheduler-local observation sink retained for legacy unit tests.
//!
//! Production replay derives reports from neutral engine effects. Keeping this
//! small sink lets the mechanically moved scheduler tests continue to exercise
//! admission and token timing without making the engine depend on Replayer.

use std::collections::HashMap;

use uuid::Uuid;

use crate::engine::KvEvent;
use crate::engine::common::protocols::KvEventPublishers;
use crate::engine::common::sequence::RequestSequence;
use crate::engine::kv_manager::{BlockRequestLease, G1Acquire, G1Manager};
use crate::engine::scheduler::capture_kv_event_sink;

/// Build the neutral native-G1 event chain used to verify that a promoted
/// tail block references an already-published parent.
///
/// Dynamo's test adapter converts these events to Router wire types; the
/// scheduler/KV fixture itself remains owned by the generalized engine.
#[doc(hidden)]
pub fn g1_parent_chain_events(block_size: usize) -> Vec<KvEvent> {
    assert!(block_size >= 2, "block size must be at least 2");
    let computed_after = block_size
        .checked_mul(3)
        .expect("ordering regression token count overflow");
    let prompt_len = computed_after - 2;
    let prompt_len_u32 =
        u32::try_from(prompt_len).expect("ordering regression prompt length must fit in u32");
    let (mut sequence, identities) = RequestSequence::new(
        (0..prompt_len_u32).collect(),
        2,
        2,
        block_size,
        true,
        true,
        false,
        Some(vec![prompt_len_u32, prompt_len_u32 + 1]),
    );
    let owner = Uuid::from_u128(1);
    let mut lease = BlockRequestLease::new(owner, identities);
    let (events, sink) = capture_kv_event_sink();
    let mut manager =
        G1Manager::new_with_event_sink(3, block_size, KvEventPublishers::new(Some(sink)), 0);
    assert!(matches!(
        manager.allocate_native(owner, &mut lease, prompt_len, 0),
        G1Acquire::Ready(3)
    ));
    let prompt_complete = prompt_len / block_size * block_size;
    manager.finalize_native_computed_prefix(owner, 0, prompt_complete, &mut sequence, &mut lease);

    for expected in [prompt_len_u32, prompt_len_u32 + 1] {
        let (generated, opened_partial) = sequence.generate_token();
        assert_eq!(generated, expected);
        assert!(!opened_partial);
    }
    manager.finalize_native_computed_prefix(
        owner,
        prompt_complete,
        computed_after,
        &mut sequence,
        &mut lease,
    );
    events.drain()
}

#[derive(Clone, Debug, Default)]
pub struct TraceSnapshot {
    #[cfg(test)]
    pub arrival_ms: Option<f64>,
    pub first_admit_ms: Option<f64>,
    pub first_token_ms: Option<f64>,
    pub last_token_ms: Option<f64>,
}

#[derive(Default)]
pub struct TraceCollector {
    snapshots: HashMap<Uuid, TraceSnapshot>,
}

impl TraceCollector {
    #[cfg(test)]
    pub fn on_arrival(
        &mut self,
        request_id: Uuid,
        arrival_ms: f64,
        _input_tokens: usize,
        _output_tokens: usize,
    ) {
        self.snapshots.entry(request_id).or_default().arrival_ms = Some(arrival_ms);
    }

    pub fn on_admit(&mut self, request_id: Uuid, now_ms: f64, _reused_input_tokens: usize) {
        self.snapshots
            .entry(request_id)
            .or_default()
            .first_admit_ms
            .get_or_insert(now_ms);
    }

    pub fn on_token(&mut self, request_id: Uuid, now_ms: f64) {
        let snapshot = self.snapshots.entry(request_id).or_default();
        snapshot.first_token_ms.get_or_insert(now_ms);
        snapshot.last_token_ms = Some(now_ms);
    }

    #[cfg(test)]
    pub fn snapshot(&self, request_id: Uuid) -> Option<&TraceSnapshot> {
        self.snapshots.get(&request_id)
    }
}
