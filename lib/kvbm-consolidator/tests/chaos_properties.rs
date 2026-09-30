// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Property-based chaos test (test 17).
//!
//! Drives the Tracker directly — no ZMQ, no Tokio — and verifies that the emitted
//! ConsolidatedEvent stream satisfies:
//!   - Every STORE is for a block not currently live.
//!   - Every REMOVE is for a block that is currently live.
//!   - ClearAll empties the live set.
//!
//! Run with: cargo test -p kvbm-consolidator --test chaos_properties -- --test-threads=1

use std::collections::{HashMap, HashSet};

use dynamo_tokens::{PositionalLineageHash, compute_hash_v2};
use kvbm_consolidator::tracker::ConsolidatedEvent;
use kvbm_consolidator::{EventSource, Tracker};
use proptest::prelude::*;

const SALT: u64 = 1337;
const CATALOG_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Pre-built catalog of 8 chained PositionalLineageHash values.
// ---------------------------------------------------------------------------

fn build_catalog() -> Vec<(PositionalLineageHash, u64)> {
    let mut result = Vec::with_capacity(CATALOG_SIZE);
    let mut parent_seq: Option<u64> = None;
    let mut parent_pos: Option<PositionalLineageHash> = None;
    for i in 0u32..CATALOG_SIZE as u32 {
        let bytes: Vec<u8> = i.to_le_bytes().to_vec();
        let block_hash = compute_hash_v2(&bytes, SALT);
        let seq_u64 = match parent_seq {
            None => block_hash,
            Some(p) => {
                let combined: Vec<u8> = [p, block_hash]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                compute_hash_v2(&combined, SALT)
            }
        };
        let pos = parent_pos.map(|p| p.position() + 1).unwrap_or(0);
        let plh = PositionalLineageHash::new(seq_u64, parent_seq, pos);
        result.push((plh, seq_u64));
        parent_seq = Some(seq_u64);
        parent_pos = Some(plh);
    }
    result
}

// ---------------------------------------------------------------------------
// Op strategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Action {
    VllmStore { idx: usize },
    VllmRemove { idx: usize },
    TrtllmStore { idx: usize },
    TrtllmRemove { idx: usize },
    KvbmStore { idx: usize },
    KvbmRemove { idx: usize },
    ClearAll,
}

fn ops_strategy() -> impl Strategy<Value = Vec<Action>> {
    prop::collection::vec(
        prop_oneof![
            (0usize..CATALOG_SIZE).prop_map(|idx| Action::VllmStore { idx }),
            (0usize..CATALOG_SIZE).prop_map(|idx| Action::VllmRemove { idx }),
            (0usize..CATALOG_SIZE).prop_map(|idx| Action::TrtllmStore { idx }),
            (0usize..CATALOG_SIZE).prop_map(|idx| Action::TrtllmRemove { idx }),
            (0usize..CATALOG_SIZE).prop_map(|idx| Action::KvbmStore { idx }),
            (0usize..CATALOG_SIZE).prop_map(|idx| Action::KvbmRemove { idx }),
            Just(Action::ClearAll),
        ],
        0..60,
    )
}

// ---------------------------------------------------------------------------
// Replay ops into tracker
// ---------------------------------------------------------------------------

fn replay(tracker: &mut Tracker, actions: &[Action]) {
    let catalog = build_catalog();

    // Track which indices are currently stored per source (for valid Remove ops).
    let mut vllm_stored: HashSet<usize> = HashSet::new();
    let mut trtllm_stored: HashSet<usize> = HashSet::new();
    let mut kvbm_stored: HashSet<usize> = HashSet::new();

    for action in actions {
        match action {
            Action::VllmStore { idx } => {
                let ext = format!("vllm_{idx}");
                let tokens = vec![*idx as u32];
                let parent_ext = if *idx > 0 && vllm_stored.contains(&(idx - 1)) {
                    Some(format!("vllm_{}", idx - 1))
                } else {
                    None
                };
                tracker.handle_store(EventSource::Vllm, ext, parent_ext, tokens, 1, None);
                vllm_stored.insert(*idx);
            }
            Action::VllmRemove { idx } => {
                if vllm_stored.contains(idx) {
                    tracker.handle_remove(EventSource::Vllm, &format!("vllm_{idx}"));
                    vllm_stored.remove(idx);
                }
            }
            Action::TrtllmStore { idx } => {
                let ext = format!("trt_{idx}");
                let tokens = vec![*idx as u32];
                let parent_ext = if *idx > 0 && trtllm_stored.contains(&(idx - 1)) {
                    Some(format!("trt_{}", idx - 1))
                } else {
                    None
                };
                tracker.handle_store(EventSource::Trtllm, ext, parent_ext, tokens, 1, None);
                trtllm_stored.insert(*idx);
            }
            Action::TrtllmRemove { idx } => {
                if trtllm_stored.contains(idx) {
                    tracker.handle_remove(EventSource::Trtllm, &format!("trt_{idx}"));
                    trtllm_stored.remove(idx);
                }
            }
            Action::KvbmStore { idx } => {
                // Store all ancestors first to maintain valid chain.
                for (ancestor, &(plh, _)) in catalog.iter().enumerate().take(*idx + 1) {
                    if !kvbm_stored.contains(&ancestor) {
                        tracker.handle_kvbm_store(plh, vec![ancestor as u32], 1, None);
                        kvbm_stored.insert(ancestor);
                    }
                }
            }
            Action::KvbmRemove { idx } => {
                if kvbm_stored.contains(idx) {
                    let (plh, _) = catalog[*idx];
                    tracker.handle_kvbm_remove(plh);
                    kvbm_stored.remove(idx);
                }
            }
            Action::ClearAll => {
                tracker.handle_clear_all();
                vllm_stored.clear();
                trtllm_stored.clear();
                kvbm_stored.clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant checker
// ---------------------------------------------------------------------------

fn check_invariants(emitted: &[ConsolidatedEvent]) -> Result<(), TestCaseError> {
    let mut live: HashSet<PositionalLineageHash> = HashSet::new();

    for ev in emitted {
        match ev {
            ConsolidatedEvent::Store { seq_hash, .. } => {
                prop_assert!(
                    !live.contains(seq_hash),
                    "duplicate STORE for {:?}",
                    seq_hash
                );
                live.insert(*seq_hash);
            }
            ConsolidatedEvent::Remove { seq_hash, .. } => {
                prop_assert!(
                    live.remove(seq_hash),
                    "REMOVE without prior STORE for {:?}",
                    seq_hash
                );
            }
            ConsolidatedEvent::ClearAll => {
                live.clear();
            }
        }
    }
    Ok(())
}

/// Verify parent-chain integrity: every STORE at position > 0 references a parent
/// that was itself emitted as a STORE at some earlier point (not necessarily still live).
fn check_parent_chain(emitted: &[ConsolidatedEvent]) -> Result<(), TestCaseError> {
    let mut ever_stored: HashMap<u64, PositionalLineageHash> = HashMap::new();

    for ev in emitted {
        if let ConsolidatedEvent::Store { seq_hash, .. } = ev {
            if seq_hash.position() > 0 {
                let parent_frag = seq_hash.parent_hash_fragment();
                prop_assert!(
                    ever_stored.contains_key(&parent_frag),
                    "STORE for {:?} at pos {} references parent fragment {} not yet emitted",
                    seq_hash,
                    seq_hash.position(),
                    parent_frag
                );
            }
            ever_stored.insert(
                kvbm_consolidator::hash::router_block_hash(*seq_hash),
                *seq_hash,
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    #[test]
    fn chaos_invariants_hold(actions in ops_strategy()) {
        let mut tracker = Tracker::new(None);
        replay(&mut tracker, &actions);
        let events = tracker.drain_events();
        check_invariants(&events)?;
    }

    #[test]
    fn chaos_parent_chain_integrity(actions in ops_strategy()) {
        let mut tracker = Tracker::new(None);
        replay(&mut tracker, &actions);
        let events = tracker.drain_events();
        check_parent_chain(&events)?;
    }

    #[test]
    fn chaos_drain_is_idempotent(actions in ops_strategy()) {
        // After draining events once, a second drain must return empty.
        let mut tracker = Tracker::new(None);
        replay(&mut tracker, &actions);
        let _first = tracker.drain_events();
        let second = tracker.drain_events();
        prop_assert!(
            second.is_empty(),
            "second drain should be empty but got {} events",
            second.len()
        );
    }
}
