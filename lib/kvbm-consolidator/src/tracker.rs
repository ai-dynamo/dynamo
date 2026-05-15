// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedup and lineage tracker.
//!
//! State machine:
//! - Key: [`kvbm_logical::SequenceHash`] (a 128-bit `PositionalLineageHash`).
//! - Per-block: set of [`EventSource`]s that currently hold the block, plus an optional
//!   [`BlockRegistrationHandle`] if a kvbm-logical registry is wired in.
//! - External-hash bridge: `(EventSource, String) → SequenceHash` for vLLM / TRT-LLM parent
//!   resolution (those sources reference parents by their opaque string hash).
//!
//! Emission rules:
//! - Publish STORE on the first source for a given sequence hash.
//! - Publish REMOVE only when the last source drops the block.
//! - Repeated stores from the same source with the same external hash are idempotent.
//!
//! Hash synthesis for external (vLLM / TRT-LLM) sources delegates entirely to
//! [`dynamo_kv_hashing`]: the consolidator builds a single-block [`Request`] for each
//! event, gets the [`BlockHash`] via [`Request::block_hashes`], and chains via
//! [`PositionalLineageHash::extend`] from the parent's PLH. KVBM events arrive with a
//! PLH already computed by the upstream registry.

use std::collections::{HashMap, HashSet, VecDeque};

use dynamo_kv_hashing::Request;
use dynamo_tokens::PositionalLineageHash;
use kvbm_logical::{BlockRegistry, SequenceHash, registry::BlockRegistrationHandle};

use crate::source::EventSource;

/// Internal event queued for publication. Carries the 128-bit hash so the publisher can
/// project to the router u64 wire shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsolidatedEvent {
    Store {
        seq_hash: SequenceHash,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_name: Option<String>,
        source: EventSource,
    },
    Remove {
        seq_hash: SequenceHash,
        source: EventSource,
    },
    ClearAll,
}

/// Per-block state: which sources have it, an optional registry handle keeping the
/// block present in kvbm-logical's shared radix tree, and whether a publishable
/// `ConsolidatedEvent::Store` has been emitted downstream.
///
/// `published` is `false` when only KVBM-bridge events (which carry no tokens /
/// block_size) have registered the block. The router can only accept stores whose
/// `block_size` matches its configured KV block size, so an empty placeholder store
/// would be rejected — we suppress it and wait for a real source.
pub struct BlockState {
    pub sources: HashSet<EventSource>,
    pub registry_handle: Option<BlockRegistrationHandle>,
    pub published: bool,
}

impl std::fmt::Debug for BlockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockState")
            .field("sources", &self.sources)
            .field("registry_handle", &self.registry_handle.is_some())
            .field("published", &self.published)
            .finish()
    }
}

/// Tracker state. Single-writer; wrap in `RwLock` for cross-task sharing.
pub struct Tracker {
    pub(crate) blocks: HashMap<SequenceHash, BlockState>,
    /// Maps `(source, external_string_hash) → SequenceHash`.
    pub(crate) external_to_seq: HashMap<(EventSource, String), SequenceHash>,
    pub(crate) event_queue: VecDeque<ConsolidatedEvent>,
    pub(crate) registry: Option<BlockRegistry>,
}

impl Tracker {
    /// Construct an empty tracker. Wire `registry` in to populate kvbm-logical's shared
    /// radix tree as side-effect of dedup (G1-block-manager pattern).
    pub fn new(registry: Option<BlockRegistry>) -> Self {
        Self {
            blocks: HashMap::new(),
            external_to_seq: HashMap::new(),
            event_queue: VecDeque::new(),
            registry,
        }
    }

    /// Compute the PLH for a single block given its parent PLH (or none for root).
    /// Hashing goes through [`dynamo_kv_hashing::Request`] so vLLM / TRT-LLM ingress
    /// matches what kvbm-logical's `BlockRegistry::register_block` produces for the
    /// same `(tokens, lora_name)` input.
    fn compute_plh(
        parent_plh: Option<PositionalLineageHash>,
        token_ids: &[u32],
        block_size: usize,
        lora_name: Option<&str>,
    ) -> Option<PositionalLineageHash> {
        let request = Request::builder()
            .tokens(token_ids.to_vec())
            .lora_name(lora_name.map(str::to_string))
            .build()
            .ok()?;
        let blocks = request.into_blocks(block_size as u32).ok()?;
        let first = blocks.first()?;
        Some(match parent_plh {
            None => first.plh,
            Some(parent) => parent.extend(first.block_hash),
        })
    }

    /// Handle a STORE event from a string-hashed source (vLLM / TRT-LLM).
    ///
    /// Returns `true` if a new block was registered, `false` if the block was already
    /// known (duplicate from same source, dedup against another source, or invalid input).
    pub fn handle_store(
        &mut self,
        source: EventSource,
        external_hash: String,
        parent_external_hash: Option<String>,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_name: Option<String>,
    ) -> bool {
        let parent_plh = match parent_external_hash {
            None => None,
            Some(ref peh) => match self.external_to_seq.get(&(source, peh.clone())) {
                Some(&p) => Some(p),
                None => {
                    tracing::warn!(
                        "Unresolved parent external hash {:?} for source {:?}; treating as root",
                        peh,
                        source
                    );
                    None
                }
            },
        };

        let plh = match Self::compute_plh(parent_plh, &token_ids, block_size, lora_name.as_deref())
        {
            Some(h) => h,
            None => {
                tracing::warn!(
                    "handle_store: hash computation failed for {} tokens at block_size {}",
                    token_ids.len(),
                    block_size
                );
                return false;
            }
        };

        // Idempotence: same (source, external_hash) already maps to this plh.
        if let Some(&existing_plh) = self.external_to_seq.get(&(source, external_hash.clone()))
            && existing_plh == plh
        {
            return false;
        }

        match self.blocks.get_mut(&plh) {
            Some(state) => {
                state.sources.insert(source);
                self.external_to_seq.insert((source, external_hash), plh);
                // A prior source registered the block without publishable metadata
                // (KVBM-bridge create with empty tokens / block_size 0). This is the
                // first real source — publish now.
                if !state.published {
                    state.published = true;
                    self.event_queue.push_back(ConsolidatedEvent::Store {
                        seq_hash: plh,
                        token_ids,
                        block_size,
                        lora_name,
                        source,
                    });
                    return true;
                }
                false
            }
            None => {
                let registry_handle = self
                    .registry
                    .as_ref()
                    .map(|r| r.register_sequence_hash(plh));
                let mut sources = HashSet::new();
                sources.insert(source);
                self.blocks.insert(
                    plh,
                    BlockState {
                        sources,
                        registry_handle,
                        published: true,
                    },
                );
                self.external_to_seq.insert((source, external_hash), plh);
                self.event_queue.push_back(ConsolidatedEvent::Store {
                    seq_hash: plh,
                    token_ids,
                    block_size,
                    lora_name,
                    source,
                });
                true
            }
        }
    }

    /// Handle a REMOVE event from a string-hashed source. Emits REMOVE only when the
    /// last source releases the block AND the block was previously published — if no
    /// publishable Store ever made it downstream, there is nothing for downstream to
    /// remove.
    pub fn handle_remove(&mut self, source: EventSource, external_hash: &str) -> bool {
        let key = (source, external_hash.to_string());
        let plh = match self.external_to_seq.remove(&key) {
            Some(p) => p,
            None => {
                tracing::warn!(
                    "handle_remove: unknown external hash {:?} for source {:?}",
                    external_hash,
                    source
                );
                return false;
            }
        };

        let (empty, published) = match self.blocks.get_mut(&plh) {
            Some(state) => {
                state.sources.remove(&source);
                (state.sources.is_empty(), state.published)
            }
            None => return false,
        };

        if empty {
            self.blocks.remove(&plh);
            if published {
                self.event_queue.push_back(ConsolidatedEvent::Remove {
                    seq_hash: plh,
                    source,
                });
                return true;
            }
        }
        false
    }

    /// Handle a STORE from KVBM — the PLH is already known.
    ///
    /// The KVBM broadcast bridge calls this with empty `token_ids` and `block_size = 0`
    /// because `kvbm_logical::events::protocol::KvCacheEvent::Create` only carries the
    /// PLH. Such calls register the block but do **not** publish a Store: an empty
    /// `BlockStored` is invalid for the kv-router (block_size must match its config).
    /// A subsequent vLLM / TRT-LLM store for the same PLH publishes with real metadata.
    /// Direct callers (tests, integrations that have real tokens) get the normal
    /// publish-on-first-source behavior.
    pub fn handle_kvbm_store(
        &mut self,
        seq_hash: SequenceHash,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_name: Option<String>,
    ) -> bool {
        let publishable = !token_ids.is_empty() && block_size > 0;
        match self.blocks.get_mut(&seq_hash) {
            Some(state) => {
                state.sources.insert(EventSource::Kvbm);
                if publishable && !state.published {
                    state.published = true;
                    self.event_queue.push_back(ConsolidatedEvent::Store {
                        seq_hash,
                        token_ids,
                        block_size,
                        lora_name,
                        source: EventSource::Kvbm,
                    });
                    return true;
                }
                false
            }
            None => {
                let registry_handle = self
                    .registry
                    .as_ref()
                    .map(|r| r.register_sequence_hash(seq_hash));
                let mut sources = HashSet::new();
                sources.insert(EventSource::Kvbm);
                self.blocks.insert(
                    seq_hash,
                    BlockState {
                        sources,
                        registry_handle,
                        published: publishable,
                    },
                );
                if publishable {
                    self.event_queue.push_back(ConsolidatedEvent::Store {
                        seq_hash,
                        token_ids,
                        block_size,
                        lora_name,
                        source: EventSource::Kvbm,
                    });
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Handle a REMOVE from KVBM. Emits Remove only when the block was previously
    /// published (see [`Self::handle_kvbm_store`] for why).
    pub fn handle_kvbm_remove(&mut self, seq_hash: SequenceHash) -> bool {
        let (empty, published) = match self.blocks.get_mut(&seq_hash) {
            Some(state) => {
                state.sources.remove(&EventSource::Kvbm);
                (state.sources.is_empty(), state.published)
            }
            None => {
                tracing::warn!("handle_kvbm_remove: unknown seq_hash {:?}", seq_hash);
                return false;
            }
        };

        if empty {
            self.blocks.remove(&seq_hash);
            if published {
                self.event_queue.push_back(ConsolidatedEvent::Remove {
                    seq_hash,
                    source: EventSource::Kvbm,
                });
                return true;
            }
        }
        false
    }

    /// Wipe all tracked state and enqueue a ClearAll event.
    pub fn handle_clear_all(&mut self) {
        self.blocks.clear();
        self.external_to_seq.clear();
        self.event_queue.push_back(ConsolidatedEvent::ClearAll);
    }

    /// Drain all queued events in FIFO order, leaving the queue empty.
    pub fn drain_events(&mut self) -> Vec<ConsolidatedEvent> {
        self.event_queue.drain(..).collect()
    }

    /// Test-only snapshot of tracked block count.
    #[cfg(any(test, feature = "testing"))]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Test-only snapshot of external-hash mapping size.
    #[cfg(any(test, feature = "testing"))]
    pub fn num_external_mappings(&self) -> usize {
        self.external_to_seq.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn tracker() -> Tracker {
        Tracker::new(None)
    }

    /// Reference PLH chain for a contiguous token stream, computed via the canonical
    /// `dynamo_kv_hashing::Request::into_blocks` path. The consolidator's vLLM ingress
    /// must produce the same PLH for the same `(tokens, lora_name)` slice.
    fn reference_chain(
        tokens: &[u32],
        block_size: usize,
        lora_name: Option<&str>,
    ) -> Vec<PositionalLineageHash> {
        let request = Request::builder()
            .tokens(tokens.to_vec())
            .lora_name(lora_name.map(str::to_string))
            .build()
            .expect("valid request");
        request
            .into_blocks(block_size as u32)
            .expect("into_blocks")
            .into_iter()
            .map(|b| b.plh)
            .collect()
    }

    #[test]
    fn tracker_first_store_from_any_source_emits_store_event() {
        let mut t = tracker();
        let ret = t.handle_store(
            EventSource::Vllm,
            "h1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        assert!(ret);
        let events = t.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ConsolidatedEvent::Store {
                source: EventSource::Vllm,
                ..
            }
        ));
    }

    #[test]
    fn tracker_repeat_store_same_source_and_hash_is_noop() {
        let mut t = tracker();
        t.handle_store(
            EventSource::Vllm,
            "h1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        t.drain_events();
        let ret = t.handle_store(
            EventSource::Vllm,
            "h1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        assert!(!ret);
        assert_eq!(t.drain_events().len(), 0);
        assert_eq!(t.num_blocks(), 1);
    }

    #[test]
    fn tracker_second_source_store_for_same_seq_hash_is_silent() {
        let mut t = tracker();
        t.handle_store(
            EventSource::Vllm,
            "v1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        t.drain_events();
        let ret = t.handle_store(
            EventSource::Trtllm,
            "t1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        assert!(!ret);
        assert_eq!(t.drain_events().len(), 0);
        assert_eq!(t.num_blocks(), 1);
    }

    #[test]
    fn tracker_emits_single_store_when_three_sources_see_same_hash() {
        let mut t = tracker();
        let tokens = vec![10u32, 20, 30, 40];

        t.handle_store(EventSource::Vllm, "v".into(), None, tokens.clone(), 4, None);
        t.handle_store(
            EventSource::Trtllm,
            "r".into(),
            None,
            tokens.clone(),
            4,
            None,
        );

        // KVBM sends the same logical block — compute its canonical PLH the same way.
        let kvbm_plh = reference_chain(&tokens, 4, None)[0];
        t.handle_kvbm_store(kvbm_plh, tokens, 4, None);

        let events = t.drain_events();
        let stores: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ConsolidatedEvent::Store { .. }))
            .collect();
        assert_eq!(stores.len(), 1);
        assert_eq!(t.num_blocks(), 1);

        let seq_hash = *t.external_to_seq.values().next().unwrap();
        assert_eq!(t.blocks[&seq_hash].sources.len(), 3);
    }

    #[test]
    fn tracker_remove_from_last_source_emits_remove_event() {
        let mut t = tracker();
        t.handle_store(
            EventSource::Vllm,
            "h1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        t.drain_events();
        let ret = t.handle_remove(EventSource::Vllm, "h1");
        assert!(ret);
        let events = t.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ConsolidatedEvent::Remove { .. }));
        assert_eq!(t.num_blocks(), 0);
    }

    #[test]
    fn tracker_partial_remove_two_of_three_sources_is_silent() {
        let mut t = tracker();
        let tokens = vec![1u32, 2, 3, 4];

        t.handle_store(EventSource::Vllm, "v".into(), None, tokens.clone(), 4, None);
        t.handle_store(
            EventSource::Trtllm,
            "r".into(),
            None,
            tokens.clone(),
            4,
            None,
        );
        let kvbm_plh = reference_chain(&tokens, 4, None)[0];
        t.handle_kvbm_store(kvbm_plh, tokens, 4, None);
        t.drain_events();

        assert!(!t.handle_remove(EventSource::Vllm, "v"));
        assert!(!t.handle_remove(EventSource::Trtllm, "r"));
        assert_eq!(t.drain_events().len(), 0);
        assert_eq!(t.num_blocks(), 1);

        assert!(t.handle_kvbm_remove(kvbm_plh));
        assert_eq!(t.drain_events().len(), 1);
        assert_eq!(t.num_blocks(), 0);
    }

    #[test]
    fn tracker_remove_unknown_external_hash_is_noop_not_panic() {
        use tracing_test::traced_test;

        #[traced_test]
        fn inner() {
            let mut t = tracker();
            let ret = t.handle_remove(EventSource::Vllm, "no_such_hash");
            assert!(!ret);
            assert!(logs_contain("handle_remove"));
        }
        inner();
    }

    #[test]
    fn tracker_clear_all_wipes_state_and_enqueues_clear_event() {
        let mut t = tracker();
        t.handle_store(EventSource::Vllm, "a".into(), None, vec![1, 2], 2, None);
        // For Kvbm path we need a real PLH; reuse the canonical chain.
        let plh = reference_chain(&[3u32, 4], 2, None)[0];
        t.handle_kvbm_store(plh, vec![3, 4], 2, None);
        t.drain_events();
        t.handle_clear_all();
        assert_eq!(t.num_blocks(), 0);
        assert_eq!(t.num_external_mappings(), 0);
        let events = t.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ConsolidatedEvent::ClearAll));
    }

    #[test]
    fn tracker_cross_source_parent_resolution_uses_same_positional_hash() {
        let mut t = tracker();
        let parent_tokens = vec![1u32, 2, 3, 4];
        let child_tokens = vec![5u32, 6, 7, 8];

        t.handle_store(
            EventSource::Vllm,
            "vllm_a".into(),
            None,
            parent_tokens.clone(),
            4,
            None,
        );
        t.handle_store(
            EventSource::Trtllm,
            "trtllm_a".into(),
            None,
            parent_tokens.clone(),
            4,
            None,
        );

        let parent_plh = *t
            .external_to_seq
            .get(&(EventSource::Vllm, "vllm_a".to_string()))
            .unwrap();

        t.handle_store(
            EventSource::Vllm,
            "vllm_b".into(),
            Some("vllm_a".into()),
            child_tokens.clone(),
            4,
            None,
        );
        t.drain_events();

        let child_plh = *t
            .external_to_seq
            .get(&(EventSource::Vllm, "vllm_b".to_string()))
            .unwrap();

        // Across the parent/child edge, the child's parent_fragment must agree with the
        // parent's current fragment as seen from the child's position+0-mode.
        assert_eq!(
            child_plh.parent_hash_fragment(),
            parent_plh.parent_fragment_for_child_position(child_plh.position())
        );
        assert_eq!(t.num_blocks(), 2);
    }

    #[test]
    fn tracker_chain_matches_kv_hashing_reference_chain() {
        let mut t = tracker();
        let tokens: Vec<u32> = (0u32..12).collect();
        let block_size = 4;
        let reference = reference_chain(&tokens, block_size, None);

        let mut parent: Option<String> = None;
        for (i, chunk) in tokens.chunks(block_size).enumerate() {
            let ext = format!("b{i}");
            t.handle_store(
                EventSource::Vllm,
                ext.clone(),
                parent.clone(),
                chunk.to_vec(),
                block_size,
                None,
            );
            parent = Some(ext);
        }
        t.drain_events();

        for (i, expected) in reference.iter().enumerate() {
            let got = *t
                .external_to_seq
                .get(&(EventSource::Vllm, format!("b{i}")))
                .unwrap();
            assert_eq!(&got, expected, "chain mismatch at block {i}");
        }
    }

    #[test]
    fn tracker_lora_isolation_produces_distinct_chains() {
        let tokens: Vec<u32> = (100..108).collect();
        let chain_base = reference_chain(&tokens, 4, None);
        let chain_a = reference_chain(&tokens, 4, Some("adapter-a"));
        let chain_b = reference_chain(&tokens, 4, Some("adapter-b"));

        assert_ne!(chain_base[0], chain_a[0]);
        assert_ne!(chain_base[0], chain_b[0]);
        assert_ne!(chain_a[0], chain_b[0]);

        // And the tracker agrees under each lora.
        for lora in [None, Some("adapter-a"), Some("adapter-b")] {
            let mut t = tracker();
            let mut parent: Option<String> = None;
            for (i, chunk) in tokens.chunks(4).enumerate() {
                let ext = format!("b{i}");
                t.handle_store(
                    EventSource::Vllm,
                    ext.clone(),
                    parent.clone(),
                    chunk.to_vec(),
                    4,
                    lora.map(str::to_string),
                );
                parent = Some(ext);
            }
            let expected = reference_chain(&tokens, 4, lora);
            for (i, expected_plh) in expected.iter().enumerate() {
                let got = *t
                    .external_to_seq
                    .get(&(EventSource::Vllm, format!("b{i}")))
                    .unwrap();
                assert_eq!(&got, expected_plh, "lora={lora:?} block {i}");
            }
        }
    }

    #[test]
    fn tracker_same_tokens_different_position_produce_distinct_blocks() {
        let mut t = tracker();
        let tokens = vec![1u32, 2, 3, 4];

        t.handle_store(
            EventSource::Vllm,
            "p0".into(),
            None,
            tokens.clone(),
            4,
            None,
        );
        t.drain_events();

        t.handle_store(
            EventSource::Vllm,
            "p1".into(),
            Some("p0".into()),
            tokens.clone(),
            4,
            None,
        );

        assert_eq!(t.num_blocks(), 2);
        let events = t.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ConsolidatedEvent::Store { .. }));
    }

    #[test]
    fn tracker_external_to_seq_cleaned_after_full_remove() {
        let mut t = tracker();
        t.handle_store(
            EventSource::Vllm,
            "h1".into(),
            None,
            vec![1, 2, 3, 4],
            4,
            None,
        );
        t.drain_events();
        assert_eq!(t.num_external_mappings(), 1);
        t.handle_remove(EventSource::Vllm, "h1");
        assert_eq!(t.num_external_mappings(), 0);
        assert_eq!(t.num_blocks(), 0);
    }

    #[test]
    fn tracker_drain_events_returns_fifo_and_empties_queue() {
        let mut t = tracker();
        t.handle_store(EventSource::Vllm, "a".into(), None, vec![1, 2], 2, None);
        t.handle_store(EventSource::Vllm, "b".into(), None, vec![3, 4], 2, None);
        t.handle_remove(EventSource::Vllm, "a");

        let events = t.drain_events();
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], ConsolidatedEvent::Store { .. }));
        assert!(matches!(&events[1], ConsolidatedEvent::Store { .. }));
        assert!(matches!(&events[2], ConsolidatedEvent::Remove { .. }));

        assert_eq!(t.drain_events().len(), 0);
    }

    #[test]
    fn tracker_lora_and_tokens_round_trip_on_store_event() {
        let mut t = tracker();
        t.handle_store(
            EventSource::Vllm,
            "h1".into(),
            None,
            vec![10, 20, 30, 40],
            4,
            Some("my-adapter".to_string()),
        );
        let events = t.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ConsolidatedEvent::Store {
                token_ids,
                lora_name,
                block_size,
                ..
            } => {
                assert_eq!(token_ids, &[10u32, 20, 30, 40]);
                assert_eq!(lora_name.as_deref(), Some("my-adapter"));
                assert_eq!(*block_size, 4);
            }
            other => panic!("expected Store, got {:?}", other),
        }
    }
}

#[cfg(test)]
mod proptest_tracker {
    use super::*;
    use proptest::prelude::*;

    /// Catalogue of canonical PLHs for a deterministic chain of 12 blocks, used by
    /// KVBM-side replay to feed real PLHs into the tracker.
    fn catalogue() -> Vec<PositionalLineageHash> {
        // 12 blocks of 1 token each so block index == token value.
        let tokens: Vec<u32> = (0u32..12).collect();
        let request = Request::builder()
            .tokens(tokens)
            .build()
            .expect("valid request");
        request
            .into_blocks(1)
            .expect("into_blocks")
            .into_iter()
            .map(|b| b.plh)
            .collect()
    }

    #[derive(Debug, Clone)]
    enum Op {
        StoreVllm { idx: usize },
        StoreTrtllm { idx: usize },
        StoreKvbm { idx: usize },
        RemoveVllm { idx: usize },
        RemoveTrtllm { idx: usize },
        RemoveKvbm { idx: usize },
    }

    fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
        prop::collection::vec(
            prop_oneof![
                (0usize..12).prop_map(|idx| Op::StoreVllm { idx }),
                (0usize..12).prop_map(|idx| Op::StoreTrtllm { idx }),
                (0usize..12).prop_map(|idx| Op::StoreKvbm { idx }),
                (0usize..12).prop_map(|idx| Op::RemoveVllm { idx }),
                (0usize..12).prop_map(|idx| Op::RemoveTrtllm { idx }),
                (0usize..12).prop_map(|idx| Op::RemoveKvbm { idx }),
            ],
            0..50,
        )
    }

    fn replay_ops(ops: &[Op]) -> (Tracker, Vec<ConsolidatedEvent>) {
        let cat = catalogue();
        let mut t = Tracker::new(None);
        let mut vllm_stored: HashSet<usize> = HashSet::new();
        let mut trtllm_stored: HashSet<usize> = HashSet::new();
        let mut kvbm_stored: HashSet<usize> = HashSet::new();

        for op in ops {
            match op {
                Op::StoreVllm { idx } => {
                    let ext = format!("vllm_{idx}");
                    let tokens_u32 = vec![*idx as u32];
                    let parent_ext = if *idx > 0 && vllm_stored.contains(&(idx - 1)) {
                        Some(format!("vllm_{}", idx - 1))
                    } else {
                        None
                    };
                    t.handle_store(EventSource::Vllm, ext, parent_ext, tokens_u32, 1, None);
                    vllm_stored.insert(*idx);
                }
                Op::StoreTrtllm { idx } => {
                    let ext = format!("trt_{idx}");
                    let tokens_u32 = vec![*idx as u32];
                    let parent_ext = if *idx > 0 && trtllm_stored.contains(&(idx - 1)) {
                        Some(format!("trt_{}", idx - 1))
                    } else {
                        None
                    };
                    t.handle_store(EventSource::Trtllm, ext, parent_ext, tokens_u32, 1, None);
                    trtllm_stored.insert(*idx);
                }
                Op::StoreKvbm { idx } => {
                    for (ancestor, seq_hash) in cat.iter().enumerate().take(*idx + 1) {
                        if !kvbm_stored.contains(&ancestor) {
                            t.handle_kvbm_store(*seq_hash, vec![ancestor as u32], 1, None);
                            kvbm_stored.insert(ancestor);
                        }
                    }
                }
                Op::RemoveVllm { idx } => {
                    if vllm_stored.contains(idx) {
                        let ext = format!("vllm_{idx}");
                        t.handle_remove(EventSource::Vllm, &ext);
                        vllm_stored.remove(idx);
                    }
                }
                Op::RemoveTrtllm { idx } => {
                    if trtllm_stored.contains(idx) {
                        let ext = format!("trt_{idx}");
                        t.handle_remove(EventSource::Trtllm, &ext);
                        trtllm_stored.remove(idx);
                    }
                }
                Op::RemoveKvbm { idx } => {
                    if kvbm_stored.contains(idx) {
                        t.handle_kvbm_remove(cat[*idx]);
                        kvbm_stored.remove(idx);
                    }
                }
            }
        }

        let events = t.drain_events();
        (t, events)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 4096,
            ..ProptestConfig::default()
        })]

        #[test]
        fn dedup_invariant_random_interleaving(ops in ops_strategy()) {
            let (_, events) = replay_ops(&ops);
            let mut live: HashSet<PositionalLineageHash> = HashSet::new();
            for ev in &events {
                match ev {
                    ConsolidatedEvent::Store { seq_hash, .. } => {
                        prop_assert!(
                            !live.contains(seq_hash),
                            "duplicate Store (already live) for {:?}",
                            seq_hash
                        );
                        live.insert(*seq_hash);
                    }
                    ConsolidatedEvent::Remove { seq_hash, .. } => {
                        prop_assert!(
                            live.remove(seq_hash),
                            "Remove without prior Store for {:?}",
                            seq_hash
                        );
                    }
                    ConsolidatedEvent::ClearAll => {
                        live.clear();
                    }
                }
            }
        }

        #[test]
        fn parent_chain_integrity(ops in ops_strategy()) {
            let (_, events) = replay_ops(&ops);
            let mut ever_stored: HashSet<PositionalLineageHash> = HashSet::new();
            for ev in &events {
                if let ConsolidatedEvent::Store { seq_hash, .. } = ev {
                    ever_stored.insert(*seq_hash);
                }
            }
            for ev in &events {
                if let ConsolidatedEvent::Store { seq_hash, .. } = ev
                    && seq_hash.position() > 0
                {
                    // Some ancestor PLH must have been Stored at some point with matching
                    // current-fragment-for-child-position alignment.
                    let parent_frag = seq_hash.parent_hash_fragment();
                    let parent_pos = seq_hash.position() - 1;
                    let found = ever_stored.iter().any(|p| {
                        p.position() == parent_pos
                            && p.parent_fragment_for_child_position(seq_hash.position())
                                == parent_frag
                    });
                    prop_assert!(
                        found,
                        "Store for {:?} at pos {} references parent fragment {} not in ever_stored",
                        seq_hash,
                        seq_hash.position(),
                        parent_frag
                    );
                }
            }
        }
    }
}
