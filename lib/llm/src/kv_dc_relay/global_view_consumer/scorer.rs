// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fresh, fenced in-memory CKF overlap scores for routing DGDs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dynamo_kv_router::global_view::PoolId as RoutingPoolId;
use dynamo_kv_router::global_view::overlap::KvOverlapScorer;
use dynamo_kv_router::identity::PoolId as CkfPoolId;
use dynamo_kv_router::indexer::cuckoo::GlobalCkfIndexer;
use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
use parking_lot::RwLock;

#[derive(Clone)]
pub(super) struct ReadyCkf {
    pub(super) model: String,
    pub(super) producer_pool_id: CkfPoolId,
    pub(super) block_size: u32,
    pub(super) is_eagle: bool,
    pub(super) indexer: Arc<GlobalCkfIndexer>,
    pub(super) received_at: Instant,
}

#[derive(Default)]
struct SessionState {
    generation: u64,
    ready: Option<ReadyCkf>,
}

/// One in-memory overlap source per routing DGD. New exact-producer sessions
/// fence older sessions; a stale session cannot republish or withdraw new data.
pub struct RelayCkfOverlapStore {
    max_age: Duration,
    sessions: RwLock<HashMap<RoutingPoolId, SessionState>>,
}

impl RelayCkfOverlapStore {
    pub fn new(max_age: Duration) -> Self {
        Self {
            max_age,
            sessions: RwLock::new(HashMap::new()),
        }
    }

    pub(super) fn begin(&self, pool_id: &RoutingPoolId) -> Result<u64> {
        let mut sessions = self.sessions.write();
        let session = sessions.entry(pool_id.clone()).or_default();
        session.generation = session
            .generation
            .checked_add(1)
            .context("CKF session generation exhausted")?;
        session.ready = None;
        Ok(session.generation)
    }

    pub(super) fn publish(
        &self,
        pool_id: &RoutingPoolId,
        generation: u64,
        ready: ReadyCkf,
    ) -> bool {
        let mut sessions = self.sessions.write();
        let Some(session) = sessions.get_mut(pool_id) else {
            return false;
        };
        if session.generation != generation {
            return false;
        }
        session.ready = Some(ready);
        true
    }

    pub(super) fn touch(&self, pool_id: &RoutingPoolId, generation: u64) -> bool {
        let mut sessions = self.sessions.write();
        let Some(session) = sessions.get_mut(pool_id) else {
            return false;
        };
        if session.generation != generation {
            return false;
        }
        let Some(ready) = session.ready.as_mut() else {
            return false;
        };
        ready.received_at = Instant::now();
        true
    }

    pub(super) fn withdraw(&self, pool_id: &RoutingPoolId, generation: u64) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(pool_id)
            && session.generation == generation
        {
            session.ready = None;
        }
    }
}

impl KvOverlapScorer for RelayCkfOverlapStore {
    fn estimate_matched_prefix_tokens(
        &self,
        pool_id: &RoutingPoolId,
        model: &str,
        token_ids: &[u32],
    ) -> Option<u64> {
        let ready = self.sessions.read().get(pool_id)?.ready.clone()?;
        if ready.model != model || ready.received_at.elapsed() > self.max_age {
            return None;
        }
        let hashes = compute_block_hash_for_seq(
            token_ids,
            ready.block_size,
            BlockHashOptions {
                is_eagle: Some(ready.is_eagle),
                ..Default::default()
            },
        );
        let matches = ready.indexer.find_prefix_matches(&hashes).ok()?;
        let depth = matches
            .lanes()
            .iter()
            .flatten()
            .find(|lane| lane.pool_id() == ready.producer_pool_id)?
            .prefix_depth();
        Some(u64::from(depth) * u64::from(ready.block_size))
    }
}
