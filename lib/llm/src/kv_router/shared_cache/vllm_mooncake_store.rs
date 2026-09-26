// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::{
    SharedCacheQuery, SharedKvCache,
    indexer::KvRouterError,
    protocols::{KvCacheStoreData, LocalBlockHash, SharedCacheHits, WorkerWithDpRank},
};
use futures::StreamExt;
use lru::LruCache;
use parking_lot::Mutex;
use rustc_hash::{FxBuildHasher, FxHashMap};
use tokio_util::sync::CancellationToken;

pub(crate) use super::mooncake_store_contract::ValidatedContract;
use super::{
    MOONCAKE_EVENT_RECONNECT_DELAY, MooncakeObjectEvent, connect_sub_socket,
    mooncake_store_contract::GroupKind, multipart_message, parse_mooncake_event_frames,
    record_subscriber_error,
};

const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MAX_BATCH_EVENTS: usize = 4096;
// At these caps, measured map allocations on aarch64/rustc 1.96 were 100.5 MiB live
// and 119 MiB during growth (LRU edge map plus reverse map, digest map, and
// one-medium objects). This excludes allocator metadata, runtime descriptors,
// input/query buffers, and RSS. Reaching a cap evicts or discards evidence and
// the hint keeps running; it never disables itself.
const MAX_LEARNED_EDGES: usize = 262_144;
const MAX_DIGEST_IDENTITIES: usize = 262_144;
const MAX_OBJECT_MEMBERSHIPS: usize = 524_288;

#[derive(Clone, Copy)]
struct Limits {
    edges: usize,
    digests: usize,
    memberships: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            edges: MAX_LEARNED_EDGES,
            digests: MAX_DIGEST_IDENTITIES,
            memberships: MAX_OBJECT_MEMBERSHIPS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Edge {
    parent: Option<u64>,
    local: LocalBlockHash,
}

#[derive(Clone, Copy, Debug)]
struct LearnedEdge {
    child: u64,
    conflicted: bool,
}

/// One unambiguous full digest behind a GPU event hash, kept only while at
/// least one of its objects is resident.
#[derive(Clone, Copy, Debug)]
struct DigestIdentity {
    digest: [u8; 32],
    objects: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ObjectIdentity {
    digest: [u8; 32],
    prefix: u16,
}

struct State {
    contract: Option<Arc<ValidatedContract>>,
    allowed_sources: HashSet<WorkerWithDpRank>,
    generation: u64,
    sequence: Option<u64>,
    /// Set once the event subscriber has stopped; the cache is retired.
    disabled: bool,
    /// Least recently used order; `limits.edges` evicts the coldest chain link.
    edges: LruCache<Edge, LearnedEdge, FxBuildHasher>,
    reverse_edges: FxHashMap<u64, Edge>,
    digests: HashMap<u64, DigestIdentity>,
    /// GPU event hashes seen with two different full digests. These survive
    /// residency clears because either digest may still be resident unseen.
    ambiguous: HashSet<u64>,
    objects: HashMap<ObjectIdentity, u8>,
    memberships: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            contract: None,
            allowed_sources: HashSet::new(),
            generation: 0,
            sequence: None,
            disabled: false,
            edges: LruCache::unbounded_with_hasher(FxBuildHasher),
            reverse_edges: FxHashMap::default(),
            digests: HashMap::new(),
            ambiguous: HashSet::new(),
            objects: HashMap::new(),
            memberships: 0,
        }
    }
}

impl State {
    /// Forget which objects are resident. Digest identities go with them:
    /// every resident object re-announces its digest, so nothing is lost.
    fn clear_residency(&mut self) {
        self.objects.clear();
        self.memberships = 0;
        self.digests.clear();
    }

    fn disable(&mut self, reason: &'static str) {
        if !self.disabled {
            tracing::warn!(reason, "Mooncake Store hints disabled");
            record_subscriber_error();
        }
        self.disabled = true;
        self.clear_residency();
    }

    fn complete(&self, contract: &ValidatedContract, group: usize, external: u64) -> bool {
        let Some(identity) = self.digests.get(&external) else {
            return false;
        };
        contract.group_prefixes[group].iter().all(|&prefix| {
            self.objects.contains_key(&ObjectIdentity {
                digest: identity.digest,
                prefix,
            })
        })
    }

    fn evict_coldest_edge(&mut self) {
        if let Some((_, evicted)) = self.edges.pop_lru() {
            self.reverse_edges.remove(&evicted.child);
        }
    }

    /// Drop every resident object of `digest`; the identity behind `low64`
    /// is no longer trustworthy.
    fn mark_ambiguous(&mut self, contract: &ValidatedContract, low64: u64, digest: [u8; 32]) {
        self.digests.remove(&low64);
        for &prefix in contract.prefixes.values() {
            if let Some(mediums) = self.objects.remove(&ObjectIdentity { digest, prefix }) {
                self.memberships -= mediums.count_ones() as usize;
            }
        }
        self.ambiguous.insert(low64);
    }
}

pub(crate) struct MooncakeStoreCache {
    state: Mutex<State>,
    pending: Box<dyn Fn() -> bool + Send + Sync>,
    limits: Limits,
}

impl MooncakeStoreCache {
    pub(crate) fn new(pending: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self {
            state: Mutex::new(State::default()),
            pending: Box::new(pending),
            limits: Limits::default(),
        }
    }

    pub(crate) fn update_contract(
        &self,
        contract: Option<ValidatedContract>,
        allowed_sources: HashSet<WorkerWithDpRank>,
    ) {
        let mut state = self.state.lock();
        let generation = state
            .generation
            .checked_add(1)
            .expect("Mooncake Store generation exhausted");
        if state.contract.as_deref() != contract.as_ref() {
            *state = State {
                contract: contract.map(Arc::new),
                generation,
                allowed_sources,
                ..State::default()
            };
        } else {
            state.allowed_sources = allowed_sources;
        }
    }

    pub(crate) fn spawn_subscriber(
        self: &Arc<Self>,
        endpoint: String,
        cancellation: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let cache = Arc::clone(self);
        tokio::spawn(async move {
            cache.run_subscriber(endpoint, cancellation).await;
            cache.state.lock().disable("router subscriber stopped");
        })
    }

    #[cfg(test)]
    pub(crate) fn apply_test_frames(&self, frames: &[Vec<u8>]) {
        let generation = self.state.lock().generation;
        self.apply_frames(generation, frames);
    }

    fn invalidate_residency(&self) {
        self.state.lock().clear_residency();
    }

    fn apply_frames(&self, generation: u64, frames: &[Vec<u8>]) {
        let decoded = if frames
            .iter()
            .try_fold(0usize, |sum, frame| sum.checked_add(frame.len()))
            .is_none_or(|size| size > MAX_FRAME_BYTES)
        {
            Err(anyhow::anyhow!("Mooncake Store batch exceeds input limit"))
        } else {
            parse_mooncake_event_frames(frames)
        };
        match decoded {
            Ok((sequence, events)) if events.len() <= MAX_BATCH_EVENTS => {
                self.apply_batch(generation, sequence, events);
            }
            _ => {
                let mut state = self.state.lock();
                if generation == state.generation {
                    state.clear_residency();
                    record_subscriber_error();
                }
            }
        }
    }

    fn apply_batch(&self, generation: u64, sequence: u64, events: Vec<MooncakeObjectEvent>) {
        let pending = (self.pending)();
        let mut state = self.state.lock();
        if generation != state.generation {
            return;
        }
        if pending || (self.pending)() {
            state.clear_residency();
            return;
        }
        if state.sequence == Some(sequence) {
            return;
        }
        if state
            .sequence
            .is_some_and(|previous| previous.checked_add(1) != Some(sequence))
        {
            state.clear_residency();
            record_subscriber_error();
        }
        state.sequence = Some(sequence);
        let Some(contract) = state.contract.clone().filter(|_| !state.disabled) else {
            return;
        };
        if events.len() > MAX_BATCH_EVENTS {
            state.clear_residency();
            record_subscriber_error();
            return;
        }
        for event in events {
            if !event.tenant_id.is_empty() && event.tenant_id != "default" {
                continue;
            }
            if matches!(event.event_type.as_str(), "cleared" | "AllBlocksCleared") {
                state.clear_residency();
                continue;
            }
            let Some(key) = event.object_key.as_deref() else {
                state.clear_residency();
                record_subscriber_error();
                return;
            };
            if contract.prefixes.contains_key(key) {
                state.clear_residency();
                record_subscriber_error();
                return;
            }
            let Some((prefix, hash)) = key.rsplit_once('@') else {
                continue;
            };
            let Some(&prefix) = contract.prefixes.get(prefix) else {
                continue;
            };
            let Some(digest) = decode_digest(hash) else {
                state.clear_residency();
                record_subscriber_error();
                return;
            };
            let medium = match event.medium.as_deref() {
                Some("cpu") => 1,
                Some("disk") => 2,
                _ => {
                    state.clear_residency();
                    record_subscriber_error();
                    return;
                }
            };
            let stored = match event.event_type.as_str() {
                "stored" | "BlockStored" => true,
                "removed" | "BlockRemoved" => false,
                _ => {
                    state.clear_residency();
                    record_subscriber_error();
                    return;
                }
            };
            let low64 = u64::from_be_bytes(digest[24..].try_into().unwrap());
            if state.ambiguous.contains(&low64) {
                continue;
            }
            if let Some(identity) = state.digests.get(&low64)
                && identity.digest != digest
            {
                let conflicting = identity.digest;
                state.mark_ambiguous(&contract, low64, conflicting);
                if state.ambiguous.len() > self.limits.digests {
                    state.ambiguous.clear();
                }
                record_subscriber_error();
                tracing::warn!("Conflicting Mooncake Store full digests share a GPU event hash");
                continue;
            }
            let key = ObjectIdentity { digest, prefix };
            let mut previous = state.objects.get(&key).copied().unwrap_or(0);
            if stored {
                if previous & medium != 0 {
                    continue;
                }
                let new_digest = !state.digests.contains_key(&low64);
                if state.memberships >= self.limits.memberships
                    || (new_digest && state.digests.len() >= self.limits.digests)
                {
                    // Residency evidence is discarded and rebuilt from later
                    // events; the current event seeds the rebuild.
                    state.clear_residency();
                    record_subscriber_error();
                    previous = 0;
                }
                let identity = state
                    .digests
                    .entry(low64)
                    .or_insert(DigestIdentity { digest, objects: 0 });
                if previous == 0 {
                    identity.objects += 1;
                }
                state.objects.insert(key, previous | medium);
                state.memberships += 1;
            } else if previous & medium != 0 {
                let remaining = previous & !medium;
                if remaining == 0 {
                    state.objects.remove(&key);
                    if let Some(identity) = state.digests.get_mut(&low64) {
                        identity.objects -= 1;
                        if identity.objects == 0 {
                            state.digests.remove(&low64);
                        }
                    }
                } else {
                    state.objects.insert(key, remaining);
                }
                state.memberships -= 1;
            }
        }
    }

    fn lookup(&self, query: SharedCacheQuery<'_>) -> SharedCacheHits {
        if (self.pending)() || !query.shared_cache_eligible {
            return SharedCacheHits::default();
        }
        let mut guard = self.state.lock();
        if (self.pending)() || guard.disabled {
            return SharedCacheHits::default();
        }
        let state = &mut *guard;
        let Some(contract) = state.contract.as_deref() else {
            return SharedCacheHits::default();
        };
        let b = contract.main_event_block_size() as usize;
        if query.block_size as usize != b || query.tokens.is_empty() {
            return SharedCacheHits::default();
        }
        let count = query.block_hashes.len().min((query.tokens.len() - 1) / b);
        let mut chain = Vec::with_capacity(count.min(self.limits.edges));
        let mut parent = None;
        // `get` promotes each walked link so requested prefixes outlive
        // chains nobody asks for.
        for &local in query.block_hashes.iter().take(count.min(self.limits.edges)) {
            let Some(edge) = state
                .edges
                .get(&Edge { parent, local })
                .filter(|e| !e.conflicted)
            else {
                break;
            };
            parent = Some(edge.child);
            chain.push(edge.child);
        }
        let state = &*state;
        let alignment = contract.alignment as usize;
        let candidates = chain.len() * b / alignment;
        if candidates == 0 {
            return SharedCacheHits::default();
        }
        let mut ready = vec![true; candidates];
        for (group_id, group) in contract.groups().iter().enumerate() {
            let span = group.block_size as usize;
            let complete_at = |end: usize| {
                end > 0
                    && end.is_multiple_of(b)
                    && chain
                        .get(end / b - 1)
                        .is_some_and(|&hash| state.complete(contract, group_id, hash))
            };
            match group.kind {
                GroupKind::FullAttention => {
                    let mut next = span;
                    let mut complete = true;
                    for (idx, candidate) in ready.iter_mut().enumerate() {
                        let end = (idx + 1) * alignment;
                        while next <= end {
                            complete &= complete_at(next);
                            next += span;
                        }
                        *candidate &= complete;
                    }
                }
                GroupKind::SlidingWindow => {
                    let window = (group.sliding_window.unwrap() as usize - 1).div_ceil(span);
                    if window == 1 {
                        for (idx, candidate) in ready.iter_mut().enumerate() {
                            *candidate &= complete_at((idx + 1) * alignment);
                        }
                    } else {
                        let chunks = candidates * alignment / span;
                        let mut missing_prefix = Vec::with_capacity(chunks + 1);
                        missing_prefix.push(0usize);
                        for idx in 1..=chunks {
                            missing_prefix.push(
                                missing_prefix[idx - 1] + usize::from(!complete_at(idx * span)),
                            );
                        }
                        for (idx, candidate) in ready.iter_mut().enumerate() {
                            let end = (idx + 1) * alignment / span;
                            *candidate &=
                                missing_prefix[end] == missing_prefix[end.saturating_sub(window)];
                        }
                    }
                }
                GroupKind::Mamba => {
                    for (idx, candidate) in ready.iter_mut().enumerate() {
                        *candidate &= complete_at((idx + 1) * alignment);
                    }
                }
            }
        }
        if (self.pending)() {
            return SharedCacheHits::default();
        }
        ready
            .iter()
            .rposition(|hit| *hit)
            .map_or_else(SharedCacheHits::default, |idx| {
                SharedCacheHits::from_ranges(Vec::from_iter(std::iter::once(
                    0..((idx + 1) * alignment / b) as u32,
                )))
            })
    }

    fn apply_message(
        &self,
        generation: u64,
        message: Option<Result<tmq::Multipart, tmq::TmqError>>,
    ) -> bool {
        match message {
            Some(Ok(frames)) => {
                self.apply_frames(generation, &multipart_message(frames));
                true
            }
            Some(Err(_)) | None => {
                self.invalidate_residency();
                record_subscriber_error();
                false
            }
        }
    }

    async fn run_subscriber(&self, endpoint: String, cancellation: CancellationToken) {
        loop {
            self.invalidate_residency();
            let connected = tokio::select! {
                _ = cancellation.cancelled() => return,
                result = connect_sub_socket(&endpoint, None) => result,
            };
            match connected {
                Ok(mut socket) => loop {
                    let generation = self.state.lock().generation;
                    tokio::select! {
                        _ = cancellation.cancelled() => return,
                        message = socket.next() => {
                            if !self.apply_message(generation, message) {
                                break;
                            }
                        }
                    }
                },
                Err(error) => {
                    self.invalidate_residency();
                    record_subscriber_error();
                    tracing::warn!(%error, "Failed to connect to Mooncake Store KV events; retrying");
                }
            }
            tokio::select! {
                _ = cancellation.cancelled() => return,
                _ = tokio::time::sleep(MOONCAKE_EVENT_RECONNECT_DELAY) => {}
            }
        }
    }
}

#[async_trait]
impl SharedKvCache for MooncakeStoreCache {
    async fn check_blocks(
        &self,
        query: SharedCacheQuery<'_>,
    ) -> Result<SharedCacheHits, KvRouterError> {
        Ok(self.lookup(query))
    }

    fn observe_stored(&self, source: WorkerWithDpRank, data: &KvCacheStoreData) {
        if (self.pending)()
            || !data.shared_cache_eligible
            || data.blocks.iter().any(|b| b.mm_extra_info.is_some())
        {
            return;
        }
        if data.parent_hash.is_none() && data.start_position.is_some_and(|p| p != 0)
            || data.parent_hash.is_some() && data.start_position == Some(0)
        {
            return;
        }
        let mut state = self.state.lock();
        if (self.pending)()
            || state.disabled
            || state.contract.is_none()
            || !state.allowed_sources.contains(&source)
        {
            return;
        }
        let mut parent = data.parent_hash.map(|h| h.0);
        for block in &data.blocks {
            let key = Edge {
                parent,
                local: block.tokens_hash,
            };
            let child = block.block_hash.0;
            if let Some(existing) = state.edges.get_mut(&key) {
                if existing.child != child {
                    existing.conflicted = true;
                    record_subscriber_error();
                    tracing::warn!(?source, "Conflicting Mooncake Store learned hash edge");
                    return;
                }
                if existing.conflicted {
                    return;
                }
            } else {
                if let Some(&existing_key) = state.reverse_edges.get(&child) {
                    // The same external hash already names another local chain.
                    // Neither chain can be trusted past this block, so the known
                    // one is marked and the new one is not learned.
                    if let Some(existing) = state.edges.peek_mut(&existing_key)
                        && !existing.conflicted
                    {
                        existing.conflicted = true;
                        record_subscriber_error();
                        tracing::warn!(
                            ?source,
                            "External Mooncake Store event hash names different local chains"
                        );
                    }
                    return;
                }
                if state.edges.len() >= self.limits.edges {
                    state.evict_coldest_edge();
                }
                state.edges.put(
                    key,
                    LearnedEdge {
                        child,
                        conflicted: false,
                    },
                );
                state.reverse_edges.insert(child, key);
            }
            parent = Some(child);
        }
    }
}

fn decode_digest(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut digest = [0u8; 32];
    for (byte, pair) in digest
        .iter_mut()
        .zip(value.as_bytes().as_chunks::<2>().0.iter())
    {
        let decode = |c| match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        };
        *byte = decode(pair[0])? * 16 + decode(pair[1])?;
    }
    Some(digest)
}

#[cfg(test)]
pub(crate) fn test_runtime_config(
    block_size: u32,
) -> crate::local_model::runtime_config::ModelRuntimeConfig {
    super::mooncake_store_contract::test_contract(
        &[(GroupKind::FullAttention, block_size, None)],
        block_size,
        2,
    )
    .runtime_config()
}

#[cfg(test)]
mod tests;
