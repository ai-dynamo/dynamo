// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format types for vLLM ZMQ KV event streams.
//!
//! These types mirror the Python `msgspec`-defined structures emitted by vLLM
//! engines over ZMQ PUB sockets. They are independent of the dynamo runtime
//! and can be used by any crate that needs to decode the raw ZMQ payloads.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use rmp_serde as rmps;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{
    DpRank, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, Placement, PlacementEvent,
    StorageTier, WorkerWithDpRank,
};

mod convert;
mod dedup;
#[cfg(test)]
mod dedup_tests;
mod deserialize;
mod extra_keys;
mod filter;
mod mm_exclusion;
#[cfg(test)]
mod mm_exclusion_tests;
#[cfg(test)]
mod tests;
mod trtllm;
mod types;

pub use convert::{
    StoredBlockOptions, convert_event, create_stored_block_from_parts, create_stored_blocks,
    normalize_mm_placeholder_runs, normalize_mm_token_runs,
};
pub use dedup::{EventDedupFilter, EventDedupPolicy};
pub use extra_keys::{
    DYNAMO_CACHE_SALT_PREFIX, extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace,
    mark_mm_hash_for_extra_key, parse_mm_hash_from_extra_key,
};
pub use filter::KvCacheSpecKind;
pub use trtllm::IMAGE_ROUTING_UUID_PREFIX;
pub use types::{
    BlockHashValue, ExtraKeyItem, KvEventBatch, KvEventOwnership, KvTokenIds, Locality, RawKvEvent,
    WIDE_TOKEN_SENTINEL,
};

use filter::KvCacheEventMetadata;

pub fn decode_event_batch(payload: &[u8]) -> Result<KvEventBatch, rmps::decode::Error> {
    rmps::from_slice(payload)
}

fn storage_tier_of(raw: &RawKvEvent) -> Option<StorageTier> {
    match raw.medium() {
        None => Some(StorageTier::Device),
        Some(medium) => StorageTier::from_kv_medium(medium),
    }
}

/// A no-op placement event that keeps the event id sequence intact when a
/// store is dropped entirely.
fn empty_removal(event_id: u64, worker: WorkerWithDpRank, tier: StorageTier) -> PlacementEvent {
    PlacementEvent::new(
        Placement::local_worker(worker.worker_id, worker.dp_rank, tier),
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: Vec::new(),
            }),
            dp_rank: worker.dp_rank,
        },
    )
}

#[derive(Debug, Clone)]
pub struct ZmqEventNormalizer {
    kv_block_size: u32,
    /// Model's image placeholder token id, when MM-aware routing is active.
    /// Lets `convert_event` normalize vLLM BlockStored events to the canonical
    /// pad_value scheme. `None` for text-only models / non-MM deployments.
    image_token_id: Option<u32>,
    /// Model's video placeholder token id. When an event contains this token,
    /// image and video objects are normalized with modality-aware mapping.
    video_token_id: Option<u32>,
    warning_count: Arc<AtomicU32>,
    group_metadata: FxHashMap<(DpRank, u32), KvCacheGroupMetadata>,
    cache_namespaces: FxHashMap<(WorkerWithDpRank, u64), CacheNamespaceState>,
    /// Exclusion of multimodal values the request side cannot reproduce. See
    /// `mm_exclusion` and [`Self::with_engine_owned_mm_identity`].
    engine_mm: mm_exclusion::EngineMmState,
    /// Workers that have published at least one full-size block. Once a worker
    /// has proven its block size matches, a smaller `BlockStored` event is a
    /// partial tail (vLLM `--prefix-match-unit` below the block size), not a
    /// misconfiguration, and is filtered quietly instead of warned about.
    full_block_workers: FxHashSet<WorkerWithDpRank>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CacheNamespaceState {
    Namespaced(Arc<str>),
    Ambiguous,
}

#[derive(Debug, Clone, Copy)]
struct KvCacheGroupMetadata {
    kind: KvCacheSpecKind,
    sliding_window: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZmqEventFilterReason {
    IgnoredEvent,
    NonLocalLocality,
    UnknownMedium,
    UnsupportedOwnership,
    UnknownOwnership,
    AmbiguousCacheNamespace,
    NonMainAttentionKind,
    UnknownKind,
    NonMainAttentionGroup,
    UnlearnedGroupIdx,
    PartialBlock,
}

impl ZmqEventFilterReason {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::IgnoredEvent => "ignored_event",
            Self::NonLocalLocality => "non_local_locality",
            Self::UnknownMedium => "unknown_medium",
            Self::UnsupportedOwnership => "unsupported_ownership",
            Self::UnknownOwnership => "unknown_ownership",
            Self::AmbiguousCacheNamespace => "ambiguous_cache_namespace",
            Self::NonMainAttentionKind => "non_main_attention_kind",
            Self::UnknownKind => "unknown_kind",
            Self::NonMainAttentionGroup => "non_main_attention_group",
            Self::UnlearnedGroupIdx => "unlearned_group_idx",
            Self::PartialBlock => "partial_block",
        }
    }
}

impl ZmqEventNormalizer {
    pub fn new(kv_block_size: u32) -> Self {
        Self {
            kv_block_size,
            image_token_id: None,
            video_token_id: None,
            warning_count: Arc::new(AtomicU32::new(0)),
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
            engine_mm: mm_exclusion::EngineMmState::new(false),
            full_block_workers: FxHashSet::default(),
        }
    }

    pub fn with_warning_count(kv_block_size: u32, warning_count: Arc<AtomicU32>) -> Self {
        Self {
            kv_block_size,
            image_token_id: None,
            video_token_id: None,
            warning_count,
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
            engine_mm: mm_exclusion::EngineMmState::new(false),
            full_block_workers: FxHashSet::default(),
        }
    }

    /// Mark this source as deriving multimodal pad values from engine
    /// identities (sgl-project/sglang#43043): every event token at or above
    /// `MM_PAD_SHIFT_VALUE` is excluded with the blocks chained after it.
    /// Unflagged sources exclude only values that did not fit in `u32`.
    pub fn with_engine_owned_mm_identity(mut self, enabled: bool) -> Self {
        self.engine_mm = mm_exclusion::EngineMmState::new(enabled);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_engine_mm_excluded_capacity(mut self, excluded_capacity: usize) -> Self {
        self.engine_mm = mm_exclusion::EngineMmState::with_excluded_capacity(
            self.engine_mm.engine_owned(),
            excluded_capacity,
        );
        self
    }

    /// Set the model's image placeholder token id so vLLM BlockStored events
    /// get normalized to the canonical pad_value scheme. No-op for text-only
    /// models (leave unset).
    pub fn with_image_token_id(mut self, image_token_id: Option<u32>) -> Self {
        self.image_token_id = image_token_id;
        self
    }

    pub fn with_video_token_id(mut self, video_token_id: Option<u32>) -> Self {
        self.video_token_id = video_token_id;
        self
    }

    pub fn preprocess(&mut self, raw: RawKvEvent, worker: WorkerWithDpRank) -> Option<RawKvEvent> {
        self.preprocess_with_reason(raw, worker).ok()
    }

    pub fn preprocess_with_reason(
        &mut self,
        raw: RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<RawKvEvent, ZmqEventFilterReason> {
        match raw.ownership() {
            Ok(KvEventOwnership::Framework) => {}
            Ok(KvEventOwnership::Kvcr) => {
                return Err(ZmqEventFilterReason::UnsupportedOwnership);
            }
            Err(_) => return Err(ZmqEventFilterReason::UnknownOwnership),
        }
        self.preprocess_residency_with_reason(raw, worker)
    }

    /// Normalize a version-gated state-agent stream which may contain both
    /// framework and vLLM-enriched KVCR transitions.
    pub fn preprocess_residency_with_reason(
        &mut self,
        mut raw: RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<RawKvEvent, ZmqEventFilterReason> {
        let Ok(ownership) = raw.ownership() else {
            return Err(ZmqEventFilterReason::UnknownOwnership);
        };
        if raw.is_ignored() {
            return Err(ZmqEventFilterReason::IgnoredEvent);
        }

        // Non-local events are dropped by policy (no shared-index consumer yet).
        // Classify them here, before the lower-tier bypass, so the gate covers
        // every tier (including Disk/External). Otherwise the listener would
        // accept the event, burn a next_event_id, and only drop it in
        // conversion, leaving an id gap the event processor mistakes for an
        // engine drop (engines_dropped_events).
        if matches!(raw.locality(), Some(Locality::Remote | Locality::Unknown)) {
            return Err(ZmqEventFilterReason::NonLocalLocality);
        }

        // Classify by medium before touching normalizer state:
        //  - Device / HostPinned (GPU, CPU offload #10368) stay on the normalizer
        //    path so their salted namespaces still propagate.
        //  - Disk / External (STORAGE) are hash-only lower-tier events with no
        //    extra_keys/cache_namespace, so they must not mutate per-group
        //    metadata or the salted-namespace chain for framework events. KVCR
        //    still needs group classification before the conversion bypass.
        //  - Unrecognized media (e.g. vLLM 0.26.0 FS/OBJ) fail closed here so the
        //    listener records an intentional filter. Bypassing to conversion,
        //    which drops them, would instead accept the event, burn a
        //    next_event_id, and leave an id gap the event processor mistakes for
        //    an engine drop -- the same trap the locality gate above avoids.
        let mut is_lower_tier = false;
        if let Some(m) = raw.medium() {
            match StorageTier::from_kv_medium(m) {
                Some(StorageTier::Device | StorageTier::HostPinned) => {}
                Some(_) if ownership == KvEventOwnership::Framework => return Ok(raw),
                Some(_) => is_lower_tier = true,
                None => return Err(ZmqEventFilterReason::UnknownMedium),
            }
        }

        let metadata = raw.metadata();
        if matches!(raw, RawKvEvent::BlockStored { .. }) {
            self.learn_metadata(metadata, worker.dp_rank);
        }
        match self.filter_reason(metadata, worker.dp_rank) {
            // A removal names only its group, which a normalizer attached
            // after the store never learned. KVCR residency outlives the
            // attachment, so the removal goes through: hash-only, it applies
            // without a kind, and an owner with no such block is a no-op.
            Some(ZmqEventFilterReason::UnlearnedGroupIdx)
                if ownership == KvEventOwnership::Kvcr
                    && matches!(raw, RawKvEvent::BlockRemoved { .. }) => {}
            Some(reason) => return Err(reason),
            None => {}
        }
        // Only blocks that survived group filtering count toward a worker's
        // full-block history, so a filtered (non-main-attention) group cannot
        // suppress a size-mismatch warning for the routed group. A clear resets
        // the history: the engine behind this worker may come back with a
        // different block size.
        match &raw {
            RawKvEvent::BlockStored { block_size, .. } => {
                if let Some(reason) = self.filter_partial_block(*block_size, worker) {
                    return Err(reason);
                }
            }
            RawKvEvent::AllBlocksCleared { .. } => {
                self.full_block_workers.remove(&worker);
            }
            _ => {}
        }
        if !is_lower_tier {
            self.propagate_cache_namespace(&mut raw, worker)?;
        }
        Ok(raw)
    }

    pub fn normalize_preprocessed(
        &mut self,
        mut raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        // Unknown media fail closed in conversion; keep that behavior.
        let Some(tier) = storage_tier_of(&raw) else {
            return convert_event(
                raw,
                event_id,
                self.kv_block_size,
                worker,
                &self.warning_count,
                self.image_token_id,
                self.video_token_id,
            );
        };
        // Engine-owned pads replace the placeholder tokens that vLLM-style
        // normalization rewrites, so only legacy sources use those ids.
        let (image_token_id, video_token_id) = if self.engine_mm.engine_owned() {
            (None, None)
        } else {
            (self.image_token_id, self.video_token_id)
        };
        let convert = |raw| {
            convert_event(
                raw,
                event_id,
                self.kv_block_size,
                worker,
                &self.warning_count,
                image_token_id,
                video_token_id,
            )
        };
        match raw {
            RawKvEvent::BlockStored { .. } => {
                // An excluded store still yields an event so the event id is
                // not mistaken for an engine-side drop.
                if !self.engine_mm.prepare_store(&mut raw, worker, tier) {
                    return Some(empty_removal(event_id, worker, tier));
                }
                convert(raw)
            }
            RawKvEvent::BlockRemoved { .. } => {
                let mut event = convert(raw)?;
                self.engine_mm.filter_remove(&mut event, worker, tier);
                Some(event)
            }
            RawKvEvent::AllBlocksCleared { .. } => {
                self.engine_mm.clear(worker);
                convert(raw)
            }
            RawKvEvent::Ignored => None,
        }
    }

    pub fn normalize(
        &mut self,
        raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        let raw = self.preprocess(raw, worker)?;
        self.normalize_preprocessed(raw, event_id, worker)
    }

    /// Filter partial-tail `BlockStored` events from engines whose hash unit is
    /// smaller than the KV block size (vLLM `--prefix-match-unit`).
    ///
    /// vLLM publishes full blocks at the engine block size and the trailing
    /// partial block of a prompt at the hash-unit size. The router can only
    /// index full blocks, so conversion drops the partial one, but it does so
    /// with a block-size-mismatch warning that users read as a misconfiguration.
    /// Once a worker has published a full-size block its block size is known to
    /// match, so a smaller event from that worker is a partial tail and is
    /// filtered here at debug level. A mismatch on the first event from a worker
    /// is left to conversion, which still warns, so a real block-size
    /// misconfiguration stays visible.
    fn filter_partial_block(
        &mut self,
        block_size: usize,
        worker: WorkerWithDpRank,
    ) -> Option<ZmqEventFilterReason> {
        let kv_block_size = self.kv_block_size as usize;
        if block_size == kv_block_size {
            self.full_block_workers.insert(worker);
            return None;
        }
        if block_size < kv_block_size && self.full_block_workers.contains(&worker) {
            tracing::debug!(
                block_size,
                kv_block_size,
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                "Filtering partial-tail KV block event (hash unit smaller than block size)"
            );
            return Some(ZmqEventFilterReason::PartialBlock);
        }
        None
    }

    fn learn_metadata(&mut self, metadata: KvCacheEventMetadata, dp_rank: DpRank) {
        let (Some(group_idx), Some(kind)) = (metadata.group_idx, metadata.kv_cache_spec_kind)
        else {
            return;
        };

        self.group_metadata.insert(
            (dp_rank, group_idx),
            KvCacheGroupMetadata {
                kind,
                sliding_window: metadata.kv_cache_spec_sliding_window,
            },
        );
    }

    fn propagate_cache_namespace(
        &mut self,
        raw: &mut RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<(), ZmqEventFilterReason> {
        match raw {
            RawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                cache_namespace,
                ..
            } => {
                if cache_namespace.as_deref() == Some("") {
                    *cache_namespace = None;
                }
                let namespace = if let Some(namespace) = cache_namespace.as_deref() {
                    parent_block_hash
                        .as_ref()
                        .and_then(|parent| {
                            self.cache_namespaces.get(&(worker, (*parent).into_u64()))
                        })
                        .and_then(|state| match state {
                            CacheNamespaceState::Namespaced(parent_namespace)
                                if parent_namespace.as_ref() == namespace =>
                            {
                                Some(Arc::clone(parent_namespace))
                            }
                            _ => None,
                        })
                        .or_else(|| Some(Arc::from(namespace)))
                } else if let Some(parent) = parent_block_hash.as_ref() {
                    match self.cache_namespaces.get(&(worker, (*parent).into_u64())) {
                        Some(CacheNamespaceState::Namespaced(namespace)) => {
                            *cache_namespace = Some(namespace.to_string());
                            Some(Arc::clone(namespace))
                        }
                        Some(CacheNamespaceState::Ambiguous) => {
                            return Err(ZmqEventFilterReason::AmbiguousCacheNamespace);
                        }
                        None => {
                            // Deliberately preserve the unsalted interpretation when a
                            // listener joins in the middle of a chain. The vLLM wire
                            // format cannot distinguish an unknown salted parent from a
                            // genuinely unsalted one, and retaining every unsalted block
                            // here would duplicate the index on the event hot path. The
                            // backend still enforces its own cache isolation; this narrow
                            // fail-open case can only pollute the router overlap score.
                            None
                        }
                    }
                } else {
                    None
                };

                if let Some(namespace) = namespace {
                    let state = CacheNamespaceState::Namespaced(namespace);
                    for block_hash in block_hashes.iter() {
                        self.cache_namespaces
                            .entry((worker, (*block_hash).into_u64()))
                            .and_modify(|existing| {
                                if *existing != state {
                                    *existing = CacheNamespaceState::Ambiguous;
                                }
                            })
                            .or_insert_with(|| state.clone());
                    }
                } else {
                    // Do not retain every unsalted block. If this hash already
                    // belongs to a namespace, however, fail closed on future
                    // propagation because the external hash is now ambiguous.
                    for block_hash in block_hashes.iter() {
                        if let Some(existing) = self
                            .cache_namespaces
                            .get_mut(&(worker, (*block_hash).into_u64()))
                        {
                            *existing = CacheNamespaceState::Ambiguous;
                        }
                    }
                }
            }
            RawKvEvent::BlockRemoved { block_hashes, .. } => {
                for block_hash in block_hashes.iter() {
                    let key = (worker, (*block_hash).into_u64());
                    if !matches!(
                        self.cache_namespaces.get(&key),
                        Some(CacheNamespaceState::Ambiguous)
                    ) {
                        self.cache_namespaces.remove(&key);
                    }
                }
            }
            RawKvEvent::AllBlocksCleared { .. } => {
                self.cache_namespaces
                    .retain(|(known_worker, _), _| *known_worker != worker);
            }
            RawKvEvent::Ignored => {}
        }
        Ok(())
    }

    fn filter_reason(
        &self,
        metadata: KvCacheEventMetadata,
        dp_rank: DpRank,
    ) -> Option<ZmqEventFilterReason> {
        if let Some(kind) = metadata.kv_cache_spec_kind {
            if kind.is_main_attention() {
                return None;
            }
            if kind == KvCacheSpecKind::Unknown {
                return Some(ZmqEventFilterReason::UnknownKind);
            }
            return Some(ZmqEventFilterReason::NonMainAttentionKind);
        }

        let group_idx = metadata.group_idx?;

        if let Some(metadata) = self.group_metadata.get(&(dp_rank, group_idx)) {
            let _sliding_window = metadata.sliding_window;
            if metadata.kind.is_main_attention() {
                return None;
            }
            if metadata.kind == KvCacheSpecKind::Unknown {
                return Some(ZmqEventFilterReason::UnknownKind);
            }
            return Some(ZmqEventFilterReason::NonMainAttentionGroup);
        }

        if group_idx == 0 {
            None
        } else {
            Some(ZmqEventFilterReason::UnlearnedGroupIdx)
        }
    }
}
