// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bench")]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::kv_hints::KvTransferCandidates;
use crate::prelude::*;
use dynamo_tokens::SequenceHash;
use rustc_hash::{FxHashMap, FxHashSet};

#[cfg(feature = "bench")]
use super::{EventCompletionBuffer, EventCompletionWriter, ObservationSeal};

/// Errors that can occur in the KV Router.
#[derive(Debug, thiserror::Error)]
pub enum KvRouterError {
    #[error("Block not found")]
    BlockNotFound,

    #[error("Indexer is offline")]
    IndexerOffline,

    #[error("Indexer dropped the request")]
    IndexerDroppedRequest,

    #[error("Prune operation failed: {0}")]
    PruneFailed(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Shared structural anchor used by branch-sharded routing when a routed
/// subtree starts on a different shard from its parent prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AnchorRef {
    pub anchor_id: ExternalSequenceBlockHash,
    pub anchor_local_hash: LocalBlockHash,
    pub anchor_depth: usize,
}

/// Worker task payload that installs an [`AnchorRef`] into a shard-local
/// backend before dependent suffix events are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AnchorTask {
    pub anchor_id: ExternalSequenceBlockHash,
    pub anchor_local_hash: LocalBlockHash,
    pub anchor_depth: usize,
}

/// Wire-friendly overlap scores for JSON serialization.
/// `OverlapScores` uses `FxHashMap<WorkerWithDpRank, _>` which can't be
/// serialized as JSON (struct keys aren't valid JSON map keys), so we flatten
/// to vecs of tuples for the wire protocol.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireOverlapScores {
    pub scores: Vec<(WorkerWithDpRank, u32)>,
    pub frequencies: Vec<usize>,
}

impl From<OverlapScores> for WireOverlapScores {
    fn from(s: OverlapScores) -> Self {
        Self {
            scores: s.scores.into_iter().collect(),
            frequencies: s.frequencies,
        }
    }
}

impl From<WireOverlapScores> for OverlapScores {
    fn from(w: WireOverlapScores) -> Self {
        Self {
            scores: w.scores.into_iter().collect(),
            frequencies: w.frequencies,
        }
    }
}

/// Wire-friendly lower-tier match payload for JSON serialization.
///
/// Mirrors `LowerTierMatchDetails.hits`. `next_continuations` and KV transfer
/// candidates are server-side intermediate state and are not carried over the wire.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireLowerTierMatchDetails {
    pub hits: Vec<(WorkerWithDpRank, usize)>,
}

/// Wire-friendly tiered match payload: device overlap plus per-tier hits.
///
/// Lower tiers are a `Vec<(StorageTier, _)>` rather than a map so we never
/// depend on `StorageTier` being a JSON-legal map key. Each `StorageTier` is
/// expected to appear at most once; the inbound conversion warns and keeps the
/// last entry if duplicates are observed.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct WireTieredMatchDetails {
    pub device: WireOverlapScores,
    pub lower_tier: Vec<(StorageTier, WireLowerTierMatchDetails)>,
    /// Confirmed device resume caps, including zero for known hybrid workers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hybrid_resume_caps: Vec<(WorkerWithDpRank, u32)>,
}

/// Precomputed hashes for recording a route-time indexer update.
#[derive(Debug, Clone)]
pub struct RoutingDecisionHashes {
    pub local_hashes: Vec<LocalBlockHash>,
    pub sequence_hashes: Vec<SequenceHash>,
}

impl RoutingDecisionHashes {
    pub fn from_local_hashes(local_hashes: Vec<LocalBlockHash>) -> Self {
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        Self {
            local_hashes,
            sequence_hashes,
        }
    }
}

/// Confirmed device caps independent of mutable overlap scores.
///
/// A uniform result owns the exact query worker set, allowing the tree's set to
/// move here without allocating and hashing another map. Zero remains a known
/// cap; a worker outside the set has no cap. Nonuniform updates materialize the
/// explicit representation and preserve every previously confirmed entry.
#[derive(Debug, Clone)]
pub enum HybridResumeCaps {
    Explicit(FxHashMap<WorkerWithDpRank, u32>),
    UniformConfirmed {
        workers: FxHashSet<WorkerWithDpRank>,
        cap: u32,
    },
    SparseConfirmed(Box<SparseHybridResumeCaps>),
}

/// Immutable caps for one exact scope. Workers without an override use the
/// default, including zero; workers outside the scope remain unknown.
#[derive(Debug, Clone)]
pub struct SparseHybridResumeCaps {
    workers: FxHashSet<WorkerWithDpRank>,
    default_cap: u32,
    overrides: Vec<(WorkerWithDpRank, u32)>,
}

impl SparseHybridResumeCaps {
    fn cap_for_scoped_worker(&self, worker: &WorkerWithDpRank) -> &u32 {
        self.overrides
            .binary_search_by_key(worker, |entry| entry.0)
            .ok()
            .map(|position| &self.overrides[position].1)
            .unwrap_or(&self.default_cap)
    }
}

// Keep the ordinary result carrier unchanged; only sparse results own a box.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<HybridResumeCaps>() == 40);

impl Default for HybridResumeCaps {
    fn default() -> Self {
        Self::Explicit(FxHashMap::default())
    }
}

impl HybridResumeCaps {
    pub fn uniform_confirmed(workers: FxHashSet<WorkerWithDpRank>, cap: u32) -> Self {
        Self::UniformConfirmed { workers, cap }
    }

    /// Canonicalize scoped overrides; duplicate workers keep their last value.
    /// Entries outside the exact scope and entries equal to the default need
    /// no storage. Neither representation derives caps from mutable scores.
    pub fn sparse_confirmed(
        workers: FxHashSet<WorkerWithDpRank>,
        default_cap: u32,
        mut overrides: Vec<(WorkerWithDpRank, u32)>,
    ) -> Self {
        overrides.retain(|(worker, _)| workers.contains(worker));
        // Stable sorting preserves last-value semantics for duplicate keys.
        overrides.sort_by_key(|entry| entry.0);
        overrides.dedup_by(|current, previous| {
            if current.0 == previous.0 {
                previous.1 = current.1;
                true
            } else {
                false
            }
        });
        overrides.retain(|(_, cap)| *cap != default_cap);
        if overrides.is_empty() {
            return Self::uniform_confirmed(workers, default_cap);
        }
        Self::SparseConfirmed(Box::new(SparseHybridResumeCaps {
            workers,
            default_cap,
            overrides,
        }))
    }

    pub fn get(&self, worker: &WorkerWithDpRank) -> Option<&u32> {
        match self {
            Self::Explicit(caps) => caps.get(worker),
            Self::UniformConfirmed { workers, cap } => workers.contains(worker).then_some(cap),
            Self::SparseConfirmed(sparse) => sparse
                .workers
                .contains(worker)
                .then(|| sparse.cap_for_scoped_worker(worker)),
        }
    }

    pub fn contains_key(&self, worker: &WorkerWithDpRank) -> bool {
        self.get(worker).is_some()
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Explicit(caps) => caps.len(),
            Self::UniformConfirmed { workers, .. } => workers.len(),
            Self::SparseConfirmed(sparse) => sparse.workers.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> HybridResumeCapsIter<'_> {
        match self {
            Self::Explicit(caps) => HybridResumeCapsIter::Explicit(caps.iter()),
            Self::UniformConfirmed { workers, cap } => HybridResumeCapsIter::Uniform {
                workers: workers.iter(),
                cap,
            },
            Self::SparseConfirmed(sparse) => HybridResumeCapsIter::Sparse {
                workers: sparse.workers.iter(),
                descriptor: sparse,
            },
        }
    }

    pub fn insert(&mut self, worker: WorkerWithDpRank, cap: u32) -> Option<u32> {
        self.materialize().insert(worker, cap)
    }

    pub fn reserve(&mut self, additional: usize) {
        if additional > 0 {
            self.materialize().reserve(additional);
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Explicit(caps) => caps.clear(),
            Self::UniformConfirmed { workers, .. } => workers.clear(),
            Self::SparseConfirmed(sparse) => {
                sparse.workers.clear();
                sparse.overrides.clear();
            }
        }
    }

    fn materialize(&mut self) -> &mut FxHashMap<WorkerWithDpRank, u32> {
        if let Self::UniformConfirmed { workers, cap } = self {
            let cap = *cap;
            let entries = std::mem::take(workers)
                .into_iter()
                .map(|worker| (worker, cap))
                .collect();
            *self = Self::Explicit(entries);
        }
        if matches!(self, Self::SparseConfirmed(_)) {
            let Self::SparseConfirmed(sparse) = std::mem::take(self) else {
                unreachable!("a sparse representation was selected")
            };
            let SparseHybridResumeCaps {
                workers,
                default_cap,
                overrides,
            } = *sparse;
            *self = Self::Explicit(
                workers
                    .into_iter()
                    .map(|worker| {
                        let cap = overrides
                            .binary_search_by_key(&worker, |entry| entry.0)
                            .ok()
                            .map(|position| overrides[position].1)
                            .unwrap_or(default_cap);
                        (worker, cap)
                    })
                    .collect(),
            );
        }
        let Self::Explicit(caps) = self else {
            unreachable!("uniform caps were materialized")
        };
        caps
    }
}

impl FromIterator<(WorkerWithDpRank, u32)> for HybridResumeCaps {
    fn from_iter<T: IntoIterator<Item = (WorkerWithDpRank, u32)>>(entries: T) -> Self {
        Self::Explicit(entries.into_iter().collect())
    }
}

impl Extend<(WorkerWithDpRank, u32)> for HybridResumeCaps {
    fn extend<T: IntoIterator<Item = (WorkerWithDpRank, u32)>>(&mut self, entries: T) {
        self.materialize().extend(entries);
    }
}

impl PartialEq for HybridResumeCaps {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(worker, cap)| other.get(worker) == Some(cap))
    }
}

impl Eq for HybridResumeCaps {}

impl PartialEq<FxHashMap<WorkerWithDpRank, u32>> for HybridResumeCaps {
    fn eq(&self, other: &FxHashMap<WorkerWithDpRank, u32>) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(worker, cap)| other.get(worker) == Some(cap))
    }
}

impl PartialEq<HybridResumeCaps> for FxHashMap<WorkerWithDpRank, u32> {
    fn eq(&self, other: &HybridResumeCaps) -> bool {
        other == self
    }
}

#[derive(Debug, Clone)]
pub enum HybridResumeCapsIter<'a> {
    Explicit(std::collections::hash_map::Iter<'a, WorkerWithDpRank, u32>),
    Uniform {
        workers: std::collections::hash_set::Iter<'a, WorkerWithDpRank>,
        cap: &'a u32,
    },
    Sparse {
        workers: std::collections::hash_set::Iter<'a, WorkerWithDpRank>,
        descriptor: &'a SparseHybridResumeCaps,
    },
}

impl<'a> Iterator for HybridResumeCapsIter<'a> {
    type Item = (&'a WorkerWithDpRank, &'a u32);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Explicit(entries) => entries.next(),
            Self::Uniform { workers, cap } => workers.next().map(|worker| (worker, *cap)),
            Self::Sparse {
                workers,
                descriptor,
            } => {
                let descriptor = *descriptor;
                workers
                    .next()
                    .map(|worker| (worker, descriptor.cap_for_scoped_worker(worker)))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Explicit(entries) => entries.size_hint(),
            Self::Uniform { workers, .. } => workers.size_hint(),
            Self::Sparse { workers, .. } => workers.size_hint(),
        }
    }
}

impl ExactSizeIterator for HybridResumeCapsIter<'_> {}
impl std::iter::FusedIterator for HybridResumeCapsIter<'_> {}

impl<'a> IntoIterator for &'a HybridResumeCaps {
    type Item = (&'a WorkerWithDpRank, &'a u32);
    type IntoIter = HybridResumeCapsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Rich non-wire query result for router-local device tier lookups.
#[derive(Debug, Clone, Default)]
pub struct MatchDetails {
    /// Existing overlap scores used by scheduling.
    pub overlap_scores: OverlapScores,
    /// Last matched device sequence hash per worker, used to seed lower-tier queries.
    pub last_matched_hashes: FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
    /// Optional root-aligned device candidates that KV copy sources are found from.
    pub kv_transfer_candidates: Option<KvTransferCandidates>,
    /// Confirmed device resume caps, including zero for known hybrid workers.
    pub hybrid_resume_caps: HybridResumeCaps,
}

impl MatchDetails {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retain_kv_transfer_candidates(
        &mut self,
        mut block_hashes: Vec<ExternalSequenceBlockHash>,
    ) {
        let mut owner_prefix_blocks: Vec<_> = self
            .overlap_scores
            .scores
            .iter()
            .filter_map(|(worker, blocks)| {
                let blocks = usize::try_from(*blocks).ok()?;
                (blocks > 0 && blocks <= block_hashes.len()).then_some(((*worker).into(), blocks))
            })
            .collect();
        if block_hashes.is_empty() || owner_prefix_blocks.is_empty() {
            return;
        }
        let max_owner_prefix_blocks = owner_prefix_blocks
            .iter()
            .map(|(_, blocks)| *blocks)
            .max()
            .unwrap_or(0);
        block_hashes.truncate(max_owner_prefix_blocks);
        owner_prefix_blocks.sort_unstable_by_key(|(worker, _)| *worker);
        self.kv_transfer_candidates = Some(KvTransferCandidates {
            block_hashes,
            owner_prefix_blocks,
            source_intervals: Vec::new(),
            routing_snapshot: None,
        });
    }
}

/// A request to find matches in the Radix Tree.
pub struct MatchRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// A channel sender to send the `OverlapScores` response.
    pub resp: oneshot::Sender<OverlapScores>,
    /// Timestamp when the request was created (for queue wait time measurement)
    #[cfg(feature = "bench")]
    pub created_at: Instant,
}

impl MatchRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        resp: oneshot::Sender<OverlapScores>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            resp,
            #[cfg(feature = "bench")]
            created_at: Instant::now(),
        }
    }
}

/// A request to find matches while also returning continuation metadata.
pub struct MatchDetailsRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// When true, retain the matched root-aligned external hash chain for KV copy sources.
    pub retain_kv_transfer_chain: bool,
    /// A channel sender to send the `MatchDetails` response.
    pub resp: oneshot::Sender<MatchDetails>,
}

/// A request to test one worker's ownership of an external block hash.
pub struct ContainsWorkerBlockRequest {
    pub worker: WorkerWithDpRank,
    pub block_hash: ExternalSequenceBlockHash,
    pub resp: oneshot::Sender<bool>,
}

impl MatchDetailsRequest {
    pub(super) fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        retain_kv_transfer_chain: bool,
        resp: oneshot::Sender<MatchDetails>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            retain_kv_transfer_chain,
            resp,
        }
    }
}

/// A request to dump the tree as events
pub struct DumpRequest {
    /// Channel to send the dumped events
    pub resp: oneshot::Sender<Vec<RouterEvent>>,
}

/// A request to wait until all previously submitted work is applied.
pub struct FlushRequest {
    /// Channel to acknowledge completion.
    pub resp: oneshot::Sender<()>,
}

/// A request to get all workers currently tracked
pub struct GetWorkersRequest {
    /// Channel to send the worker IDs
    pub resp: oneshot::Sender<Vec<WorkerId>>,
}

#[derive(Debug, Default)]
pub struct WorkerLookupStats {
    pub worker_blocks: Vec<(WorkerWithDpRank, usize)>,
}

impl WorkerLookupStats {
    pub fn from_worker_block_counts(
        counts: impl IntoIterator<Item = (WorkerWithDpRank, usize)>,
    ) -> Self {
        Self {
            worker_blocks: counts
                .into_iter()
                .filter(|(_, block_count)| *block_count > 0)
                .collect(),
        }
    }

    pub fn worker_count(&self) -> usize {
        self.worker_blocks.len()
    }

    pub fn block_count(&self) -> usize {
        self.worker_blocks
            .iter()
            .map(|(_, block_count)| *block_count)
            .sum()
    }

    pub fn block_count_for_worker(&self, worker: WorkerWithDpRank) -> Option<usize> {
        self.worker_blocks
            .iter()
            .find_map(|(candidate, block_count)| (*candidate == worker).then_some(*block_count))
    }
}

pub enum WorkerTask {
    Event(RouterEvent),
    EventWithAck {
        event: RouterEvent,
        resp: oneshot::Sender<bool>,
    },
    ApproximateLru(super::ApproximateLruTask),
    #[cfg(feature = "bench")]
    InstallObservation {
        writer: EventCompletionWriter,
        resp: oneshot::Sender<bool>,
    },
    #[cfg(feature = "bench")]
    ObservedEvent {
        event: RouterEvent,
        correlation_id: u32,
    },
    #[cfg(feature = "bench")]
    SealObservation(oneshot::Sender<Option<ObservationSeal>>),
    #[cfg(feature = "bench")]
    HarvestObservation(oneshot::Sender<EventCompletionBuffer>),
    Anchor {
        worker: WorkerWithDpRank,
        anchor: AnchorTask,
    },
    /// Permanently remove a worker from tracking.
    RemoveWorker {
        worker_id: WorkerId,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
        /// Acknowledges completion of this lane's cold-path removal phase.
        resp: oneshot::Sender<()>,
    },
    /// Remove a single dp_rank for a worker.
    RemoveWorkerDpRank {
        worker_id: WorkerId,
        dp_rank: DpRank,
        /// True for the one shared-state backend task that owns structural cleanup.
        sweep_tree: bool,
    },
    DumpEvents(oneshot::Sender<anyhow::Result<Vec<RouterEvent>>>),
    Stats(oneshot::Sender<WorkerLookupStats>),
    ContainsWorkerBlock {
        worker: WorkerWithDpRank,
        block_hash: ExternalSequenceBlockHash,
        resp: oneshot::Sender<bool>,
    },
    Flush(oneshot::Sender<()>),
    /// Keep live per-rank block counts in these counters from now on.
    TrackResidentBlocks(crate::ResidentBlockCountsHandle),
    Terminate,
}

/// A request to process a routing decision.
pub(super) struct RoutingDecisionRequest {
    pub(super) worker: WorkerWithDpRank,
    pub(super) local_hashes: Vec<LocalBlockHash>,
    pub(super) sequence_hashes: Vec<SequenceHash>,
}

#[cfg(test)]
mod hybrid_resume_caps_tests {
    use super::*;

    #[test]
    fn sparse_caps_keep_exact_dp_scope_zero_and_score_independent_references() {
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(7, 1);
        let zero = WorkerWithDpRank::new(8, 0);
        let outside = WorkerWithDpRank::new(7, 2);
        let mut details = MatchDetails {
            hybrid_resume_caps: HybridResumeCaps::sparse_confirmed(
                FxHashSet::from_iter([first, second, zero]),
                0,
                vec![(second, 2), (outside, 99), (second, 4), (first, 0)],
            ),
            ..Default::default()
        };
        let default_reference = details.hybrid_resume_caps.get(&first).unwrap();
        let override_reference = details.hybrid_resume_caps.get(&second).unwrap();
        details
            .overlap_scores
            .scores
            .extend((0..128).map(|id| (WorkerWithDpRank::new(id, 0), 99)));
        assert_eq!(*default_reference, 0);
        assert_eq!(*override_reference, 4);
        assert_eq!(details.hybrid_resume_caps.get(&zero), Some(&0));
        assert_eq!(details.hybrid_resume_caps.get(&outside), None);
        assert_eq!(details.hybrid_resume_caps.len(), 3);
        let HybridResumeCaps::SparseConfirmed(sparse) = &details.hybrid_resume_caps else {
            panic!("a nondefault scoped cap needs the sparse descriptor")
        };
        assert_eq!(sparse.overrides, [(second, 4)]);
        let mut entries = details.hybrid_resume_caps.iter();
        assert_eq!(entries.len(), 3);
        entries.next().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries.clone().count(), 2);
        assert_eq!(
            details.hybrid_resume_caps,
            FxHashMap::from_iter([(first, 0), (second, 4), (zero, 0)])
        );
        let mut cloned = details.hybrid_resume_caps.clone();
        cloned.clear();
        assert!(cloned.is_empty());
        assert_eq!(details.hybrid_resume_caps.get(&second), Some(&4));
    }

    #[test]
    fn sparse_caps_materialize_without_losing_default_zero_or_foreign_entries() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let zero = WorkerWithDpRank::new(3, 0);
        let outside = WorkerWithDpRank::new(4, 1);
        let mut caps = HybridResumeCaps::sparse_confirmed(
            FxHashSet::from_iter([first, second, zero]),
            8,
            vec![(second, 4), (zero, 0)],
        );
        caps.reserve(0);
        assert!(matches!(caps, HybridResumeCaps::SparseConfirmed(_)));
        let mut reserved = caps.clone();
        reserved.reserve(1);
        assert!(matches!(reserved, HybridResumeCaps::Explicit(_)));
        assert_eq!(reserved, caps);
        assert_eq!(caps.insert(first, 2), Some(8));
        assert_eq!(caps.get(&second), Some(&4));
        assert_eq!(caps.get(&zero), Some(&0));
        assert_eq!(caps.insert(outside, 12), None);
        caps.extend([(second, 6)]);
        assert_eq!(
            caps,
            FxHashMap::from_iter([(first, 2), (second, 6), (zero, 0), (outside, 12)])
        );
    }

    #[test]
    fn sparse_default_only_normalizes_without_losing_known_zero() {
        let worker = WorkerWithDpRank::new(1, 0);
        let caps = HybridResumeCaps::sparse_confirmed(
            FxHashSet::from_iter([worker]),
            0,
            vec![(worker, 8), (worker, 0)],
        );
        assert!(matches!(
            caps,
            HybridResumeCaps::UniformConfirmed { cap: 0, .. }
        ));
        assert_eq!(caps.get(&worker), Some(&0));
        assert_eq!(caps.len(), 1);
        let empty = HybridResumeCaps::sparse_confirmed(FxHashSet::default(), 8, vec![(worker, 1)]);
        assert!(empty.is_empty());
        assert_eq!(empty.get(&worker), None);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<HybridResumeCaps>(), 40);
    }

    #[test]
    fn sparse_native_caps_keep_zero_and_scope_on_the_existing_wire_format() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(1, 1);
        let zero = WorkerWithDpRank::new(2, 0);
        let outside = WorkerWithDpRank::new(2, 1);
        let mut native = crate::TieredMatchDetails::default();
        native.device.hybrid_resume_caps = HybridResumeCaps::sparse_confirmed(
            FxHashSet::from_iter([first, second, zero]),
            8,
            vec![(second, 4), (zero, 0)],
        );
        native
            .device
            .overlap_scores
            .scores
            .extend([(first, 8), (second, 4)]);
        let wire = WireTieredMatchDetails::from(&native);
        assert_eq!(wire.hybrid_resume_caps.len(), 3);
        assert!(wire.hybrid_resume_caps.contains(&(zero, 0)));
        let json = serde_json::to_value(wire).unwrap();
        assert!(json["hybrid_resume_caps"].is_array());
        let decoded = serde_json::from_value::<WireTieredMatchDetails>(json).unwrap();
        let restored = crate::TieredMatchDetails::from(decoded);
        assert_eq!(
            restored.device.hybrid_resume_caps,
            native.device.hybrid_resume_caps
        );
        assert_eq!(restored.device.hybrid_resume_caps.get(&outside), None);
        assert_eq!(
            restored.device.overlap_scores.scores,
            native.device.overlap_scores.scores
        );
    }

    #[test]
    fn uniform_caps_remain_confirmed_when_scores_change() {
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(7, 1);
        let outside = WorkerWithDpRank::new(8, 0);
        let workers = FxHashSet::from_iter([first, second]);
        let capacity = workers.capacity();
        let mut details = MatchDetails {
            hybrid_resume_caps: HybridResumeCaps::UniformConfirmed { workers, cap: 8 },
            ..Default::default()
        };
        details
            .overlap_scores
            .scores
            .extend([(first, 8), (second, 8)]);
        details.overlap_scores.scores.insert(first, 99);
        details.overlap_scores.scores.remove(&second);
        details.overlap_scores.scores.insert(outside, 100);

        assert_eq!(details.hybrid_resume_caps.get(&first), Some(&8));
        assert_eq!(details.hybrid_resume_caps.get(&second), Some(&8));
        assert_eq!(details.hybrid_resume_caps.get(&outside), None);
        assert_eq!(details.hybrid_resume_caps.len(), 2);
        let HybridResumeCaps::UniformConfirmed { workers, .. } = &details.hybrid_resume_caps else {
            panic!("reading caps must preserve their uniform representation")
        };
        assert_eq!(workers.capacity(), capacity);
        let mut entries = details.hybrid_resume_caps.iter();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries.next().map(|(_, cap)| *cap), Some(8));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries.clone().count(), 1);
    }

    #[test]
    fn materializing_uniform_caps_preserves_zero_and_unrelated_workers() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let outside = WorkerWithDpRank::new(3, 1);
        let mut caps = HybridResumeCaps::UniformConfirmed {
            workers: FxHashSet::from_iter([first, second]),
            cap: 0,
        };
        assert_eq!(caps.insert(first, 4), Some(0));
        assert!(matches!(caps, HybridResumeCaps::Explicit(_)));
        assert_eq!(caps.get(&second), Some(&0));
        assert_eq!(caps.insert(outside, 12), None);
        caps.extend([(first, 5), (outside, 13)]);
        assert_eq!(
            caps,
            FxHashMap::from_iter([(first, 5), (second, 0), (outside, 13)])
        );
        caps.clear();
        assert!(caps.is_empty());
        assert_eq!(caps.get(&second), None);
    }

    #[test]
    fn explicit_and_uniform_caps_have_the_same_entry_semantics() {
        let worker = WorkerWithDpRank::new(1, 2);
        let uniform = HybridResumeCaps::UniformConfirmed {
            workers: FxHashSet::from_iter([worker]),
            cap: 0,
        };
        let explicit = HybridResumeCaps::from_iter([(worker, 0)]);
        assert_eq!(uniform, explicit);
        assert_eq!(FxHashMap::from_iter([(worker, 0)]), uniform);
        let mut empty = uniform.clone();
        empty.clear();
        assert_eq!(uniform.get(&worker), Some(&0));
        assert_eq!(empty, HybridResumeCaps::default());
        assert_eq!(empty.iter().len(), 0);
        assert!(!empty.contains_key(&worker));
        assert!(matches!(
            HybridResumeCaps::default(),
            HybridResumeCaps::Explicit(_)
        ));
    }
}
