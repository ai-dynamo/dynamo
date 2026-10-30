// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV fetch source each candidate worker would use.
//!
//! Worker selection credits every candidate for the prefix it could fetch, and
//! the picked worker's hint fetches from that same source, so both read one
//! [`KvTransferSourceSummary`]. A source must share the target's hint worker
//! type, have a control endpoint, and be neither the target nor a cache owner
//! attached to it. When the target sets a `kv_transfer_domain`, the source must
//! also share the target's value for that domain (its island); with `preferred`
//! enforcement, a cross-island source is used only when no same-island source
//! qualifies.

use std::collections::HashMap;
use std::collections::hash_map::{DefaultHasher, Entry};
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use rustc_hash::FxHashMap;

use super::{KvTransferCandidateSource, KvTransferCandidates};
use crate::protocols::{
    KvTransferEnforcement, RouterHintSourceMetadata, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};

/// Hash a label once, when its config or snapshot is published, so matching
/// sources to targets per request compares integers instead of strings.
fn label_key(label: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    hasher.finish()
}

/// Sorted, so keys built from equal labels compare equal.
fn topology_keys(topology: &HashMap<String, String>) -> Vec<(u64, u64)> {
    let mut keys: Vec<_> = topology
        .iter()
        .map(|(domain, value)| (label_key(domain), label_key(value)))
        .collect();
    keys.sort_unstable();
    keys
}

/// A worker's router-hint identity, with its labels hashed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerKvTransferKeys {
    /// Hint worker type; `None` when the worker cannot consume hints.
    worker_type: Option<u64>,
    /// DP ranks with a source control endpoint, ascending.
    source_ranks: Vec<u32>,
    /// `(domain, value)` topology labels.
    topology: Vec<(u64, u64)>,
    /// The `kv_transfer_domain` this worker fetches within, and its value for it.
    transfer_domain: Option<u64>,
    transfer_island: Option<u64>,
    preferred: bool,
}

impl WorkerKvTransferKeys {
    pub fn new<C: WorkerConfigLike + ?Sized>(config: &C) -> Self {
        let start = config.data_parallel_start_rank();
        let mut worker_type = None;
        let mut source_ranks = Vec::new();
        for rank in start..start.saturating_add(config.data_parallel_size()) {
            let Some(metadata) = config.kv_hint_transfer_metadata_for_dp_rank(rank) else {
                continue;
            };
            worker_type.get_or_insert_with(|| label_key(metadata.worker_type));
            if metadata
                .source_control_endpoint
                .is_some_and(|endpoint| !endpoint.is_empty())
            {
                source_ranks.push(rank);
            }
        }
        let topology = config
            .topology_domains()
            .map(topology_keys)
            .unwrap_or_default();
        let transfer_domain = config.kv_transfer_domain().map(label_key);
        let transfer_island = transfer_domain.and_then(|domain| island(&topology, domain));
        Self {
            worker_type,
            source_ranks,
            topology,
            transfer_domain,
            transfer_island,
            preferred: config.kv_transfer_enforcement() == Some(KvTransferEnforcement::Preferred),
        }
    }

    fn is_source(&self, dp_rank: u32) -> bool {
        self.source_ranks.binary_search(&dp_rank).is_ok()
    }

    /// What a selection needs to find this worker's source; `None` when the
    /// worker cannot consume hints.
    pub fn target(&self) -> Option<KvTransferTarget> {
        Some(KvTransferTarget {
            worker_type: self.worker_type?,
            transfer_domain: self.transfer_domain,
            transfer_island: self.transfer_island,
            preferred: self.preferred,
        })
    }
}

/// The hashed labels that choose a target worker's source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTransferTarget {
    worker_type: u64,
    transfer_domain: Option<u64>,
    transfer_island: Option<u64>,
    preferred: bool,
}

/// The targets a [`KvTransferSourceSummary`] answers, so it groups only the
/// sources they can use.
#[derive(Clone, Debug)]
pub struct KvTransferScope {
    /// The shortest local prefix of any target; shorter sources are skipped.
    min_local_prefix_blocks: usize,
    /// No target can use a continuation starting beyond this depth.
    max_local_prefix_blocks: usize,
    /// Whether some target takes a source from any island: it has no
    /// transfer domain, or `preferred` enforcement allows a fallback.
    any_island: bool,
    /// The islands targets fetch within.
    islands: IslandScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IslandScope {
    /// No target fetches within an island.
    Unused,
    /// Every target that fetches within an island uses this `(domain, island)`.
    One(u64, u64),
    Any,
}

impl KvTransferScope {
    /// Every target, for sources summarized before the targets are known.
    pub fn all() -> Self {
        Self {
            min_local_prefix_blocks: 0,
            max_local_prefix_blocks: usize::MAX,
            any_island: true,
            islands: IslandScope::Any,
        }
    }

    /// Exactly these targets, each with its local prefix in blocks.
    pub fn of(targets: impl IntoIterator<Item = (KvTransferTarget, usize)>) -> Self {
        let mut scope = Self {
            min_local_prefix_blocks: usize::MAX,
            max_local_prefix_blocks: 0,
            any_island: false,
            islands: IslandScope::Unused,
        };
        for (target, local_prefix_blocks) in targets {
            scope.min_local_prefix_blocks = scope.min_local_prefix_blocks.min(local_prefix_blocks);
            scope.max_local_prefix_blocks = scope.max_local_prefix_blocks.max(local_prefix_blocks);
            scope.any_island |= target.transfer_domain.is_none() || target.preferred;
            if let (Some(domain), Some(island)) = (target.transfer_domain, target.transfer_island) {
                let one = IslandScope::One(domain, island);
                scope.islands = match scope.islands {
                    IslandScope::Unused => one,
                    islands if islands == one => one,
                    _ => IslandScope::Any,
                };
            }
        }
        scope
    }

    fn covers(&self, domain: u64, island: u64) -> bool {
        self.islands == IslandScope::Any || self.islands == IslandScope::One(domain, island)
    }
}

fn island(topology: &[(u64, u64)], domain: u64) -> Option<u64> {
    topology
        .iter()
        .find_map(|&(label, value)| (label == domain).then_some(value))
}

/// [`WorkerKvTransferKeys`] derived from a worker config on first use. The
/// config's other fields decide equality, and it must not change afterwards.
#[derive(Clone, Debug, Default)]
pub struct WorkerKvTransferKeysCache(OnceLock<WorkerKvTransferKeys>);

impl WorkerKvTransferKeysCache {
    pub fn get_or_init<C: WorkerConfigLike + ?Sized>(&self, config: &C) -> &WorkerKvTransferKeys {
        self.0.get_or_init(|| WorkerKvTransferKeys::new(config))
    }
}

impl PartialEq for WorkerKvTransferKeysCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// A cache owner's router-hint identity, hashed when its routing snapshot is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheOwnerKvTransferKeys {
    worker_type: u64,
    is_source: bool,
    topology: Vec<(u64, u64)>,
}

impl CacheOwnerKvTransferKeys {
    pub fn new(metadata: &RouterHintSourceMetadata) -> Self {
        Self {
            worker_type: label_key(&metadata.worker_type),
            is_source: !metadata.source_control_endpoint.is_empty(),
            topology: topology_keys(&metadata.topology_domains),
        }
    }
}

/// The source a target fetches from, and the resulting prefix length in blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTransferSourceChoice {
    pub source: KvTransferCandidateSource,
    pub prefix_blocks: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RankedSource {
    choice: KvTransferSourceChoice,
    /// The worker this source belongs to, which never fetches from it.
    attachment: Option<WorkerWithDpRank>,
}

impl RankedSource {
    /// A longer prefix wins, then the lower source.
    fn outranks(&self, other: &Self) -> bool {
        (
            self.choice.prefix_blocks,
            std::cmp::Reverse(self.choice.source),
        ) > (
            other.choice.prefix_blocks,
            std::cmp::Reverse(other.choice.source),
        )
    }

    fn shares_attachment(&self, other: &Self) -> bool {
        self.attachment.is_some() && self.attachment == other.attachment
    }
}

/// A group's best source and the best one attached to a different worker, so
/// the best source not attached to any given target is one of the two.
#[derive(Clone, Copy, PartialEq, Eq)]
struct TopSources {
    best: RankedSource,
    runner_up: Option<RankedSource>,
}

impl TopSources {
    fn offer(&mut self, source: RankedSource) {
        if source.outranks(&self.best) {
            if !source.shares_attachment(&self.best) {
                self.runner_up = Some(self.best);
            }
            self.best = source;
        } else if !source.shares_attachment(&self.best)
            && self
                .runner_up
                .is_none_or(|runner_up| source.outranks(&runner_up))
        {
            self.runner_up = Some(source);
        }
    }

    fn best_for(&self, target: WorkerWithDpRank) -> Option<KvTransferSourceChoice> {
        if self.best.attachment == Some(target) {
            self.runner_up.map(|source| source.choice)
        } else {
            Some(self.best.choice)
        }
    }
}

/// Root-only queries keep the existing inline top two. Continuations retain
/// cumulative top two choices at each start depth where the ranking changes.
/// Expired intervals need no removal: their ends rank below every live interval.
enum SourceBucket {
    Root(TopSources),
    Continuations(Vec<(usize, TopSources)>),
}

impl SourceBucket {
    /// Offers must arrive in nondecreasing start order.
    fn offer(entry: Entry<'_, impl Hash + Eq, Self>, start: usize, source: RankedSource) {
        let bucket = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let top = TopSources {
                    best: source,
                    runner_up: None,
                };
                entry.insert(if start == 0 {
                    Self::Root(top)
                } else {
                    Self::Continuations(vec![(start, top)])
                });
                return;
            }
        };
        match bucket {
            Self::Root(top) if start == 0 => top.offer(source),
            Self::Root(top) => {
                let mut next = *top;
                next.offer(source);
                if next != *top {
                    *bucket = Self::Continuations(vec![(0, *top), (start, next)]);
                }
            }
            Self::Continuations(by_start) => {
                if let Some((last_start, top)) = by_start.last_mut() {
                    debug_assert!(start >= *last_start);
                    let mut next = *top;
                    next.offer(source);
                    if start == *last_start {
                        *top = next;
                    } else if next != *top {
                        by_start.push((start, next));
                    }
                }
            }
        }
    }

    fn best_for(
        &self,
        worker: WorkerWithDpRank,
        local_prefix: usize,
    ) -> Option<KvTransferSourceChoice> {
        match self {
            Self::Root(top) => top.best_for(worker),
            Self::Continuations(by_start) => {
                let idx = by_start.partition_point(|(start, _)| *start <= local_prefix);
                by_start
                    .get(idx.checked_sub(1)?)
                    .and_then(|(_, top)| top.best_for(worker))
            }
        }
    }
}

/// The eligible sources of one request, grouped so the source for any target
/// in its scope is found without scanning sources. Root-prefix lookup is O(1);
/// continuation lookup is O(log distinct start depths) per applicable bucket.
pub struct KvTransferSourceSummary {
    by_type: FxHashMap<u64, SourceBucket>,
    by_island: FxHashMap<(u64, u64, u64), SourceBucket>,
    scope: KvTransferScope,
}

impl KvTransferSourceSummary {
    /// Summarize the sources the targets in `scope` can use. Sources no
    /// longer than every target's local prefix are skipped before their
    /// metadata is read.
    pub fn new<C: WorkerConfigLike>(
        candidates: &KvTransferCandidates,
        workers: &HashMap<WorkerId, C>,
        scope: KvTransferScope,
    ) -> Self {
        let min_local_prefix_blocks = scope.min_local_prefix_blocks;
        let max_local_prefix_blocks = scope.max_local_prefix_blocks;
        let needs_islands = scope.islands != IslandScope::Unused;
        let mut summary = Self {
            by_type: FxHashMap::default(),
            by_island: FxHashMap::default(),
            scope,
        };
        let snapshot = candidates.routing_snapshot.as_deref();
        let usable = |start, end| {
            start <= max_local_prefix_blocks
                && end > min_local_prefix_blocks
                && end <= candidates.block_hashes.len()
                && start < end
        };
        let roots = candidates
            .owner_prefix_blocks
            .iter()
            .map(|&(source, end)| (source, 0, end))
            .filter(|&(_, start, end)| usable(start, end));
        let mut intervals: Vec<_> = candidates
            .source_intervals
            .iter()
            .filter(|interval| usable(interval.start, interval.end))
            .collect();
        intervals.sort_unstable_by_key(|interval| interval.start);
        for (source, start, prefix_blocks) in roots.chain(
            intervals
                .into_iter()
                .map(|interval| (interval.source, interval.start, interval.end)),
        ) {
            let worker_keys;
            let attached_worker_keys;
            let (worker_type, attachment, topology) = match source {
                KvTransferCandidateSource::Worker(worker) => {
                    let Some(config) = workers.get(&worker.worker_id) else {
                        continue;
                    };
                    worker_keys = config.kv_transfer_keys();
                    let Some(worker_type) = worker_keys.worker_type else {
                        continue;
                    };
                    if !worker_keys.is_source(worker.dp_rank) {
                        continue;
                    }
                    (worker_type, Some(worker), worker_keys.topology.as_slice())
                }
                KvTransferCandidateSource::CacheOwner(owner) => {
                    let Some(owner) =
                        snapshot.and_then(|snapshot| snapshot.router_hint_source(owner))
                    else {
                        continue;
                    };
                    let keys = &owner.kv_transfer_keys;
                    if !keys.is_source {
                        continue;
                    }
                    // An owner that publishes no topology is in its attached worker's island.
                    let topology = if keys.topology.is_empty() && needs_islands {
                        attached_worker_keys = owner
                            .attached_worker
                            .and_then(|worker| workers.get(&worker.worker_id))
                            .map(|config| config.kv_transfer_keys());
                        attached_worker_keys
                            .as_deref()
                            .map_or(&[][..], |keys| keys.topology.as_slice())
                    } else {
                        keys.topology.as_slice()
                    };
                    (keys.worker_type, owner.attached_worker, topology)
                }
            };
            let ranked = RankedSource {
                choice: KvTransferSourceChoice {
                    source,
                    prefix_blocks,
                },
                attachment,
            };
            if summary.scope.any_island {
                SourceBucket::offer(summary.by_type.entry(worker_type), start, ranked);
            }
            for &(domain, value) in topology {
                if summary.scope.covers(domain, value) {
                    SourceBucket::offer(
                        summary.by_island.entry((worker_type, domain, value)),
                        start,
                        ranked,
                    );
                }
            }
        }
        summary
    }

    /// The source `worker` fetches from: the eligible source with the longest
    /// prefix beyond `local_prefix_blocks`, or `None` when there is none.
    pub fn source_for(
        &self,
        worker: WorkerWithDpRank,
        target: KvTransferTarget,
        local_prefix_blocks: usize,
    ) -> Option<KvTransferSourceChoice> {
        debug_assert!(local_prefix_blocks >= self.scope.min_local_prefix_blocks);
        let pick = |top: Option<&SourceBucket>| {
            top?.best_for(worker, local_prefix_blocks)
                .filter(|choice| choice.prefix_blocks > local_prefix_blocks)
        };
        let worker_type = target.worker_type;
        let Some(domain) = target.transfer_domain else {
            return pick(self.by_type.get(&worker_type));
        };
        target
            .transfer_island
            .and_then(|island| pick(self.by_island.get(&(worker_type, domain, island))))
            .or_else(|| {
                target
                    .preferred
                    .then(|| pick(self.by_type.get(&worker_type)))
                    .flatten()
            })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::identity::{
        CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };
    use crate::protocols::{
        ExternalSequenceBlockHash, KvHintTransferWorkerMetadata, ResidencyOwner,
        ResidencyProjection, ResidencyRoutingSnapshot,
    };

    struct Worker;

    impl WorkerConfigLike for Worker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            2
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn kv_hint_transfer_metadata_for_dp_rank(
            &self,
            _dp_rank: u32,
        ) -> Option<KvHintTransferWorkerMetadata<'_>> {
            Some(KvHintTransferWorkerMetadata {
                worker_type: "prefill",
                source_control_endpoint: Some("tcp://worker:23280"),
            })
        }
    }

    fn workers() -> HashMap<WorkerId, Worker> {
        (7..=9).map(|worker_id| (worker_id, Worker)).collect()
    }

    fn candidates(sources: Vec<(KvTransferCandidateSource, usize)>) -> KvTransferCandidates {
        KvTransferCandidates {
            block_hashes: (101..=104).map(ExternalSequenceBlockHash).collect(),
            owner_prefix_blocks: sources,
            source_intervals: Vec::new(),
            routing_snapshot: None,
        }
    }

    fn source_for(
        candidates: &KvTransferCandidates,
        target: WorkerWithDpRank,
        local_prefix_blocks: usize,
    ) -> Option<(KvTransferCandidateSource, usize)> {
        let workers = workers();
        let target_keys = workers[&target.worker_id].kv_transfer_keys().target()?;
        let scope = KvTransferScope::of([(target_keys, local_prefix_blocks)]);
        KvTransferSourceSummary::new(candidates, &workers, scope)
            .source_for(target, target_keys, local_prefix_blocks)
            .map(|choice| (choice.source, choice.prefix_blocks))
    }

    #[test]
    fn longest_prefix_beyond_the_local_one_wins() {
        let target = WorkerWithDpRank::new(7, 0);
        let other_rank = WorkerWithDpRank::new(7, 1);
        let (eight, nine) = (WorkerWithDpRank::new(8, 0), WorkerWithDpRank::new(9, 0));
        let candidates = candidates(vec![
            (target.into(), 4),
            (nine.into(), 3),
            (eight.into(), 3),
            (other_rank.into(), 2),
            // Longer than the retained chain, so it could not be fetched.
            (WorkerWithDpRank::new(9, 1).into(), 5),
        ]);

        // The target is never its own source; equal prefixes go to the lower source.
        assert_eq!(source_for(&candidates, target, 0), Some((eight.into(), 3)));
        assert_eq!(source_for(&candidates, target, 2), Some((eight.into(), 3)));
        assert_eq!(source_for(&candidates, target, 3), None);
        // Another DP rank of the same worker is a separate source.
        let candidates = self::candidates(vec![(target.into(), 4), (other_rank.into(), 2)]);
        assert_eq!(
            source_for(&candidates, target, 0),
            Some((other_rank.into(), 2))
        );
    }

    #[test]
    fn continuation_choices_match_single_source_inventory() {
        use crate::kv_hints::KvTransferSourceInterval;

        let workers = workers();
        let mut rng = fastrand::Rng::with_seed(59);
        for _ in 0..128 {
            // Independent block inventories, including holes, shared starts and
            // expired intervals. Query every target frontier against the same summary.
            let inventories: Vec<_> = (7..=10)
                .flat_map(|id| (0..2).map(move |rank| WorkerWithDpRank::new(id, rank)))
                .map(|worker| (worker, (0..12).map(|_| rng.bool()).collect::<Vec<_>>()))
                .collect();
            let mut candidates = KvTransferCandidates {
                block_hashes: (0..12).map(ExternalSequenceBlockHash).collect(),
                ..Default::default()
            };
            for (worker, blocks) in &inventories {
                let mut pos = 0;
                while pos < blocks.len() {
                    if !blocks[pos] {
                        pos += 1;
                        continue;
                    }
                    let start = pos;
                    while pos < blocks.len() && blocks[pos] {
                        pos += 1;
                    }
                    if start == 0 {
                        candidates.owner_prefix_blocks.push(((*worker).into(), pos));
                    } else {
                        candidates.source_intervals.push(KvTransferSourceInterval {
                            source: (*worker).into(),
                            start,
                            end: pos,
                        });
                    }
                }
            }
            rng.shuffle(&mut candidates.source_intervals);
            let summary =
                KvTransferSourceSummary::new(&candidates, &workers, KvTransferScope::all());
            for target in
                (7..=9).flat_map(|id| (0..2).map(move |rank| WorkerWithDpRank::new(id, rank)))
            {
                for local in 0..=12 {
                    let expected = inventories
                        .iter()
                        .filter(|(worker, _)| {
                            *worker != target && workers.contains_key(&worker.worker_id)
                        })
                        .filter_map(|(worker, blocks)| {
                            let end = local
                                + blocks[local..]
                                    .iter()
                                    .take_while(|&&present| present)
                                    .count();
                            (end > local)
                                .then_some((KvTransferCandidateSource::Worker(*worker), end))
                        })
                        .max_by_key(|&(source, end)| (end, std::cmp::Reverse(source)));
                    let keys = workers[&target.worker_id].kv_transfer_keys();
                    let actual = summary
                        .source_for(target, keys.target().unwrap(), local)
                        .map(|choice| (choice.source, choice.prefix_blocks));
                    assert_eq!(actual, expected, "target={target:?}, local={local}");
                    let scoped = KvTransferSourceSummary::new(
                        &candidates,
                        &workers,
                        KvTransferScope::of([(keys.target().unwrap(), local)]),
                    );
                    let actual = scoped
                        .source_for(target, keys.target().unwrap(), local)
                        .map(|choice| (choice.source, choice.prefix_blocks));
                    assert_eq!(actual, expected, "scoped target={target:?}, local={local}");
                }
            }
        }
    }

    #[test]
    fn cache_owners_attached_to_the_target_are_skipped() {
        let owner = |slot: u8| {
            CacheOwnerId::new(
                PoolId::new(
                    IndexerDomainId::new(
                        CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                        RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                    ),
                    ClusterId::new(3),
                ),
                StableDpSlotId::new([slot; 16], IdentitySource::Explicit),
            )
        };
        let key = |owner| ResidencyOwner::cache_owner(owner).compact_key();
        let target = WorkerWithDpRank::new(7, 0);
        let other = WorkerWithDpRank::new(8, 0);
        let metadata = RouterHintSourceMetadata {
            source_control_endpoint: "tcp://owner:23280".to_string(),
            worker_type: "prefill".to_string(),
            topology_domains: HashMap::new(),
        };
        // The two longest sources both belong to the target.
        let mut candidates = candidates(vec![
            (KvTransferCandidateSource::CacheOwner(key(owner(4))), 4),
            (KvTransferCandidateSource::CacheOwner(key(owner(5))), 3),
            (other.into(), 2),
        ]);
        candidates.routing_snapshot = Some(Arc::new(ResidencyRoutingSnapshot::new(
            ResidencyProjection::default(),
            [
                (owner(4), metadata.clone(), Some(target)),
                (owner(5), metadata, Some(target)),
            ],
        )));

        assert_eq!(source_for(&candidates, target, 0), Some((other.into(), 2)));
        assert_eq!(
            source_for(&candidates, other, 0),
            Some((KvTransferCandidateSource::CacheOwner(key(owner(4))), 4))
        );

        candidates.owner_prefix_blocks.clear();
        candidates.source_intervals = vec![
            crate::kv_hints::KvTransferSourceInterval {
                source: KvTransferCandidateSource::CacheOwner(key(owner(4))),
                start: 1,
                end: 4,
            },
            crate::kv_hints::KvTransferSourceInterval {
                source: other.into(),
                start: 1,
                end: 3,
            },
        ];
        assert_eq!(source_for(&candidates, target, 1), Some((other.into(), 3)));
        assert_eq!(
            source_for(&candidates, other, 1),
            Some((KvTransferCandidateSource::CacheOwner(key(owner(4))), 4))
        );
    }
}
