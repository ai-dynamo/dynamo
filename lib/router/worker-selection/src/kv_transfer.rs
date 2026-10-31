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

use std::collections::hash_map::Entry;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use dynamo_router_protocol::kv_hints::KvTransferCandidateSource;
use dynamo_router_protocol::worker::WorkerWithDpRank;

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
    /// Assemble the keys from already-hashed labels. The router derives them from a worker's
    /// runtime config; see `WorkerKvTransferKeysExt` in `dynamo_kv_router::kv_hints`.
    pub fn from_parts(
        worker_type: Option<u64>,
        source_ranks: Vec<u32>,
        topology: Vec<(u64, u64)>,
        transfer_domain: Option<u64>,
        preferred: bool,
    ) -> Self {
        let transfer_island = transfer_domain.and_then(|domain| island(&topology, domain));
        Self {
            worker_type,
            source_ranks,
            topology,
            transfer_domain,
            transfer_island,
            preferred,
        }
    }

    /// Whether `dp_rank` publishes a source control endpoint.
    pub fn is_source(&self, dp_rank: u32) -> bool {
        self.source_ranks.binary_search(&dp_rank).is_ok()
    }

    /// Hashed hint worker type; `None` when the worker cannot consume hints.
    pub fn worker_type(&self) -> Option<u64> {
        self.worker_type
    }

    /// `(domain, value)` topology labels, hashed and sorted.
    pub fn topology(&self) -> &[(u64, u64)] {
        &self.topology
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
            any_island: true,
            islands: IslandScope::Any,
        }
    }

    /// Exactly these targets, each with its local prefix in blocks.
    pub fn of(targets: impl IntoIterator<Item = (KvTransferTarget, usize)>) -> Self {
        let mut scope = Self::empty();
        for (target, local_prefix_blocks) in targets {
            scope.add_target(target, local_prefix_blocks);
        }
        scope
    }

    /// No targets yet; add them with [`Self::add_target`].
    pub fn empty() -> Self {
        Self {
            min_local_prefix_blocks: usize::MAX,
            any_island: false,
            islands: IslandScope::Unused,
        }
    }

    /// Include `target`, with its local prefix in blocks.
    pub fn add_target(&mut self, target: KvTransferTarget, local_prefix_blocks: usize) {
        self.min_local_prefix_blocks = self.min_local_prefix_blocks.min(local_prefix_blocks);
        self.any_island |= target.transfer_domain.is_none() || target.preferred;
        if let (Some(domain), Some(island)) = (target.transfer_domain, target.transfer_island) {
            let one = IslandScope::One(domain, island);
            self.islands = match self.islands {
                IslandScope::Unused => one,
                islands if islands == one => one,
                _ => IslandScope::Any,
            };
        }
    }

    /// The shortest local prefix of any target in this scope.
    pub fn min_local_prefix_blocks(&self) -> usize {
        self.min_local_prefix_blocks
    }

    /// Whether some target fetches within an island, so sources need their topology.
    pub fn needs_islands(&self) -> bool {
        self.islands != IslandScope::Unused
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

/// One eligible KV source as the router resolved it for a selection: who holds the prefix,
/// how long it is, the hashed hint worker type, the worker it is attached to (which never
/// fetches from it), and its topology labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedTransferSource {
    /// The source.
    pub source: KvTransferCandidateSource,
    /// Prefix the source holds, in blocks.
    pub prefix_blocks: usize,
    /// Hashed hint worker type the source serves.
    pub worker_type: u64,
    /// The worker this source belongs to; it never fetches from it.
    pub attachment: Option<WorkerWithDpRank>,
    /// `(domain, value)` topology labels, hashed.
    pub topology: Vec<(u64, u64)>,
}

/// The source a target fetches from, and the prefix it holds in blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTransferSourceChoice {
    pub source: KvTransferCandidateSource,
    pub prefix_blocks: usize,
}

#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
struct TopSources {
    best: RankedSource,
    runner_up: Option<RankedSource>,
}

impl TopSources {
    fn offer(entry: Entry<'_, impl Hash + Eq, TopSources>, source: RankedSource) {
        let top = match entry {
            Entry::Occupied(top) => top.into_mut(),
            Entry::Vacant(top) => {
                top.insert(TopSources {
                    best: source,
                    runner_up: None,
                });
                return;
            }
        };
        if source.outranks(&top.best) {
            if !source.shares_attachment(&top.best) {
                top.runner_up = Some(top.best);
            }
            top.best = source;
        } else if !source.shares_attachment(&top.best)
            && top
                .runner_up
                .is_none_or(|runner_up| source.outranks(&runner_up))
        {
            top.runner_up = Some(source);
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

/// The eligible sources of one request, grouped so the source for any target
/// in its scope is found in O(1) after one pass over the sources.
pub struct KvTransferSourceSummary {
    by_type: FxHashMap<u64, TopSources>,
    by_island: FxHashMap<(u64, u64, u64), TopSources>,
    scope: KvTransferScope,
}

impl KvTransferSourceSummary {
    /// Summarize `sources` for the targets in `scope`. The router resolves the request's
    /// candidates into [`ResolvedTransferSource`]s (skipping any no longer than every target's
    /// local prefix) before calling this.
    pub fn new(
        sources: impl IntoIterator<Item = ResolvedTransferSource>,
        scope: KvTransferScope,
    ) -> Self {
        let mut summary = Self {
            by_type: FxHashMap::default(),
            by_island: FxHashMap::default(),
            scope,
        };
        for source in sources {
            let ranked = RankedSource {
                choice: KvTransferSourceChoice {
                    source: source.source,
                    prefix_blocks: source.prefix_blocks,
                },
                attachment: source.attachment,
            };
            if summary.scope.any_island {
                TopSources::offer(summary.by_type.entry(source.worker_type), ranked);
            }
            for &(domain, value) in &source.topology {
                if summary.scope.covers(domain, value) {
                    TopSources::offer(
                        summary.by_island.entry((source.worker_type, domain, value)),
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
        let pick = |top: Option<&TopSources>| {
            top?.best_for(worker)
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
    use super::*;

    #[test]
    fn scope_keeps_one_island_until_targets_differ() {
        let target = |island| KvTransferTarget {
            worker_type: 1,
            transfer_domain: Some(2),
            transfer_island: Some(island),
            preferred: false,
        };
        let one = KvTransferScope::of([(target(3), 0), (target(3), 1)]);
        assert!(one.covers(2, 3) && !one.covers(2, 4) && !one.covers(5, 3));
        let both = KvTransferScope::of([(target(3), 0), (target(4), 0)]);
        assert!(both.covers(2, 3) && both.covers(2, 6));
    }
}
