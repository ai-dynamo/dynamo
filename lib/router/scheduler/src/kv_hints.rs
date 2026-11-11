// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests: the kernel's view of
//! `dynamo-kv-hints` (re-exported by name), the [`WorkerConfigLike`] adapter that derives a
//! worker's transfer keys, and [`resolve_transfer_sources`], which joins the indexer's
//! candidates with worker keys under a selection scope.

pub use dynamo_kv_hints::{
    KvHint, KvHintAction, KvSourceLocationsPayload, KvTransferScope, KvTransferSourceChoice,
    KvTransferSourceSummary, ResolvedTransferSource, WorkerKvTransferKeys,
    WorkerKvTransferKeysCache,
};
pub use dynamo_kv_indexer::{KvTransferCandidateSource, KvTransferCandidates};

use std::collections::HashMap;

use crate::protocols::WorkerConfigLike;
use dynamo_router_protocol::worker::WorkerId;

/// Derives a worker's hashed hint identity from its runtime config. Lives beside
/// [`WorkerConfigLike`] because the vocabulary crate does not know the config trait; the
/// body is [`WorkerKvTransferKeys::from_worker`] over the config's fields.
pub trait WorkerKvTransferKeysExt: Sized {
    /// Hash the worker's hint labels and collect its source ranks.
    fn of<C: WorkerConfigLike + ?Sized>(config: &C) -> Self;
}

impl WorkerKvTransferKeysExt for WorkerKvTransferKeys {
    fn of<C: WorkerConfigLike + ?Sized>(config: &C) -> Self {
        WorkerKvTransferKeys::from_worker(
            config.data_parallel_start_rank(),
            config.data_parallel_size(),
            |rank| config.kv_hint_transfer_metadata_for_dp_rank(rank),
            config.topology_domains(),
            config.kv_transfer_domain(),
            config.kv_transfer_enforcement(),
        )
    }
}

/// Resolve a request's transfer candidates into the sources a selection can rank: the root
/// prefixes and the suffix-only intervals, skipping any no target in `scope` can use (ending
/// at or before every target's local prefix, starting beyond every target's local prefix, or
/// longer than the retained chain), dropping worker ranks that publish no source endpoint,
/// and looking cache owners up in the routing snapshot (an owner without topology sits in
/// its attached worker's island).
pub fn resolve_transfer_sources<C: WorkerConfigLike>(
    candidates: &KvTransferCandidates,
    workers: &HashMap<WorkerId, C>,
    scope: &KvTransferScope,
) -> Vec<ResolvedTransferSource> {
    let min_local_prefix_blocks = scope.min_local_prefix_blocks();
    let max_local_prefix_blocks = scope.max_local_prefix_blocks();
    let needs_islands = scope.needs_islands();
    let snapshot = candidates.routing_snapshot.as_deref();
    let mut resolved = Vec::new();
    // A source cannot share more of the prompt than the retained chain.
    let usable = |start: usize, end: usize| {
        start <= max_local_prefix_blocks
            && end > min_local_prefix_blocks
            && end <= candidates.block_hashes.len()
            && start < end
    };
    let roots = candidates
        .owner_prefix_blocks
        .iter()
        .map(|&(source, end)| (source, 0, end));
    let intervals = candidates
        .source_intervals
        .iter()
        .map(|interval| (interval.source, interval.start, interval.end));
    for (source, start, prefix_blocks) in roots
        .chain(intervals)
        .filter(|&(_, start, end)| usable(start, end))
    {
        let (worker_type, attachment, topology) = match source {
            KvTransferCandidateSource::Worker(worker) => {
                let Some(config) = workers.get(&worker.worker_id) else {
                    continue;
                };
                let keys = config.kv_transfer_keys();
                let Some(worker_type) = keys.worker_type() else {
                    continue;
                };
                if !keys.is_source(worker.dp_rank) {
                    continue;
                }
                (worker_type, Some(worker), keys.topology().to_vec())
            }
            KvTransferCandidateSource::CacheOwner(owner) => {
                let Some(owner) = snapshot.and_then(|snapshot| snapshot.router_hint_source(owner))
                else {
                    continue;
                };
                let keys = &owner.kv_transfer_keys;
                if !keys.is_source {
                    continue;
                }
                // An owner that publishes no topology is in its attached worker's island.
                let topology = if keys.topology.is_empty() && needs_islands {
                    owner
                        .attached_worker
                        .and_then(|worker| workers.get(&worker.worker_id))
                        .map(|config| config.kv_transfer_keys().topology().to_vec())
                        .unwrap_or_default()
                } else {
                    keys.topology.clone()
                };
                (keys.worker_type, owner.attached_worker, topology)
            }
        };
        resolved.push(ResolvedTransferSource {
            source,
            start,
            prefix_blocks,
            worker_type,
            attachment,
            topology,
        });
    }
    resolved
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use dynamo_kv_hints::{KvHintTransferWorkerMetadata, KvTransferEnforcement};

    /// Two ranks, the second a source, two topology labels, a transfer domain, `preferred`.
    struct Worker {
        topology: HashMap<String, String>,
    }

    impl WorkerConfigLike for Worker {
        fn data_parallel_start_rank(&self) -> u32 {
            4
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
            dp_rank: u32,
        ) -> Option<KvHintTransferWorkerMetadata<'_>> {
            Some(KvHintTransferWorkerMetadata {
                worker_type: "prefill",
                source_control_endpoint: (dp_rank == 5).then_some("tcp://worker:23280"),
            })
        }

        fn topology_domains(&self) -> Option<&HashMap<String, String>> {
            Some(&self.topology)
        }

        fn kv_transfer_domain(&self) -> Option<&str> {
            Some("rdma-island")
        }

        fn kv_transfer_enforcement(&self) -> Option<KvTransferEnforcement> {
            Some(KvTransferEnforcement::Preferred)
        }
    }

    #[test]
    fn adapter_equals_the_direct_call() {
        let config = Worker {
            topology: HashMap::from([
                ("rdma-island".to_string(), "a".to_string()),
                ("zone".to_string(), "z1".to_string()),
            ]),
        };
        let direct = WorkerKvTransferKeys::from_worker(
            4,
            2,
            |rank| config.kv_hint_transfer_metadata_for_dp_rank(rank),
            Some(&config.topology),
            Some("rdma-island"),
            Some(KvTransferEnforcement::Preferred),
        );

        let keys = WorkerKvTransferKeys::of(&config);

        assert_eq!(keys, direct);
        assert!(keys.worker_type().is_some());
        assert!(keys.is_source(5) && !keys.is_source(4));
        assert_eq!(keys.topology().len(), 2);
        assert_eq!(keys.target(), direct.target());
    }
}

#[cfg(test)]
mod summary_tests {
    use dynamo_kv_indexer::KvTransferSourceInterval;
    use std::collections::HashMap;

    use dynamo_router_protocol::worker::WorkerWithDpRank;

    use super::KvTransferCandidates;
    use crate::protocols::WorkerConfigLike;
    use dynamo_kv_indexer::ResidencyOwnerKeyExt;
    use dynamo_router_protocol::worker::WorkerId;
    use std::sync::Arc;

    use super::*;
    use dynamo_kv_events::residency::ResidencyOwner;
    use dynamo_kv_hashing::ExternalSequenceBlockHash;
    use dynamo_kv_hints::KvHintTransferWorkerMetadata;
    use dynamo_kv_indexer::{
        ResidencyProjection, ResidencyRoutingSnapshot, RouterHintSourceMetadata,
    };
    use dynamo_router_protocol::identity::{
        CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
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
        KvTransferSourceSummary::new(
            crate::kv_hints::resolve_transfer_sources(candidates, &workers, &scope),
            scope,
        )
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
            let scope = KvTransferScope::all();
            let summary = KvTransferSourceSummary::new(
                resolve_transfer_sources(&candidates, &workers, &scope),
                scope,
            );
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
                    let scope = KvTransferScope::of([(keys.target().unwrap(), local)]);
                    let scoped = KvTransferSourceSummary::new(
                        resolve_transfer_sources(&candidates, &workers, &scope),
                        scope,
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
            KvTransferSourceInterval {
                source: KvTransferCandidateSource::CacheOwner(key(owner(4))),
                start: 1,
                end: 4,
            },
            KvTransferSourceInterval {
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
