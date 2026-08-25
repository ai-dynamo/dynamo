// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session-affinity policies used by deterministic offline replay experiments.

use std::num::NonZeroUsize;

use dynamo_kv_router::{
    protocols::{OverlapScores, WorkerWithDpRank},
    rendezvous,
};
use lru::LruCache;

const MODULO_DOMAIN: &[u8] = b"dynamo-replay-session-modulo-v1";
const HRW_DOMAIN: &[u8] = b"dynamo-replay-session-hrw-v1/aggregated";
const MAX_LOCAL_BINDINGS: usize = 65_536;

/// How much of the engine's KV event stream is visible to the replay router index.
///
/// The engine cache and actual reuse accounting are never changed by this control. Observation is
/// filtered as a complete worker/rank stream so unseen stores cannot be followed by orphan remove
/// events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplayKvObservationMode {
    #[default]
    Complete,
    DropAll,
    /// Keep complete event streams from a deterministic percentage of worker/rank targets. This
    /// preserves each visible target's parent chain while modeling a router with incomplete
    /// per-worker event subscriptions. Values above 100 are treated as 100.
    DeterministicWorkerEventKeepPercent(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReplaySessionSimulationOptions {
    pub(crate) affinity_mode: ReplaySessionAffinityMode,
    pub(crate) kv_observation_mode: ReplayKvObservationMode,
}

impl ReplaySessionSimulationOptions {
    pub(crate) fn new(
        affinity_mode: ReplaySessionAffinityMode,
        kv_observation_mode: ReplayKvObservationMode,
    ) -> Self {
        Self {
            affinity_mode,
            kv_observation_mode,
        }
    }
}

/// Session routing policy applied by the offline KV-router simulator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplaySessionAffinityMode {
    /// Preserve normal KV/load-aware selection on every turn.
    #[default]
    Disabled,
    /// Bind to the first successful normal selection in this replay router.
    ///
    /// The table is capped like production but intentionally has no TTL because this first A/B
    /// fixture has no virtual-time affinity configuration. It is an idealized local baseline.
    Local,
    /// Deterministically map the session hash modulo the canonical target list.
    Modulo,
    /// Deterministically select the highest Rendezvous/HRW-scored replay worker/rank target.
    /// Replay workers currently expose ephemeral worker IDs rather than production stable IDs.
    Hrw,
    /// At arrival, restrict candidates to targets whose KV-cache deficit from the best-overlap
    /// target is within `max_extra_uncached_tokens`, then choose the highest HRW-scored target.
    ///
    /// This simulation-only control becomes an exact queue pin and is not reevaluated after
    /// arrival. It deliberately composes a physical cache-work budget with HRW as a deterministic
    /// ordering; it does not model the proposed soft, queue-reevaluable production spill policy.
    HrwKvOverlapBoundedArrivalPin { max_extra_uncached_tokens: usize },
    /// Prefer a router-local LRU binding, falling back to HRW on a miss. At arrival, the replay
    /// router compares that preferred target with the existing KV/load-aware selector cost and
    /// exact-pins it only when its extra cost is within `max_cost_regret_blocks`.
    ///
    /// The LRU records the latest successful actual dispatch, including a spill, because that is
    /// where the newest turn's KV resides; the separate HRW home remains unchanged. The arrival
    /// decision is not reevaluated after queueing, so this remains a simulation proxy for a
    /// production soft preference. `max_entries` is clamped to `[1, 65_536]`.
    LocalLruHrwKvCostBoundedArrivalPin {
        max_entries: usize,
        max_cost_regret_blocks: u32,
    },
}

pub(crate) struct ReplaySessionAffinity {
    mode: ReplaySessionAffinityMode,
    local_bindings: Option<LruCache<String, WorkerWithDpRank>>,
}

impl ReplaySessionAffinity {
    pub(crate) fn new(mode: ReplaySessionAffinityMode) -> Self {
        let capacity = match mode {
            ReplaySessionAffinityMode::Local => Some(MAX_LOCAL_BINDINGS),
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_entries, ..
            } => Some(max_entries.clamp(1, MAX_LOCAL_BINDINGS)),
            _ => None,
        };
        Self {
            mode,
            local_bindings: capacity.map(|capacity| {
                // `LruCache::new` reserves the full capacity immediately. Start unbounded and
                // resize while empty to retain the cap without preallocating attacker-controlled
                // or unused capacity.
                let mut cache = LruCache::unbounded();
                cache
                    .resize(NonZeroUsize::new(capacity).expect("session LRU capacity is non-zero"));
                cache
            }),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.mode != ReplaySessionAffinityMode::Disabled
    }

    pub(crate) fn preferred_target(
        &mut self,
        session_id: Option<&str>,
        candidates: &[WorkerWithDpRank],
    ) -> Option<WorkerWithDpRank> {
        let session_id = session_id?;
        if candidates.is_empty() {
            return None;
        }
        match self.mode {
            ReplaySessionAffinityMode::Disabled => None,
            ReplaySessionAffinityMode::Local => {
                let bindings = self
                    .local_bindings
                    .as_mut()
                    .expect("local affinity has an LRU");
                let target = bindings.get(session_id).copied()?;
                if candidates.contains(&target) {
                    Some(target)
                } else {
                    bindings.pop(session_id);
                    None
                }
            }
            ReplaySessionAffinityMode::Modulo => {
                let mut canonical = candidates.to_vec();
                canonical.sort_unstable();
                let hash = rendezvous::score(MODULO_DOMAIN, session_id.as_bytes(), b"modulo", 0);
                let index = (hash % canonical.len() as u64) as usize;
                Some(canonical[index])
            }
            ReplaySessionAffinityMode::Hrw
            | ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin { .. } => {
                Self::hrw_target(session_id, candidates.iter().copied())
            }
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin { .. } => {
                let bindings = self
                    .local_bindings
                    .as_mut()
                    .expect("hybrid affinity has an LRU");
                if let Some(target) = bindings.get(session_id).copied() {
                    if candidates.contains(&target) {
                        return Some(target);
                    }
                    bindings.pop(session_id);
                }
                Self::hrw_target(session_id, candidates.iter().copied())
            }
        }
    }

    pub(crate) fn max_cost_regret_blocks(&self) -> Option<f64> {
        match self.mode {
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_cost_regret_blocks,
                ..
            } => Some(f64::from(max_cost_regret_blocks)),
            _ => None,
        }
    }

    pub(crate) fn preferred_target_with_overlap(
        &mut self,
        session_id: Option<&str>,
        candidates: &[WorkerWithDpRank],
        overlaps: &OverlapScores,
        block_size: u32,
    ) -> Option<WorkerWithDpRank> {
        let ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
            max_extra_uncached_tokens,
        } = self.mode
        else {
            return self.preferred_target(session_id, candidates);
        };
        let session_id = session_id?;
        let best_overlap_blocks = candidates
            .iter()
            .map(|worker| overlaps.scores.get(worker).copied().unwrap_or(0))
            .max()?;
        let block_size = block_size as usize;
        let admissible = candidates.iter().copied().filter(|worker| {
            let overlap_blocks = overlaps.scores.get(worker).copied().unwrap_or(0);
            let extra_uncached_tokens =
                ((best_overlap_blocks - overlap_blocks) as usize).saturating_mul(block_size);
            extra_uncached_tokens <= max_extra_uncached_tokens
        });
        Self::hrw_target(session_id, admissible)
    }

    fn hrw_target(
        session_id: &str,
        candidates: impl Iterator<Item = WorkerWithDpRank>,
    ) -> Option<WorkerWithDpRank> {
        candidates
            .map(|worker| {
                let worker_id = worker.worker_id.to_le_bytes();
                let score = rendezvous::score(
                    HRW_DOMAIN,
                    session_id.as_bytes(),
                    &worker_id,
                    worker.dp_rank,
                );
                (worker, score)
            })
            .max_by(|(left_worker, left_score), (right_worker, right_score)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| right_worker.cmp(left_worker))
            })
            .map(|(worker, _)| worker)
    }

    pub(crate) fn observe_selection(
        &mut self,
        session_id: Option<&str>,
        selected: WorkerWithDpRank,
    ) {
        match (self.mode, session_id) {
            (ReplaySessionAffinityMode::Local, Some(session_id)) => {
                let bindings = self
                    .local_bindings
                    .as_mut()
                    .expect("local affinity has an LRU");
                if bindings.get(session_id).is_none() {
                    bindings.put(session_id.to_string(), selected);
                }
            }
            (
                ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin { .. },
                Some(session_id),
            ) => {
                // The logical HRW home remains immutable; the LRU separately records the latest
                // successful physical dispatch because that is where the newest turn's KV lives.
                self.local_bindings
                    .as_mut()
                    .expect("hybrid affinity has an LRU")
                    .put(session_id.to_string(), selected);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workers(count: u64) -> Vec<WorkerWithDpRank> {
        worker_rank_targets(count, 1)
    }

    fn worker_rank_targets(worker_count: u64, ranks_per_worker: u32) -> Vec<WorkerWithDpRank> {
        (0..worker_count)
            .flat_map(|worker_id| {
                (0..ranks_per_worker).map(move |dp_rank| WorkerWithDpRank::new(worker_id, dp_rank))
            })
            .collect()
    }

    #[test]
    fn hrw_is_independent_of_candidate_order() {
        let forward = workers(8);
        let mut reverse = forward.clone();
        reverse.reverse();
        let mut left = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        let mut right = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        assert_eq!(
            left.preferred_target(Some("session-a"), &forward),
            right.preferred_target(Some("session-a"), &reverse)
        );
    }

    #[test]
    fn hrw_agrees_on_the_same_worker_and_dp_rank_across_routers() {
        let forward = worker_rank_targets(3, 4);
        let mut reverse = forward.clone();
        reverse.reverse();
        let mut left = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        let mut right = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);

        for idx in 0..10_000 {
            let session = format!("session-{idx}");
            let left_target = left.preferred_target(Some(&session), &forward).unwrap();
            let right_target = right.preferred_target(Some(&session), &reverse).unwrap();
            assert_eq!(left_target, right_target, "routers disagreed for {session}");
            assert!(left_target.worker_id < 3);
            assert!(left_target.dp_rank < 4);
        }
    }

    #[test]
    fn hrw_removal_moves_only_sessions_on_removed_target() {
        let before = workers(8);
        let removed = before[3];
        let after = before
            .iter()
            .copied()
            .filter(|worker| *worker != removed)
            .collect::<Vec<_>>();
        let mut router = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        for idx in 0..10_000 {
            let session = format!("session-{idx}");
            let old = router.preferred_target(Some(&session), &before).unwrap();
            let new = router.preferred_target(Some(&session), &after).unwrap();
            if old != removed {
                assert_eq!(old, new, "unaffected session {session} remapped");
            }
        }
    }

    #[test]
    fn hrw_dp_rank_removal_moves_only_sessions_homed_on_that_rank() {
        let before = worker_rank_targets(3, 4);
        let removed = WorkerWithDpRank::new(1, 2);
        let after = before
            .iter()
            .copied()
            .filter(|target| *target != removed)
            .collect::<Vec<_>>();
        let mut router = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        let mut affected = 0;

        for idx in 0..10_000 {
            let session = format!("session-{idx}");
            let old = router.preferred_target(Some(&session), &before).unwrap();
            let new = router.preferred_target(Some(&session), &after).unwrap();
            if old == removed {
                affected += 1;
                assert_ne!(new, removed);
            } else {
                assert_eq!(old, new, "unaffected session {session} remapped");
            }
        }

        assert!(affected > 0, "fixture did not exercise the removed DP rank");
    }

    #[test]
    fn kv_overlap_bound_controls_when_hrw_home_can_override_best_overlap() {
        let candidates = workers(3);
        let session = "session-kv-conflict";
        let mut strict = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        let home = strict.preferred_target(Some(session), &candidates).unwrap();
        let kv_best = candidates
            .iter()
            .copied()
            .find(|candidate| *candidate != home)
            .unwrap();
        let mut overlaps = dynamo_kv_router::protocols::OverlapScores::default();
        overlaps.scores.insert(kv_best, 16);
        overlaps.scores.insert(home, 8);

        let mut tight =
            ReplaySessionAffinity::new(ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 64,
            });
        assert_eq!(
            tight.preferred_target_with_overlap(Some(session), &candidates, &overlaps, 16),
            Some(kv_best),
            "the HRW home is eight blocks colder and must be outside a four-block budget"
        );

        let mut relaxed =
            ReplaySessionAffinity::new(ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
                max_extra_uncached_tokens: 128,
            });
        assert_eq!(
            relaxed.preferred_target_with_overlap(Some(session), &candidates, &overlaps, 16),
            Some(home),
            "the HRW home becomes admissible at an eight-block budget"
        );
    }

    #[test]
    fn kv_overlap_bounded_dispatch_can_disagree_on_divergent_cache_snapshots() {
        let candidates = workers(3);
        let session = "session-divergent-cache";
        let mut left_overlap = dynamo_kv_router::protocols::OverlapScores::default();
        left_overlap.scores.insert(candidates[0], 16);
        let mut right_overlap = dynamo_kv_router::protocols::OverlapScores::default();
        right_overlap.scores.insert(candidates[1], 16);
        let mode = ReplaySessionAffinityMode::HrwKvOverlapBoundedArrivalPin {
            max_extra_uncached_tokens: 0,
        };
        let mut left = ReplaySessionAffinity::new(mode);
        let mut right = ReplaySessionAffinity::new(mode);

        assert_ne!(
            left.preferred_target_with_overlap(Some(session), &candidates, &left_overlap, 16,),
            right.preferred_target_with_overlap(Some(session), &candidates, &right_overlap, 16,),
            "dynamic overlap can change dispatch even though strict HRW home remains identical"
        );

        let mut strict_left = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        let mut strict_right = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        assert_eq!(
            strict_left.preferred_target(Some(session), &candidates),
            strict_right.preferred_target(Some(session), &candidates),
        );
    }

    #[test]
    fn hrw_scale_out_remaps_far_less_than_modulo() {
        let before = workers(8);
        let after = workers(9);
        let count_remaps = |mode| {
            let mut router = ReplaySessionAffinity::new(mode);
            (0..10_000)
                .filter(|idx| {
                    let session = format!("session-{idx}");
                    router.preferred_target(Some(&session), &before)
                        != router.preferred_target(Some(&session), &after)
                })
                .count()
        };
        let hrw_remaps = count_remaps(ReplaySessionAffinityMode::Hrw);
        let modulo_remaps = count_remaps(ReplaySessionAffinityMode::Modulo);
        let expected_hrw = 10_000.0 / 9.0;
        assert!((hrw_remaps as f64 - expected_hrw).abs() < 150.0);
        assert!(
            modulo_remaps > hrw_remaps * 5,
            "modulo={modulo_remaps}, hrw={hrw_remaps}"
        );
    }

    #[test]
    fn local_binding_is_committed_only_after_observation() {
        let candidates = workers(4);
        let mut router = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Local);
        assert_eq!(
            router.preferred_target(Some("session-a"), &candidates),
            None
        );
        router.observe_selection(Some("session-a"), candidates[2]);
        assert_eq!(
            router.preferred_target(Some("session-a"), &candidates),
            Some(candidates[2])
        );
    }

    #[test]
    fn hybrid_lru_refreshes_hits_and_evicts_the_least_recent_session() {
        let candidates = workers(4);
        let mode = ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
            max_entries: 2,
            max_cost_regret_blocks: 0,
        };
        let mut router = ReplaySessionAffinity::new(mode);

        router.observe_selection(Some("session-a"), candidates[0]);
        router.observe_selection(Some("session-b"), candidates[1]);
        assert_eq!(
            router.preferred_target(Some("session-a"), &candidates),
            Some(candidates[0]),
            "reading session-a must refresh its LRU position"
        );
        router.observe_selection(Some("session-c"), candidates[2]);

        assert_eq!(
            router.preferred_target(Some("session-a"), &candidates),
            Some(candidates[0])
        );
        assert_eq!(
            router.preferred_target(Some("session-c"), &candidates),
            Some(candidates[2])
        );
        assert_ne!(
            router.preferred_target(Some("session-b"), &candidates),
            Some(candidates[1]),
            "session-b was least recently used and must have been evicted"
        );
    }

    #[test]
    fn lru_is_allocated_only_for_local_modes_and_capacity_is_bounded() {
        let hrw = ReplaySessionAffinity::new(ReplaySessionAffinityMode::Hrw);
        assert!(hrw.local_bindings.is_none());

        let hybrid = ReplaySessionAffinity::new(
            ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
                max_entries: usize::MAX,
                max_cost_regret_blocks: 0,
            },
        );
        assert_eq!(
            hybrid.local_bindings.as_ref().unwrap().cap().get(),
            MAX_LOCAL_BINDINGS
        );
        assert_eq!(hybrid.local_bindings.as_ref().unwrap().len(), 0);
    }

    #[test]
    fn hybrid_uses_hrw_on_miss_then_remembers_the_admitted_target() {
        let candidates = workers(4);
        let mode = ReplaySessionAffinityMode::LocalLruHrwKvCostBoundedArrivalPin {
            max_entries: 8,
            max_cost_regret_blocks: 4,
        };
        let mut router = ReplaySessionAffinity::new(mode);
        let initial_home = router
            .preferred_target(Some("session-a"), &candidates)
            .unwrap();
        let selected = candidates
            .iter()
            .copied()
            .find(|candidate| *candidate != initial_home)
            .unwrap();

        router.observe_selection(Some("session-a"), selected);

        assert_eq!(
            router.preferred_target(Some("session-a"), &candidates),
            Some(selected)
        );

        let later_selection = candidates
            .iter()
            .copied()
            .find(|candidate| *candidate != selected)
            .unwrap();
        router.observe_selection(Some("session-a"), later_selection);
        assert_eq!(
            router.preferred_target(Some("session-a"), &candidates),
            Some(later_selection),
            "the active anchor must follow the latest successful dispatch"
        );
    }
}
