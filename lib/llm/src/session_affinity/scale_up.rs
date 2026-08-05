// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use dynamo_kv_router::protocols::WorkerId;
use parking_lot::RwLock;
use xxhash_rust::xxh3::xxh3_64;

use crate::{
    discovery::RuntimeConfigWatch, local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::extensions::SessionAffinityId,
};

/// One immutable view of the workers that contribute usable KV capacity to a
/// model. Affinity entries retain an `Arc` to the view they last evaluated so
/// additions that occur between two turns can be identified without scanning
/// or rewriting every entry when topology changes.
#[derive(Debug, Default)]
pub(crate) struct ScaleUpSnapshot {
    generation: u64,
    workers: BTreeMap<WorkerId, u128>,
    total_capacity: u128,
}

impl ScaleUpSnapshot {
    fn new(generation: u64, workers: BTreeMap<WorkerId, u128>) -> Self {
        let total_capacity = workers.values().copied().sum();
        Self {
            generation,
            workers,
            total_capacity,
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg(test)]
    pub(crate) fn from_workers(generation: u64, workers: &[(WorkerId, u128)]) -> Arc<Self> {
        Arc::new(Self::new(generation, workers.iter().copied().collect()))
    }
}

/// Result of comparing an affinity entry's previous topology with the current
/// topology. `migration_workers` is populated only for the deterministic
/// capacity-proportional cohort selected to move on this scale-up.
pub(crate) struct ScaleUpEvaluation {
    pub(crate) snapshot: Arc<ScaleUpSnapshot>,
    pub(crate) migration_workers: Option<Arc<HashSet<WorkerId>>>,
}

#[derive(Clone)]
pub(crate) struct ScaleUpMigrationTracker {
    scope: Arc<str>,
    current: Arc<RwLock<Arc<ScaleUpSnapshot>>>,
}

impl ScaleUpMigrationTracker {
    pub(crate) fn new(scope: String, mut runtime_configs: RuntimeConfigWatch) -> Self {
        let initial_workers = worker_capacities(&runtime_configs.borrow());
        let has_initial_baseline = initial_workers
            .as_ref()
            .is_some_and(|workers| !workers.is_empty());
        let initial = Arc::new(ScaleUpSnapshot::new(0, initial_workers.unwrap_or_default()));
        let tracker = Self {
            scope: Arc::from(scope),
            current: Arc::new(RwLock::new(initial)),
        };

        let background = tracker.clone();
        tokio::spawn(async move {
            let mut has_baseline = has_initial_baseline;
            loop {
                if runtime_configs.changed().await.is_err() {
                    return;
                }

                let configs = runtime_configs.borrow_and_update().clone();
                let Some(workers) = worker_capacities(&configs) else {
                    tracing::warn!(
                        model = %background.scope,
                        "Skipping affinity scale-up update because at least one worker has missing or zero KV capacity"
                    );
                    continue;
                };

                let previous = background.snapshot();
                if !has_baseline {
                    *background.current.write() = Arc::new(ScaleUpSnapshot::new(0, workers));
                    has_baseline = true;
                    continue;
                }

                let added_workers: Vec<_> = workers
                    .keys()
                    .filter(|worker_id| !previous.workers.contains_key(worker_id))
                    .copied()
                    .collect();
                let generation = previous
                    .generation
                    .saturating_add(u64::from(!added_workers.is_empty()));
                let next = Arc::new(ScaleUpSnapshot::new(generation, workers));

                if !added_workers.is_empty() {
                    let added_capacity: u128 = added_workers
                        .iter()
                        .filter_map(|worker_id| next.workers.get(worker_id))
                        .copied()
                        .sum();
                    tracing::info!(
                        model = %background.scope,
                        generation,
                        ?added_workers,
                        added_capacity,
                        total_capacity = next.total_capacity,
                        "Detected affinity scale-up"
                    );
                }

                *background.current.write() = next;
            }
        });

        tracker
    }

    pub(crate) fn snapshot(&self) -> Arc<ScaleUpSnapshot> {
        self.current.read().clone()
    }

    #[cfg(test)]
    pub(crate) fn for_test(scope: &str, current: Arc<ScaleUpSnapshot>) -> Self {
        Self {
            scope: Arc::from(scope),
            current: Arc::new(RwLock::new(current)),
        }
    }

    #[cfg(test)]
    pub(crate) fn set_snapshot_for_test(&self, snapshot: Arc<ScaleUpSnapshot>) {
        *self.current.write() = snapshot;
    }

    pub(crate) fn evaluate(
        &self,
        session_id: &SessionAffinityId,
        previous: &Arc<ScaleUpSnapshot>,
    ) -> ScaleUpEvaluation {
        let current = self.snapshot();
        // An entry created before the runtime-config watch established its
        // startup baseline must adopt the first real snapshot without treating
        // every startup worker as newly added.
        if previous.total_capacity == 0
            || current.generation <= previous.generation
            || current.total_capacity == 0
        {
            return ScaleUpEvaluation {
                snapshot: current,
                migration_workers: None,
            };
        }

        let migration_workers: HashSet<_> = current
            .workers
            .keys()
            .filter(|worker_id| !previous.workers.contains_key(worker_id))
            .copied()
            .collect();
        let added_capacity: u128 = migration_workers
            .iter()
            .filter_map(|worker_id| current.workers.get(worker_id))
            .copied()
            .sum();
        if migration_workers.is_empty() || added_capacity == 0 {
            return ScaleUpEvaluation {
                snapshot: current,
                migration_workers: None,
            };
        }

        let selected = selected_for_scale_up(
            &self.scope,
            session_id.as_str(),
            current.generation,
            added_capacity,
            current.total_capacity,
        );
        ScaleUpEvaluation {
            snapshot: current,
            migration_workers: selected.then(|| Arc::new(migration_workers)),
        }
    }
}

fn worker_capacities(
    configs: &std::collections::HashMap<WorkerId, ModelRuntimeConfig>,
) -> Option<BTreeMap<WorkerId, u128>> {
    let mut capacities = BTreeMap::new();
    for (&worker_id, config) in configs {
        let per_rank = config.total_kv_blocks.filter(|capacity| *capacity > 0)? as u128;
        let ranks = u128::from(config.data_parallel_size.max(1));
        capacities.insert(worker_id, per_rank.saturating_mul(ranks));
    }
    Some(capacities)
}

fn selected_for_scale_up(
    scope: &str,
    session_id: &str,
    generation: u64,
    added_capacity: u128,
    total_capacity: u128,
) -> bool {
    debug_assert!(added_capacity <= total_capacity);
    if added_capacity == 0 || total_capacity == 0 {
        return false;
    }
    if added_capacity == total_capacity {
        return true;
    }

    // Length delimiters avoid ambiguous concatenations such as ("ab", "c")
    // and ("a", "bc"). XXH3 is already the stable routing hash used by the
    // LLM crate and is inexpensive on short session IDs.
    let mut bytes = Vec::with_capacity(scope.len() + session_id.len() + 24);
    bytes.extend_from_slice(&(scope.len() as u64).to_le_bytes());
    bytes.extend_from_slice(scope.as_bytes());
    bytes.extend_from_slice(&(session_id.len() as u64).to_le_bytes());
    bytes.extend_from_slice(session_id.as_bytes());
    bytes.extend_from_slice(&generation.to_le_bytes());
    let hash = u128::from(xxh3_64(&bytes));
    let hash_space = u128::from(u64::MAX) + 1;
    let cutoff = hash_space.saturating_mul(added_capacity) / total_capacity;
    hash < cutoff
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(value: &str) -> SessionAffinityId {
        SessionAffinityId::new(value.to_string())
    }

    #[test]
    fn capacity_multiplies_per_rank_blocks_by_dp_size() {
        let mut configs = std::collections::HashMap::new();
        let config = ModelRuntimeConfig {
            total_kv_blocks: Some(100),
            data_parallel_size: 4,
            ..Default::default()
        };
        configs.insert(7, config);

        assert_eq!(worker_capacities(&configs).unwrap()[&7], 400);
    }

    #[tokio::test]
    async fn runtime_config_watch_detects_a_ready_worker_addition() {
        let first_config = ModelRuntimeConfig {
            total_kv_blocks: Some(70),
            ..Default::default()
        };
        let initial = std::collections::HashMap::from([(10, first_config.clone())]);
        let (sender, receiver) = tokio::sync::watch::channel(initial.clone());
        let tracker = ScaleUpMigrationTracker::new("model".to_string(), receiver);
        let previous = tracker.snapshot();

        let mut scaled = initial;
        let second_config = ModelRuntimeConfig {
            total_kv_blocks: Some(30),
            ..Default::default()
        };
        scaled.insert(20, second_config);
        sender.send(scaled).unwrap();

        let current = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let current = tracker.snapshot();
                if current.generation > previous.generation {
                    break current;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("scale-up watcher did not observe the new worker");

        assert_eq!(current.total_capacity, 100);
        assert_eq!(
            current.workers.keys().copied().collect::<Vec<_>>(),
            [10, 20]
        );
    }

    #[test]
    fn missing_capacity_disables_snapshot() {
        let mut configs = std::collections::HashMap::new();
        configs.insert(7, ModelRuntimeConfig::default());
        assert!(worker_capacities(&configs).is_none());
    }

    #[test]
    fn empty_startup_snapshot_is_adopted_without_migration() {
        let startup = ScaleUpSnapshot::from_workers(0, &[]);
        let ready = ScaleUpSnapshot::from_workers(1, &[(10, 100), (20, 100)]);
        let tracker = ScaleUpMigrationTracker::for_test("model", ready.clone());

        let evaluation = tracker.evaluate(&session("existing-session"), &startup);
        assert!(evaluation.migration_workers.is_none());
        assert!(Arc::ptr_eq(&evaluation.snapshot, &ready));
    }

    #[test]
    fn evaluation_only_targets_new_workers() {
        let previous = ScaleUpSnapshot::from_workers(1, &[(10, 100)]);
        let current = ScaleUpSnapshot::from_workers(2, &[(10, 100), (20, 200)]);
        let tracker = ScaleUpMigrationTracker::for_test("model", current.clone());

        let mut selected = None;
        for index in 0..10_000 {
            let evaluation = tracker.evaluate(&session(&format!("session-{index}")), &previous);
            if evaluation.migration_workers.is_some() {
                selected = Some(evaluation);
                break;
            }
        }
        let evaluation = selected.expect("two-thirds cohort should select at least one key");
        assert_eq!(evaluation.snapshot.generation, 2);
        assert_eq!(
            evaluation.migration_workers.unwrap().as_ref(),
            &HashSet::from([20])
        );
    }

    #[test]
    fn selection_is_deterministic_and_capacity_proportional() {
        let selected = (0..20_000)
            .filter(|index| selected_for_scale_up("model", &format!("session-{index}"), 8, 30, 100))
            .count();
        assert!((5_800..=6_200).contains(&selected), "selected {selected}");

        assert_eq!(
            selected_for_scale_up("model", "same", 8, 30, 100),
            selected_for_scale_up("model", "same", 8, 30, 100)
        );
    }

    #[test]
    fn missed_generations_union_all_new_workers() {
        let previous = ScaleUpSnapshot::from_workers(1, &[(10, 100)]);
        let current = ScaleUpSnapshot::from_workers(3, &[(10, 100), (20, 100), (30, 100)]);
        let tracker = ScaleUpMigrationTracker::for_test("model", current);

        let mut selected = None;
        for index in 0..10_000 {
            let evaluation = tracker.evaluate(&session(&format!("session-{index}")), &previous);
            if let Some(workers) = evaluation.migration_workers {
                selected = Some(workers);
                break;
            }
        }
        assert_eq!(selected.unwrap().as_ref(), &HashSet::from([20, 30]));
    }
}
