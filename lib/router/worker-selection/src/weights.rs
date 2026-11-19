// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};

use dynamo_router_protocol::worker::WorkerId;
use serde::{Deserialize, Serialize};

use crate::RuntimePolicyError;

/// A batch of process-local overrides. Null restores weight 1. IDs must belong to this pool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerWeightUpdate {
    expected_revision: u64,
    updates: BTreeMap<WorkerId, Option<f64>>,
}

impl WorkerWeightUpdate {
    pub fn new(expected_revision: u64, updates: BTreeMap<WorkerId, Option<f64>>) -> Self {
        Self {
            expected_revision,
            updates,
        }
    }
}

/// Effective weights of current pool members, ordered by worker ID.
#[derive(Debug, Serialize)]
pub struct WorkerWeightConfig {
    revision: u64,
    weights: BTreeMap<WorkerId, f64>,
}

impl WorkerWeightConfig {
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn weights(&self) -> &BTreeMap<WorkerId, f64> {
        &self.weights
    }
}

/// Typed, actor-owned traffic weights. Unspecified workers have weight 1.
///
/// A picker opts in through `WorkerPicker::worker_weights`. The host updates this state
/// between selections and reconciles discovered membership independently of request eligibility.
/// Values are relative shares, not rate or concurrency limits. No state survives a router rebuild.
#[derive(Default)]
pub struct WorkerWeights {
    revision: u64,
    members: BTreeSet<WorkerId>,
    overrides: HashMap<WorkerId, f64>,
}

impl WorkerWeights {
    /// Read the configured weight. This is a borrowed in-memory lookup without synchronization.
    pub fn weight(&self, worker: WorkerId) -> f64 {
        self.overrides.get(&worker).copied().unwrap_or(1.0)
    }

    pub(crate) fn snapshot(&self) -> WorkerWeightConfig {
        WorkerWeightConfig {
            revision: self.revision,
            weights: self
                .members
                .iter()
                .map(|&id| (id, self.weight(id)))
                .collect(),
        }
    }

    pub(crate) fn reconcile(&mut self, members: &[WorkerId]) {
        let members: BTreeSet<_> = members.iter().copied().collect();
        if self.members != members {
            self.overrides.retain(|id, _| members.contains(id));
            self.members = members;
            self.revision += 1;
        }
    }

    pub(crate) fn update(
        &mut self,
        update: WorkerWeightUpdate,
    ) -> Result<WorkerWeightConfig, RuntimePolicyError> {
        if update.expected_revision != self.revision {
            return Err(RuntimePolicyError::Conflict {
                expected: update.expected_revision,
                current: self.revision,
            });
        }
        if update.updates.is_empty() {
            return Err(RuntimePolicyError::InvalidParameters(
                "empty weight update".into(),
            ));
        }
        for (&id, weight) in &update.updates {
            if !self.members.contains(&id) {
                return Err(RuntimePolicyError::InvalidParameters(format!(
                    "worker {id} is not a member of this router"
                )));
            }
            if weight.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(RuntimePolicyError::InvalidParameters(
                    "weights must be finite and non-negative".into(),
                ));
            }
        }
        for (id, weight) in update.updates {
            match weight {
                Some(value) if value != 1.0 => {
                    self.overrides.insert(id, value);
                }
                _ => {
                    self.overrides.remove(&id);
                }
            }
        }
        self.revision += 1;
        Ok(self.snapshot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_are_atomic_and_removed_workers_lose_overrides() {
        let mut weights = WorkerWeights::default();
        weights.reconcile(&[1, 2]);
        let revision = weights.snapshot().revision;
        let update = |revision, entries| WorkerWeightUpdate {
            expected_revision: revision,
            updates: entries,
        };
        assert!(
            weights
                .update(update(
                    revision,
                    BTreeMap::from([(1, Some(0.2)), (2, Some(f64::NAN)),])
                ))
                .is_err()
        );
        assert_eq!(weights.weight(1), 1.0);
        assert_eq!(weights.snapshot().revision, revision);
        let config = weights
            .update(update(revision, BTreeMap::from([(1, Some(0.2))])))
            .unwrap();
        assert_eq!(config.weights[&2], 1.0);
        assert!(matches!(
            weights.update(update(revision, BTreeMap::from([(1, None)]))),
            Err(RuntimePolicyError::Conflict { .. })
        ));
        assert!(
            weights
                .update(update(config.revision, BTreeMap::from([(99, Some(0.0))])))
                .is_err()
        );
        weights.reconcile(&[2]);
        weights.reconcile(&[1, 2]);
        assert_eq!(weights.weight(1), 1.0);
        assert!(weights.overrides.is_empty());
        let config = weights
            .update(update(
                weights.snapshot().revision,
                BTreeMap::from([(1, Some(0.0))]),
            ))
            .unwrap();
        weights
            .update(update(config.revision, BTreeMap::from([(1, None)])))
            .unwrap();
        assert!(weights.overrides.is_empty());
    }
}
