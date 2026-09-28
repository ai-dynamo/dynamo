// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load-only selection of aggregated DGD pools from one Global View.
//!
//! This mirrors the worker selector's eligibility, score, and pick stages at
//! pool scope. Worker-local scheduler inputs and bookings have different
//! semantics from Relay snapshots, so a shared numeric cost kernel can be
//! extracted only after those signals are mapped explicitly.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::PoolId;
use super::eligibility::eligible_pools;
use super::state::{
    FreshnessPolicy, PoolRole, PoolState, PoolStateRepository, SignalState, SignalStatus,
};

/// Whether the chosen pool had comparable, fresh request load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadBasis {
    /// Observed request pressure, optionally divided by max concurrency.
    Observed,
    /// Catalog and readiness were usable, but request load was unavailable.
    Fallback,
}

/// A decision trace and the frontend endpoint to which the caller can forward.
#[derive(Clone, Debug)]
pub struct PoolDecision {
    pub pool_id: PoolId,
    pub region: String,
    pub frontend_endpoint: String,
    pub basis: LoadBasis,
    pub observed_requests: Option<u64>,
    pub assignments_since_sample: u64,
    pub normalized_by_max_concurrency: bool,
    pub cost: f64,
    pub load_status: SignalStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LoadEpoch {
    state: SignalState,
    source_observed_at_unix_ms: Option<u64>,
    received_at_unix_ms: Option<u64>,
}

impl From<&SignalStatus> for LoadEpoch {
    fn from(status: &SignalStatus) -> Self {
        Self {
            state: status.state,
            source_observed_at_unix_ms: status.source_observed_at_unix_ms,
            received_at_unix_ms: status.received_at_unix_ms,
        }
    }
}

#[derive(Clone, Copy)]
struct LocalAssignments {
    epoch: LoadEpoch,
    count: u64,
}

/// Selects one aggregated pool and counts local assignments until the next
/// Relay load sample. This prevents a burst of choices from all seeing the same
/// remote minimum. It is local to one Global Router replica.
pub struct LoadPoolSelector {
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
    assignments: Mutex<HashMap<(PoolId, String), LocalAssignments>>,
}

impl LoadPoolSelector {
    pub fn new(repository: Arc<dyn PoolStateRepository>, freshness: FreshnessPolicy) -> Self {
        Self {
            repository,
            freshness,
            assignments: Mutex::new(HashMap::new()),
        }
    }

    /// Choose the least pressured ready aggregated pool. Fresh model-level
    /// request counts are preferred; missing load enters a fallback tier rather
    /// than looking like zero. If every scored pool has a positive concurrency
    /// capacity, compare pressure per slot; otherwise compare raw counts for
    /// the first homogeneous Mocker deployment.
    pub fn select(&self, model: &str, now_unix_ms: u64) -> Option<PoolDecision> {
        let pools: Vec<_> = eligible_pools(
            self.repository.as_ref(),
            model,
            now_unix_ms,
            &self.freshness,
        )
        .into_iter()
        .filter(|pool| pool.descriptors.roles.contains(&PoolRole::Aggregated))
        .collect();
        if pools.is_empty() {
            return None;
        }

        let normalize = pools
            .iter()
            .filter(|pool| observed_requests(pool, model).is_some())
            .all(|pool| pool.capacity.max_concurrency.is_some_and(|value| value > 0));
        let mut assignments = self.assignments.lock();
        let mut best: Option<PoolDecision> = None;
        for pool in &pools {
            let key = (pool.pool_id.clone(), model.to_owned());
            let epoch = LoadEpoch::from(&pool.signal_status.load);
            let local = assignments
                .entry(key)
                .or_insert(LocalAssignments { epoch, count: 0 });
            if local.epoch != epoch {
                *local = LocalAssignments { epoch, count: 0 };
            }

            let observed = observed_requests(pool, model);
            let pressure = observed.unwrap_or(0).saturating_add(local.count);
            let cost = if observed.is_some() && normalize {
                pressure as f64 / pool.capacity.max_concurrency.unwrap() as f64
            } else {
                pressure as f64
            };
            let decision = PoolDecision {
                pool_id: pool.pool_id.clone(),
                region: pool.descriptors.location.region.clone(),
                frontend_endpoint: pool.descriptors.frontend_endpoint.clone().unwrap(),
                basis: if observed.is_some() {
                    LoadBasis::Observed
                } else {
                    LoadBasis::Fallback
                },
                observed_requests: observed,
                assignments_since_sample: local.count,
                normalized_by_max_concurrency: observed.is_some() && normalize,
                cost,
                load_status: pool.signal_status.load.clone(),
            };
            if best
                .as_ref()
                .is_none_or(|current| cheaper(&decision, current))
            {
                best = Some(decision);
            }
        }
        let selected = best?;
        let local = assignments
            .get_mut(&(selected.pool_id.clone(), model.to_owned()))
            .expect("selected pool had an assignment entry");
        local.count = local.count.saturating_add(1);
        Some(selected)
    }
}

fn observed_requests(pool: &PoolState, model: &str) -> Option<u64> {
    let status = &pool.signal_status.load;
    if !matches!(status.state, SignalState::Complete | SignalState::Degraded) {
        return None;
    }
    // Load status may be degraded because a scheduler signal is absent, while
    // model counts still cover every frontend. Partial model coverage is not a
    // comparable pool-level count and must use the fallback tier instead.
    match (status.expected_sources, status.observed_sources) {
        (Some(expected), Some(observed)) if expected > 0 && expected == observed => {}
        (None, None) => {}
        _ => return None,
    }
    let load = pool.load.request_plane.get(model)?;
    // PR #13187 counts input_processing_requests within pending_first_output_requests.
    // These two lifecycle stages are disjoint; adding input_processing would double count.
    load.pending_first_output_requests?
        .checked_add(load.output_generation_requests?)
}

fn cheaper(candidate: &PoolDecision, current: &PoolDecision) -> bool {
    let tier = |basis| match basis {
        LoadBasis::Observed => 0,
        LoadBasis::Fallback => 1,
    };
    (
        tier(candidate.basis),
        candidate.cost,
        candidate.pool_id.as_str(),
    ) < (tier(current.basis), current.cost, current.pool_id.as_str())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::global_view::state::{
        InMemoryPoolStateRepository, ModelRequestLoad, PoolCapacity, PoolDescriptors, PoolLoad,
        PoolLocation, PoolSignalStatus, PoolStateSink, ServingReadiness,
    };
    use crate::global_view::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};

    fn status(at: u64) -> SignalStatus {
        SignalStatus {
            state: SignalState::Complete,
            source_observed_at_unix_ms: Some(at),
            received_at_unix_ms: Some(at),
            ..Default::default()
        }
    }

    fn freshness() -> FreshnessPolicy {
        FreshnessPolicy {
            catalog_max_age_ms: 1_000,
            readiness_max_age_ms: 1_000,
            capacity_max_age_ms: 1_000,
            load_max_age_ms: 1_000,
            kv_usage_max_age_ms: 1_000,
            kv_overlap_max_age_ms: 1_000,
        }
    }

    fn pool(
        site: &str,
        pending: Option<u64>,
        generating: Option<u64>,
        cap: Option<u64>,
    ) -> PoolState {
        PoolState {
            pool_id: V1PoolIdDeriver.derive(&PoolKey::new(site, "dynamo", "mocker").unwrap()),
            descriptors: PoolDescriptors {
                site_id: site.into(),
                namespace: "dynamo".into(),
                dgd_name: "mocker".into(),
                location: PoolLocation {
                    region: site.into(),
                    availability_zone: None,
                    cluster: None,
                    datacenter: None,
                },
                models: vec!["model".into()],
                model_readiness: BTreeMap::from([("model".into(), ServingReadiness::Ready)]),
                roles: vec![PoolRole::Aggregated],
                frontend_endpoint: Some(format!("{site}.frontend.generate")),
                hardware: Vec::new(),
            },
            capacity: PoolCapacity {
                max_concurrency: cap,
                ..Default::default()
            },
            load: PoolLoad {
                request_plane: BTreeMap::from([(
                    "model".into(),
                    ModelRequestLoad {
                        pending_first_output_requests: pending,
                        output_generation_requests: generating,
                        input_processing_requests: Some(99),
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            },
            signal_status: PoolSignalStatus {
                catalog: status(100),
                readiness: status(100),
                capacity: status(100),
                load: status(100),
                ..Default::default()
            },
        }
    }

    #[test]
    fn ranks_observed_load_and_does_not_double_count_processing() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let a = pool("a", Some(2), Some(3), None);
        let b = pool("b", Some(6), Some(0), None);
        let expected = a.pool_id.clone();
        repo.replace(a);
        repo.replace(b);
        let selector = LoadPoolSelector::new(repo, freshness());
        let decision = selector.select("model", 100).unwrap();
        assert_eq!(decision.pool_id, expected);
        assert_eq!(decision.region, "a");
        assert_eq!(decision.observed_requests, Some(5));
        assert_eq!(decision.basis, LoadBasis::Observed);
    }

    #[test]
    fn capacity_normalizes_only_when_every_observed_pool_has_it() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let a = pool("a", Some(4), Some(0), Some(8));
        let b = pool("b", Some(3), Some(0), Some(3));
        let expected = a.pool_id.clone();
        repo.replace(a);
        repo.replace(b);
        let decision = LoadPoolSelector::new(repo, freshness())
            .select("model", 100)
            .unwrap();
        assert_eq!(decision.pool_id, expected);
        assert!(decision.normalized_by_max_concurrency);
        assert_eq!(decision.cost, 0.5);
    }

    #[test]
    fn missing_load_stays_eligible_as_fallback() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let observed = pool("observed", Some(20), Some(0), None);
        let mut missing = pool("missing", Some(0), Some(0), None);
        let expected = observed.pool_id.clone();
        missing.signal_status.load = SignalStatus::default();
        repo.replace(observed);
        repo.replace(missing);
        let selector = LoadPoolSelector::new(repo.clone(), freshness());
        assert_eq!(selector.select("model", 100).unwrap().pool_id, expected);
        repo.remove(&expected);
        let decision = selector.select("model", 100).unwrap();
        assert_eq!(decision.basis, LoadBasis::Fallback);
        assert_eq!(decision.observed_requests, None);
    }

    #[test]
    fn partial_frontend_coverage_is_fallback_not_low_load() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let mut partial = pool("partial", Some(0), Some(0), None);
        partial.signal_status.load.state = SignalState::Degraded;
        partial.signal_status.load.expected_sources = Some(2);
        partial.signal_status.load.observed_sources = Some(1);
        let mut complete = pool("complete", Some(20), Some(0), None);
        complete.signal_status.load.expected_sources = Some(2);
        complete.signal_status.load.observed_sources = Some(2);
        let expected = complete.pool_id.clone();
        repo.replace(partial);
        repo.replace(complete);
        let decision = LoadPoolSelector::new(repo, freshness())
            .select("model", 100)
            .unwrap();
        assert_eq!(decision.pool_id, expected);
        assert_eq!(decision.basis, LoadBasis::Observed);
    }

    #[test]
    fn degraded_scheduler_with_complete_frontend_coverage_uses_load() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let mut state = pool("pool", Some(2), Some(1), None);
        state.signal_status.load.state = SignalState::Degraded;
        state.signal_status.load.expected_sources = Some(2);
        state.signal_status.load.observed_sources = Some(2);
        repo.replace(state);
        let decision = LoadPoolSelector::new(repo, freshness())
            .select("model", 100)
            .unwrap();
        assert_eq!(decision.basis, LoadBasis::Observed);
        assert_eq!(decision.observed_requests, Some(3));
    }

    #[test]
    fn local_assignments_spread_ties_and_reset_on_new_sample() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let a = pool("a", Some(0), Some(0), None);
        let b = pool("b", Some(0), Some(0), None);
        let a_id = a.pool_id.clone();
        let b_id = b.pool_id.clone();
        repo.replace(a.clone());
        repo.replace(b.clone());
        let selector = LoadPoolSelector::new(repo.clone(), freshness());
        let first = selector.select("model", 100).unwrap();
        let second = selector.select("model", 100).unwrap();
        assert_ne!(first.pool_id, second.pool_id);

        let mut refreshed = if first.pool_id == a_id { a } else { b };
        refreshed.signal_status.load = status(200);
        repo.replace(refreshed);
        let third = selector.select("model", 200).unwrap();
        assert_eq!(third.pool_id, first.pool_id);
        assert_eq!(third.assignments_since_sample, 0);
        assert!(third.pool_id == a_id || third.pool_id == b_id);
    }

    #[test]
    fn non_aggregated_or_unready_pools_are_excluded() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let mut prefill = pool("prefill", Some(0), Some(0), None);
        prefill.descriptors.roles = vec![PoolRole::Prefill];
        repo.replace(prefill);
        assert!(
            LoadPoolSelector::new(repo, freshness())
                .select("model", 100)
                .is_none()
        );
    }
}
