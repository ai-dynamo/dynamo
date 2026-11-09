// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral pool snapshots for the Global View.
//!
//! Relay adapters assemble these snapshots from independent source streams.
//! The repository exposes complete per-pool replacements to routing and
//! diagnostics consumers without exposing relay protobuf types.

use std::collections::{BTreeMap, HashMap};

use parking_lot::RwLock;
use serde::Serialize;

use super::PoolId;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PoolLocation {
    pub region: String,
    pub availability_zone: Option<String>,
    pub cluster: Option<String>,
    pub datacenter: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolRole {
    Aggregated,
    Prefill,
    Decode,
    Encode,
    Legacy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingReadiness {
    Ready,
    Unavailable,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PoolDescriptors {
    pub site_id: String,
    /// Kubernetes namespace containing the DGD.
    pub namespace: String,
    pub dgd_name: String,
    pub location: PoolLocation,
    pub models: Vec<String>,
    /// Readiness is model-specific, independent of signal freshness.
    pub model_readiness: BTreeMap<String, ServingReadiness>,
    pub roles: Vec<PoolRole>,
    pub frontend_endpoint: Option<String>,
    pub hardware: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PoolCapacity {
    pub live_workers: Option<u64>,
    pub max_concurrency: Option<u64>,
    pub kv_capacity_blocks: Option<u64>,
    pub expected_ranks: Option<u64>,
    pub observed_ranks: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ModelRequestLoad {
    pub ready_frontends: Option<u64>,
    pub pending_first_output_requests: Option<u64>,
    pub pending_first_output_input_tokens: Option<u64>,
    pub live_input_tokens: Option<u64>,
    pub input_processing_requests: Option<u64>,
    pub output_generation_requests: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SchedulerLoad {
    pub active_prefill_tokens: Option<u64>,
    pub active_decode_blocks: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvUsage {
    pub used_blocks: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PoolLoad {
    /// Keyed by canonical model name. Model-level load is counted once per DGD.
    pub request_plane: BTreeMap<String, ModelRequestLoad>,
    pub scheduler: BTreeMap<PoolRole, SchedulerLoad>,
    pub kv_usage: Option<KvUsage>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalState {
    Complete,
    Degraded,
    #[default]
    Unavailable,
    Stale,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SignalStatus {
    pub state: SignalState,
    pub source_observed_at_unix_ms: Option<u64>,
    pub received_at_unix_ms: Option<u64>,
    pub expected_sources: Option<u64>,
    pub observed_sources: Option<u64>,
}

impl SignalStatus {
    fn expire_at(&mut self, now_unix_ms: u64, max_age_ms: u64) {
        if !matches!(self.state, SignalState::Complete | SignalState::Degraded) {
            return;
        }
        let fresh = self
            .received_at_unix_ms
            .and_then(|received| now_unix_ms.checked_sub(received))
            .is_some_and(|age| age <= max_age_ms);
        if !fresh {
            self.state = SignalState::Stale;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FreshnessPolicy {
    pub catalog_max_age_ms: u64,
    pub readiness_max_age_ms: u64,
    pub capacity_max_age_ms: u64,
    pub load_max_age_ms: u64,
    pub kv_usage_max_age_ms: u64,
    pub kv_overlap_max_age_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PoolSignalStatus {
    pub catalog: SignalStatus,
    pub readiness: SignalStatus,
    pub capacity: SignalStatus,
    pub load: SignalStatus,
    pub kv_usage: SignalStatus,
    /// Availability and freshness only; no CKF payload is exposed.
    pub kv_overlap: SignalStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PoolState {
    pub pool_id: PoolId,
    pub descriptors: PoolDescriptors,
    pub capacity: PoolCapacity,
    pub load: PoolLoad,
    pub signal_status: PoolSignalStatus,
}

impl PoolState {
    fn with_freshness(mut self, now_unix_ms: u64, policy: &FreshnessPolicy) -> Self {
        let status = &mut self.signal_status;
        status
            .catalog
            .expire_at(now_unix_ms, policy.catalog_max_age_ms);
        status
            .readiness
            .expire_at(now_unix_ms, policy.readiness_max_age_ms);
        status
            .capacity
            .expire_at(now_unix_ms, policy.capacity_max_age_ms);
        status.load.expire_at(now_unix_ms, policy.load_max_age_ms);
        status
            .kv_usage
            .expire_at(now_unix_ms, policy.kv_usage_max_age_ms);
        status
            .kv_overlap
            .expire_at(now_unix_ms, policy.kv_overlap_max_age_ms);
        if !matches!(
            status.capacity.state,
            SignalState::Complete | SignalState::Degraded
        ) {
            self.capacity = PoolCapacity::default();
        }
        if !matches!(
            status.load.state,
            SignalState::Complete | SignalState::Degraded
        ) {
            self.load.request_plane.clear();
            self.load.scheduler.clear();
        }
        if !matches!(
            status.kv_usage.state,
            SignalState::Complete | SignalState::Degraded
        ) {
            self.load.kv_usage = None;
        }
        self
    }
}

/// Read side used by routing and the optional POC diagnostics endpoints.
/// Every read applies explicit per-signal freshness limits.
pub trait PoolStateRepository: Send + Sync {
    fn get(
        &self,
        pool_id: &PoolId,
        now_unix_ms: u64,
        freshness: &FreshnessPolicy,
    ) -> Option<PoolState>;
    fn list(&self, now_unix_ms: u64, freshness: &FreshnessPolicy) -> Vec<PoolState>;
}

/// Write side used by a relay adapter after it assembles a full pool snapshot.
pub trait PoolStateSink: Send + Sync {
    fn replace(&self, state: PoolState);
    fn remove(&self, pool_id: &PoolId);
}

/// In-memory state owned by one Global Router replica.
#[derive(Default)]
pub struct InMemoryPoolStateRepository {
    pools: RwLock<HashMap<PoolId, PoolState>>,
}

impl PoolStateRepository for InMemoryPoolStateRepository {
    fn get(
        &self,
        pool_id: &PoolId,
        now_unix_ms: u64,
        freshness: &FreshnessPolicy,
    ) -> Option<PoolState> {
        self.pools
            .read()
            .get(pool_id)
            .cloned()
            .map(|state| state.with_freshness(now_unix_ms, freshness))
    }

    fn list(&self, now_unix_ms: u64, freshness: &FreshnessPolicy) -> Vec<PoolState> {
        let mut pools: Vec<_> = self
            .pools
            .read()
            .values()
            .cloned()
            .map(|state| state.with_freshness(now_unix_ms, freshness))
            .collect();
        pools.sort_by(|a, b| a.pool_id.as_str().cmp(b.pool_id.as_str()));
        pools
    }
}

impl PoolStateSink for InMemoryPoolStateRepository {
    fn replace(&self, state: PoolState) {
        self.pools.write().insert(state.pool_id.clone(), state);
    }

    fn remove(&self, pool_id: &PoolId) {
        self.pools.write().remove(pool_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_view::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};

    fn complete_at(received_at_unix_ms: u64) -> SignalStatus {
        SignalStatus {
            state: SignalState::Complete,
            received_at_unix_ms: Some(received_at_unix_ms),
            ..Default::default()
        }
    }

    fn policy() -> FreshnessPolicy {
        FreshnessPolicy {
            catalog_max_age_ms: 1_000,
            readiness_max_age_ms: 1_000,
            capacity_max_age_ms: 1_000,
            load_max_age_ms: 100,
            kv_usage_max_age_ms: 1_000,
            kv_overlap_max_age_ms: 1_000,
        }
    }

    fn pool() -> PoolState {
        let key = PoolKey::new("ohio", "dynamo", "mocker-1").unwrap();
        let status = complete_at(1_000);
        PoolState {
            pool_id: V1PoolIdDeriver.derive(&key),
            descriptors: PoolDescriptors {
                site_id: key.site_id().into(),
                namespace: key.namespace().into(),
                dgd_name: key.dgd_name().into(),
                location: PoolLocation {
                    region: "us-east-2".into(),
                    availability_zone: Some("us-east-2a".into()),
                    cluster: None,
                    datacenter: None,
                },
                models: vec!["model".into()],
                model_readiness: BTreeMap::from([("model".into(), ServingReadiness::Ready)]),
                roles: vec![PoolRole::Aggregated],
                frontend_endpoint: Some("dynamo-mocker-1.frontend.generate".into()),
                hardware: Vec::new(),
            },
            capacity: PoolCapacity {
                live_workers: Some(2),
                ..Default::default()
            },
            load: PoolLoad {
                request_plane: BTreeMap::from([("model".into(), ModelRequestLoad::default())]),
                kv_usage: Some(KvUsage { used_blocks: 10 }),
                ..Default::default()
            },
            signal_status: PoolSignalStatus {
                catalog: status.clone(),
                readiness: status.clone(),
                capacity: status.clone(),
                load: status.clone(),
                kv_usage: status.clone(),
                kv_overlap: status,
            },
        }
    }

    #[test]
    fn read_expires_only_the_stale_signal() {
        let repo = InMemoryPoolStateRepository::default();
        let original = pool();
        repo.replace(original.clone());

        let read = repo.get(&original.pool_id, 1_200, &policy()).unwrap();
        assert_eq!(read.signal_status.load.state, SignalState::Stale);
        assert!(read.load.request_plane.is_empty());
        assert_eq!(read.capacity.live_workers, Some(2));
        assert_eq!(read.load.kv_usage, Some(KvUsage { used_blocks: 10 }));
        assert_eq!(read.signal_status.readiness.state, SignalState::Complete);
    }

    #[test]
    fn missing_or_future_receive_time_is_not_fresh() {
        let mut missing = SignalStatus {
            state: SignalState::Complete,
            ..Default::default()
        };
        missing.expire_at(1_000, 100);
        assert_eq!(missing.state, SignalState::Stale);

        let mut future = complete_at(1_100);
        future.expire_at(1_000, 100);
        assert_eq!(future.state, SignalState::Stale);
    }

    #[test]
    fn unavailable_capacity_is_not_exposed_as_a_value() {
        let repo = InMemoryPoolStateRepository::default();
        let mut state = pool();
        state.signal_status.capacity.state = SignalState::Unavailable;
        let pool_id = state.pool_id.clone();
        repo.replace(state);

        let read = repo.get(&pool_id, 1_000, &policy()).unwrap();
        assert_eq!(read.capacity, PoolCapacity::default());
    }
}
