// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic model admission for pool selection. Cost ranking is a separate step.

use super::state::{
    FreshnessPolicy, PoolRole, PoolState, PoolStateRepository, ServingReadiness, SignalState,
    SignalStatus,
};

fn usable(status: &SignalStatus) -> bool {
    matches!(status.state, SignalState::Complete | SignalState::Degraded)
}

/// Return pools that can serve the model at this instant.
///
/// Catalog and model readiness must be fresh and usable. The selected local
/// frontend endpoint must be known. PoolStateRepository already omits expired
/// load, capacity, and KV values; missing cost signals do not exclude a pool.
pub fn eligible_pools(
    repository: &dyn PoolStateRepository,
    model: &str,
    now_unix_ms: u64,
    freshness: &FreshnessPolicy,
) -> Vec<PoolState> {
    repository
        .list(now_unix_ms, freshness)
        .into_iter()
        .filter(|pool| eligible_for_model(pool, model))
        .collect()
}

fn eligible_for_model(pool: &PoolState, model: &str) -> bool {
    usable(&pool.signal_status.catalog)
        && usable(&pool.signal_status.readiness)
        && pool.descriptors.models.iter().any(|name| name == model)
        && pool.descriptors.model_readiness.get(model) == Some(&ServingReadiness::Ready)
        && pool
            .descriptors
            .frontend_endpoint
            .as_deref()
            .is_some_and(|endpoint| !endpoint.is_empty())
}

/// Readiness for the first aggregated router. Stale load and KV signals do
/// not make a ready, cataloged pool unavailable to inference.
pub fn has_ready_aggregated_pool(
    repository: &dyn PoolStateRepository,
    now_unix_ms: u64,
    freshness: &FreshnessPolicy,
) -> bool {
    repository.list(now_unix_ms, freshness).iter().any(|pool| {
        pool.descriptors.roles.contains(&PoolRole::Aggregated)
            && pool
                .descriptors
                .models
                .iter()
                .any(|model| eligible_for_model(pool, model))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::global_view::state::{
        InMemoryPoolStateRepository, KvUsage, PoolCapacity, PoolDescriptors, PoolLoad,
        PoolLocation, PoolRole, PoolSignalStatus, PoolStateSink,
    };
    use crate::global_view::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};

    fn status(received_at_unix_ms: u64) -> SignalStatus {
        SignalStatus {
            state: SignalState::Complete,
            received_at_unix_ms: Some(received_at_unix_ms),
            ..Default::default()
        }
    }

    fn pool(site: &str, readiness_at: u64) -> PoolState {
        let key = PoolKey::new(site, "dynamo", "mocker").unwrap();
        PoolState {
            pool_id: V1PoolIdDeriver.derive(&key),
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
                frontend_endpoint: Some("frontend.generate".into()),
                hardware: Vec::new(),
            },
            capacity: PoolCapacity::default(),
            load: PoolLoad {
                kv_usage: Some(KvUsage { used_blocks: 100 }),
                ..Default::default()
            },
            signal_status: PoolSignalStatus {
                catalog: status(1_000),
                readiness: status(readiness_at),
                kv_usage: status(1_000),
                ..Default::default()
            },
        }
    }

    #[test]
    fn stale_cost_signal_does_not_exclude_ready_pool() {
        let repo = InMemoryPoolStateRepository::default();
        repo.replace(pool("ohio", 1_000));
        repo.replace(pool("west", 0));
        let freshness = FreshnessPolicy {
            catalog_max_age_ms: 2_000,
            readiness_max_age_ms: 700,
            capacity_max_age_ms: 500,
            load_max_age_ms: 500,
            kv_usage_max_age_ms: 500,
            kv_overlap_max_age_ms: 500,
        };
        let pools = eligible_pools(&repo, "model", 1_600, &freshness);
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].descriptors.site_id, "ohio");
        assert_eq!(pools[0].load.kv_usage, None);
        assert_eq!(pools[0].signal_status.kv_usage.state, SignalState::Stale);
        assert!(has_ready_aggregated_pool(&repo, 1_600, &freshness));
        assert!(eligible_pools(&repo, "other", 1_600, &freshness).is_empty());
        assert!(!has_ready_aggregated_pool(&repo, 2_100, &freshness));
    }
}
