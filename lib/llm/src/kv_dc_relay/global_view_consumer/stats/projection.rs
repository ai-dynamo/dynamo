// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PR #13187 stats service adapter. The service reports worker KV pools; the
//! catalog is the authority for assigning those records to a routing DGD.

use std::collections::BTreeMap;

use dynamo_kv_router::global_view::source::PoolObservation;
use dynamo_kv_router::global_view::state::{
    KvUsage, ModelRequestLoad, PoolCapacity, PoolRole, SchedulerLoad, SignalState, SignalStatus,
};

use crate::global_view::CatalogProjection;
use crate::kv_dc_relay::wan::grpc::protocol::{KvPoolId, validate_pool_id};

use super::proto;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoolKey {
    cache_digest: Vec<u8>,
    cache_source: i32,
    routing_digest: Vec<u8>,
    routing_source: i32,
    dc_id: u64,
}

impl PoolKey {
    fn from_catalog(pool: &KvPoolId) -> Option<Self> {
        validate_pool_id(pool).ok()?;
        let domain = pool.indexer_domain.as_ref()?;
        let cache = domain.cache_semantics.as_ref()?;
        let routing = domain.routing_scope.as_ref()?;
        Some(Self {
            cache_digest: cache.digest.to_vec(),
            cache_source: cache.source,
            routing_digest: routing.digest.to_vec(),
            routing_source: routing.source,
            dc_id: pool.dc_id,
        })
    }

    fn from_stats(pool: &proto::PoolIdentity) -> Option<Self> {
        if pool.cache_semantics_digest.len() != 16
            || pool.routing_scope_digest.len() != 16
            || !matches!(
                proto::IdentitySource::try_from(pool.cache_semantics_source).ok()?,
                proto::IdentitySource::DefaultDerived | proto::IdentitySource::Explicit
            )
            || !matches!(
                proto::IdentitySource::try_from(pool.routing_scope_source).ok()?,
                proto::IdentitySource::DefaultDerived | proto::IdentitySource::Explicit
            )
        {
            return None;
        }
        Some(Self {
            cache_digest: pool.cache_semantics_digest.to_vec(),
            cache_source: pool.cache_semantics_source,
            routing_digest: pool.routing_scope_digest.to_vec(),
            routing_source: pool.routing_scope_source,
            dc_id: pool.dc_id,
        })
    }
}

/// The current DGD's cataloged producers. Model request load can span several
/// producers in this DGD and is counted once. KV usage and worker scheduler
/// load use one aggregated producer until multi-producer semantics are agreed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatsCatalog {
    relay: (u64, u64),
    producer: Option<PoolKey>,
    scoped_pools: Vec<PoolKey>,
    model: String,
}

impl StatsCatalog {
    pub fn from_projection(
        projection: &CatalogProjection,
        model: &str,
        relay: (u64, u64),
    ) -> Option<Self> {
        let mut scoped_pools = Vec::new();
        for descriptor in &projection.producers {
            let pool = descriptor.producer.as_ref()?.pool_id.as_ref()?;
            let key = PoolKey::from_catalog(pool)?;
            if !scoped_pools.contains(&key) {
                scoped_pools.push(key);
            }
        }
        let has_model = projection.producers.iter().any(|descriptor| {
            descriptor
                .registrations
                .iter()
                .any(|registration| registration.canonical_model_id == model)
        });
        if !has_model {
            return None;
        }
        let producer = projection
            .sole_aggregated_overlap_producer(model)
            .and_then(|descriptor| descriptor.producer.as_ref()?.pool_id.as_ref())
            .and_then(PoolKey::from_catalog);
        Some(Self {
            relay,
            producer,
            scoped_pools,
            model: model.to_owned(),
        })
    }

    fn accepts(&self, pool: &proto::PoolIdentity) -> bool {
        PoolKey::from_stats(pool).as_ref() == self.producer.as_ref()
    }

    fn accepts_scoped(&self, pool: &proto::PoolIdentity) -> bool {
        PoolKey::from_stats(pool).is_some_and(|key| self.scoped_pools.contains(&key))
    }

    pub(super) fn accepts_metadata(&self, metadata: Option<&proto::RelayMessageMetadata>) -> bool {
        metadata.is_some_and(|metadata| {
            (metadata.drt_instance_id, metadata.relay_incarnation) == self.relay
        })
    }
}

#[derive(Clone)]
pub struct UsageProjection {
    pub capacity: PoolCapacity,
    pub capacity_status: Option<SignalStatus>,
    pub usage: Option<PoolObservation>,
}

#[derive(Clone)]
pub struct LoadProjection {
    pub capacity: PoolCapacity,
    pub capacity_status: Option<SignalStatus>,
    pub load: Option<PoolObservation>,
}

fn signal_status(data_status: i32, source_at: u64, received_at: u64) -> Option<SignalStatus> {
    let state = match proto::DataStatus::try_from(data_status).ok()? {
        proto::DataStatus::Complete => SignalState::Complete,
        proto::DataStatus::Degraded => SignalState::Degraded,
        _ => return None,
    };
    Some(SignalStatus {
        state,
        source_observed_at_unix_ms: (source_at > 0).then_some(source_at),
        received_at_unix_ms: Some(received_at),
        ..Default::default()
    })
}

pub fn project_usage(
    snapshot: &proto::KvUsageSnapshot,
    catalog: &StatsCatalog,
    received_at: u64,
) -> Option<UsageProjection> {
    if !catalog.accepts_metadata(snapshot.metadata.as_ref()) {
        return None;
    }
    let mut matching = snapshot.pools.iter().filter(|entry| {
        entry.role == proto::WorkerRole::Aggregated as i32
            && entry
                .pool
                .as_ref()
                .is_some_and(|pool| catalog.accepts(pool))
            && entry.models.iter().any(|model| {
                model.model == catalog.model
                    && model.base_model == catalog.model
                    && model.adapter.is_none()
            })
    });
    let entry = matching.next()?;
    if matching.next().is_some() {
        return None;
    }
    let status = signal_status(entry.status, entry.source_observed_at_unix_ms, received_at).map(
        |mut status| {
            status.expected_sources = Some(entry.expected_ranks);
            status.observed_sources = Some(entry.observed_ranks);
            status
        },
    );
    let capacity = PoolCapacity {
        kv_capacity_blocks: entry.capacity_blocks,
        expected_ranks: status.as_ref().map(|_| entry.expected_ranks),
        observed_ranks: status.as_ref().map(|_| entry.observed_ranks),
        ..Default::default()
    };
    let capacity_status = entry.capacity_blocks.and_then(|_| status.clone());
    let usage =
        entry
            .used_blocks
            .zip(status)
            .map(|(used_blocks, status)| PoolObservation::KvUsage {
                value: Some(KvUsage { used_blocks }),
                status,
            });
    Some(UsageProjection {
        capacity,
        capacity_status,
        usage,
    })
}

pub fn project_load(
    snapshot: &proto::LoadSnapshot,
    catalog: &StatsCatalog,
    received_at: u64,
) -> Option<LoadProjection> {
    if !catalog.accepts_metadata(snapshot.metadata.as_ref()) {
        return None;
    }
    let mut matching_pools = snapshot.pools.iter().filter(|entry| {
        entry.role == proto::WorkerRole::Aggregated as i32
            && entry
                .pool
                .as_ref()
                .is_some_and(|pool| catalog.accepts(pool))
    });
    let pool = matching_pools.next();
    if matching_pools.next().is_some() {
        return None;
    }
    let pool_status = pool.and_then(|pool| {
        signal_status(
            pool.scheduler_status,
            pool.scheduler_observed_at_unix_ms,
            received_at,
        )
    });
    let capacity = PoolCapacity {
        live_workers: pool_status.as_ref().and_then(|_| pool?.live_workers),
        max_concurrency: pool_status.as_ref().and_then(|_| pool?.max_concurrency),
        ..Default::default()
    };
    let capacity_status = (capacity.live_workers.is_some() || capacity.max_concurrency.is_some())
        .then_some(pool_status.clone())
        .flatten();
    let mut matching_models = snapshot.models.iter().filter(|entry| {
        entry.model.as_ref().is_some_and(|registration| {
            registration.model == catalog.model
                && registration.base_model == catalog.model
                && registration.adapter.is_none()
        })
    });
    let model = matching_models.next();
    if matching_models.next().is_some() {
        return None;
    }
    let model_status = model
        .filter(|model| {
            !model.serving_pools.is_empty()
                && model
                    .serving_pools
                    .iter()
                    .all(|identity| catalog.accepts_scoped(identity))
        })
        .and_then(|model| {
            signal_status(model.status, model.source_observed_at_unix_ms, received_at).map(
                |mut status| {
                    status.expected_sources = Some(u64::from(model.expected_frontends));
                    status.observed_sources = Some(u64::from(model.observed_frontends));
                    status
                },
            )
        });
    let mut request_plane = BTreeMap::new();
    if let (Some(model), Some(_)) = (model, model_status.as_ref()) {
        request_plane.insert(
            catalog.model.clone(),
            ModelRequestLoad {
                ready_frontends: model.ready_frontends,
                pending_first_output_requests: model.pending_first_output_requests,
                pending_first_output_input_tokens: model.pending_first_output_input_tokens,
                live_input_tokens: model.live_input_tokens,
                input_processing_requests: model.input_processing_requests,
                output_generation_requests: model.output_generation_requests,
            },
        );
    }
    let mut scheduler = BTreeMap::new();
    if let (Some(pool), Some(_)) = (pool, pool_status.as_ref()) {
        scheduler.insert(
            PoolRole::Aggregated,
            SchedulerLoad {
                active_prefill_tokens: pool.active_prefill_tokens,
                active_decode_blocks: pool.active_decode_blocks,
            },
        );
    }
    let load_status = match (model_status, pool_status) {
        (None, None) => None,
        (Some(status), None) | (None, Some(status)) => Some(SignalStatus {
            state: SignalState::Degraded,
            ..status
        }),
        (Some(model_status), Some(pool_status)) => Some(SignalStatus {
            state: if model_status.state == SignalState::Complete
                && pool_status.state == SignalState::Complete
            {
                SignalState::Complete
            } else {
                SignalState::Degraded
            },
            source_observed_at_unix_ms: match (
                model_status.source_observed_at_unix_ms,
                pool_status.source_observed_at_unix_ms,
            ) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            received_at_unix_ms: Some(received_at),
            expected_sources: model_status.expected_sources,
            observed_sources: model_status.observed_sources,
        }),
    };
    let load = load_status.map(|status| PoolObservation::Load {
        request_plane,
        scheduler,
        status,
    });
    Some(LoadProjection {
        capacity,
        capacity_status,
        load,
    })
}

/// Merge independent capacity signals without replacing a known field with an
/// invented zero. One missing source makes the combined capacity degraded.
pub fn combined_capacity(
    usage: Option<&UsageProjection>,
    load: Option<&LoadProjection>,
    received_at: u64,
) -> Option<PoolObservation> {
    let usage = usage.filter(|value| value.capacity_status.is_some());
    let load = load.filter(|value| value.capacity_status.is_some());
    if usage.is_none() && load.is_none() {
        return None;
    }
    let source_at = [
        usage.and_then(|value| value.capacity_status.as_ref()?.source_observed_at_unix_ms),
        load.and_then(|value| value.capacity_status.as_ref()?.source_observed_at_unix_ms),
    ]
    .into_iter()
    .flatten()
    .min();
    let state = if usage
        .is_some_and(|value| value.capacity_status.as_ref().unwrap().state == SignalState::Complete)
        && load.is_some_and(|value| {
            value.capacity_status.as_ref().unwrap().state == SignalState::Complete
        }) {
        SignalState::Complete
    } else {
        SignalState::Degraded
    };
    let received_at = [
        usage.and_then(|value| value.capacity_status.as_ref()?.received_at_unix_ms),
        load.and_then(|value| value.capacity_status.as_ref()?.received_at_unix_ms),
    ]
    .into_iter()
    .flatten()
    .min()
    .unwrap_or(received_at);
    Some(PoolObservation::Capacity {
        value: PoolCapacity {
            live_workers: load.and_then(|value| value.capacity.live_workers),
            max_concurrency: load.and_then(|value| value.capacity.max_concurrency),
            kv_capacity_blocks: usage.and_then(|value| value.capacity.kv_capacity_blocks),
            expected_ranks: usage.and_then(|value| value.capacity.expected_ranks),
            observed_ranks: usage.and_then(|value| value.capacity.observed_ranks),
        },
        status: SignalStatus {
            state,
            source_observed_at_unix_ms: source_at,
            received_at_unix_ms: Some(received_at),
            expected_sources: usage
                .and_then(|value| value.capacity_status.as_ref()?.expected_sources),
            observed_sources: usage
                .and_then(|value| value.capacity_status.as_ref()?.observed_sources),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> StatsCatalog {
        StatsCatalog {
            relay: (5, 7),
            producer: Some(PoolKey {
                cache_digest: vec![1; 16],
                cache_source: proto::IdentitySource::Explicit as i32,
                routing_digest: vec![2; 16],
                routing_source: proto::IdentitySource::DefaultDerived as i32,
                dc_id: 9,
            }),
            scoped_pools: vec![PoolKey {
                cache_digest: vec![1; 16],
                cache_source: proto::IdentitySource::Explicit as i32,
                routing_digest: vec![2; 16],
                routing_source: proto::IdentitySource::DefaultDerived as i32,
                dc_id: 9,
            }],
            model: "m".into(),
        }
    }

    fn pool() -> proto::PoolIdentity {
        proto::PoolIdentity {
            cache_semantics_digest: vec![1; 16],
            cache_semantics_source: proto::IdentitySource::Explicit as i32,
            routing_scope_digest: vec![2; 16],
            routing_scope_source: proto::IdentitySource::DefaultDerived as i32,
            dc_id: 9,
        }
    }

    fn metadata() -> Option<proto::RelayMessageMetadata> {
        Some(proto::RelayMessageMetadata {
            drt_instance_id: 5,
            relay_incarnation: 7,
            observed_at_unix_ms: 100,
        })
    }

    fn registration() -> proto::ModelRegistration {
        proto::ModelRegistration {
            model: "m".into(),
            base_model: "m".into(),
            adapter: None,
            aliases: Vec::new(),
        }
    }

    #[test]
    fn missing_capacity_is_not_zero_but_zero_usage_is_valid() {
        let snapshot = proto::KvUsageSnapshot {
            metadata: metadata(),
            pools: vec![proto::PoolKvUsage {
                pool: Some(pool()),
                models: vec![registration()],
                role: proto::WorkerRole::Aggregated as i32,
                capacity_blocks: None,
                used_blocks: Some(0),
                status: proto::DataStatus::Complete as i32,
                source_observed_at_unix_ms: 100,
                ..Default::default()
            }],
        };
        let projected = project_usage(&snapshot, &catalog(), 200).unwrap();
        assert!(projected.capacity_status.is_none());
        assert_eq!(projected.capacity.kv_capacity_blocks, None);
        assert!(matches!(
            projected.usage,
            Some(PoolObservation::KvUsage {
                value: Some(KvUsage { used_blocks: 0 }),
                ..
            })
        ));
        assert!(combined_capacity(Some(&projected), None, 200).is_none());
    }

    #[test]
    fn model_request_load_is_counted_once_and_foreign_pools_are_rejected() {
        let model = proto::ModelLoad {
            model: Some(registration()),
            ready_frontends: Some(1),
            pending_first_output_requests: Some(4),
            serving_pools: vec![pool()],
            status: proto::DataStatus::Complete as i32,
            source_observed_at_unix_ms: 100,
            ..Default::default()
        };
        let snapshot = proto::LoadSnapshot {
            metadata: metadata(),
            pools: vec![proto::PoolLoad {
                pool: Some(pool()),
                role: proto::WorkerRole::Aggregated as i32,
                live_workers: Some(2),
                max_concurrency: Some(8),
                scheduler_status: proto::DataStatus::Complete as i32,
                scheduler_observed_at_unix_ms: 100,
                ..Default::default()
            }],
            models: vec![model.clone()],
        };
        let projected = project_load(&snapshot, &catalog(), 200).unwrap();
        let Some(PoolObservation::Load { request_plane, .. }) = projected.load else {
            panic!("expected load observation");
        };
        assert_eq!(request_plane.len(), 1);
        assert_eq!(request_plane["m"].pending_first_output_requests, Some(4));
        let second = proto::PoolIdentity {
            dc_id: 10,
            ..pool()
        };
        let mut same_dgd = catalog();
        same_dgd
            .scoped_pools
            .push(PoolKey::from_stats(&second).unwrap());
        let mut multi_pool = snapshot.clone();
        multi_pool.models[0].serving_pools.push(second.clone());
        let projected = project_load(&multi_pool, &same_dgd, 200).unwrap();
        let Some(PoolObservation::Load { request_plane, .. }) = projected.load else {
            panic!("expected same-DGD model load");
        };
        assert_eq!(request_plane.len(), 1);
        assert_eq!(request_plane["m"].pending_first_output_requests, Some(4));
        let mut ambiguous = same_dgd.clone();
        ambiguous.producer = None;
        let projected = project_load(&multi_pool, &ambiguous, 200).unwrap();
        let Some(PoolObservation::Load {
            request_plane,
            scheduler,
            ..
        }) = projected.load
        else {
            panic!("expected model-only load observation");
        };
        assert_eq!(request_plane.len(), 1);
        assert!(scheduler.is_empty());
        let mut foreign = snapshot;
        foreign.models[0].serving_pools.push(proto::PoolIdentity {
            dc_id: 10,
            ..pool()
        });
        let projected = project_load(&foreign, &catalog(), 200).unwrap();
        let Some(PoolObservation::Load { request_plane, .. }) = projected.load else {
            panic!("expected scheduler-only load observation");
        };
        assert!(request_plane.is_empty());
        let mut duplicate = multi_pool;
        duplicate.models.push(model);
        assert!(project_load(&duplicate, &same_dgd, 200).is_none());
    }

    #[test]
    fn absent_stats_source_degrades_combined_capacity() {
        let load = LoadProjection {
            capacity: PoolCapacity {
                live_workers: Some(2),
                ..Default::default()
            },
            capacity_status: signal_status(proto::DataStatus::Complete as i32, 100, 200),
            load: None,
        };
        let Some(PoolObservation::Capacity { value, status }) =
            combined_capacity(None, Some(&load), 300)
        else {
            panic!("expected partial capacity");
        };
        assert_eq!(value.live_workers, Some(2));
        assert_eq!(value.kv_capacity_blocks, None);
        assert_eq!(status.state, SignalState::Degraded);
        assert_eq!(status.received_at_unix_ms, Some(200));
    }

    #[test]
    fn relay_incarnation_mismatch_is_not_attributed() {
        let snapshot = proto::KvUsageSnapshot {
            metadata: Some(proto::RelayMessageMetadata {
                relay_incarnation: 8,
                ..metadata().unwrap()
            }),
            pools: Vec::new(),
        };
        assert!(project_usage(&snapshot, &catalog(), 200).is_none());
    }
}
