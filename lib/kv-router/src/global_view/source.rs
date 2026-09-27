// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assemble transport-neutral relay observations for one configured DGD.
//!
//! A transport adapter validates wire messages and maps all producers within
//! its configured DGD to these complete, per-plane observations. Each stream
//! starts a new lease; late messages from an older stream cannot restore state.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use thiserror::Error;

use super::state::{
    KvUsage, ModelRequestLoad, PoolCapacity, PoolDescriptors, PoolLoad, PoolLocation, PoolRole,
    PoolSignalStatus, PoolState, PoolStateSink, SchedulerLoad, ServingReadiness, SignalState,
    SignalStatus,
};
use super::{PoolId, PoolIdDeriver, PoolKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SourcePlane {
    Catalog,
    Readiness,
    Capacity,
    Load,
    KvUsage,
    KvOverlap,
}

impl SourcePlane {
    const fn index(self) -> usize {
        self as usize
    }
}

/// Issued locally when a stream connects. Relay incarnation changes require
/// opening fresh leases for all planes from that relay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLease {
    plane: SourcePlane,
    generation: u64,
}

pub enum PoolObservation {
    Catalog {
        models: Vec<String>,
        roles: Vec<PoolRole>,
        frontend_endpoint: Option<String>,
        hardware: Vec<String>,
        status: SignalStatus,
    },
    Readiness {
        models: BTreeMap<String, ServingReadiness>,
        status: SignalStatus,
    },
    Capacity {
        value: PoolCapacity,
        status: SignalStatus,
    },
    Load {
        request_plane: BTreeMap<String, ModelRequestLoad>,
        scheduler: BTreeMap<PoolRole, SchedulerLoad>,
        status: SignalStatus,
    },
    KvUsage {
        value: Option<KvUsage>,
        status: SignalStatus,
    },
    /// CKF replicas live in the indexer, not in serializable pool state.
    KvOverlap { status: SignalStatus },
}

impl PoolObservation {
    pub fn plane(&self) -> SourcePlane {
        match self {
            Self::Catalog { .. } => SourcePlane::Catalog,
            Self::Readiness { .. } => SourcePlane::Readiness,
            Self::Capacity { .. } => SourcePlane::Capacity,
            Self::Load { .. } => SourcePlane::Load,
            Self::KvUsage { .. } => SourcePlane::KvUsage,
            Self::KvOverlap { .. } => SourcePlane::KvOverlap,
        }
    }

    fn status(&self) -> &SignalStatus {
        match self {
            Self::Catalog { status, .. }
            | Self::Readiness { status, .. }
            | Self::Capacity { status, .. }
            | Self::Load { status, .. }
            | Self::KvUsage { status, .. }
            | Self::KvOverlap { status } => status,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ObservationError {
    #[error("observation belongs to a different source plane")]
    WrongPlane,
    #[error("usable observation must have a local receive timestamp")]
    MissingReceiveTime,
    #[error("observation must be complete or degraded")]
    InvalidState,
    #[error("source generation exhausted")]
    GenerationExhausted,
}

struct Assembly {
    state: PoolState,
    generations: [u64; 6],
    active: [bool; 6],
}

/// One assembler per configured DGD. A source adapter owns stream connections;
/// this object owns cross-plane replacement and publishes coherent pool copies.
pub struct PoolObservationAssembler {
    inner: Mutex<Assembly>,
    sink: Arc<dyn PoolStateSink>,
}

impl PoolObservationAssembler {
    pub fn new(
        key: &PoolKey,
        location: PoolLocation,
        deriver: &dyn PoolIdDeriver,
        sink: Arc<dyn PoolStateSink>,
    ) -> Self {
        let state = PoolState {
            pool_id: deriver.derive(key),
            descriptors: PoolDescriptors {
                site_id: key.site_id().into(),
                namespace: key.namespace().into(),
                dgd_name: key.dgd_name().into(),
                location,
                models: Vec::new(),
                model_readiness: BTreeMap::new(),
                roles: Vec::new(),
                frontend_endpoint: None,
                hardware: Vec::new(),
            },
            capacity: PoolCapacity::default(),
            load: PoolLoad::default(),
            signal_status: PoolSignalStatus::default(),
        };
        sink.replace(state.clone());
        Self {
            inner: Mutex::new(Assembly {
                state,
                generations: [0; 6],
                active: [false; 6],
            }),
            sink,
        }
    }

    pub fn pool_id(&self) -> PoolId {
        self.inner.lock().state.pool_id.clone()
    }

    /// Invalidates previous values immediately, including on reconnect.
    pub fn open(&self, plane: SourcePlane) -> Result<PlaneLease, ObservationError> {
        let mut inner = self.inner.lock();
        let generation = inner.generations[plane.index()]
            .checked_add(1)
            .ok_or(ObservationError::GenerationExhausted)?;
        inner.generations[plane.index()] = generation;
        inner.active[plane.index()] = true;
        clear_plane(&mut inner.state, plane);
        self.sink.replace(inner.state.clone());
        Ok(PlaneLease { plane, generation })
    }

    /// Returns false for a late observation from an old stream.
    pub fn apply(
        &self,
        lease: PlaneLease,
        observation: PoolObservation,
    ) -> Result<bool, ObservationError> {
        if lease.plane != observation.plane() {
            return Err(ObservationError::WrongPlane);
        }
        let status = observation.status();
        if !matches!(status.state, SignalState::Complete | SignalState::Degraded) {
            return Err(ObservationError::InvalidState);
        }
        if status.received_at_unix_ms.is_none() {
            return Err(ObservationError::MissingReceiveTime);
        }
        let mut inner = self.inner.lock();
        if inner.generations[lease.plane.index()] != lease.generation
            || !inner.active[lease.plane.index()]
        {
            return Ok(false);
        }
        let state = &mut inner.state;
        match observation {
            PoolObservation::Catalog {
                models,
                roles,
                frontend_endpoint,
                hardware,
                status,
            } => {
                state.descriptors.models = models;
                state.descriptors.roles = roles;
                state.descriptors.frontend_endpoint = frontend_endpoint;
                state.descriptors.hardware = hardware;
                state.signal_status.catalog = status;
            }
            PoolObservation::Readiness { models, status } => {
                state.descriptors.model_readiness = models;
                state.signal_status.readiness = status;
            }
            PoolObservation::Capacity { value, status } => {
                state.capacity = value;
                state.signal_status.capacity = status;
            }
            PoolObservation::Load {
                request_plane,
                scheduler,
                status,
            } => {
                state.load.request_plane = request_plane;
                state.load.scheduler = scheduler;
                state.signal_status.load = status;
            }
            PoolObservation::KvUsage { value, status } => {
                state.load.kv_usage = value;
                state.signal_status.kv_usage = status;
            }
            PoolObservation::KvOverlap { status } => {
                state.signal_status.kv_overlap = status;
            }
        }
        self.sink.replace(state.clone());
        Ok(true)
    }

    /// A disconnect only invalidates the plane if this is its current stream.
    pub fn disconnect(&self, lease: PlaneLease) -> bool {
        let mut inner = self.inner.lock();
        if inner.generations[lease.plane.index()] != lease.generation
            || !inner.active[lease.plane.index()]
        {
            return false;
        }
        inner.active[lease.plane.index()] = false;
        clear_plane(&mut inner.state, lease.plane);
        self.sink.replace(inner.state.clone());
        true
    }
}

fn clear_plane(state: &mut PoolState, plane: SourcePlane) {
    match plane {
        SourcePlane::Catalog => {
            state.descriptors.models.clear();
            state.descriptors.roles.clear();
            state.descriptors.frontend_endpoint = None;
            state.descriptors.hardware.clear();
            state.signal_status.catalog = SignalStatus::default();
        }
        SourcePlane::Readiness => {
            state.descriptors.model_readiness.clear();
            state.signal_status.readiness = SignalStatus::default();
        }
        SourcePlane::Capacity => {
            state.capacity = PoolCapacity::default();
            state.signal_status.capacity = SignalStatus::default();
        }
        SourcePlane::Load => {
            state.load.request_plane.clear();
            state.load.scheduler.clear();
            state.signal_status.load = SignalStatus::default();
        }
        SourcePlane::KvUsage => {
            state.load.kv_usage = None;
            state.signal_status.kv_usage = SignalStatus::default();
        }
        SourcePlane::KvOverlap => {
            state.signal_status.kv_overlap = SignalStatus::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_view::V1PoolIdDeriver;
    use crate::global_view::state::{
        FreshnessPolicy, InMemoryPoolStateRepository, PoolStateRepository,
    };

    fn assembler() -> (PoolObservationAssembler, Arc<InMemoryPoolStateRepository>) {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let key = PoolKey::new("ohio", "dynamo", "mocker").unwrap();
        let assembler = PoolObservationAssembler::new(
            &key,
            PoolLocation {
                region: "us-east-2".into(),
                availability_zone: None,
                cluster: None,
                datacenter: None,
            },
            &V1PoolIdDeriver,
            repo.clone(),
        );
        (assembler, repo)
    }

    fn freshness() -> FreshnessPolicy {
        FreshnessPolicy {
            catalog_max_age_ms: 1000,
            readiness_max_age_ms: 1000,
            capacity_max_age_ms: 1000,
            load_max_age_ms: 1000,
            kv_usage_max_age_ms: 1000,
            kv_overlap_max_age_ms: 1000,
        }
    }

    fn status() -> SignalStatus {
        SignalStatus {
            state: SignalState::Complete,
            received_at_unix_ms: Some(100),
            ..Default::default()
        }
    }

    #[test]
    fn independent_planes_preserve_each_other_and_replace_complete_values() {
        let (assembler, repo) = assembler();
        let catalog = assembler.open(SourcePlane::Catalog).unwrap();
        let usage = assembler.open(SourcePlane::KvUsage).unwrap();
        assembler
            .apply(
                catalog,
                PoolObservation::Catalog {
                    models: vec!["model".into()],
                    roles: vec![PoolRole::Aggregated],
                    frontend_endpoint: Some("frontend.generate".into()),
                    hardware: Vec::new(),
                    status: status(),
                },
            )
            .unwrap();
        assembler
            .apply(
                usage,
                PoolObservation::KvUsage {
                    value: Some(KvUsage { used_blocks: 7 }),
                    status: status(),
                },
            )
            .unwrap();
        assembler
            .apply(
                catalog,
                PoolObservation::Catalog {
                    models: vec!["other".into()],
                    roles: vec![PoolRole::Decode],
                    frontend_endpoint: Some("new.generate".into()),
                    hardware: Vec::new(),
                    status: status(),
                },
            )
            .unwrap();
        let state = repo.get(&assembler.pool_id(), 100, &freshness()).unwrap();
        assert_eq!(state.descriptors.models, vec!["other"]);
        assert_eq!(state.load.kv_usage, Some(KvUsage { used_blocks: 7 }));
    }

    #[test]
    fn reconnect_fences_late_messages_and_invalidates_only_its_plane() {
        let (assembler, repo) = assembler();
        let old = assembler.open(SourcePlane::KvUsage).unwrap();
        assembler
            .apply(
                old,
                PoolObservation::KvUsage {
                    value: Some(KvUsage { used_blocks: 7 }),
                    status: status(),
                },
            )
            .unwrap();
        let current = assembler.open(SourcePlane::KvUsage).unwrap();
        assert!(
            !assembler
                .apply(
                    old,
                    PoolObservation::KvUsage {
                        value: Some(KvUsage { used_blocks: 99 }),
                        status: status(),
                    },
                )
                .unwrap()
        );
        assert!(!assembler.disconnect(old));
        let state = repo.get(&assembler.pool_id(), 100, &freshness()).unwrap();
        assert_eq!(state.load.kv_usage, None);
        assert_eq!(state.signal_status.kv_usage.state, SignalState::Unavailable);
        assembler
            .apply(
                current,
                PoolObservation::KvUsage {
                    value: Some(KvUsage { used_blocks: 8 }),
                    status: status(),
                },
            )
            .unwrap();
        assert!(assembler.disconnect(current));
        assert!(
            !assembler
                .apply(
                    current,
                    PoolObservation::KvUsage {
                        value: Some(KvUsage { used_blocks: 99 }),
                        status: status(),
                    },
                )
                .unwrap()
        );
        assert_eq!(
            repo.get(&assembler.pool_id(), 100, &freshness())
                .unwrap()
                .load
                .kv_usage,
            None
        );
    }

    #[test]
    fn catalog_reconnect_preserves_independent_readiness_stream() {
        let (assembler, repo) = assembler();
        let readiness = assembler.open(SourcePlane::Readiness).unwrap();
        assembler
            .apply(
                readiness,
                PoolObservation::Readiness {
                    models: BTreeMap::from([("new-model".into(), ServingReadiness::Ready)]),
                    status: status(),
                },
            )
            .unwrap();
        let catalog = assembler.open(SourcePlane::Catalog).unwrap();
        assembler
            .apply(
                catalog,
                PoolObservation::Catalog {
                    models: vec!["old-model".into()],
                    roles: vec![PoolRole::Aggregated],
                    frontend_endpoint: None,
                    hardware: Vec::new(),
                    status: status(),
                },
            )
            .unwrap();
        let next_catalog = assembler.open(SourcePlane::Catalog).unwrap();
        assembler
            .apply(
                next_catalog,
                PoolObservation::Catalog {
                    models: vec!["new-model".into()],
                    roles: vec![PoolRole::Aggregated],
                    frontend_endpoint: None,
                    hardware: Vec::new(),
                    status: status(),
                },
            )
            .unwrap();
        let state = repo.get(&assembler.pool_id(), 100, &freshness()).unwrap();
        assert_eq!(state.signal_status.catalog.state, SignalState::Complete);
        assert_eq!(state.signal_status.readiness.state, SignalState::Complete);
        assert_eq!(
            state.descriptors.model_readiness.get("new-model"),
            Some(&ServingReadiness::Ready)
        );
    }

    #[test]
    fn rejects_wrong_plane_and_unfreshenable_observation() {
        let (assembler, _) = assembler();
        let lease = assembler.open(SourcePlane::Capacity).unwrap();
        assert_eq!(
            assembler.apply(lease, PoolObservation::KvOverlap { status: status() }),
            Err(ObservationError::WrongPlane)
        );
        assert_eq!(
            assembler.apply(
                lease,
                PoolObservation::Capacity {
                    value: PoolCapacity::default(),
                    status: SignalStatus {
                        state: SignalState::Complete,
                        ..Default::default()
                    },
                }
            ),
            Err(ObservationError::MissingReceiveTime)
        );
    }
}
