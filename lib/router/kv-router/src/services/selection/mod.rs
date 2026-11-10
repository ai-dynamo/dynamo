// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free worker selection service.
//!
//! The service owns worker selection and reservation state, but never forwards
//! model requests and never owns model responses.

// The HTTP server drives every adapter: the peer-recovery endpoint, the replica-peer
// routes, and the ZMQ listener registry.
#[cfg(all(
    feature = "http",
    feature = "ingest-zmq",
    feature = "replica-sync-zmq",
    feature = "peer-recovery"
))]
mod server;
mod service;
#[cfg(feature = "ingest-zmq")]
mod zmq_ingress;

// Five of these six tests drive the ZMQ-direct ingress.
#[cfg(all(test, feature = "ingest-zmq"))]
mod core_host_tests;
#[cfg(all(
    test,
    feature = "http",
    feature = "ingest-zmq",
    feature = "replica-sync-zmq",
    feature = "peer-recovery"
))]
mod tests;

#[cfg(feature = "replica-sync-zmq")]
pub use crate::services::common::replica_sync::ReplicaPeerError;
#[cfg(all(
    feature = "http",
    feature = "ingest-zmq",
    feature = "replica-sync-zmq",
    feature = "peer-recovery"
))]
pub use server::{AppState, run_server};
#[cfg(all(
    feature = "ingest-zmq",
    feature = "replica-sync-zmq",
    feature = "peer-recovery"
))]
pub use service::SelectionServiceConfig;
pub use service::{
    SelectionService, SelectionServiceBuilder, SessionBindingUpdates, local_selection_core,
    local_selection_core_with_host, warn_for_unserved_worker_selection_policies,
};
#[cfg(feature = "ingest-zmq")]
pub use zmq_ingress::ZmqDirectIngress;

// Pre-split paths kept through 1.9 and removed in 1.10; the named list is the inventory in compat.rs.
#[allow(deprecated)]
pub use crate::compat::services::selection::{
    AffinityBindingEvent, CatalogObserver, CatalogReconciler, DEFAULT_MODEL_NAME,
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    HostCache, HostEligibility, HostLoad, HostReplicaChannels, HostReplicaSyncFactory,
    HostReplication, HostTelemetry, KvEventIngress, KvIndexSource, ModelLoadResponse,
    OutputBlockRequest, OverlapScoresRequest, OverlapScoresResponse, PotentialLoadsRequest,
    PromptRequest, PromptView, ReadyResponse, ReplicaIngressObserver, ReservationRequest,
    ReservationResponse, RouterPluginRegistry, SchedulerLoadSink, SelectAndReserveRequest,
    SelectRequest, SelectResponse, Selected, SelectionAdmission, SelectionCacheConfig,
    SelectionCore, SelectionError, SelectionHost, SelectionInputTrigger, SelectionOperation,
    SelectionOutcome, SelectionPartition, SelectionRun, SelectionScheduler,
    SelectionSessionContext, SelectionWorkerConfig, SelectionWorkerLoad, SessionBinding,
    SharedCacheOverlapScore, WorkerCatalogRecord, WorkerCatalogSource, WorkerLifecycle,
    WorkerOverlapScore, WorkerPatchRequest, WorkerRequest, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProvider, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError, affinity,
};
