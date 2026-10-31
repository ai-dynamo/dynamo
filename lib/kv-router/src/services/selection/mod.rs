// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free worker selection service.
//!
//! The service owns worker selection and reservation state, but never forwards
//! model requests and never owns model responses.

pub use dynamo_session_affinity as affinity;
// Shims over the kernel's selection modules, so the server, the service, the ZMQ ingress, and
// the HTTP tests keep their `super::` paths until step 8 flips hosts to the owners.
mod core {
    pub use dynamo_scheduler::selection::core::*;
}
mod error {
    pub use dynamo_scheduler::selection::error::*;
}
mod ingress {
    pub use dynamo_scheduler::selection::ingress::*;
}
mod input {
    pub use dynamo_scheduler::selection::input::*;
}
mod pending {
    pub use dynamo_scheduler::selection::pending::*;
}
mod types {
    pub use dynamo_scheduler::selection::types::*;
}
mod server;
mod service;
mod zmq_ingress;

#[cfg(test)]
mod core_host_tests;
#[cfg(test)]
mod tests;

// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
pub use crate::plugins::worker_selection::WorkerSelectionPolicyFactory;
pub use crate::plugins::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
pub use crate::services::common::replica_sync::ReplicaPeerError;
pub use core::{
    HostCache, HostEligibility, HostLoad, HostReplication, HostTelemetry, KvIndexSource, Selected,
    SelectionAdmission, SelectionCore, SelectionHost, SelectionOperation, SelectionOutcome,
    SelectionPartition, SelectionRun, SelectionScheduler, SessionBinding,
};
pub use dynamo_scheduler::replication::{
    AffinityBindingEvent, HostReplicaChannels, HostReplicaSyncFactory, ReplicaIngressObserver,
    SchedulerLoadSink,
};
pub use dynamo_scheduler::selection::membership::{
    CatalogObserver, CatalogReconciler, WorkerCatalogSource,
};
pub use error::SelectionError;
pub use ingress::{KvEventIngress, LocalIndexIngress};
pub use input::{PromptRequest, PromptView};
pub use pending::SelectionCacheConfig;
pub use server::{AppState, run_server};
pub use service::{
    SelectionService, SelectionServiceBuilder, SelectionServiceConfig, SessionBindingUpdates,
    local_selection_core, local_selection_core_with_host,
    warn_for_unserved_worker_selection_policies,
};
pub use types::{
    DEFAULT_MODEL_NAME, ModelLoadResponse, OutputBlockRequest, OverlapScoresRequest,
    OverlapScoresResponse, PotentialLoadsRequest, ReadyResponse, ReservationRequest,
    ReservationResponse, SelectAndReserveRequest, SelectRequest, SelectResponse,
    SelectionInputTrigger, SelectionSessionContext, SelectionWorkerConfig, SelectionWorkerLoad,
    SharedCacheOverlapScore, WorkerCatalogRecord, WorkerLifecycle, WorkerOverlapScore,
    WorkerPatchRequest, WorkerRequest,
};
pub use zmq_ingress::ZmqDirectIngress;
