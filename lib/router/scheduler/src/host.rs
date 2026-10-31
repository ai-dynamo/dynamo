// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a host implements or calls to run the kernel in-process, in one place: the core and
//! its constructor inputs, the capability structs a host fills, the hook traits it
//! implements, and the operation it runs. Everything here is re-exported from its owning
//! module; this list is the contract (hosts inventory section 3.6).

pub use crate::config::{RouterConfig, RouterConfigOverride, Validated};
pub use crate::overlap_summary::MooncakeOverlapSummary;
pub use crate::replication::{
    AffinityBindingEvent, HostReplicaChannels, HostReplicaSyncFactory, ReplicaIngressObserver,
    ReplicaSyncConfig, SchedulerLoadSink,
};
pub use crate::selection::SelectionError;
pub use crate::selection::core::{
    HostCache, HostEligibility, HostLoad, HostReplication, HostTelemetry, KvIndexSource, Selected,
    SelectionAdmission, SelectionCore, SelectionHost, SelectionOperation, SelectionOutcome,
    SelectionPartition, SelectionRun, SelectionScheduler, SessionBinding,
};
pub use crate::selection::ingress::{KvEventIngress, LocalIndexIngress};
pub use crate::selection::input::{PromptRequest, PromptView};
pub use crate::selection::membership::{CatalogObserver, CatalogReconciler, WorkerCatalogSource};
pub use crate::selection::pending::SelectionCacheConfig;
pub use crate::selection::types::{SelectionWorkerConfig, WorkerCatalogRecord, WorkerLifecycle};
pub use crate::tracking_hash::{TrackingHashConfig, TrackingHashContext};
pub use crate::types::{RejectThresholdsProvider, WorkerSelectionPolicyFactory};
pub use dynamo_admission::classifier::RequestLifecycle;
pub use dynamo_kv_indexer::backend::IndexerPolicy;
pub use dynamo_kv_indexer::registry::IndexRegistry;
