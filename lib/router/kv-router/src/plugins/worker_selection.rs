// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.
//!
//! The stage lives in `dynamo-worker-selection`; this module re-exports it at the paths
//! plugins and hosts already name and adds what only the composed router knows: the
//! policy factory keyed by routing partition, the host's instance config, the registry,
//! and the kernel's config- and projection-derived row columns.

mod config;

#[allow(deprecated)]
pub use dynamo_worker_selection::WorkerTransferInput;
pub use dynamo_worker_selection::experimental;
pub use dynamo_worker_selection::selection_config;
pub use dynamo_worker_selection::{
    ActivePrefillSums, CacheSnapshot, CandidateData, CandidatePreparer, CandidateRank,
    CandidateRowSource, HostedSelectionRequest, HostedWorkerPicker, PreparedRows, RankVisitor,
    RowPick, RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
    ScoredWorkerCandidate, SelectionConfig, SelectionFailure, SelectionTrace, SelectionTraceRow,
    SelectionTraceSpec, SessionContext, SharedCacheHits, SharedCacheType, WorkerCacheData,
    WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates, WorkerCapacity,
    WorkerCopyInput, WorkerFilter, WorkerInfo, WorkerInputRows, WorkerInputView, WorkerInputs,
    WorkerLoadInput, WorkerPicker, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};
#[cfg(any(test, feature = "bench"))]
pub use dynamo_worker_selection::{DefaultPick, DefaultWorkerPicker};

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
pub(crate) use config::RawWorkerSelectionConfig;
pub use config::{WorkerSelectionConfig, WorkerSelectionInstance, WorkerSelectionPolicySelection};
pub use dynamo_router_protocol::session::WorkerSelectionInputTrigger;
pub use dynamo_scheduler::hint_policy::Commitments;
pub use dynamo_scheduler::kv_hints::{KvCopyLocations, KvCopySource};
pub use dynamo_scheduler::selector::{WorkerCapacityExt, WorkerInfoExt, WorkerLoadInputExt};

/// Factory that creates one worker-selection policy per routing partition.
pub use dynamo_scheduler::types::WorkerSelectionPolicyFactory;
