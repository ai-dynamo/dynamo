// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.
//!
//! The stage lives in `dynamo-worker-selection`; this module re-exports it at the paths
//! plugins and hosts already name and adds what only the composed router knows: the
//! policy factory keyed by routing partition, the host's instance config, the registry,
//! and the kernel's config- and projection-derived row columns.

mod config;

pub use dynamo_worker_selection::selection_config;
pub use dynamo_worker_selection::{
    CacheSnapshot, CandidateData, CandidatePreparer, CandidateRank, CandidateRowSource,
    PreparedRows, RankVisitor, RowPick, RuntimePolicyBuilder, RuntimePolicyConfig,
    RuntimePolicyError, RuntimePolicyUpdate, ScoredWorkerCandidate, SelectionConfig,
    SelectionFailure, SelectionTrace, SelectionTraceRow, SelectionTraceSpec, SessionContext,
    SharedCacheHits, SharedCacheType, WorkerCacheData, WorkerCacheInput, WorkerCacheInputs,
    WorkerCandidate, WorkerCandidates, WorkerCapacity, WorkerFilter, WorkerInfo, WorkerInputRows,
    WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerTransferInput,
};
#[cfg(any(test, feature = "bench"))]
pub use dynamo_worker_selection::{DefaultPick, DefaultWorkerPicker};

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
pub use crate::scheduling::WorkerSelectionInputTrigger;
pub use crate::scheduling::selector::{WorkerCapacityExt, WorkerInfoExt, WorkerLoadInputExt};
pub(crate) use config::RawWorkerSelectionConfig;
pub use config::{WorkerSelectionConfig, WorkerSelectionInstance};

use crate::{RoutingPartitionRef, WorkerType};
use std::sync::Arc;

/// Factory that creates one worker-selection policy per routing partition.
pub type WorkerSelectionPolicyFactory = Arc<
    dyn for<'a> Fn(&SelectionConfig, WorkerType, RoutingPartitionRef<'a>) -> WorkerSelectionPolicy
        + Send
        + Sync,
>;
