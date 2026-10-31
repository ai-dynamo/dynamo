// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The scheduler kernel: run the stages atomically for one request. Owns the request and
//! response vocabulary, eligibility, the queue actor and `LocalScheduler`, the overlap and
//! prefill-load analysis, the selection kernel over the worker-selection stage, the validated
//! kernel configuration, and the tracking hash. Hosts and the composed router
//! (`dynamo-kv-router`) sit above it; the five stages, protocol, and kv-hashing sit below.

pub mod classifier_rows;
pub mod config;
pub mod filter;
pub mod identity;
pub mod kv_hints;
pub mod local;
pub mod overlap;
pub mod overlap_refresh;
pub mod overlap_summary;
pub mod prefill_load;
pub mod protocols;
pub mod queue;
pub mod selector;
pub mod shared_cache;
pub mod tracking_hash;
pub mod types;

#[cfg(any(test, feature = "testing", feature = "bench"))]
pub mod test_utils;

pub use config::{
    CacheHitWeights, PolicyProfileSource, RouterConfig, RouterConfigError, Validated,
};
pub use config::{RouterConfigOverride, RouterPrefillLoadModel};
pub use filter::*;
pub use local::LocalScheduler;
pub use overlap::{
    CacheHitEstimates, OverlapAnalysis, OverlapScoresResponse, OverlapSignals,
    SelectedWorkerTierSnapshot, SharedCacheOverlapScore, WorkerOverlapScore,
};
pub use overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap, TieredOverlapRefresher,
};
pub use overlap_summary::MooncakeOverlapSummary;
pub use prefill_load::{
    InvalidEffectivePrefillTokens, PrefillLoadEstimator, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
pub use types::*;
