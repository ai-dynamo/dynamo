// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental router plugin APIs.
//!
//! Items here have no compatibility guarantee: Dynamo can change or remove one in any release,
//! without a deprecation period. Each API has its own submodule and is imported by name, so
//! that a plugin's dependence on it is visible where it is used:
//!
//! ```
//! use dynamo_kv_router::plugins::experimental::dispatch_gate::{DispatchDecision, DispatchGate};
//! ```
//!
//! An experimental API's names move into the module for its kind of plugin,
//! [`super::worker_selection`] or [`super::request_classifier`], when it is ready.

/// How a request classifier holds a queued request back from dispatch.
pub mod dispatch_gate {
    pub use dynamo_admission::{DispatchDecision, DispatchGate, DispatchHead, DispatchRecheck};
}

/// Hint policies: engine-directed KV actions for a booked worker, their registration, and the
/// KV copies they send.
pub mod hints {
    pub use crate::plugins::hint_policy::{
        Commitments, HintContext, HintHost, HintPolicy, HintPolicyConstructor, HintPolicyError,
        HintPolicyParameters, HintPolicyProvider, HintPolicyProviderError, HintPolicyRegistryError,
        KvHintAction,
    };
    pub use dynamo_kv_hashing::ExternalSequenceBlockHash;
    pub use dynamo_kv_hints::KvSourceLocationsPayload;
    pub use dynamo_kv_indexer::KvTransferCandidateSource;
    pub use dynamo_scheduler::{KvCopyLocations, KvCopySource};
}

/// Per-rank resident block counts and cache-owner capacity, read by a worker-selection policy.
pub use dynamo_worker_selection::experimental::resident_blocks;

/// Live frontend-local decode progress, read by a worker-selection picker.
pub use dynamo_worker_selection::experimental::decode_progress;
