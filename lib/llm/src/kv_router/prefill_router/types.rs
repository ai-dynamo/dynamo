// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::config::RouterConfigOverride;

use crate::protocols::common::preprocessor::{BootstrapInfo, PrefillResult};

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result of the prefill phase in `generate()`.
pub(super) enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready
    Bootstrap(BootstrapInfo),
    /// Synchronous prefill completed with result
    Completed(PrefillResult),
}

pub(super) enum PrefillResolveDecision {
    Resolved {
        worker_id: u64,
        dp_rank: Option<u32>,
        bootstrap_info: BootstrapInfo,
    },
    Unavailable,
    NotActivated,
    NoBootstrapEndpoint,
}

pub(super) fn build_decode_router_override(
    existing_override: Option<RouterConfigOverride>,
) -> RouterConfigOverride {
    // Experimental gate: when DYN_ROUTER_DECODE_KV_AWARE is truthy, let decode routing
    // inherit the frontend's overlap_score_weight and assume_kv_reuse so repeat-prefix
    // traffic is steered toward the decode worker already holding those blocks. Requires
    // SGLang's disaggregation-decode-enable-radix-cache on the worker side.
    let decode_kv_aware = std::env::var("DYN_ROUTER_DECODE_KV_AWARE")
        .ok()
        .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "FALSE"))
        .unwrap_or(false);

    let base = RouterConfigOverride {
        track_prefill_tokens: Some(false),
        ..existing_override.unwrap_or_default()
    };

    if decode_kv_aware {
        base
    } else {
        RouterConfigOverride {
            overlap_score_weight: Some(0.0),
            assume_kv_reuse: Some(false),
            ..base
        }
    }
}
