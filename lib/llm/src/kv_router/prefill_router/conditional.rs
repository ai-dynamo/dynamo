// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::protocols::common::llm_backend::PreprocessedRequest;

/// Key used in `extra_args` to store the effective ISL (total tokens minus KV cache overlap).
pub const EFFECTIVE_ISL_KEY: &str = "effective_isl";

/// Trait for deciding per-request whether to skip prefill and route directly to decode.
///
/// Implementations receive the preprocessed request and return `true` to skip prefill
/// (routing directly to decode as an aggregated request) or `false` to proceed with
/// the normal disaggregated prefill→decode flow.
pub trait ConditionalPrefillStrategy: Send + Sync + std::fmt::Debug {
    fn should_skip_prefill(&self, req: &PreprocessedRequest) -> bool;

    /// Whether this strategy needs KV cache overlap information to make its decision.
    /// When true, the router queries the prefill-side KV indexer and stores the
    /// effective ISL in `extra_args["effective_isl"]` before calling `should_skip_prefill`.
    fn needs_kv_overlap(&self) -> bool {
        false
    }
}

/// Skip prefill for requests with input sequence length below a threshold.
///
/// Short prompts are cheap to prefill locally on the decode worker, so bypassing
/// the remote prefill worker avoids the KV transfer overhead.
#[derive(Debug, Clone)]
pub struct IslThresholdStrategy {
    pub max_tokens: usize,
}

impl IslThresholdStrategy {
    pub fn new(max_tokens: usize) -> Arc<dyn ConditionalPrefillStrategy> {
        Arc::new(Self { max_tokens })
    }
}

impl ConditionalPrefillStrategy for IslThresholdStrategy {
    fn should_skip_prefill(&self, req: &PreprocessedRequest) -> bool {
        // Use effective ISL (after KV cache overlap) if available, else fall back to full ISL
        let effective_isl = req
            .extra_args
            .as_ref()
            .and_then(|args| args.get(EFFECTIVE_ISL_KEY))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(req.token_ids.len());

        effective_isl < self.max_tokens
    }

    fn needs_kv_overlap(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::preprocessor::PreprocessedRequest;

    fn make_request(num_tokens: usize) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model(String::new())
            .token_ids(vec![0; num_tokens])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    #[test]
    fn test_isl_threshold_skips_short_requests() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        let req = make_request(64);
        assert!(strategy.should_skip_prefill(&req));
    }

    #[test]
    fn test_isl_threshold_keeps_long_requests() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        let req = make_request(256);
        assert!(!strategy.should_skip_prefill(&req));
    }

    #[test]
    fn test_isl_threshold_boundary_equal() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        let req = make_request(128);
        assert!(!strategy.should_skip_prefill(&req), "equal to threshold should NOT skip");
    }

    #[test]
    fn test_isl_threshold_boundary_one_below() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        let req = make_request(127);
        assert!(strategy.should_skip_prefill(&req), "one below threshold should skip");
    }

    #[test]
    fn test_isl_threshold_zero_tokens() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        let req = make_request(0);
        assert!(strategy.should_skip_prefill(&req));
    }

    #[test]
    fn test_isl_threshold_uses_effective_isl_from_extra_args() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        // 256 tokens total, but effective ISL is 64 (high cache overlap)
        let mut req = make_request(256);
        req.extra_args = Some(serde_json::json!({ "effective_isl": 64 }));
        assert!(strategy.should_skip_prefill(&req), "should skip based on effective ISL");
    }

    #[test]
    fn test_isl_threshold_effective_isl_above_threshold() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        // 256 tokens total, effective ISL is 200 (low cache overlap)
        let mut req = make_request(256);
        req.extra_args = Some(serde_json::json!({ "effective_isl": 200 }));
        assert!(!strategy.should_skip_prefill(&req), "should NOT skip when effective ISL above threshold");
    }

    #[test]
    fn test_isl_threshold_falls_back_to_full_isl_without_extra_args() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        let req = make_request(64);
        assert!(req.extra_args.is_none());
        assert!(strategy.should_skip_prefill(&req), "should fall back to token_ids.len()");
    }

    #[test]
    fn test_needs_kv_overlap() {
        let strategy = IslThresholdStrategy { max_tokens: 128 };
        assert!(strategy.needs_kv_overlap());
    }
}
