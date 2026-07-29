// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::protocols::PrefillLoadHint;

pub trait PrefillLoadEstimator: Send + Sync {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<Duration>;
}

pub fn effective_prefill_tokens(isl_tokens: usize, weighted_cached_tokens: usize) -> usize {
    isl_tokens.saturating_sub(weighted_cached_tokens.min(isl_tokens))
}

pub fn prefill_load_hint_from_effective_tokens(
    isl_tokens: usize,
    effective_prefill_tokens: usize,
) -> Result<PrefillLoadHint, InvalidEffectivePrefillTokens> {
    if effective_prefill_tokens > isl_tokens {
        return Err(InvalidEffectivePrefillTokens {
            effective_prefill_tokens,
            isl_tokens,
        });
    }

    Ok(PrefillLoadHint {
        initial_effective_prefill_tokens: effective_prefill_tokens,
        expected_prefill_duration: None,
    })
}

#[derive(Debug, thiserror::Error)]
#[error(
    "effective_prefill_tokens ({effective_prefill_tokens}) must not exceed isl_tokens ({isl_tokens})"
)]
pub struct InvalidEffectivePrefillTokens {
    pub effective_prefill_tokens: usize,
    pub isl_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_prefill_clamps_weighted_cache_credit() {
        assert_eq!(effective_prefill_tokens(100, 40), 60);
        assert_eq!(effective_prefill_tokens(100, 120), 0);
    }

    #[test]
    fn direct_hint_validates_against_isl() {
        let hint = prefill_load_hint_from_effective_tokens(100, 60).unwrap();
        assert_eq!(hint.initial_effective_prefill_tokens, 60);
        assert!(prefill_load_hint_from_effective_tokens(100, 101).is_err());
    }
}
