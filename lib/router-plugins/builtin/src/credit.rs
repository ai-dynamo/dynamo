// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared arithmetic for Dynamo's aggregated prefill/decode cost.
//!
//! Callers own eligibility, snapshot freshness, token-to-block conversion, cache
//! tier weights/decay, and booking. This function never mutates scheduling state.

/// Combine prefill work, cache credit, decode work, and the optional request cost.
/// All work and credit inputs are in KV blocks; `prefill_load_scale` is dimensionless.
/// Inputs must be finite. Cache credit only reduces prefill, with a floor of zero;
/// it cannot cancel decode work. The host validates the resulting finite score.
/// Keeping this arithmetic shared prevents pool and worker experiments drifting.
#[inline]
pub fn aggregated_cost(
    prefill_blocks: f64,
    cache_credit: f64,
    prefill_load_scale: f64,
    decode_blocks: f64,
    request_cost: f64,
) -> f64 {
    let prefill = (prefill_blocks - cache_credit).max(0.0);
    prefill_load_scale * prefill + decode_blocks + request_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_credit_cannot_erase_decode_load() {
        assert_eq!(aggregated_cost(4.0, 8.0, 2.0, 9.0, 3.0), 12.0);
        assert_eq!(aggregated_cost(10.0, 4.0, 2.0, 9.0, 3.0), 24.0);
    }
}
