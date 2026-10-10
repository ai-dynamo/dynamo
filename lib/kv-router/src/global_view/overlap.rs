// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-specific KV overlap read boundary for the Global View.

use super::PoolId;

/// Returns an approximate matched text-token prefix for a routing pool.
///
/// Implementations hash tokens with each producer's advertised query semantics
/// and read only a compatible, fresh CKF replica. None means no usable estimate;
/// zero means a usable replica found no matching prefix.
pub trait KvOverlapScorer: Send + Sync {
    fn estimate_matched_prefix_tokens(
        &self,
        pool_id: &PoolId,
        model: &str,
        token_ids: &[u32],
    ) -> Option<u64>;
}
