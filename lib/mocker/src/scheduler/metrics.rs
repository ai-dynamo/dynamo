// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-neutral metrics exposed by the Mocker scheduler boundary.

/// Latest observable state for one Mocker scheduler rank.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct MockerMetrics {
    pub dp_rank: dynamo_kv_router::protocols::DpRank,
    pub active_decode_blocks: u64,
    pub total_blocks: u64,
    pub gpu_cache_usage_perc: f64,
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub vllm_preemptions_total: u64,
    pub sglang_cache_hit_tokens: u64,
    pub sglang_cache_total_tokens: u64,
}

impl MockerMetrics {
    pub fn new(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
    ) -> Self {
        Self::from_parts(dp_rank, active_decode_blocks, total_blocks, 0, 0, 0, 0, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
        running_requests: u64,
        waiting_requests: u64,
        vllm_preemptions_total: u64,
        sglang_cache_hit_tokens: u64,
        sglang_cache_total_tokens: u64,
    ) -> Self {
        let gpu_cache_usage_perc = if total_blocks == 0 {
            0.0
        } else {
            active_decode_blocks as f64 / total_blocks as f64
        };
        Self {
            dp_rank,
            active_decode_blocks,
            total_blocks,
            gpu_cache_usage_perc,
            running_requests,
            waiting_requests,
            vllm_preemptions_total,
            sglang_cache_hit_tokens,
            sglang_cache_total_tokens,
        }
    }
}
