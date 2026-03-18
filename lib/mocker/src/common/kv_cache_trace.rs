// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared KV cache trace logging for both vLLM and SGLang backends.
//!
//! Enabled by setting `DYN_MOCKER_KV_CACHE_TRACE=1` or `true`.

use dynamo_runtime::config::environment_names::mocker;
use std::env;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Check the env var to enable KV cache allocation/eviction trace logs.
pub static KV_CACHE_TRACE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    env::var(mocker::DYN_MOCKER_KV_CACHE_TRACE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Log a vLLM KV cache trace event.
pub fn log_vllm_trace(
    event: &str,
    dp_rank: u32,
    block_size: usize,
    active_blocks: usize,
    inactive_blocks: usize,
    total_blocks: usize,
) {
    if !*KV_CACHE_TRACE_ENABLED {
        return;
    }
    let free_blocks = total_blocks
        .saturating_sub(active_blocks)
        .saturating_sub(inactive_blocks);
    let utilization = if total_blocks > 0 {
        (active_blocks + inactive_blocks) as f64 / total_blocks as f64
    } else {
        0.0
    };
    tracing::info!(
        engine_type = "vllm",
        event,
        timestamp_ms = timestamp_ms(),
        dp_rank,
        block_size,
        free_blocks,
        active_blocks,
        inactive_blocks,
        total_blocks,
        utilization,
        "KV cache trace"
    );
}

/// SGLang cache state snapshot for trace logging.
pub struct SglangCacheState<'a> {
    pub event: &'a str,
    pub dp_rank: u32,
    pub num_tokens: usize,
    pub page_size: usize,
    pub available_tokens: usize,
    pub evictable_tokens: usize,
    pub protected_tokens: usize,
    pub total_tokens: usize,
}

/// Log an SGLang KV cache trace event.
pub fn log_sglang_trace(state: &SglangCacheState) {
    if !*KV_CACHE_TRACE_ENABLED {
        return;
    }
    let utilization = if state.total_tokens > 0 {
        (state.total_tokens - state.available_tokens) as f64 / state.total_tokens as f64
    } else {
        0.0
    };
    tracing::info!(
        engine_type = "sglang",
        event = state.event,
        timestamp_ms = timestamp_ms(),
        dp_rank = state.dp_rank,
        num_tokens = state.num_tokens,
        page_size = state.page_size,
        available_tokens = state.available_tokens,
        evictable_tokens = state.evictable_tokens,
        protected_tokens = state.protected_tokens,
        total_tokens = state.total_tokens,
        utilization,
        "KV cache trace"
    );
}
