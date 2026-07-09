// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded direct-Valkey publisher tuning.

pub(super) const MAX_BATCHING_TIMEOUT_MS: u64 = 15_000;
pub const DEFAULT_BATCHING_TIMEOUT_MS: Option<u64> = None;
pub const MAX_VALKEY_WORKER_RANKS: u32 = crate::kv_router::indexer::valkey::MAX_WORKER_RANKS as u32;
/// Direct Valkey writers synchronously wait for replica acknowledgement. A
/// short window coalesces bursty engine events into one ordered APPLY/WAIT
/// without meaningfully delaying cache-affinity visibility.
pub(super) const DEFAULT_VALKEY_BATCHING_TIMEOUT_MS: u64 = 1;
pub(super) const VALKEY_BATCHING_TIMEOUT_ENV: &str = "DYN_ROUTER_VALKEY_EVENT_BATCHING_TIMEOUT_MS";
pub(super) const VALKEY_EVENT_INPUT_BUFFER_SIZE_ENV: &str =
    "DYN_ROUTER_VALKEY_EVENT_INPUT_BUFFER_SIZE";
/// Direct worker events are affinity metadata, so overload is handled by a
/// worker-wide permanent integrity fence instead of unbounded buffering.
/// One 4,096-concurrency, 1,024-token wave can produce 65,536 sixteen-token
/// block events on a worker. Two such waves cover an in-flight wave plus the
/// replacement wave that can arrive while Sentinel promotes a replica.
pub(super) const DEFAULT_VALKEY_EVENT_INPUT_BUFFER_SIZE: usize = 131_072;
pub(super) const MIN_VALKEY_EVENT_INPUT_BUFFER_SIZE: usize = 1_024;
pub(super) const MAX_VALKEY_EVENT_INPUT_BUFFER_SIZE: usize = 1_048_576;

pub(super) fn valkey_batching_timeout_ms(
    configured_timeout_ms: Option<u64>,
    env_timeout_ms: Option<&str>,
) -> Option<u64> {
    configured_timeout_ms.or_else(|| match env_timeout_ms {
        None => Some(DEFAULT_VALKEY_BATCHING_TIMEOUT_MS),
        Some(value) => match value.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(timeout_ms) => Some(timeout_ms.min(MAX_BATCHING_TIMEOUT_MS)),
            Err(_) => Some(DEFAULT_VALKEY_BATCHING_TIMEOUT_MS),
        },
    })
}

pub(super) fn valkey_event_input_buffer_size(env_value: Option<&str>) -> usize {
    let Some(value) = env_value else {
        return DEFAULT_VALKEY_EVENT_INPUT_BUFFER_SIZE;
    };

    match value.trim().parse::<usize>() {
        Ok(size) if size >= MIN_VALKEY_EVENT_INPUT_BUFFER_SIZE => {
            size.min(MAX_VALKEY_EVENT_INPUT_BUFFER_SIZE)
        }
        _ => DEFAULT_VALKEY_EVENT_INPUT_BUFFER_SIZE,
    }
}
