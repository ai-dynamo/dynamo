// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_runtime::config::environment_names::kvbm::remote_storage as env_g4;
use once_cell::sync::Lazy;

/// Default timeout in seconds for G4 (remote storage) transfers.
const DEFAULT_G4_TRANSFER_TIMEOUT_SECS: u64 = 30;
/// Minimum number of G4 candidate blocks required before triggering object lookup.
const DEFAULT_G4_MIN_CANDIDATE_BLOCKS: usize = 8;
/// Default batch size for flushing remaining blocks on request finish.
const DEFAULT_FLUSH_BATCH_SIZE: usize = 512;

/// Timeout for G4 transfers - cached from env var.
static G4_TRANSFER_TIMEOUT: Lazy<Duration> = Lazy::new(|| {
    let secs: u64 = std::env::var(env_g4::DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_G4_TRANSFER_TIMEOUT_SECS);
    Duration::from_secs(secs)
});

/// Minimum number of G4 candidate blocks required to trigger object lookup.
/// Set to 0 to disable gating.
static G4_MIN_CANDIDATE_BLOCKS: Lazy<usize> = Lazy::new(|| {
    std::env::var(env_g4::DYN_KVBM_G4_MIN_CANDIDATE_BLOCKS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_G4_MIN_CANDIDATE_BLOCKS)
});

/// Flush batch size for post-request D2H flushing.
static FLUSH_BATCH_SIZE: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_FLUSH_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_FLUSH_BATCH_SIZE)
});

#[inline]
pub fn g4_transfer_timeout() -> Duration {
    *G4_TRANSFER_TIMEOUT
}

#[inline]
pub fn g4_min_candidate_blocks() -> usize {
    *G4_MIN_CANDIDATE_BLOCKS
}

#[inline]
pub fn flush_batch_size() -> usize {
    *FLUSH_BATCH_SIZE
}
