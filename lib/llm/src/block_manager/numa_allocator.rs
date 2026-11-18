// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-export NUMA utilities from dynamo-memory.
pub use dynamo_memory::numa::*;

/// Check if NUMA optimization is explicitly opted-in for the block manager.
///
/// Set `DYN_KVBM_ENABLE_NUMA=1` to enable NUMA-aware allocation in the
/// KV cache block manager. This is opt-in because the block manager
/// manages its own pinned memory allocations separately from `PinnedStorage`.
///
/// The global kill switch `DYN_MEMORY_DISABLE_NUMA` always takes precedence:
/// if it is set truthy, this function returns `false` regardless of
/// `DYN_KVBM_ENABLE_NUMA`.
///
/// TODO(KVBM-336): remove this function in the future
#[deprecated(
    since = "1.0.0",
    note = "Use dynamo_memory::numa::is_numa_enabled instead"
)]
pub fn is_numa_enabled() -> bool {
    // Global kill switch always wins
    if dynamo_memory::numa::is_numa_disabled() {
        return false;
    }
    dynamo_runtime::config::env_is_truthy("DYN_KVBM_ENABLE_NUMA")
}
