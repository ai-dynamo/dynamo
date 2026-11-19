// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::plugins::prelude::{WorkerCacheInputs, WorkerLoadInput};

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

// The builtin two-tier picker is private; parity tests keep this fixed-default copy aligned.
pub(super) fn select(
    cache: WorkerCacheInputs<'_>,
    load: &[WorkerLoadInput],
    request_blocks: u64,
) -> Option<usize> {
    if cache.is_empty() || cache.len() != load.len() {
        return None;
    }
    let min_load = load.iter().map(|row| row.active_requests()).min()?;
    let max_load = load.iter().map(|row| row.active_requests()).max()?;
    if max_load.saturating_sub(min_load) > 32 && (max_load as f64) > 1.1 * (min_load as f64) {
        return least_loaded(load, 0..load.len());
    }
    let max_overlap = cache
        .iter()
        .map(|row| row.device_overlap_blocks())
        .max_by(f64::total_cmp)?;
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    if cache_ratio > 0.5 {
        return least_loaded(
            load,
            cache.iter().enumerate().filter_map(|(row, item)| {
                (item.device_overlap_blocks() == max_overlap).then_some(row)
            }),
        );
    }
    least_loaded(load, 0..load.len())
}
