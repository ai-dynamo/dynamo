// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashSet;

use crate::protocols::WorkerWithDpRank;

#[inline]
pub(crate) fn reconcile_active_workers(
    active: &mut FxHashSet<WorkerWithDpRank>,
    next: &FxHashSet<WorkerWithDpRank>,
    mut on_drop: impl FnMut(WorkerWithDpRank),
) {
    let active_count = active.len();
    let next_count = next.len();

    if next_count == active_count {
        return;
    }

    if next_count < active_count && next.iter().all(|worker| active.contains(worker)) {
        for &worker in active.iter() {
            if !next.contains(&worker) {
                on_drop(worker);
            }
        }
        active.clone_from(next);
        return;
    }

    active.retain(|worker| {
        if next.contains(worker) {
            true
        } else {
            on_drop(*worker);
            false
        }
    });
}
