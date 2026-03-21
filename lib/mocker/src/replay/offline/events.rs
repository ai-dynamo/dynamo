// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub(crate) struct WorkerCompletion {
    pub(crate) at_ms: f64,
    pub(crate) worker_idx: usize,
    pub(crate) completed_requests: usize,
}

impl PartialEq for WorkerCompletion {
    fn eq(&self, other: &Self) -> bool {
        self.at_ms.to_bits() == other.at_ms.to_bits()
            && self.worker_idx == other.worker_idx
            && self.completed_requests == other.completed_requests
    }
}

impl Eq for WorkerCompletion {}

impl PartialOrd for WorkerCompletion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkerCompletion {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at_ms
            .partial_cmp(&self.at_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.worker_idx.cmp(&self.worker_idx))
            .then_with(|| other.completed_requests.cmp(&self.completed_requests))
    }
}
