// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::connector::protocol::WorkerTransferRequest;

#[derive(Debug, Default)]
pub struct OperationTracker {
    pending_operations: Option<Vec<WorkerTransferRequest>>,
    dispatched_operations_count: usize,
}

impl OperationTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_any(&self) -> bool {
        self.pending_count() > 0 || self.dispatched_operations_count > 0
    }

    pub fn pending_count(&self) -> usize {
        self.pending_operations
            .as_ref()
            .map(|ops| ops.len())
            .unwrap_or(0)
    }

    pub fn dispatched_count(&self) -> usize {
        self.dispatched_operations_count
    }

    pub fn append_pending(&mut self, operation: WorkerTransferRequest) {
        if let Some(pending_operations) = self.pending_operations.as_mut() {
            pending_operations.push(operation);
        } else {
            self.pending_operations = Some(vec![operation]);
        }
    }

    /// Moves pending operations into metadata-dispatch state by incrementing
    /// `dispatched_operations_count`.
    pub fn take_pending_for_dispatch(&mut self) -> Option<Vec<WorkerTransferRequest>> {
        let ops = self.pending_operations.take();
        if let Some(ref ops) = ops {
            self.dispatched_operations_count += ops.len();
        }
        ops
    }

    /// Drops pending operations without increasing dispatched count.
    pub fn discard_pending(&mut self) -> usize {
        self.pending_operations.take().map(|ops| ops.len()).unwrap_or(0)
    }

    /// Clears pending and dispatched operation accounting.
    pub fn clear_all(&mut self) {
        self.pending_operations = None;
        self.dispatched_operations_count = 0;
    }
}
