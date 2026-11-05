// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::plugins::worker_selection::WorkerInfo;

/// Cached discovery inputs for one router. Workers appear and disappear as the
/// host processes registration updates; this does not probe worker health.
#[derive(Clone)]
pub struct RequestClassifierContext {
    block_size: u32,
    workers: Arc<dyn Fn() -> Vec<WorkerInfo> + Send + Sync>,
}

impl RequestClassifierContext {
    /// The callback must return cached state without blocking I/O.
    pub fn new(
        block_size: u32,
        workers: impl Fn() -> Vec<WorkerInfo> + Send + Sync + 'static,
    ) -> Self {
        Self {
            block_size,
            workers: Arc::new(workers),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn workers(&self) -> Vec<WorkerInfo> {
        (self.workers)()
    }
}

impl std::fmt::Debug for RequestClassifierContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestClassifierContext")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}
