// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::protocols::WorkerWithDpRank;

/// One registered worker rank and its advertised capacity, which may be absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestClassifierWorker {
    worker: WorkerWithDpRank,
    total_kv_blocks: Option<u64>,
}

impl RequestClassifierWorker {
    pub fn new(worker: WorkerWithDpRank, total_kv_blocks: Option<u64>) -> Self {
        Self {
            worker,
            total_kv_blocks,
        }
    }

    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Total capacity from the worker's runtime config, not currently free blocks.
    /// Each block contains [`RequestClassifierContext::block_size`] tokens.
    pub fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }
}

/// Cached discovery inputs for one router. Workers appear and disappear as the
/// host processes registration updates; this does not probe worker health.
#[derive(Clone)]
pub struct RequestClassifierContext {
    block_size: u32,
    workers: Arc<dyn Fn() -> Vec<RequestClassifierWorker> + Send + Sync>,
}

impl RequestClassifierContext {
    /// The callback must return cached state without blocking I/O.
    pub fn new(
        block_size: u32,
        workers: impl Fn() -> Vec<RequestClassifierWorker> + Send + Sync + 'static,
    ) -> Self {
        Self {
            block_size,
            workers: Arc::new(workers),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn workers(&self) -> Vec<RequestClassifierWorker> {
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
