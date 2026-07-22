// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerCapacity {
    pub worker: WorkerWithDpRank,
    /// GPU KV capacity available to requests that may be delivered to the backend.
    pub device_tokens: usize,
    /// Combined GPU KV and native-offload capacity used for session retention.
    pub tokens: usize,
}

pub trait WorkerCapacityProvider: Send {
    fn snapshot(&mut self) -> Arc<[WorkerCapacity]>;
}

impl<F> WorkerCapacityProvider for F
where
    F: FnMut() -> Vec<WorkerCapacity> + Send,
{
    fn snapshot(&mut self) -> Arc<[WorkerCapacity]> {
        self().into()
    }
}

pub struct WatchWorkerCapacity<C> {
    workers: watch::Receiver<HashMap<WorkerId, C>>,
    block_size: u32,
    cached: Option<Arc<[WorkerCapacity]>>,
    warned_missing_capacity: bool,
}

impl<C> WatchWorkerCapacity<C> {
    pub fn new(workers: watch::Receiver<HashMap<WorkerId, C>>, block_size: u32) -> Self {
        Self {
            workers,
            block_size,
            cached: None,
            warned_missing_capacity: false,
        }
    }
}

impl<C> WorkerCapacityProvider for WatchWorkerCapacity<C>
where
    C: WorkerConfigLike + Send + Sync,
{
    fn snapshot(&mut self) -> Arc<[WorkerCapacity]> {
        if let Some(cached) = &self.cached
            && !self.workers.has_changed().unwrap_or(false)
        {
            return Arc::clone(cached);
        }

        let workers = self.workers.borrow_and_update();
        let mut capacities = Vec::new();
        let mut missing_capacity_workers = 0;
        for (&worker_id, config) in workers.iter() {
            let Some(blocks) = config.total_kv_blocks() else {
                missing_capacity_workers += 1;
                continue;
            };
            if blocks == 0 {
                missing_capacity_workers += 1;
                continue;
            }
            let device_tokens = blocks.saturating_mul(u64::from(self.block_size));
            let tokens = device_tokens
                .saturating_add(config.native_offloading_capacity_tokens().unwrap_or(0));
            let device_tokens = usize::try_from(device_tokens).unwrap_or(usize::MAX);
            let tokens = usize::try_from(tokens).unwrap_or(usize::MAX);
            let start = config.data_parallel_start_rank();
            let end = start.saturating_add(config.data_parallel_size());
            capacities.extend((start..end).map(|dp_rank| WorkerCapacity {
                worker: WorkerWithDpRank::new(worker_id, dp_rank),
                device_tokens,
                tokens,
            }));
        }
        if missing_capacity_workers > 0 && !self.warned_missing_capacity {
            tracing::warn!(
                worker_count = workers.len(),
                missing_capacity_workers,
                "Session-aware admission-control capacity gating excludes workers without usable KV capacity"
            );
            self.warned_missing_capacity = true;
        }
        capacities.sort_unstable_by_key(|capacity| capacity.worker);
        let capacities = Arc::from(capacities);
        self.cached = Some(Arc::clone(&capacities));
        capacities
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    struct Config;

    impl WorkerConfigLike for Config {
        fn data_parallel_start_rank(&self) -> u32 {
            2
        }

        fn data_parallel_size(&self) -> u32 {
            2
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(10)
        }

        fn native_offloading_capacity_tokens(&self) -> Option<u64> {
            Some(7)
        }

        fn taints(&self) -> &HashSet<String> {
            static EMPTY: std::sync::LazyLock<HashSet<String>> =
                std::sync::LazyLock::new(HashSet::new);
            &EMPTY
        }
    }

    struct ZeroConfig;

    impl WorkerConfigLike for ZeroConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(0)
        }

        fn native_offloading_capacity_tokens(&self) -> Option<u64> {
            Some(1_000)
        }

        fn taints(&self) -> &HashSet<String> {
            static EMPTY: std::sync::LazyLock<HashSet<String>> =
                std::sync::LazyLock::new(HashSet::new);
            &EMPTY
        }
    }

    #[test]
    fn watch_provider_expands_dp_ranks_and_offload_capacity() {
        let (_tx, rx) = watch::channel(HashMap::from([(11, Config)]));
        let mut provider = WatchWorkerCapacity::new(rx, 16);
        assert_eq!(
            provider.snapshot().as_ref(),
            &[
                WorkerCapacity {
                    worker: WorkerWithDpRank::new(11, 2),
                    device_tokens: 160,
                    tokens: 167,
                },
                WorkerCapacity {
                    worker: WorkerWithDpRank::new(11, 3),
                    device_tokens: 160,
                    tokens: 167,
                },
            ]
        );
    }

    #[test]
    fn watch_provider_treats_zero_kv_blocks_as_missing_capacity() {
        let (_tx, rx) = watch::channel(HashMap::from([(11, ZeroConfig)]));
        let mut provider = WatchWorkerCapacity::new(rx, 16);
        assert!(provider.snapshot().is_empty());
        assert!(provider.warned_missing_capacity);
    }
}
