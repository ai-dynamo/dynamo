// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use dynamo_global_common::PoolId;

/// Counts the requests this process has sent to each pool and not yet
/// finished. Other Global Frontend replicas keep their own counts.
#[derive(Debug, Default)]
pub struct InFlightTracker {
    pools: DashMap<PoolId, Arc<AtomicU64>>,
}

impl InFlightTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts one more request to `pool` until the returned guard drops.
    pub fn start(&self, pool: &PoolId) -> InFlightGuard {
        let count = match self.pools.get(pool) {
            Some(count) => count.clone(),
            None => self.pools.entry(pool.clone()).or_default().clone(),
        };
        count.fetch_add(1, Ordering::Relaxed);
        InFlightGuard { count }
    }

    /// Requests to `pool` that have started and not finished.
    pub fn get(&self, pool: &PoolId) -> u64 {
        self.pools
            .get(pool)
            .map_or(0, |count| count.load(Ordering::Relaxed))
    }
}

/// One request in flight. Dropping it ends the request's count.
#[derive(Debug)]
pub struct InFlightGuard {
    count: Arc<AtomicU64>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};

    use super::*;

    fn id(dgd: &str) -> PoolId {
        V1PoolIdDeriver.derive(&PoolKey::new("ohio", "dynamo", dgd).unwrap())
    }

    #[test]
    fn counts_each_pool_until_its_guards_drop() {
        let tracker = InFlightTracker::new();
        let (blue, green) = (id("blue"), id("green"));

        let first = tracker.start(&blue);
        let second = tracker.start(&blue);
        let other = tracker.start(&green);
        assert_eq!((tracker.get(&blue), tracker.get(&green)), (2, 1));

        drop(first);
        assert_eq!(tracker.get(&blue), 1);
        drop((second, other));
        assert_eq!((tracker.get(&blue), tracker.get(&green)), (0, 0));
        assert_eq!(tracker.get(&id("unseen")), 0);
    }
}
