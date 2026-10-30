// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which pool each of the core's workers is.
//!
//! The core's plugins see worker ids and the core's own load columns. The
//! Global Router's signals (Relay occupancy, region round trips) are keyed by
//! pool, so a picker that weighs them needs the way back from a worker id to
//! its pool and region. [`PoolCatalogSource`](super::PoolCatalogSource)
//! keeps this directory in step with the snapshots it hands the core.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use parking_lot::RwLock;

use super::catalog::worker_id;
use crate::Pool;

/// What a picker needs to know about a worker's pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolEntry {
    pub id: PoolId,
    pub region: String,
}

/// Worker id to pool, for every pool of the last snapshot. Replaced as a
/// whole on each snapshot and read on every selection.
#[derive(Debug, Default)]
pub struct PoolDirectory {
    entries: RwLock<Arc<HashMap<WorkerId, PoolEntry>>>,
}

impl PoolDirectory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace(&self, pools: &[Pool]) {
        let entries = pools
            .iter()
            .map(|pool| {
                (
                    worker_id(&pool.id),
                    PoolEntry {
                        id: pool.id.clone(),
                        region: pool.location.region.clone(),
                    },
                )
            })
            .collect();
        *self.entries.write() = Arc::new(entries);
    }

    /// The directory as one consistent set.
    pub fn snapshot(&self) -> Arc<HashMap<WorkerId, PoolEntry>> {
        self.entries.read().clone()
    }

    /// The pool `worker` stands for, if the last snapshot had it.
    pub fn pool(&self, worker: WorkerId) -> Option<PoolEntry> {
        self.entries.read().get(&worker).cloned()
    }
}

#[cfg(test)]
mod tests {
    use crate::selector::tests::regional_pool;

    use super::*;

    #[test]
    fn maps_workers_back_to_their_pool_and_region() {
        let blue = regional_pool("us-east-2", "blue", "qwen", &["http://blue:8000"]);
        let tokyo = regional_pool("ap-northeast-1", "tokyo", "qwen", &["http://tokyo:8000"]);
        let directory = PoolDirectory::new();
        assert_eq!(directory.pool(worker_id(&blue.id)), None);

        directory.replace(&[blue.clone(), tokyo.clone()]);
        assert_eq!(
            directory.pool(worker_id(&tokyo.id)),
            Some(PoolEntry {
                id: tokyo.id.clone(),
                region: "ap-northeast-1".to_owned(),
            })
        );
        assert_eq!(directory.snapshot().len(), 2);

        directory.replace(std::slice::from_ref(&blue));
        assert_eq!(directory.pool(worker_id(&tokyo.id)), None);
        assert_eq!(
            directory.pool(worker_id(&blue.id)).map(|entry| entry.id),
            Some(blue.id)
        );
    }
}
