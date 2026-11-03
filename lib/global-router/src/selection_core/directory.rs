// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker id to pool and region, for pickers that weigh per-pool signals.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use parking_lot::RwLock;

use super::catalog::worker_id;
use crate::Pool;

#[derive(Debug)]
pub(super) struct PoolEntry {
    pub(super) id: PoolId,
    pub(super) region: String,
}

/// Replaced whole on each snapshot; read on every pick.
#[derive(Debug, Default)]
pub(super) struct PoolDirectory {
    pools: RwLock<Arc<HashMap<WorkerId, PoolEntry>>>,
}

impl PoolDirectory {
    pub(super) fn replace(&self, pools: &[Pool]) {
        let pools = pools
            .iter()
            .map(|pool| {
                let entry = PoolEntry {
                    id: pool.id.clone(),
                    region: pool.location.region.clone(),
                };
                (worker_id(&pool.id), entry)
            })
            .collect();
        *self.pools.write() = Arc::new(pools);
    }

    pub(super) fn snapshot(&self) -> Arc<HashMap<WorkerId, PoolEntry>> {
        self.pools.read().clone()
    }
}
