// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker id to pool, for pickers that weigh per-pool signals.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;
use parking_lot::RwLock;

use super::catalog::worker_id;
use crate::Pool;

/// Replaced whole on each snapshot; read on every pick.
#[derive(Debug, Default)]
pub(super) struct PoolDirectory {
    pools: RwLock<Arc<HashMap<WorkerId, PoolId>>>,
}

impl PoolDirectory {
    pub(super) fn replace(&self, pools: &[Pool]) {
        let pools = pools
            .iter()
            .map(|pool| (worker_id(&pool.id), pool.id.clone()))
            .collect();
        *self.pools.write() = Arc::new(pools);
    }

    pub(super) fn snapshot(&self) -> Arc<HashMap<WorkerId, PoolId>> {
        self.pools.read().clone()
    }
}
