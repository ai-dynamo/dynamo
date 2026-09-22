// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous adapter for direct selection and deterministic replay.

use dynamo_kv_router::plugins::worker_selection::{WorkerInputs, WorkerSelectionPolicy};
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerSelectionResult};
use dynamo_kv_router::{KvRouterConfig, KvSchedulerError, WorkerSelectionInput, WorkerSelector};
use parking_lot::Mutex;
use std::sync::Arc;

use super::policy_with_rng;

/// Synchronous adapter for direct callers and replay. Scheduler actors use `default_policy`
/// directly and do not acquire this adapter's mutex.
pub struct DefaultWorkerSelector {
    kv_router_config: KvRouterConfig,
    worker_type: &'static str,
    policy: Mutex<WorkerSelectionPolicy>,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
}
impl DefaultWorkerSelector {
    pub fn new(config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self::with_rng(config.unwrap_or_default(), worker_type, None)
    }
    /// Construct a reproducible selector. Clones share its random stream.
    pub fn new_seeded(
        config: Option<KvRouterConfig>,
        worker_type: &'static str,
        seed: u64,
    ) -> Self {
        Self::with_rng(
            config.unwrap_or_default(),
            worker_type,
            Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
        )
    }
    fn with_rng(
        config: KvRouterConfig,
        worker_type: &'static str,
        rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        Self {
            policy: Mutex::new(policy_with_rng(
                config.clone(),
                super::PolicyParameters::from(&config),
                worker_type,
                rng.clone(),
                false,
            )),
            kv_router_config: config,
            worker_type,
            rng,
        }
    }
}
impl Clone for DefaultWorkerSelector {
    fn clone(&self) -> Self {
        Self::with_rng(
            self.kv_router_config.clone(),
            self.worker_type,
            self.rng.clone(),
        )
    }
}
impl std::fmt::Debug for DefaultWorkerSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultWorkerSelector")
            .field("kv_router_config", &self.kv_router_config)
            .field("worker_type", &self.worker_type)
            .finish_non_exhaustive()
    }
}
impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        <WorkerSelectionPolicy as WorkerSelector<C>>::required_worker_inputs(&self.policy.lock())
    }
    fn uses_exclusive_affinity_target(&self) -> bool {
        true
    }
    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        self.policy.lock().select_worker(input)
    }
}
