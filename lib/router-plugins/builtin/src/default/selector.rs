// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous adapter for direct selection and deterministic replay.

use dynamo_kv_router::plugins::worker_selection::{
    RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate, WorkerInputs,
    WorkerSelectionPolicy,
};
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerSelectionResult};
use dynamo_kv_router::{KvRouterConfig, KvSchedulerError, WorkerSelectionInput, WorkerSelector};
use parking_lot::Mutex;
use std::sync::Arc;

use super::runtime_policy;

/// Synchronous adapter for direct callers and replay. Scheduler actors use `default_policy`
/// directly and do not acquire this adapter's mutex.
///
/// The policy accepts runtime parameter updates through a [`DefaultWorkerSelectorControl`] from
/// [`Self::runtime_control`]. Updates apply between selections and keep the selector's random
/// stream.
pub struct DefaultWorkerSelector {
    kv_router_config: KvRouterConfig,
    worker_type: &'static str,
    policy: Arc<Mutex<WorkerSelectionPolicy>>,
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
            policy: Arc::new(Mutex::new(runtime_policy(
                config.clone(),
                super::PolicyParameters::from(&config),
                worker_type,
                rng.clone(),
            ))),
            kv_router_config: config,
            worker_type,
            rng,
        }
    }
    /// Handle that reads and updates this selector's runtime parameters from another owner.
    ///
    /// Every handle shares this selector's policy, so an update is visible to the next
    /// selection. A clone of the selector has its own policy and needs its own handle.
    pub fn runtime_control(&self) -> DefaultWorkerSelectorControl {
        DefaultWorkerSelectorControl {
            policy: Arc::clone(&self.policy),
        }
    }
}
/// A clone starts from the startup parameters at revision 0, with its own policy, and shares
/// the random stream of the original.
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

/// Cloneable runtime-parameter handle for one [`DefaultWorkerSelector`].
///
/// Updates are serialized with selection by the selector's mutex and take effect at the next
/// selection. They follow the [`WorkerSelectionPolicy`] runtime contract: revision-checked,
/// validated before they replace the policy, and rejected when the patch is empty.
#[derive(Clone)]
pub struct DefaultWorkerSelectorControl {
    policy: Arc<Mutex<WorkerSelectionPolicy>>,
}
impl DefaultWorkerSelectorControl {
    /// Current resolved parameters and revision of the selector's policy.
    pub fn runtime_config(&self) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.policy.lock().runtime_config()
    }
    /// Validate and apply `update` to the selector's policy.
    pub fn update_runtime_config(
        &self,
        update: RuntimePolicyUpdate,
    ) -> Result<RuntimePolicyConfig, RuntimePolicyError> {
        self.policy.lock().update_runtime_config(update)
    }
}
impl std::fmt::Debug for DefaultWorkerSelectorControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultWorkerSelectorControl")
            .finish_non_exhaustive()
    }
}
