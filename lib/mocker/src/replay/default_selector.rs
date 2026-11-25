// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous adapter for direct selection and deterministic replay.

use dynamo_kv_router::plugins::worker_selection::*;
use dynamo_router_policies::default_policy_with_rng;
use dynamo_scheduler::host::WorkerSelectionInput;
use dynamo_scheduler::host::WorkerSelector;
use dynamo_scheduler::host::{KvSchedulerError, WorkerConfigLike, WorkerSelectionResult};
use parking_lot::Mutex;
use std::sync::Arc;

/// Synchronous adapter for direct callers and replay. Scheduler actors use `default_policy`
/// directly and do not acquire this adapter's mutex.
///
/// The policy accepts runtime parameter updates through a [`DefaultWorkerSelectorControl`] from
/// [`Self::runtime_control`]. Updates apply between selections and keep the selector's random
/// stream.
pub struct DefaultWorkerSelector {
    kv_router_config: SelectionConfig,
    worker_type: &'static str,
    policy: Arc<Mutex<WorkerSelectionPolicy>>,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    /// Factory and worker role of a configured policy; `None` runs Dynamo's default.
    configured: Option<(WorkerSelectionPolicyFactory, WorkerType)>,
}
impl DefaultWorkerSelector {
    pub fn new(config: Option<SelectionConfig>, worker_type: &'static str) -> Self {
        Self::with_rng(config.unwrap_or_default(), worker_type, None)
    }
    /// Construct a reproducible selector. Clones share its random stream.
    pub fn new_seeded(
        config: Option<SelectionConfig>,
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
        config: SelectionConfig,
        worker_type: &'static str,
        rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        Self {
            policy: Arc::new(Mutex::new(default_policy_with_rng(
                config.clone(),
                worker_type,
                rng.clone(),
            ))),
            kv_router_config: config,
            worker_type,
            rng,
            configured: None,
        }
    }
    /// Run the policy `factory` builds for the `role` pool instead of Dynamo's default. The
    /// factory builds it as it does for a live router, so no replay seed reaches it.
    pub fn with_policy_factory(
        config: SelectionConfig,
        worker_type: &'static str,
        factory: WorkerSelectionPolicyFactory,
        role: WorkerType,
    ) -> Self {
        let partition = RoutingPartitionRef::new("replay", "default");
        Self {
            policy: Arc::new(Mutex::new(factory(WorkerSelectionPolicyContext::new(
                &config, role, partition,
            )))),
            kv_router_config: config,
            worker_type,
            rng: None,
            configured: Some((factory, role)),
        }
    }
    /// Handle that reads and updates this selector's runtime parameters from another owner.
    ///
    /// The worker inputs whose resources a router starts for `role`: this selector's, plus
    /// those of the role's configured hint policies. Rejects what the live router's setup
    /// rejects ([`dynamo_kv_router::plugins::host::check_role`]).
    pub fn check_role(
        &self,
        role: WorkerType,
        hint_policies: dynamo_kv_router::plugins::host::RoleHintPolicies,
        config: &dynamo_kv_router::config::KvRouterConfig,
    ) -> Result<WorkerInputs, dynamo_kv_router::plugins::host::PluginConfigError> {
        dynamo_kv_router::plugins::host::check_role(
            role,
            &self.policy.lock(),
            hint_policies,
            config,
        )
    }

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
        match &self.configured {
            Some((factory, role)) => Self::with_policy_factory(
                self.kv_router_config.clone(),
                self.worker_type,
                factory.clone(),
                *role,
            ),
            None => Self::with_rng(
                self.kv_router_config.clone(),
                self.worker_type,
                self.rng.clone(),
            ),
        }
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
        self.policy.lock().uses_exclusive_affinity_target()
    }
    fn applies_worker_filters(&self) -> bool {
        <WorkerSelectionPolicy as WorkerSelector<C>>::applies_worker_filters(&self.policy.lock())
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
