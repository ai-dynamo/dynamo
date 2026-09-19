// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::WorkerSelectionResult;
use dynamo_kv_router::{DefaultWorkerSelector, WorkerSelectionInput, WorkerSelector};
#[cfg(feature = "replay-builtin")]
use dynamo_kv_router::{RoutingPartitionRef, WorkerSelectionPolicy};

use super::recent_cache;
use crate::common::protocols::WorkerType;
use crate::replay::router_shared::{ReplayWorkerConfig, replay_selector_with_seed};

pub(super) enum OfflineWorkerSelector {
    Default(DefaultWorkerSelector),
    #[cfg(feature = "replay-builtin")]
    TwoTier(WorkerSelectionPolicy),
}

impl OfflineWorkerSelector {
    pub(super) fn new(
        config: &KvRouterConfig,
        worker_type: WorkerType,
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        #[cfg(feature = "replay-builtin")]
        if let Some(instance_name) = config.selected_worker_selection_policy_instance_for(
            dynamo_kv_router::WorkerType::Aggregated,
        )? {
            anyhow::ensure!(
                worker_type == WorkerType::Aggregated,
                "offline builtin worker-selection replay requires aggregated workers"
            );
            let mut registry =
                dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry::default();
            dynamo_custom_policy_builtin::register(&mut registry)?;
            let factory = registry
                .resolve_for_worker_type(config, dynamo_kv_router::WorkerType::Aggregated)?
                .ok_or_else(|| anyhow::anyhow!("selected builtin replay policy did not resolve"))?;
            let instance = config
                .worker_selection_config()?
                .and_then(|catalog| catalog.instance(&instance_name))
                .ok_or_else(|| anyhow::anyhow!("selected builtin replay instance is missing"))?;
            anyhow::ensure!(
                instance.policy_type() == "dynamo-two-tier-cost-fn",
                "offline replay supports only the dynamo-two-tier-cost-fn builtin policy"
            );
            return Ok(Self::TwoTier(factory(
                config,
                dynamo_kv_router::WorkerType::Aggregated,
                RoutingPartitionRef::new("replay", "default"),
            )));
        }
        #[cfg(not(feature = "replay-builtin"))]
        let _ = worker_type;
        Ok(Self::Default(replay_selector_with_seed(
            config,
            selector_seed,
        )?))
    }

    pub(super) fn uses_device_tier(&self) -> bool {
        match self {
            Self::Default(_) => false,
            #[cfg(feature = "replay-builtin")]
            Self::TwoTier(_) => true,
        }
    }

    pub(super) fn validate_recent_cache(&self, config: &recent_cache::Config) -> Result<()> {
        anyhow::ensure!(
            !self.uses_device_tier()
                || (!config.predict && config.mode == recent_cache::Mode::Observe),
            "two-tier replay requires recent-cache mode=observe and predict=false"
        );
        Ok(())
    }

    pub(super) fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, ReplayWorkerConfig>,
    ) -> Result<WorkerSelectionResult, dynamo_kv_router::KvSchedulerError> {
        match self {
            Self::Default(selector) => selector.select_worker(input),
            #[cfg(feature = "replay-builtin")]
            Self::TwoTier(selector) => selector.select_worker(input),
        }
    }
}
