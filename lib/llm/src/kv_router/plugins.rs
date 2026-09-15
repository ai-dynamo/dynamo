// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host construction for the resolved router plugin bundle.

use std::sync::Arc;

use dynamo_kv_router::{
    WorkerSelectionPolicy,
    plugins::RouterPlugins,
    selector::{DefaultWorkerSelector, WorkerSelector},
};

use super::{KvRouter, WorkerSelectorFactory};
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Binds a plugin bundle to the host's concrete selector type before router construction.
pub struct RouterPluginBuilder<Sel = DefaultWorkerSelector> {
    pub(crate) selector_factory: WorkerSelectorFactory<Sel>,
    plugins: RouterPlugins,
}

impl RouterPluginBuilder<WorkerSelectionPolicy> {
    pub fn new(plugins: RouterPlugins) -> Self {
        let selector_factory = plugins.worker_selection().cloned().unwrap_or_else(|| {
            Arc::new(|config, worker_type, _partition| {
                WorkerSelectionPolicy::default(config.clone(), worker_type.default_selector_label())
            })
        });
        Self {
            selector_factory,
            plugins,
        }
    }
}

impl Default for RouterPluginBuilder<DefaultWorkerSelector> {
    fn default() -> Self {
        Self {
            selector_factory: Arc::new(|config, worker_type, _partition| {
                DefaultWorkerSelector::new(
                    Some(config.clone()),
                    worker_type.default_selector_label(),
                )
            }),
            plugins: RouterPlugins::default(),
        }
    }
}

impl RouterPluginBuilder<DefaultWorkerSelector> {
    pub fn with_default_selector(plugins: RouterPlugins) -> anyhow::Result<Self> {
        anyhow::ensure!(
            plugins.worker_selection().is_none(),
            "custom worker selection requires the policy selector"
        );
        Ok(Self {
            plugins,
            ..Self::default()
        })
    }
}

impl<Sel> RouterPluginBuilder<Sel> {
    pub(crate) fn install(&self, router: &KvRouter<Sel>) -> anyhow::Result<()>
    where
        Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
    {
        if let Some(factory) = self.plugins.request_classifier() {
            router.install_request_classifier(factory())?;
        }
        Ok(())
    }
}
