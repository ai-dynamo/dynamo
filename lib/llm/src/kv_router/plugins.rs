// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host construction for the resolved router plugin bundle.

use std::sync::OnceLock;

use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPlugins};

use super::{KvRouter, SelectionPolicySource};

static INSTALLED_PLUGINS: OnceLock<RouterPluginRegistry> = OnceLock::new();

/// Install the linked catalog once for frontend and embedded router construction.
/// Returns `false` if a catalog has already been installed.
pub fn install_router_plugin_registry(registry: RouterPluginRegistry) -> bool {
    INSTALLED_PLUGINS.set(registry).is_ok()
}

/// The installed catalog, or an empty registry for Dynamo's default policies.
pub fn router_plugin_registry() -> RouterPluginRegistry {
    INSTALLED_PLUGINS.get().cloned().unwrap_or_default()
}

/// Carries configured plugins through the host's shared router construction.
#[derive(Clone, Default)]
pub struct RouterPluginBuilder {
    plugins: RouterPlugins,
}

impl RouterPluginBuilder {
    pub fn new(plugins: RouterPlugins) -> Self {
        Self { plugins }
    }

    pub(crate) fn validate_config(
        &self,
        config: &dynamo_kv_router::KvRouterConfig,
    ) -> anyhow::Result<()> {
        if config.request_classifier_config()?.is_some()
            && self.plugins.request_classifier().is_none()
        {
            anyhow::bail!(
                "request_classifier is configured but not installed; supply a resolved RouterPlugins bundle"
            );
        }
        Ok(())
    }

    pub(crate) fn selection_policy(&self) -> SelectionPolicySource {
        self.plugins.worker_selection().cloned().map_or(
            SelectionPolicySource::Registry,
            SelectionPolicySource::Factory,
        )
    }

    pub(crate) fn install(&self, router: &KvRouter) -> anyhow::Result<()> {
        if let Some(factory) = self.plugins.request_classifier() {
            router.install_request_classifier(factory(router.request_classifier_context()))?;
        }
        Ok(())
    }
}
