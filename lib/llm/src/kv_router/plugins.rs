// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host construction for the resolved router plugin bundle.

use std::sync::{Arc, OnceLock};

use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPlugins};
use dynamo_runtime::pipeline::{InstanceSelector, ModeSelectors, PushRouterOptions};

use super::{KvRouter, SelectionPolicySource};

static INSTALLED_PLUGINS: OnceLock<RouterPluginRegistry> = OnceLock::new();

/// The worker-selection policy hosts run when no policy is configured, including for a
/// `default` selection. Every host install path reads it from here, so changing the default
/// policy is a change to this function alone.
fn host_default_factory() -> WorkerSelectionPolicyFactory {
    dynamo_router_policies::default_factory()
}

/// Install the linked catalog once for frontend and embedded router construction, adding
/// Dynamo's default and any routing mode it lacks. Returns `false` if a catalog has already
/// been installed.
pub fn install_router_plugin_registry(registry: RouterPluginRegistry) -> bool {
    INSTALLED_PLUGINS
        .set(
            dynamo_router_policies::with_defaults(registry)
                .with_default_factory(host_default_factory()),
        )
        .is_ok()
}

/// The installed catalog, or the required builtin default policy.
pub fn router_plugin_registry() -> RouterPluginRegistry {
    INSTALLED_PLUGINS.get().cloned().unwrap_or_else(|| {
        dynamo_router_policies::default_registry().with_default_factory(host_default_factory())
    })
}

/// The router-mode plugins as `PushRouter` instance selectors, resolved from the installed
/// catalog. Each call to the returned function builds a fresh plugin, so every router keeps
/// its own state, such as its round-robin rotation.
pub fn mode_selectors() -> ModeSelectors {
    Arc::new(|mode| {
        let picker = super::routing_host::hosted_picker(mode)
            .inspect_err(|error| {
                tracing::warn!(?mode, %error, "router-mode plugin unavailable");
            })
            .ok()?;
        Some(Arc::new(picker) as Arc<dyn InstanceSelector>)
    })
}

/// `PushRouter` options that pick with the router-mode plugins.
pub fn push_router_options<T, U>() -> PushRouterOptions<T, U> {
    PushRouterOptions {
        selectors: Some(mode_selectors()),
        ..PushRouterOptions::default()
    }
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

    pub(crate) fn has_custom_worker_selection(&self) -> bool {
        self.plugins.has_custom_worker_selection()
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
