// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host construction for the resolved router plugin bundle.

use std::sync::{Arc, OnceLock};

use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::host::RouterPluginRegistryHostExt;
use dynamo_kv_router::plugins::host::RouterPlugins;
use dynamo_kv_router::plugins::worker_selection::{WorkerInputs, WorkerSelectionPolicyFactory};
use dynamo_runtime::pipeline::{InstanceSelector, ModeSelectors, PushRouterOptions};
use dynamo_worker_selection::host::{HostedSelectionRequest, HostedWorkerPicker};

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
        Some(Arc::new(HostedInstanceSelector(picker)) as Arc<dyn InstanceSelector>)
    })
}

/// Lets `PushRouter` run a hosted picker. Its callers carry no LLM request, so the selection
/// sees a zero-token request. The runtime names no router crate, so the frontend host, which
/// depends on both, carries the adapter.
struct HostedInstanceSelector(HostedWorkerPicker);

impl InstanceSelector for HostedInstanceSelector {
    fn reads_active_requests(&self) -> bool {
        self.0.inputs().contains(WorkerInputs::ACTIVE_REQUESTS)
    }

    fn select_instance(
        &self,
        instance_ids: &[u64],
        active_requests: &dyn Fn(u64) -> u64,
        preview: bool,
    ) -> Option<u64> {
        let request = HostedSelectionRequest::new(0).with_preview(preview);
        let active_requests = self.reads_active_requests().then_some(active_requests);
        self.0.select(instance_ids, active_requests, &request).ok()
    }
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
        config: &dynamo_kv_router::config::KvRouterConfig,
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

    /// A policy selected in the frontend's own config applies to every worker set. The host
    /// default does not: each set resolves its policy from its own routing config, so a set that
    /// advertises a policy file is scored by it, like every other advertised setting.
    pub(crate) fn selection_policy(&self) -> SelectionPolicySource {
        match self.plugins.worker_selection() {
            Some(factory) if self.plugins.has_custom_worker_selection() => {
                SelectionPolicySource::Factory(factory.clone())
            }
            _ => SelectionPolicySource::Registry,
        }
    }

    pub(crate) fn install(&self, router: &KvRouter) -> anyhow::Result<()> {
        if let Some(factory) = self.plugins.request_classifier() {
            router.install_request_classifier(factory(router.request_classifier_context()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::config::KvRouterConfig;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::worker_type::WorkerType;

    /// A config whose policy file selects `least-loaded` for decode. That type declares
    /// `ACTIVE_REQUESTS` and the default policy does not, so the input set shows which one ran.
    fn least_loaded_decode() -> (NamedTempFile, KvRouterConfig) {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "worker_selection:\n  decode: least-loaded\n").unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        (file, config)
    }

    fn decode_reads_active_requests(source: SelectionPolicySource, set: &KvRouterConfig) -> bool {
        source
            .prepare(set, WorkerType::Decode, "decode", None)
            .unwrap()
            .inputs()
            .contains(WorkerInputs::ACTIVE_REQUESTS)
    }

    fn frontend_policy(frontend: &KvRouterConfig) -> SelectionPolicySource {
        RouterPluginBuilder::new(router_plugin_registry().resolve_plugins(frontend).unwrap())
            .selection_policy()
    }

    #[test]
    fn frontend_without_a_policy_uses_the_worker_sets_own() {
        let frontend = frontend_policy(&KvRouterConfig::default());
        let (_file, advertised) = least_loaded_decode();

        assert!(decode_reads_active_requests(frontend.clone(), &advertised));
        assert!(!decode_reads_active_requests(
            frontend,
            &KvRouterConfig::default()
        ));
    }

    #[test]
    fn frontend_selected_policy_applies_to_every_worker_set() {
        let (_file, frontend_config) = least_loaded_decode();
        let frontend = frontend_policy(&frontend_config);

        assert!(matches!(frontend, SelectionPolicySource::Factory(_)));
        assert!(decode_reads_active_requests(
            frontend,
            &KvRouterConfig::default()
        ));
    }
}
