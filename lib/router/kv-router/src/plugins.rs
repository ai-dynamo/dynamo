// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public contracts, configuration, and registration for statically linked router plugins.
//!
//! Plugin authors implement a worker-selection policy or a request classifier and register
//! their providers with [`RouterPluginRegistry`]. Scheduling owns execution, eligibility,
//! lifecycle synchronization, and capacity accounting.
//!
//! - [`v1alpha1`]: the plugin API, in one import. Alpha: it can change in a minor release, and
//!   each change is listed in the release's deprecation notes.
//! - [`experimental`]: APIs with no compatibility guarantee, one submodule each, imported by
//!   name.
//! - [`testing`]: a harness that runs a policy over a described request and workers, behind the
//!   `plugin-testing` feature.
//! - [`host`]: what a routing host names to resolve and run plugins. Plugin crates do not
//!   import it.
//!
//! [`worker_selection`], [`request_classifier`], and [`hint_policy`] are the per-contract
//! modules behind these: each re-exports its whole stage, host items included.

pub mod experimental;
pub mod hint_policy;
pub mod host;
mod registry;
pub mod request_classifier;
#[cfg(any(test, feature = "plugin-testing"))]
pub mod testing;
pub mod v1alpha1;
pub mod worker_selection;

pub use registry::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};

pub use dynamo_scheduler::PluginConfigError;

/// Check, for a routing host building a router, that one role's plugins fit together, and return
/// the worker inputs whose resources the router starts for that role: the worker-selection
/// policy's, plus those of the role's configured hint policies.
///
/// Rejects the same configurations as the router's own setup, with a message that names the fix.
pub fn check_role(
    role: dynamo_router_protocol::WorkerType,
    selection: &worker_selection::WorkerSelectionPolicy,
    hint_policies: hint_policy::RoleHintPolicies,
    config: &crate::config::KvRouterConfig,
) -> Result<worker_selection::WorkerInputs, PluginConfigError> {
    dynamo_scheduler::check_role(
        role,
        selection,
        hint_policies,
        config.supports_kv_transfer_chain_retention(),
    )
}

use hint_policy::HintPolicyRegistryError;
use request_classifier::{RequestClassifierFactory, RequestClassifierRegistryError};
use worker_selection::WorkerSelectionPolicyFactory;

/// Configured factories shared across router construction, with fresh instances per router.
#[derive(Clone, Default)]
pub struct RouterPlugins {
    worker_selection: Option<WorkerSelectionPolicyFactory>,
    custom_worker_selection: bool,
    request_classifier: Option<RequestClassifierFactory>,
}

impl RouterPlugins {
    pub fn with_worker_selection(mut self, factory: WorkerSelectionPolicyFactory) -> Self {
        self.worker_selection = Some(factory);
        self.custom_worker_selection = true;
        self
    }

    pub fn with_request_classifier(mut self, factory: RequestClassifierFactory) -> Self {
        self.request_classifier = Some(factory);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.worker_selection.is_none() && self.request_classifier.is_none()
    }

    /// Whether worker selection was explicitly supplied, rather than filled by the host default.
    /// Hosts use this to require typed worker discovery without delaying stock router startup.
    pub fn has_custom_worker_selection(&self) -> bool {
        self.custom_worker_selection
    }

    /// Whether explicit policy or classifier choices require custom-plugin frontend support.
    pub fn has_custom_plugins(&self) -> bool {
        self.custom_worker_selection || self.request_classifier.is_some()
    }

    pub fn worker_selection(&self) -> Option<&WorkerSelectionPolicyFactory> {
        self.worker_selection.as_ref()
    }

    pub fn request_classifier(&self) -> Option<&RequestClassifierFactory> {
        self.request_classifier.as_ref()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RouterPluginRegistryError {
    #[error(transparent)]
    WorkerSelection(#[from] WorkerSelectionPolicyRegistryError),
    #[error(transparent)]
    RequestClassifier(#[from] RequestClassifierRegistryError),
    #[error(transparent)]
    HintPolicy(#[from] HintPolicyRegistryError),
}
