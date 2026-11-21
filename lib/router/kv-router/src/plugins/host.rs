// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a routing host names to resolve and run plugins, beyond the plugin API in
//! [`super::v1alpha1`]: the resolved plugin set, the policy-file entries the registry resolves
//! from, and the hosted picker.
//!
//! Plugin crates do not import this module. Kernel and stage types a host also needs (the
//! scheduling request, worker configuration, candidate rows) come from each owning crate's
//! `host` module, such as `dynamo_scheduler::host`.
//!
//! Resolution is an extension trait here, [`RouterPluginRegistryHostExt`], not methods on
//! [`RouterPluginRegistry`]: a plugin's `register` receives the registry to add providers to,
//! and cannot replace the host's default policy or resolve other plugins through it.

use dynamo_router_protocol::WorkerType;

use crate::config::KvRouterConfig;
use crate::plugins::hint_policy::HintPolicyRegistryError;
use crate::plugins::request_classifier::{
    RequestClassifierFactory, RequestClassifierRegistryError,
};
use crate::plugins::worker_selection::WorkerSelectionPolicyFactory;
use crate::plugins::{
    RouterPluginRegistry, RouterPluginRegistryError, WorkerSelectionPolicyRegistryError,
};

// Resolution: the plugins a configuration selects, and the check that one role's fit together.
pub use crate::plugins::{PluginConfigError, RouterPlugins, check_role};

// The policy-file entries the registry resolves plugins from.
pub use crate::plugins::hint_policy::HintPolicyEntry;
pub use crate::plugins::request_classifier::RequestClassifierConfig;
pub use crate::plugins::worker_selection::{
    WorkerSelectionConfig, WorkerSelectionInstance, WorkerSelectionPolicySelection,
};
pub use crate::plugins::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
};

// Running a resolved worker-selection policy.
pub use crate::plugins::worker_selection::{
    HostedSelectionRequest, HostedWorkerPicker, SelectionFailure, SelectionTraceSpec,
};
pub use dynamo_worker_selection::host::WorkerSelectionPolicyHostExt;

// Running resolved hint policies.
pub use crate::plugins::hint_policy::{HintPolicyFactory, RoleHintPolicies};

/// What a routing host does with a [`RouterPluginRegistry`]: supply its default policy and
/// resolve the plugins a configuration selects.
pub trait RouterPluginRegistryHostExt: Sized {
    /// Supply the host default while preserving registered custom plugin types.
    fn with_default_factory(self, factory: WorkerSelectionPolicyFactory) -> Self;

    /// Whether this image has no linked router plugin types.
    fn is_empty(&self) -> bool;

    /// Resolve every configured plugin once before constructing any routers.
    fn resolve_plugins(
        &self,
        config: &KvRouterConfig,
    ) -> Result<RouterPlugins, RouterPluginRegistryError>;

    /// Resolve the configured policy instances once at process startup.
    ///
    /// Aggregated, prefill, decode, and encode pools can select separate instances. The returned
    /// factory dispatches by the typed role that the host supplies for each worker pool.
    fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError>;

    /// Resolve only the policy selected for one worker role.
    ///
    /// Single-pool hosts use this method to avoid loading policy types for worker pools that they
    /// do not construct.
    fn resolve_for_worker_type(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError>;

    /// Build the factory for one registered policy type with default parameters, as a routing
    /// mode names it (for example `round-robin`) without a policy file.
    fn resolve_type(
        &self,
        policy_type: &str,
    ) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyRegistryError>;

    /// Resolve the configured request classifier once at frontend startup.
    fn resolve_request_classifier(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<RequestClassifierFactory>, RequestClassifierRegistryError>;

    /// Resolve each role's hints once at startup, from its `worker_selection` entry. A role whose
    /// entry lists no `hints` runs the hints registered as on by default for its policy type.
    fn resolve_hint_policies(
        &self,
        config: &KvRouterConfig,
    ) -> Result<HintPolicyFactory, HintPolicyRegistryError>;
}

impl RouterPluginRegistryHostExt for RouterPluginRegistry {
    fn with_default_factory(self, factory: WorkerSelectionPolicyFactory) -> Self {
        RouterPluginRegistry::with_default_factory(self, factory)
    }

    fn is_empty(&self) -> bool {
        RouterPluginRegistry::is_empty(self)
    }

    fn resolve_plugins(
        &self,
        config: &KvRouterConfig,
    ) -> Result<RouterPlugins, RouterPluginRegistryError> {
        RouterPluginRegistry::resolve_plugins(self, config)
    }

    fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        RouterPluginRegistry::resolve(self, config)
    }

    fn resolve_for_worker_type(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        RouterPluginRegistry::resolve_for_worker_type(self, config, worker_type)
    }

    fn resolve_type(
        &self,
        policy_type: &str,
    ) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyRegistryError> {
        RouterPluginRegistry::resolve_type(self, policy_type)
    }

    fn resolve_request_classifier(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<RequestClassifierFactory>, RequestClassifierRegistryError> {
        RouterPluginRegistry::resolve_request_classifier(self, config)
    }

    fn resolve_hint_policies(
        &self,
        config: &KvRouterConfig,
    ) -> Result<HintPolicyFactory, HintPolicyRegistryError> {
        RouterPluginRegistry::resolve_hint_policies(self, config)
    }
}
