// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a routing host names to resolve and run plugins, beyond the plugin API in
//! [`super::worker_selection`] and [`super::request_classifier`]: the resolved plugin set, the
//! policy-file entries the registry resolves from, and the hosted picker.
//!
//! Plugin crates do not import this module. Kernel and stage types a host also needs (the
//! scheduling request, worker configuration, candidate rows) come from each owning crate's
//! `host` module, such as `dynamo_scheduler::host`.
//!
//! Resolution is an extension trait here, [`RouterPluginRegistryHostExt`], not methods on
//! [`RouterPluginRegistry`]: a plugin's `register` receives the registry to add providers to,
//! and cannot replace the host's default policy or resolve other plugins through it.
//!
//! [`super::testing`] also reaches host operations: it resolves a registered type and builds
//! the request and the progress reading a classifier is handed, so that a plugin's tests need
//! not name this module or a stage crate's `host` module. A plugin crate imports it only from
//! its tests and benches. Like the rule for this module, that is a rule for plugin crates and
//! a review item, not a compile-time check: both modules are public in every build.

use dynamo_router_protocol::WorkerType;
use dynamo_worker_selection::{WorkerInputs, WorkerSelectionPolicy};

use crate::config::KvRouterConfig;
use crate::plugins::hint_policy::HintPolicyRegistryError;
use crate::plugins::request_classifier::{
    RequestClassifierFactory, RequestClassifierRegistryError,
};
use crate::plugins::worker_selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyRegistryError,
};
use crate::plugins::{RouterPluginRegistry, RouterPluginRegistryError};

// The check that one role's plugins fit together reports this.
pub use dynamo_scheduler::PluginConfigError;

// The policy-file entries the registry resolves plugins from, and the environment variables
// that select a role's policy.
pub use crate::plugins::hint_policy::HintPolicyEntry;
pub use crate::plugins::registry::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
};
pub use crate::plugins::request_classifier::config::RequestClassifierConfig;
pub use crate::plugins::worker_selection::config::{
    WorkerSelectionConfig, WorkerSelectionInstance, WorkerSelectionPolicySelection,
};

// Running a resolved worker-selection policy.
pub use dynamo_worker_selection::host::WorkerSelectionPolicyHostExt;
pub use dynamo_worker_selection::{
    HostedSelectionRequest, HostedWorkerPicker, SelectionFailure, SelectionTraceSpec,
};

// Running resolved hint policies.
pub use crate::plugins::hint_policy::{HintPolicyFactory, RoleHintPolicies};

// The plugins a configuration selects.
pub use crate::plugins::registry::RouterPlugins;

/// Check, for a routing host building a router, that one role's plugins fit together, and return
/// the worker inputs whose resources the router starts for that role: the worker-selection
/// policy's, plus those of the role's configured hint policies.
///
/// Rejects the same configurations as the router's own setup, with a message that names the fix.
pub fn check_role(
    role: WorkerType,
    selection: &WorkerSelectionPolicy,
    hint_policies: RoleHintPolicies,
    config: &KvRouterConfig,
) -> Result<WorkerInputs, PluginConfigError> {
    dynamo_scheduler::check_role(
        role,
        selection,
        hint_policies,
        config.supports_kv_transfer_chain_retention(),
    )
}

mod sealed {
    /// Keeps the host traits implementable only here, so a method can be added to one.
    pub trait Sealed {}
}

impl sealed::Sealed for RouterPluginRegistry {}

/// What a routing host does with a [`RouterPluginRegistry`]: supply its default policy and
/// resolve the plugins a configuration selects.
pub trait RouterPluginRegistryHostExt: sealed::Sealed + Sized {
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
