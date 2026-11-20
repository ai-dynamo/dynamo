// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a routing host names to resolve and run plugins, beyond the plugin API in
//! [`super::v1alpha1`]: the resolved plugin set, the policy-file entries the registry resolves
//! from, the hosted picker, and runtime parameter updates.
//!
//! Plugin crates do not import this module. Kernel and stage types a host also needs (the
//! scheduling request, worker configuration, candidate rows) come from each owning crate's
//! `host` module, such as `dynamo_scheduler::host`.

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
    HostedSelectionRequest, HostedWorkerPicker, RuntimePolicyConfig, RuntimePolicyUpdate,
    SelectionFailure, SelectionTraceSpec,
};

// Running resolved hint policies.
pub use crate::plugins::hint_policy::{HintPolicyFactory, RoleHintPolicies};
