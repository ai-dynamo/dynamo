// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's policies as router core plugins. The core makes one
//! picker per model.

use dynamo_kv_router::services::selection::RouterPluginRegistry;
use dynamo_router_policies::router_modes;

use super::SelectionCorePolicy;

impl SelectionCorePolicy {
    fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => router_modes::ROUND_ROBIN,
        }
    }
}

/// The router's builtin plugins, which include its routing modes.
pub(super) fn registry() -> RouterPluginRegistry {
    dynamo_router_policies::default_registry()
}

/// The policy YAML selecting `policy` for aggregated workers.
pub(super) fn policy_config(policy: &SelectionCorePolicy) -> String {
    let name = policy.name();
    format!(
        "worker_selection:\n  aggregated: {name}\n  instances:\n    - name: {name}\n      type: {name}\n"
    )
}
