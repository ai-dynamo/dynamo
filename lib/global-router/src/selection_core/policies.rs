// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router's policies as router core plugins. The core makes one
//! picker per model.

use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistryError;
use dynamo_router_policies::router_modes;

use super::SelectionCorePolicy;

impl SelectionCorePolicy {
    fn name(&self) -> &'static str {
        match self {
            Self::RoundRobin => router_modes::ROUND_ROBIN,
        }
    }
}

/// The router's builtin plugin for `policy`, resolved by name as a routing
/// mode is.
pub(super) fn factory(
    policy: &SelectionCorePolicy,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyRegistryError> {
    dynamo_router_policies::default_registry().resolve_type(policy.name())
}
