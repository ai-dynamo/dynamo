// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog for the custom worker-selection policy examples.

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    basic_agg_policy::register(registry)?;
    basic_disagg_policy::register(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_both_policies() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();

        assert!(matches!(
            basic_agg_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "least-busy"
        ));
        assert!(matches!(
            basic_disagg_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "disaggregated-load"
        ));
    }
}
