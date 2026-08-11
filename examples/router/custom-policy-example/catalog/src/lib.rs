// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog for the custom worker-selection policy examples.

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    dynamo_custom_policy_example_basic::register(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_least_busy() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();

        assert!(matches!(
            register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "least-busy"
        ));
    }
}
