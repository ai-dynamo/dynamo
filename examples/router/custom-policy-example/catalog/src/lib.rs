// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog for the custom worker-selection policy examples.

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    simple_filter_score_pick_policy::register(registry)?;
    disagg_filter_score_pick_policy::register(registry)?;
    simple_stacked_score_pick_policy::register(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_all_policies() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();

        assert!(matches!(
            simple_filter_score_pick_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "simple-filter-score-pick"
        ));
        assert!(matches!(
            disagg_filter_score_pick_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "disagg-filter-score-pick"
        ));
        assert!(matches!(
            simple_stacked_score_pick_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "simple-stacked-score-pick"
        ));
    }
}
