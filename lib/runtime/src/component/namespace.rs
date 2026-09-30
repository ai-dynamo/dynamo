// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::component::Namespace;
use crate::metrics::{MetricsHierarchy, MetricsRegistry};

impl MetricsHierarchy for Namespace {
    fn basename(&self) -> String {
        self.name.clone()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        let mut parents = vec![];

        // Walk up the namespace parent chain (grandparents to immediate parent)
        let parent_chain: Vec<&Namespace> =
            std::iter::successors(self.parent.as_deref(), |ns| ns.parent.as_deref()).collect();

        // Add DRT first (root)
        parents.push(&*self.runtime as &dyn MetricsHierarchy);

        // Then add parent namespaces in reverse order (root -> leaf)
        for parent_ns in parent_chain.iter().rev() {
            parents.push(*parent_ns as &dyn MetricsHierarchy);
        }

        parents
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        &self.metrics_registry
    }

    fn connection_id(&self) -> Option<u64> {
        use crate::traits::DistributedRuntimeProvider;
        Some(self.drt().connection_id())
    }
}
