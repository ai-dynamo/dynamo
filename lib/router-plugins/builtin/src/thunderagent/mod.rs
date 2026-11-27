// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Program-aware admission and placement from the native ThunderAgent prototype.
//!
//! Ported from `ishandhanani/router-sandbox` at `0184d2d39826f7f39c0ca33ee32c4141e86c2792`.
//! The classifier owns program state; the stateless selector executes its preferred placement.

mod config;
mod request_classifier;
mod worker_selection;

use config::{ConfigError, ThunderAgentConfig};

use dynamo_kv_router::plugins::*;

const THUNDERAGENT_CLASSIFIER_TYPE: &str = "thunderagent";

/// Register ThunderAgent's classifier and worker selector in one plugin catalog.
pub(super) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RouterPluginRegistryError> {
    worker_selection::register(registry)?;
    request_classifier::register(registry)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::plugins::request_classifier::RequestClassifierContext;
    use dynamo_kv_router::plugins::testing::{registered_classifier, registered_policy};

    use super::*;

    #[test]
    fn registers_classifier_and_worker_selector() {
        let mut registry = RouterPluginRegistry::default();
        register(&mut registry).unwrap();

        assert!(registered_policy(&registry, "thunderagent", "", WorkerType::Aggregated).is_ok());
        assert!(
            registered_classifier(
                &registry,
                "thunderagent",
                "",
                RequestClassifierContext::new(16, Vec::new),
            )
            .is_ok()
        );
    }
}
