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

use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPluginRegistryError};

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
    #[test]
    fn registers_classifier_and_worker_selector() {
        let mut registry = crate::default_registry();
        crate::register(&mut registry).unwrap();
        let policy = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy.path(),
            r#"
request_classifier:
  type: thunderagent
worker_selection:
  aggregated: thunderagent
  instances:
    - name: thunderagent
      type: thunderagent
"#,
        )
        .unwrap();
        let config = dynamo_kv_router::KvRouterConfig {
            router_policy_config: Some(policy.path().display().to_string()),
            ..Default::default()
        };
        let plugins = registry.resolve_plugins(&config).unwrap();
        assert!(plugins.worker_selection().is_some());
        let factory = plugins.request_classifier().unwrap();
        let context = dynamo_kv_router::plugins::request_classifier::RequestClassifierContext::new(
            16,
            Vec::new,
        );
        let _classifier = factory(context);
    }
}
