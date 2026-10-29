// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ThunderAgent's program-aware admission and placement, extended for RL rollouts.
//!
//! A group (`x-dynamo-group`) of rollouts is stamped with the trainer weight version when its first
//! program starts. New groups start only whole, under a limit that is the smaller of a KV estimate
//! and a target-age limit fed back from finished groups. Groups waiting to start hold no version.

mod config;
mod group_priority;
mod request_classifier;
mod worker_selection;

use config::{ConfigError, ContactRouterConfig};

use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPluginRegistryError};

const CONTACT_ROUTER_CLASSIFIER_TYPE: &str = "contact_router";

/// Register ContactRouter's classifier and worker selector in one plugin catalog.
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
  type: contact_router
worker_selection:
  aggregated: contact_router
  instances:
    - name: contact_router
      type: contact_router
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
