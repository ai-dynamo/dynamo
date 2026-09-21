// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection policies Dynamo ships.
//!
//! Routing hosts always link the default through `default_registry`. The optional custom
//! catalog adds the named default and two-tier providers through `register`. The default
//! itself uses the same public candidate inputs and scorer/picker dispatch as external policies.
//! Sequence tracking, eligibility, and admission remain in dynamo-kv-router.

mod default;
mod two_tier_cost_fn;
pub use default::{DefaultWorkerSelector, default_factory, default_policy};

/// Registry containing the required default only, without an optional policy catalog.
pub fn default_registry() -> RouterPluginRegistry {
    RouterPluginRegistry::default().with_default_factory(default_factory())
}

use dynamo_kv_router::plugins::{RouterPluginRegistry, WorkerSelectionPolicyRegistryError};

/// Register the named providers Dynamo ships, without changing the host's default factory.
///
/// `default` is reserved by the registry for Dynamo's built-in worker selector, so no policy here
/// can shadow it. A later catalog that reuses one of these type names fails registration rather
/// than overriding it.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    default::register(registry)?;
    two_tier_cost_fn::register(registry)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
    use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};

    use super::*;

    /// Resolve router-policy YAML exactly as the Python bindings do at startup, so these cover the
    /// real configuration path rather than the registrars in isolation.
    fn resolve(
        yaml: &str,
    ) -> (
        KvRouterConfig,
        Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError>,
    ) {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy_file.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        let mut registry = default_registry();
        register(&mut registry).unwrap();
        let resolved = registry.resolve(&config);
        (config, resolved)
    }

    /// Catches a policy type name that drifts from its documentation, and proves the documented
    /// instance shape constructs for every stage it selects.
    #[test]
    fn resolves_documented_yaml() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  prefill: dynamo-two-tier-cost-fn
  decode: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");

        let partition = RoutingPartitionRef::new("model", "default");
        for worker_type in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            factory(&config, worker_type, partition);
        }
    }

    /// An unknown parameter key is a mistake, most often a misremembered threshold name. It must
    /// fail startup rather than silently leaving the default in place.
    #[test]
    fn rejects_an_unknown_parameter_key() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_affinity_threshold: 0.8
"#,
        );

        let Err(error) = resolved else {
            panic!("an unknown parameter must fail resolution");
        };
        assert!(
            matches!(&error, WorkerSelectionPolicyRegistryError::Provider { policy_type, .. }
                if policy_type == two_tier_cost_fn::POLICY_TYPE),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("cache_affinity_threshold"),
            "the error should name the offending key: {error}"
        );
    }
}
