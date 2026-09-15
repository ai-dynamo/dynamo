// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection policies Dynamo ships.
//!
//! Any build that enables the Python bindings' `custom-policy` feature links these, and that
//! feature is on by default, so a frontend deployment selects one through router-policy YAML
//! without rebuilding. A build with default features off, and the standalone EPP, link no catalog
//! at all and reject a configured policy type at startup.
//!
//! These do not go through the replaceable `dynamo-worker-selection-policy-catalog` alias, and they
//! register before it, so a custom image's catalog adds its policies alongside these rather than
//! displacing them.
//!
//! Each policy is one module here. To ship another, add a module and a line in [`register`], then
//! add a row to the policy table in the router configuration guide. Keep a policy in a single file
//! until it needs submodules, then promote it to a directory. If a policy ever needs a dependency
//! beyond `dynamo-kv-router`, put it behind its own default-on Cargo feature so a build can drop
//! it; every policy registered here is compiled into every artifact that links this crate.

mod soft_affinity_load_guard;
mod two_tier_cost_fn;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

/// Register every policy Dynamo ships.
///
/// `default` is reserved by the registry for Dynamo's built-in worker selector, so no policy here
/// can shadow it. A later catalog that reuses one of these type names fails registration rather
/// than overriding it.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    two_tier_cost_fn::register(registry)?;
    soft_affinity_load_guard::register(registry)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::services::selection::WorkerSelectionPolicyFactory;
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

        let mut registry = WorkerSelectionPolicyRegistry::default();
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

    /// Same contract for the soft-affinity load guard: the documented instance shape must
    /// resolve through the default build path for every stage it selects.
    #[test]
    fn resolves_documented_soft_affinity_load_guard_yaml() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-soft-affinity-load-guard
  prefill: dynamo-soft-affinity-load-guard
  decode: dynamo-soft-affinity-load-guard
  instances:
    - name: dynamo-soft-affinity-load-guard
      type: dynamo-soft-affinity-load-guard
      parameters:
        max_active_requests: 32
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

    /// Every parameter is optional, so an instance with no `parameters` mapping must still start.
    #[test]
    fn resolves_soft_affinity_load_guard_without_parameters() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-soft-affinity-load-guard
  instances:
    - name: dynamo-soft-affinity-load-guard
      type: dynamo-soft-affinity-load-guard
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");

        factory(
            &config,
            WorkerType::Aggregated,
            RoutingPartitionRef::new("model", "default"),
        );
    }

    #[test]
    fn rejects_an_unknown_soft_affinity_load_guard_parameter() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: dynamo-soft-affinity-load-guard
  instances:
    - name: dynamo-soft-affinity-load-guard
      type: dynamo-soft-affinity-load-guard
      parameters:
        group_idle_ttl_secs: 300
"#,
        );

        let Err(error) = resolved else {
            panic!("an unknown parameter must fail resolution");
        };
        assert!(
            matches!(&error, WorkerSelectionPolicyRegistryError::Provider { policy_type, .. }
                if policy_type == soft_affinity_load_guard::POLICY_TYPE),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("group_idle_ttl_secs"),
            "the error should name the offending key: {error}"
        );
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
