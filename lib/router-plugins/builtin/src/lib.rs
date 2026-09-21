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

mod non_kv;
mod two_tier_cost_fn;

use dynamo_kv_router::plugins::{RouterPluginRegistry, WorkerSelectionPolicyRegistryError};

/// Register every policy Dynamo ships.
///
/// `default` is reserved by the registry for Dynamo's built-in worker selector, so no policy here
/// can shadow it. A later catalog that reuses one of these type names fails registration rather
/// than overriding it.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    non_kv::register(registry)?;
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

        let mut registry = RouterPluginRegistry::default();
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
  aggregated: rr
  prefill: p2c
  decode: least
  encode: device
  instances:
    - name: rr
      type: dynamo-round-robin
    - name: random
      type: dynamo-random
    - name: p2c
      type: dynamo-power-of-two-choices
    - name: least
      type: dynamo-least-loaded
    - name: direct
      type: dynamo-direct
    - name: device
      type: dynamo-device-aware-weighted
    - name: two-tier
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
            WorkerType::Encode,
        ] {
            factory(&config, worker_type, partition);
        }
    }

    #[test]
    fn every_non_kv_type_resolves_for_every_stage() {
        for policy_type in [
            non_kv::ROUND_ROBIN,
            non_kv::RANDOM,
            non_kv::POWER_OF_TWO_CHOICES,
            non_kv::LEAST_LOADED,
            non_kv::DIRECT,
            non_kv::DEVICE_AWARE_WEIGHTED,
        ] {
            let yaml = format!(
                r#"
worker_selection:
  aggregated: selected
  prefill: selected
  decode: selected
  encode: selected
  instances:
    - name: selected
      type: {policy_type}
"#
            );
            let (config, resolved) = resolve(&yaml);
            let factory = resolved.unwrap().expect("configured factory");
            for worker_type in [
                WorkerType::Aggregated,
                WorkerType::Prefill,
                WorkerType::Decode,
                WorkerType::Encode,
            ] {
                factory(
                    &config,
                    worker_type,
                    RoutingPartitionRef::new("model", "default"),
                );
            }
        }
    }

    #[test]
    fn omitted_stage_still_materializes_the_default_policy() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  prefill: rr
  instances:
    - name: rr
      type: dynamo-round-robin
"#,
        );
        let factory = resolved.unwrap().expect("configured factory");
        factory(
            &config,
            WorkerType::Prefill,
            RoutingPartitionRef::new("model", "default"),
        );
        factory(
            &config,
            WorkerType::Decode,
            RoutingPartitionRef::new("model", "default"),
        );
    }

    #[test]
    fn non_kv_types_reject_parameters() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: rr
  instances:
    - name: rr
      type: dynamo-round-robin
      parameters:
        seed: 7
"#,
        );
        let Err(error) = resolved else {
            panic!("unknown parameters must fail resolution");
        };
        assert!(error.to_string().contains("seed"));
    }

    #[test]
    fn linked_type_diagnostic_lists_all_builtins() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: missing
  instances:
    - name: missing
      type: not-linked
"#,
        );
        let Err(error) = resolved else {
            panic!("unknown policy type must fail resolution");
        };
        let message = error.to_string();
        for policy_type in [
            non_kv::ROUND_ROBIN,
            non_kv::RANDOM,
            non_kv::POWER_OF_TWO_CHOICES,
            non_kv::LEAST_LOADED,
            non_kv::DIRECT,
            non_kv::DEVICE_AWARE_WEIGHTED,
            two_tier_cost_fn::POLICY_TYPE,
        ] {
            assert!(
                message.contains(policy_type),
                "missing {policy_type}: {message}"
            );
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
