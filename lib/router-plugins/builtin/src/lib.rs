// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router plugins Dynamo ships.
//!
//! Routing hosts always link the default through `default_registry`. The optional custom catalog
//! adds the named default, `kvcr` with its `kv_fetch` hint policy, two-tier, and ThunderAgent
//! providers through `register`. The default itself uses the same public candidate inputs and scorer/picker
//! dispatch as external policies. Sequence tracking, eligibility, and admission remain in
//! dynamo-kv-router.

mod default;
pub mod kv_fetch;
pub mod router_modes;
mod sticky_min_load;
mod thunderagent;
mod two_tier_cost_fn;
pub use default::{
    KVCR_POLICY_TYPE, default_factory, default_policy, default_policy_with_rng, kvcr_policy,
    startup_policy_parameters,
};

/// Registry every host links: the required default plus the routing modes
/// ([`router_modes`]), without an optional policy catalog.
pub fn default_registry() -> RouterPluginRegistry {
    with_defaults(RouterPluginRegistry::default())
}

/// `registry` plus what [`default_registry`] holds: the required default and each routing mode it
/// doesn't already provide. A host applies this to the catalog it installs, so `--router-mode`
/// resolves however that catalog was built.
pub fn with_defaults(registry: RouterPluginRegistry) -> RouterPluginRegistry {
    let mut registry = registry.with_default_factory(default_factory());
    router_modes::register_missing(&mut registry);
    registry
}

use dynamo_kv_router::plugins::prelude::*;

/// Register the named providers Dynamo ships, without changing the host's default factory.
///
/// `default` is reserved by the registry for Dynamo's built-in worker selector, so no policy here
/// can shadow it. A later catalog that reuses one of these type names fails registration rather
/// than overriding it.
pub fn register(registry: &mut RouterPluginRegistry) -> Result<(), RouterPluginRegistryError> {
    default::register(registry)?;
    sticky_min_load::register(registry)?;
    two_tier_cost_fn::register(registry)?;
    thunderagent::register(registry)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::plugins::prelude::*;

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
            factory(&SelectionConfig::from(&config), worker_type, partition);
        }
    }

    #[test]
    fn resolves_sticky_min_load_yaml() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated:
    type: dynamo-sticky-min-load
    parameters:
      max_idle_secs: 600
  prefill:
    type: dynamo-sticky-min-load
    parameters:
      max_idle_secs: 600
  decode:
    type: dynamo-sticky-min-load
    parameters:
      max_idle_secs: 600
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
            factory(&SelectionConfig::from(&config), worker_type, partition);
        }
    }

    /// `kvcr` resolves by type name and takes the default's parameters.
    #[test]
    fn resolves_kvcr_by_type_name() {
        let (config, resolved) = resolve(
            r#"
worker_selection:
  prefill:
    type: kvcr
    parameters:
      shared_cache_multiplier: 0.5
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");
        factory(
            &SelectionConfig::from(&config),
            WorkerType::Prefill,
            RoutingPartitionRef::new("model", "default"),
        );
    }

    /// `kvcr` sends `kv_fetch` unless its parameters say otherwise; the default cost sends no
    /// hints unless its parameters enable one. Both read the hint's parameters from
    /// `hint_parameters`, and reject names and values no default policy knows.
    #[test]
    fn enable_hints_and_hint_parameters_configure_kv_fetch() {
        let partition = RoutingPartitionRef::new("model", "default");
        let policy = |yaml: &str, role: WorkerType| {
            let (config, resolved) = resolve(yaml);
            resolved
                .unwrap()
                .expect("a configured instance resolves to a factory")(
                &SelectionConfig::from(&config),
                role,
                partition,
            )
        };
        let sends_fetch = |policy: WorkerSelectionPolicy| {
            policy.hint_emitter().is_some() && policy.hint_inputs().contains(WorkerInputs::KV_COPY)
        };
        let configured = r#"
worker_selection:
  prefill:
    type: kvcr
    parameters:
      shared_cache_multiplier: 0.5
      enable_hints: [kv_fetch]
      hint_parameters:
        kv_fetch: {min_additional_blocks: 2}
  decode: kvcr
  aggregated: dynamo-default-cost-fn
  encode:
    type: dynamo-default-cost-fn
    parameters:
      enable_hints: [kv_fetch]
"#;
        assert!(sends_fetch(policy(configured, WorkerType::Prefill)));
        assert!(sends_fetch(policy(configured, WorkerType::Decode)));
        assert!(!sends_fetch(policy(configured, WorkerType::Aggregated)));
        assert!(sends_fetch(policy(configured, WorkerType::Encode)));
        assert!(!sends_fetch(policy(
            "worker_selection:\n  prefill:\n    type: kvcr\n    parameters: {enable_hints: []}\n",
            WorkerType::Prefill,
        )));

        for parameters in [
            // The fetch's own parameters are validated.
            "{enable_hints: [kv_fetch], hint_parameters: {kv_fetch: {min_additional_blocks: 0}}}",
            "{enable_hints: [kv_fetch], hint_parameters: {kv_fetch: {min_blocks: 2}}}",
            // Unknown, repeated, and unenabled hints are mistakes.
            "{enable_hints: [kv_prefetch]}",
            "{enable_hints: [kv_fetch, kv_fetch]}",
            "{enable_hints: [], hint_parameters: {kv_fetch: {}}}",
            "{hint_parameters: {kv_fetch: {}}}",
            // Copy credit needs the fetch that sends the copies.
            "{shared_cache_multiplier: 0.5, enable_hints: []}",
        ] {
            let yaml = format!(
                "worker_selection:\n  prefill:\n    type: kvcr\n    parameters: {parameters}\n"
            );
            let (_config, resolved) = resolve(&yaml);
            assert!(
                matches!(
                    resolved,
                    Err(WorkerSelectionPolicyRegistryError::Provider { .. })
                ),
                "{parameters}: {:?}",
                resolved.as_ref().err().map(ToString::to_string)
            );
        }
        // The old per-role list is gone, and the error says where hints went.
        let (_config, resolved) = resolve(
            "worker_selection:\n  prefill:\n    type: kvcr\n    hints:\n      - type: kv_fetch\n",
        );
        let Err(error) = resolved else {
            panic!("the removed hints key must fail resolution");
        };
        assert!(error.to_string().contains("enable_hints"), "{error}");
    }

    /// An unknown parameter key is a mistake, most often a misremembered threshold name. It must
    /// fail startup rather than silently leaving the default in place.
    #[test]
    fn rejects_an_unknown_parameter_key() {
        let (_config, resolved) = resolve(
            r#"
worker_selection:
  aggregated:
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

    /// The routing modes take no parameters, so a `parameters` block under one is a mistake.
    #[test]
    fn routing_modes_reject_parameters() {
        for policy_type in [
            router_modes::RANDOM,
            router_modes::ROUND_ROBIN,
            router_modes::LEAST_LOADED,
        ] {
            let (_config, resolved) = resolve(&format!(
                r#"
worker_selection:
  aggregated:
    type: {policy_type}
    parameters:
      temperature: 0.5
"#
            ));
            let Err(error) = resolved else {
                panic!("{policy_type} must reject parameters");
            };
            assert!(
                matches!(&error, WorkerSelectionPolicyRegistryError::Provider { policy_type: rejected, .. }
                    if rejected == policy_type),
                "unexpected error: {error}"
            );
        }
    }
}
