// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router plugins Dynamo ships.
//!
//! Routing hosts always link the default through `default_registry`, which also installs the
//! `kv_fetch` hint policy as the default hint policy. The optional custom catalog adds the named
//! default, `kvcr`, two-tier, and ThunderAgent providers through `register`. The default itself
//! uses the same public candidate inputs and scorer/picker dispatch as external policies.
//! Sequence tracking, eligibility, and admission remain in dynamo-kv-router.

mod default;
pub mod kv_fetch;
pub mod router_modes;
mod sticky_min_load;
mod thunderagent;
mod two_tier_cost_fn;
pub use default::{
    DefaultWorkerSelector, DefaultWorkerSelectorControl, KVCR_POLICY_TYPE, default_factory,
    default_policy, kvcr_policy, startup_policy_parameters,
};
pub use kv_fetch::KvSourceLocationsPayload;

/// Registry every host links: the required default plus the routing modes
/// ([`router_modes`]), without an optional policy catalog.
pub fn default_registry() -> RouterPluginRegistry {
    with_defaults(RouterPluginRegistry::default())
}

/// `registry` plus what [`default_registry`] holds: the required default, each routing mode it
/// doesn't already provide, the `kv_fetch` hint-policy type, and `kv_fetch` as the default hint
/// policy of every role that `hint_policies` doesn't list. A host applies this to the catalog it
/// installs, so `--router-mode` and `kv_fetch` resolve however that catalog was built.
pub fn with_defaults(registry: RouterPluginRegistry) -> RouterPluginRegistry {
    let mut registry = registry.with_default_factory(default_factory());
    router_modes::register_missing(&mut registry);
    match kv_fetch::register(&mut registry) {
        Ok(()) | Err(HintPolicyRegistryError::Duplicate { .. }) => {}
        Err(error) => unreachable!("kv_fetch has a valid type name: {error}"),
    }
    registry
        .with_default_hint_policies([(
            kv_fetch::POLICY_TYPE.to_string(),
            kv_fetch::default_constructor(),
        )])
        .expect("one default hint policy has a unique name")
}

use dynamo_kv_router::plugins::hint_policy::HintPolicyRegistryError;
use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPluginRegistryError};

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
    use dynamo_kv_router::plugins::WorkerSelectionPolicyRegistryError;
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
            factory(&config, worker_type, partition);
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
            &config,
            WorkerType::Prefill,
            RoutingPartitionRef::new("model", "default"),
        );
    }

    /// `kv_fetch` resolves by type name in every host registry, validates its parameters, and is
    /// the default hint policy of every role the config doesn't list.
    #[test]
    fn resolves_kv_fetch_by_type_name_and_as_the_default() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        let config = |yaml: &str| {
            std::fs::write(policy_file.path(), yaml).unwrap();
            KvRouterConfig {
                router_policy_config: Some(policy_file.path().display().to_string()),
                ..Default::default()
            }
        };
        let registry = default_registry();
        let resolved = registry
            .resolve_hint_policies(&config(
                "hint_policies:\n  prefill:\n    - type: kv_fetch\n      parameters: {min_additional_blocks: 2}\n  decode: []\n",
            ))
            .unwrap();
        assert_eq!(resolved.names(WorkerType::Prefill), [kv_fetch::POLICY_TYPE]);
        assert!(!resolved.uses_defaults(WorkerType::Prefill));
        assert!(resolved.names(WorkerType::Decode).is_empty());
        assert_eq!(
            resolved.names(WorkerType::Aggregated),
            [kv_fetch::POLICY_TYPE]
        );
        assert!(resolved.uses_defaults(WorkerType::Aggregated));

        for parameters in ["{min_additional_blocks: 0}", "{min_blocks: 2}"] {
            let yaml = format!(
                "hint_policies:\n  prefill:\n    - type: kv_fetch\n      parameters: {parameters}\n"
            );
            assert!(
                registry.resolve_hint_policies(&config(&yaml)).is_err(),
                "{parameters}"
            );
        }
    }

    /// `kvcr` commits copies, and the default `kv_fetch` sends them, so a role that lists no hint
    /// policies resolves. Listing the role without a sender is rejected when the router is built.
    #[test]
    fn kvcr_pairs_with_kv_fetch() {
        use dynamo_kv_router::plugins::{PluginConfigError, resolve_role};

        let policy_file = tempfile::NamedTempFile::new().unwrap();
        let resolve = |hint_policies: &str| {
            std::fs::write(
                policy_file.path(),
                format!(
                    "worker_selection:\n  prefill:\n    type: kvcr\n    parameters: {{shared_cache_multiplier: 0.5}}\n{hint_policies}"
                ),
            )
            .unwrap();
            let config = KvRouterConfig {
                router_policy_config: Some(policy_file.path().display().to_string()),
                ..Default::default()
            };
            let mut registry = default_registry();
            register(&mut registry).unwrap();
            let selection = registry
                .resolve(&config)
                .unwrap()
                .expect("a configured instance resolves to a factory");
            let partition = RoutingPartitionRef::new("model", "default");
            let policy = selection(&config, WorkerType::Prefill, partition);
            let hints = registry.resolve_hint_policies(&config).unwrap();
            resolve_role(
                WorkerType::Prefill,
                &policy,
                hints.instances(&config, WorkerType::Prefill, partition),
                &config,
            )
        };

        let defaulted = resolve("").unwrap();
        assert!(defaulted.commit_kv_copy());
        resolve("hint_policies:\n  prefill:\n    - type: kv_fetch\n").unwrap();
        assert!(matches!(
            resolve("hint_policies:\n  prefill: []\n"),
            Err(PluginConfigError::UnsentCopies { .. })
        ));
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
