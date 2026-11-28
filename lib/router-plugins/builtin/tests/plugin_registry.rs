// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::host::{RouterPluginRegistryHostExt, RouterPlugins};
use dynamo_kv_router::plugins::request_classifier::*;
use dynamo_kv_router::plugins::worker_selection::*;

struct PassThrough;

impl RequestClassifier for PassThrough {
    fn classify(
        &mut self,
        request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> Result<ClassifyRequest, Box<RequestClassifierError>> {
        Ok(request)
    }
}

fn config(yaml: &str) -> (tempfile::NamedTempFile, KvRouterConfig) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), yaml).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        ..Default::default()
    };
    (file, config)
}

#[test]
fn builtin_default_does_not_opt_into_custom_frontend_restrictions() {
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let (_file, explicit_default) = config("worker_selection:\n  aggregated: default\n");
    for config in [KvRouterConfig::default(), explicit_default] {
        assert!(registry.resolve(&config).unwrap().is_some());
        let plugins = registry.resolve_plugins(&config).unwrap();
        assert!(plugins.worker_selection().is_some());
        assert!(!plugins.has_custom_worker_selection());
        assert!(!plugins.has_custom_plugins());
    }
}

#[test]
fn builtin_registration_preserves_classifiers_and_resolves_mixed_pools() {
    for selection in ["default", "dynamo-default-cost-fn"] {
        let mut registry = dynamo_router_policies::default_registry();
        registry
            .register_request_classifier(
                "pass-through",
                Arc::new(|_| Ok(Arc::new(|_| Box::new(PassThrough)))),
            )
            .unwrap();
        // Registering named providers must preserve the fallback and classifier providers.
        dynamo_router_policies::register(&mut registry).unwrap();
        let (_file, config) = config(&format!(
            r#"
request_classifier:
  type: pass-through
worker_selection:
  prefill: {selection}
"#
        ));
        let plugins = registry.resolve_plugins(&config).unwrap();
        assert!(!plugins.is_empty());
        assert!(plugins.request_classifier().is_some());
        assert!(plugins.worker_selection().is_some());
        assert!(plugins.has_custom_plugins());
        assert_eq!(
            plugins.has_custom_worker_selection(),
            selection != "default"
        );
        // With only a classifier selected, the host still resolves the builtin policy.
        let factory = registry.resolve(&config).unwrap().unwrap();
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
        ] {
            factory(WorkerSelectionPolicyContext::new(
                &SelectionConfig::from(&config),
                role,
                RoutingPartitionRef::new("model", "default"),
            ));
        }
    }
}

/// The KV-capacity-aware decode policy resolves from the YAML its documentation shows, and a
/// misspelled parameter fails startup like the other policies.
#[test]
fn resolves_kv_capacity_aware_for_decode() {
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let (_file, documented) = config(
        r#"
worker_selection:
  decode:
    type: dynamo-kv-capacity-aware
    parameters:
      capacity_fraction: 1.0
      count_priority: true
      decode_active_request_weight: 0
      capacity_blocks: 20000
"#,
    );
    let factory = registry
        .resolve(&documented)
        .unwrap()
        .expect("a configured policy resolves to a factory");
    factory(WorkerSelectionPolicyContext::new(
        &SelectionConfig::from(&documented),
        WorkerType::Decode,
        RoutingPartitionRef::new("model", "default"),
    ));

    let (_file, misspelled) = config(
        r#"
worker_selection:
  decode:
    type: dynamo-kv-capacity-aware
    parameters:
      capacity_ceiling: 20000
"#,
    );
    let Err(error) = registry.resolve(&misspelled) else {
        panic!("an unknown parameter must fail resolution");
    };
    assert!(
        matches!(&error, WorkerSelectionPolicyRegistryError::Provider { policy_type, .. }
            if policy_type == "dynamo-kv-capacity-aware"),
        "unexpected error: {error}"
    );
    assert!(error.to_string().contains("capacity_ceiling"), "{error}");
}

#[test]
fn programmatic_factory_is_an_explicit_custom_selection() {
    let plugins =
        RouterPlugins::default().with_worker_selection(dynamo_router_policies::default_factory());
    assert!(plugins.has_custom_worker_selection());
    assert!(plugins.has_custom_plugins());
}

#[test]
fn registering_named_policies_does_not_replace_the_hosts_default() {
    let fallback = dynamo_router_policies::default_factory();
    let mut registry = dynamo_kv_router::plugins::RouterPluginRegistry::default()
        .with_default_factory(fallback.clone());
    dynamo_router_policies::register(&mut registry).unwrap();
    let resolved = registry
        .resolve(&KvRouterConfig::default())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&fallback, &resolved));
}
