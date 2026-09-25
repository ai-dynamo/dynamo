// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::plugins::RouterPlugins;
use dynamo_kv_router::plugins::request_classifier::{
    ClassifyFuture, ClassifyRequest, RequestClassifier,
};
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};

struct PassThrough;

impl RequestClassifier for PassThrough {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        Box::pin(async move { Ok(request) })
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
    let mut registry = dynamo_custom_policy_builtin::default_registry();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
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
    for selection in ["default", "named-default"] {
        let mut registry = dynamo_custom_policy_builtin::default_registry();
        registry
            .register_request_classifier(
                "pass-through",
                Arc::new(|_| Ok(Arc::new(|_| Box::new(PassThrough)))),
            )
            .unwrap();
        // Registering named providers must preserve the fallback and classifier providers.
        dynamo_custom_policy_builtin::register(&mut registry).unwrap();
        let (_file, config) = config(&format!(
            r#"
request_classifier:
  type: pass-through
worker_selection:
  prefill: {selection}
  instances:
    - name: named-default
      type: dynamo-default-cost-fn
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
            factory(&config, role, RoutingPartitionRef::new("model", "default"));
        }
    }
}

#[test]
fn programmatic_factory_is_an_explicit_custom_selection() {
    let plugins = RouterPlugins::default()
        .with_worker_selection(dynamo_custom_policy_builtin::default_factory());
    assert!(plugins.has_custom_worker_selection());
    assert!(plugins.has_custom_plugins());
}

#[test]
fn registering_named_policies_does_not_replace_the_hosts_default() {
    let fallback = dynamo_custom_policy_builtin::default_factory();
    let mut registry = dynamo_kv_router::plugins::RouterPluginRegistry::default()
        .with_default_factory(fallback.clone());
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    let resolved = registry
        .resolve(&KvRouterConfig::default())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&fallback, &resolved));
}
