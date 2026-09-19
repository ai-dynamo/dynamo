// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::plugins::RouterPluginRegistry;
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
        assert!(registry.resolve_plugins(&config).unwrap().is_empty());
    }
}

#[test]
fn builtin_registration_preserves_classifiers_and_resolves_mixed_pools() {
    for selection in ["default", "named-default"] {
        let mut registry = RouterPluginRegistry::default();
        registry
            .register_request_classifier(
                "pass-through",
                Arc::new(|_| Ok(Arc::new(|_| Box::new(PassThrough)))),
            )
            .unwrap();
        // Installing the default must preserve classifier providers already in the catalog.
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
        assert_eq!(plugins.worker_selection().is_some(), selection != "default");
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
