// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::*;
use crate::WorkerType;
use crate::config::KvRouterConfig;
use crate::identity::RoutingPartitionId;
use crate::plugins::RouterPluginRegistry;
use crate::scheduling::RouterPolicyConfig;

/// A router config that loads `yaml` as its router policy config. The file must outlive the
/// config's reads.
fn config_with(yaml: &str) -> (KvRouterConfig, tempfile::NamedTempFile) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), yaml).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        ..KvRouterConfig::default()
    };
    (config, file)
}

struct ActionOnly;

impl HintPolicy for ActionOnly {
    fn requirements(&self) -> HintRequirements {
        HintRequirements::new().with_actions()
    }
}

fn action_only_provider() -> HintPolicyProvider {
    Arc::new(|_parameters| {
        Ok(HintPolicyDefinition::new(
            HintRequirements::new().with_actions(),
            Arc::new(|_, _, _| HintPolicyInstance::new("action_only", ActionOnly)),
        ))
    })
}

/// Declares cache inputs, so its role starts the KV index.
struct CacheReader;

impl HintPolicy for CacheReader {
    fn requirements(&self) -> HintRequirements {
        HintRequirements::new()
            .with_inputs(WorkerInputs::CACHE)
            .with_actions()
    }
}

fn cache_reader_provider() -> HintPolicyProvider {
    Arc::new(|_parameters| {
        Ok(HintPolicyDefinition::new(
            CacheReader.requirements(),
            Arc::new(|_, _, _| HintPolicyInstance::new("cache_reader", CacheReader)),
        ))
    })
}

/// An action-only provider that validates its parameters.
fn strict_provider() -> HintPolicyProvider {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Strict {
        #[serde(default)]
        #[allow(dead_code)]
        level: u32,
    }
    Arc::new(|parameters| {
        parameters.deserialize::<Strict>()?;
        Ok(HintPolicyDefinition::new(
            HintRequirements::new().with_actions(),
            Arc::new(|_, _, _| HintPolicyInstance::new("strict", ActionOnly)),
        ))
    })
}

#[test]
fn config_keeps_role_order_and_runs_nothing_for_omitted_roles() {
    let config = RouterPolicyConfig::from_yaml(
        r#"
hint_policies:
  aggregated: [cleanup, priority]
  prefill: [priority]
  instances:
    - name: cleanup
      type: compaction_cleanup
      parameters:
        enabled: true
    - name: priority
      type: tool_priority
"#,
    )
    .unwrap();
    let hints = config.hint_policies().unwrap();
    assert_eq!(hints.role(WorkerType::Aggregated), ["cleanup", "priority"]);
    assert_eq!(hints.role(WorkerType::Prefill), ["priority"]);
    // Hint policies are opt-in per role.
    assert!(hints.role(WorkerType::Decode).is_empty());
    assert_eq!(
        hints.instance("cleanup").unwrap().policy_type(),
        "compaction_cleanup"
    );
}

#[test]
fn config_rejects_invalid_hint_policies() {
    for (yaml, expected) in [
        (
            "hint_policies:\n  aggregated: [missing]\n",
            "does not name a configured instance",
        ),
        (
            "hint_policies:\n  aggregated: [x, x]\n  instances:\n    - name: x\n      type: action_only\n",
            "more than once",
        ),
        (
            "hint_policies:\n  instances:\n    - name: x\n      type: action_only\n    - name: x\n      type: action_only\n",
            "duplicate instance",
        ),
        (
            "hint_policies:\n  instances:\n    - name: x\n      type: action_only\n      parameters: [1]\n",
            "must be a YAML mapping",
        ),
        ("hint_policies:\n  unknown_role: []\n", "unknown field"),
    ] {
        let error = RouterPolicyConfig::from_yaml(yaml).unwrap_err().to_string();
        assert!(error.contains(expected), "{yaml}: {error}");
    }
}

#[test]
fn without_config_no_role_runs_hint_policies() {
    let factory = RouterPluginRegistry::default()
        .resolve_hint_policies(&KvRouterConfig::default())
        .unwrap();
    for worker_type in [
        WorkerType::Aggregated,
        WorkerType::Prefill,
        WorkerType::Decode,
        WorkerType::Encode,
    ] {
        assert!(factory.names(worker_type).is_empty());
        assert_eq!(factory.requirements(worker_type), HintRequirements::new());
        assert_eq!(
            factory.requirements(worker_type).worker_inputs(),
            WorkerInputs::NONE
        );
    }
}

#[test]
fn configured_policies_declare_the_resources_they_start() {
    let mut registry = RouterPluginRegistry::default();
    registry
        .register_hint_policy("cache_reader", cache_reader_provider())
        .unwrap();
    registry
        .register_hint_policy("action_only", action_only_provider())
        .unwrap();
    let (config, _file) = config_with(
        r#"
hint_policies:
  aggregated: [reader, actions]
  prefill: [actions]
  decode: []
  instances:
    - name: reader
      type: cache_reader
    - name: actions
      type: action_only
"#,
    );
    let factory = registry.resolve_hint_policies(&config).unwrap();
    let requirements = factory.requirements(WorkerType::Aggregated);
    // A policy that declares cache inputs needs the KV index even when the worker-selection
    // policy reads none.
    assert!(requirements.uses_actions());
    assert_eq!(requirements.worker_inputs(), WorkerInputs::CACHE);
    // An action-only role starts no worker-input resources.
    let requirements = factory.requirements(WorkerType::Prefill);
    assert_eq!(requirements.worker_inputs(), WorkerInputs::NONE);
    assert!(factory.names(WorkerType::Decode).is_empty());
    assert!(factory.names(WorkerType::Encode).is_empty());
}

#[test]
fn registry_rejects_empty_and_duplicate_names() {
    let mut registry = RouterPluginRegistry::default();
    assert!(matches!(
        registry.register_hint_policy("", action_only_provider()),
        Err(HintPolicyRegistryError::EmptyName)
    ));
    registry
        .register_hint_policy("action_only", action_only_provider())
        .unwrap();
    assert!(matches!(
        registry.register_hint_policy("action_only", action_only_provider()),
        Err(HintPolicyRegistryError::Duplicate { .. })
    ));
    assert!(!registry.is_empty());
}

#[test]
fn resolution_reports_unknown_types_and_invalid_parameters() {
    let mut registry = RouterPluginRegistry::default();
    registry
        .register_hint_policy("action_only", action_only_provider())
        .unwrap();
    registry
        .register_hint_policy("strict", strict_provider())
        .unwrap();

    let (config, _file) = config_with(
        "hint_policies:\n  aggregated: [x]\n  instances:\n    - name: x\n      type: missing\n",
    );
    let error = registry.resolve_hint_policies(&config).err().unwrap();
    assert!(
        matches!(&error, HintPolicyRegistryError::UnknownType { available, .. }
            if available == "action_only, strict"),
        "{error}"
    );

    for parameters in ["{level: -1}", "{unknown: 1}"] {
        let (config, _file) = config_with(&format!(
            "hint_policies:\n  aggregated: [x]\n  instances:\n    - name: x\n      type: strict\n      parameters: {parameters}\n"
        ));
        assert!(
            matches!(
                registry.resolve_hint_policies(&config),
                Err(HintPolicyRegistryError::Provider { .. })
            ),
            "{parameters}"
        );
    }
}

#[test]
fn unlisted_instances_are_still_validated() {
    let (config, _file) = config_with(
        "hint_policies:\n  aggregated: []\n  instances:\n    - name: typo\n      type: missing\n",
    );
    assert!(matches!(
        RouterPluginRegistry::default().resolve_hint_policies(&config),
        Err(HintPolicyRegistryError::UnknownType { .. })
    ));
}

#[test]
fn hosts_cannot_list_a_policy_twice() {
    let definition = || {
        HintPolicyDefinition::new(
            HintRequirements::new().with_actions(),
            Arc::new(|_, _, _| HintPolicyInstance::new("action_only", ActionOnly)),
        )
    };
    assert!(matches!(
        HintPolicyFactory::for_all_roles([
            ("same".to_string(), definition()),
            ("same".to_string(), definition()),
        ]),
        Err(HintPolicyRegistryError::DuplicateInstance { .. })
    ));
}

#[test]
fn instances_must_keep_their_declared_requirements() {
    let factory = HintPolicyFactory::for_all_roles([(
        "drifting".to_string(),
        HintPolicyDefinition::new(
            HintRequirements::new()
                .with_inputs(WorkerInputs::CACHE)
                .with_actions(),
            Arc::new(|_, _, _| HintPolicyInstance::new("drifting", ActionOnly)),
        ),
    )])
    .unwrap();
    let partition = RoutingPartitionId::new("model", "default");
    assert!(matches!(
        factory.instances(
            &KvRouterConfig::default(),
            WorkerType::Aggregated,
            partition.as_ref()
        ),
        Err(HintPolicyRegistryError::RequirementsChanged { .. })
    ));
}
