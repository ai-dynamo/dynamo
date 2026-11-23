// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::WorkerType;
use crate::config::KvRouterConfig;
use crate::plugins::RouterPluginRegistry;
use crate::plugins::worker_selection::WorkerInputs;
use crate::scheduling::RouterPolicyConfig;
use dynamo_router_protocol::identity::RoutingPartitionId;

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

/// Records how often it runs, and returns `actions` actions of type `test.<name>`.
#[derive(Clone)]
struct Recorder {
    name: &'static str,
    actions: usize,
    fail: bool,
    inputs: WorkerInputs,
    commitments: Commitments,
    calls: Arc<AtomicUsize>,
}

impl Recorder {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            actions: 1,
            fail: false,
            inputs: WorkerInputs::NONE,
            commitments: Commitments::NONE,
            calls: Arc::default(),
        }
    }
}

impl HintPolicy for Recorder {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.inputs
    }

    fn commitments(&self) -> Commitments {
        self.commitments
    }

    fn build_hints(
        &self,
        _context: &HintContext<'_>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(HintPolicyError::new("recorder failed"));
        }
        Ok((0..self.actions)
            .map(|_| KvHintAction::unnumbered(format!("test.{}", self.name), "1", BTreeMap::new()))
            .collect())
    }
}

fn recorder_provider(policy: Recorder) -> HintPolicyProvider {
    Arc::new(move |_parameters| Ok(HintPolicyConstructor::shared(policy.clone())))
}

/// A provider that validates its parameters.
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
        Ok(HintPolicyConstructor::shared(Recorder::new("strict")))
    })
}

// ---- YAML -------------------------------------------------------------------------------------

/// The `hints` a role's `worker_selection` entry lists, parsed as the router policy file is.
fn hints(yaml: &str, role: WorkerType) -> Option<Vec<HintPolicyEntry>> {
    RouterPolicyConfig::from_yaml(yaml)
        .unwrap()
        .worker_selection()
        .unwrap()
        .hints(role)
        .map(<[HintPolicyEntry]>::to_vec)
}

#[test]
fn hints_keep_list_order_and_distinguish_omitted_from_empty() {
    let yaml = r#"
worker_selection:
  aggregated:
    type: dynamo-default-cost-fn
    parameters:
      router_temperature: 0.5
    hints:
      - type: compaction_cleanup
        parameters:
          enabled: true
      - type: tool_priority
  prefill:
    type: dynamo-default-cost-fn
    hints: []
  decode:
    type: dynamo-default-cost-fn
  encode: dynamo-default-cost-fn
"#;
    let aggregated = hints(yaml, WorkerType::Aggregated).unwrap();
    let names: Vec<_> = aggregated.iter().map(HintPolicyEntry::name).collect();
    assert_eq!(names, ["compaction_cleanup", "tool_priority"]);
    assert_eq!(aggregated[0].policy_type(), "compaction_cleanup");
    assert_eq!(
        aggregated[0].parameters()["enabled"],
        serde_yaml::Value::Bool(true)
    );
    assert_eq!(
        aggregated[1].parameters(),
        &serde_yaml::Value::Mapping(Default::default())
    );
    // An empty list is a choice; an entry without `hints`, or a bare entry, takes the defaults.
    assert_eq!(hints(yaml, WorkerType::Prefill), Some(Vec::new()));
    assert_eq!(hints(yaml, WorkerType::Decode), None);
    assert_eq!(hints(yaml, WorkerType::Encode), None);
}

#[test]
fn a_role_lists_one_type_twice_only_with_distinct_names() {
    let decode = hints(
        r#"
worker_selection:
  decode:
    type: dynamo-default-cost-fn
    hints:
      - type: harness_prefetch
        name: short
      - type: harness_prefetch
        name: long
"#,
        WorkerType::Decode,
    )
    .unwrap();
    let names: Vec<_> = decode.iter().map(HintPolicyEntry::name).collect();
    assert_eq!(names, ["short", "long"]);
}

#[test]
fn config_rejects_invalid_hints() {
    let entry = |hints: &str| {
        format!(
            "worker_selection:\n  prefill:\n    type: dynamo-default-cost-fn\n    hints:\n{hints}"
        )
    };
    for (yaml, expected) in [
        (
            entry("      - type: kv_fetch\n      - type: kv_fetch\n"),
            r#"worker_selection.prefill.hints lists "kv_fetch" more than once; give each entry of the same type a distinct name"#,
        ),
        (
            entry("      - type: kv_fetch\n        parameters: [1]\n"),
            r#"worker_selection.prefill.hints entry "kv_fetch" parameters must be a YAML mapping"#,
        ),
        (entry("      - type: 'bad type'\n"), "must match"),
        (entry("      - name: x\n"), "missing field `type`"),
        // No bare names.
        (
            entry("      - kv_fetch\n"),
            r#"worker_selection.prefill: invalid type: string "kv_fetch""#,
        ),
        (
            entry("      - type: kv_fetch\n        hints: []\n"),
            "unknown field `hints`",
        ),
        // The deprecated `instances` form takes no hints.
        (
            "worker_selection:\n  prefill: legacy\n  instances:\n    - name: legacy\n      type: dynamo-default-cost-fn\n      hints: []\n".to_string(),
            "unknown field `hints`",
        ),
    ] {
        let error = RouterPolicyConfig::from_yaml(&yaml)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{yaml}: {error}");
    }
}

// ---- Registry ---------------------------------------------------------------------------------

const ANY_ON: [HintHost; 1] = [HintHost::AnyWorkerSelection];
const NO_HOSTS: [HintHost; 0] = [];

/// Register a [`Recorder`] named `name` with `hosts`.
fn register(
    registry: &mut RouterPluginRegistry,
    name: &'static str,
    hosts: impl IntoIterator<Item = HintHost>,
) {
    registry
        .register_hint_policy(name, recorder_provider(Recorder::new(name)), hosts)
        .unwrap();
}

#[test]
fn roles_without_hints_run_the_defaults_and_listed_hints_replace_them() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "fallback", ANY_ON);
    register(&mut registry, "record", NO_HOSTS);
    let (config, _file) = config_with(
        r#"
worker_selection:
  aggregated:
    type: alpha
    hints:
      - type: record
  prefill:
    type: alpha
    hints: []
  decode: alpha
"#,
    );
    let factory = registry.resolve_hint_policies(&config).unwrap();
    assert_eq!(factory.names(WorkerType::Aggregated), ["record"]);
    assert!(!factory.uses_defaults(WorkerType::Aggregated));
    assert!(factory.names(WorkerType::Prefill).is_empty());
    assert!(!factory.uses_defaults(WorkerType::Prefill));
    // A bare entry and an unlisted role run the hints that are on by default.
    for role in [WorkerType::Decode, WorkerType::Encode] {
        assert_eq!(factory.names(role), ["fallback"]);
        assert!(factory.uses_defaults(role));
    }

    let partition = RoutingPartitionId::new("model", "default");
    let instances = factory.instances(
        &crate::plugins::worker_selection::SelectionConfig::from(&config),
        WorkerType::Decode,
        partition.as_ref(),
    );
    assert!(instances.are_defaults());
    assert_eq!(instances.policies().len(), 1);
}

#[test]
fn defaults_run_in_registration_order() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "first", ANY_ON);
    register(
        &mut registry,
        "kvcr_only",
        [HintHost::worker_selection("kvcr")],
    );
    register(
        &mut registry,
        "builtin_only",
        [HintHost::worker_selection("default")],
    );
    register(&mut registry, "last", ANY_ON);
    let (config, _file) = config_with(
        r#"
worker_selection:
  aggregated: default
  prefill: kvcr
  decode:
    type: alpha
  encode: legacy
  instances:
    - name: legacy
      type: kvcr
"#,
    );
    let factory = registry.resolve_hint_policies(&config).unwrap();
    // `default`, and a role `worker_selection` doesn't list, run the built-in selector.
    assert_eq!(
        factory.names(WorkerType::Aggregated),
        ["first", "builtin_only", "last"]
    );
    assert_eq!(
        factory.names(WorkerType::Prefill),
        ["first", "kvcr_only", "last"]
    );
    assert_eq!(factory.names(WorkerType::Decode), ["first", "last"]);
    // A deprecated instance name runs under its type.
    assert_eq!(
        factory.names(WorkerType::Encode),
        ["first", "kvcr_only", "last"]
    );
}

#[test]
fn defaults_are_built_from_empty_parameters() {
    let mut registry = RouterPluginRegistry::default();
    registry
        .register_hint_policy(
            "needs_level",
            Arc::new(|parameters| {
                #[derive(serde::Deserialize)]
                #[allow(dead_code)]
                struct Required {
                    level: u32,
                }
                parameters.deserialize::<Required>()?;
                Ok(HintPolicyConstructor::shared(Recorder::new("needs_level")))
            }),
            ANY_ON,
        )
        .unwrap();
    let error = registry
        .resolve_hint_policies(&KvRouterConfig::default())
        .err()
        .unwrap();
    assert!(
        matches!(&error, HintPolicyRegistryError::Provider { policy_type, .. }
            if policy_type == "needs_level"),
        "{error}"
    );
}

#[test]
fn a_role_runs_its_listed_hints_in_order() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "kv_fetch", ANY_ON);
    register(&mut registry, "acme_priority", NO_HOSTS);
    let (config, _file) = config_with(
        r#"
worker_selection:
  prefill:
    type: kvcr
    hints:
      - type: acme_priority
      - type: kv_fetch
"#,
    );
    let factory = registry.resolve_hint_policies(&config).unwrap();
    assert_eq!(
        factory.names(WorkerType::Prefill),
        ["acme_priority", "kv_fetch"]
    );
}

#[test]
fn registry_rejects_bad_names_and_hosts() {
    let mut registry = RouterPluginRegistry::default();
    let provider = || recorder_provider(Recorder::new("x"));
    assert!(matches!(
        registry.register_hint_policy("", provider(), ANY_ON),
        Err(HintPolicyRegistryError::EmptyName)
    ));
    register(&mut registry, "record", ANY_ON);
    assert!(matches!(
        registry.register_hint_policy("record", provider(), ANY_ON),
        Err(HintPolicyRegistryError::Duplicate { .. })
    ));
    assert!(!registry.is_empty());

    let error = registry
        .register_hint_policy(
            "x",
            provider(),
            [
                HintHost::worker_selection("kvcr"),
                HintHost::worker_selection("kvcr"),
            ],
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        r#"hint-policy type "x" lists host WorkerSelection("kvcr") more than once"#
    );
    // A rejected registration leaves no type behind.
    register(&mut registry, "x", ANY_ON);
}

#[test]
fn resolve_plugins_rejects_hosts_no_linked_plugin_registers() {
    let mut registry = RouterPluginRegistry::default();
    registry
        .register_worker_selection(
            "kvcr",
            Arc::new(|_| unreachable!("resolve_plugins builds no policy here")),
        )
        .unwrap();
    register(
        &mut registry,
        "linked",
        [
            HintHost::AnyWorkerSelection,
            HintHost::worker_selection("kvcr"),
            HintHost::worker_selection("default"),
        ],
    );
    registry
        .resolve_plugins(&KvRouterConfig::default())
        .unwrap();

    register(&mut registry, "typo", [HintHost::worker_selection("kvrc")]);
    let error = registry
        .resolve_plugins(&KvRouterConfig::default())
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        r#"hint-policy type "typo" names host WorkerSelection("kvrc"), which no linked plugin registers"#
    );
}

#[test]
fn resolution_reports_unknown_types_and_invalid_parameters() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "record", NO_HOSTS);
    registry
        .register_hint_policy("strict", strict_provider(), NO_HOSTS)
        .unwrap();

    let (config, _file) = config_with(
        "worker_selection:\n  aggregated:\n    type: alpha\n    hints:\n      - type: missing\n",
    );
    let error = registry.resolve_hint_policies(&config).err().unwrap();
    assert!(
        matches!(&error, HintPolicyRegistryError::UnknownType { available, .. }
            if available == "record, strict"),
        "{error}"
    );

    for parameters in ["{level: -1}", "{unknown: 1}"] {
        let (config, _file) = config_with(&format!(
            "worker_selection:\n  aggregated:\n    type: alpha\n    hints:\n      - type: strict\n        parameters: {parameters}\n"
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
