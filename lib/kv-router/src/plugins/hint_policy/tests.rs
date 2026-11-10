// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::runtime::{BookedCopy, HintInputs, HintPolicySlot, compose_kv_hint};
use super::*;
use crate::WorkerType;
use crate::config::KvRouterConfig;
use crate::identity::RoutingPartitionId;
use crate::kv_hints::{
    KvCopySource, KvTransferCandidateSource, KvTransferCandidates, KvTransferSourceChoice,
    MAX_KV_HINT_ACTIONS, kv_copy_locations,
};
use crate::plugins::worker_selection::{
    WorkerCandidates, WorkerInputView, WorkerPicker, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicy,
};
use crate::plugins::{PluginConfigError, RouterPluginRegistry, resolve_role};
use crate::protocols::{
    ExternalSequenceBlockHash, KvHintTransferWorkerMetadata, WorkerConfigLike, WorkerId,
};
use crate::scheduling::{RouterPolicyConfig, WorkerSelectionPolicyError};

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

    fn consumer(name: &'static str) -> Self {
        Self {
            commitments: Commitments::KV_COPY,
            ..Self::new(name)
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
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

const ANY_ON: [(HintHost, HintDefault); 1] = [(HintHost::AnyWorkerSelection, HintDefault::On)];
const ANY_OFF: [(HintHost, HintDefault); 1] = [(HintHost::AnyWorkerSelection, HintDefault::Off)];

/// Register a [`Recorder`] named `name` with `hosts`.
fn register(
    registry: &mut RouterPluginRegistry,
    name: &'static str,
    hosts: impl IntoIterator<Item = (HintHost, HintDefault)>,
) {
    registry
        .register_hint_policy(name, recorder_provider(Recorder::new(name)), hosts)
        .unwrap();
}

#[test]
fn roles_without_hints_run_the_defaults_and_listed_hints_replace_them() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "fallback", ANY_ON);
    register(&mut registry, "record", ANY_OFF);
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
    let instances = factory.instances(&config, WorkerType::Decode, partition.as_ref());
    assert!(instances.are_defaults());
    assert_eq!(instances.policies().len(), 1);
}

#[test]
fn defaults_run_in_registration_order_and_a_named_host_overrides_any() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "first", ANY_ON);
    register(
        &mut registry,
        "kvcr_only",
        [
            (HintHost::AnyWorkerSelection, HintDefault::Off),
            (HintHost::worker_selection("kvcr"), HintDefault::On),
        ],
    );
    register(
        &mut registry,
        "not_under_kvcr",
        [
            (HintHost::worker_selection("kvcr"), HintDefault::Off),
            (HintHost::AnyWorkerSelection, HintDefault::On),
        ],
    );
    register(
        &mut registry,
        "builtin_only",
        [(HintHost::worker_selection("default"), HintDefault::On)],
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
        ["first", "not_under_kvcr", "builtin_only", "last"]
    );
    assert_eq!(
        factory.names(WorkerType::Prefill),
        ["first", "kvcr_only", "last"]
    );
    assert_eq!(
        factory.names(WorkerType::Decode),
        ["first", "not_under_kvcr", "last"]
    );
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
fn a_listed_hint_must_run_under_the_role_policy() {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry, "kv_fetch", ANY_ON);
    register(
        &mut registry,
        "acme_priority",
        [(HintHost::worker_selection("kvcr"), HintDefault::Off)],
    );
    let (config, _file) = config_with(
        r#"
worker_selection:
  prefill:
    type: dynamo-two-tier-cost-fn
    hints:
      - type: acme_priority
"#,
    );
    let error = registry.resolve_hint_policies(&config).err().unwrap();
    assert_eq!(
        error.to_string(),
        r#"hint-policy type "acme_priority" doesn't run under worker-selection type "dynamo-two-tier-cost-fn" (prefill); hints for that type: kv_fetch"#
    );

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
        .register_hint_policy("x", provider(), [])
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        r#"hint-policy type "x" registers no hosts"#
    );
    let error = registry
        .register_hint_policy(
            "x",
            provider(),
            [
                (HintHost::worker_selection("kvcr"), HintDefault::On),
                (HintHost::worker_selection("kvcr"), HintDefault::Off),
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
            (HintHost::AnyWorkerSelection, HintDefault::On),
            (HintHost::worker_selection("kvcr"), HintDefault::On),
            (HintHost::worker_selection("default"), HintDefault::On),
        ],
    );
    registry
        .resolve_plugins(&KvRouterConfig::default())
        .unwrap();

    register(
        &mut registry,
        "typo",
        [(HintHost::worker_selection("kvrc"), HintDefault::On)],
    );
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
    register(&mut registry, "record", ANY_OFF);
    registry
        .register_hint_policy("strict", strict_provider(), ANY_OFF)
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

#[test]
fn hosts_cannot_list_a_policy_twice() {
    let constructor = || HintPolicyConstructor::shared(Recorder::new("record"));
    assert!(matches!(
        HintPolicyFactory::for_all_roles([
            ("same".to_string(), constructor()),
            ("same".to_string(), constructor()),
        ]),
        Err(HintPolicyRegistryError::DuplicateName { .. })
    ));
}

#[test]
fn constructors_share_or_build_per_partition() {
    let built = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&built);
    let factory = HintPolicyFactory::for_all_roles([
        (
            "shared".to_string(),
            HintPolicyConstructor::shared(Recorder::new("shared")),
        ),
        (
            "fresh".to_string(),
            HintPolicyConstructor::new(move |_, _, _| {
                counter.fetch_add(1, Ordering::SeqCst);
                Arc::new(Recorder::new("fresh"))
            }),
        ),
    ])
    .unwrap();
    let config = KvRouterConfig::default();
    let first = RoutingPartitionId::new("model", "a");
    let second = RoutingPartitionId::new("model", "b");
    let a = factory.instances(&config, WorkerType::Aggregated, first.as_ref());
    let b = factory.instances(&config, WorkerType::Aggregated, second.as_ref());
    assert!(Arc::ptr_eq(&a.policies()[0].1, &b.policies()[0].1));
    assert!(!Arc::ptr_eq(&a.policies()[1].1, &b.policies()[1].1));
    assert_eq!(built.load(Ordering::SeqCst), 2);
    assert!(!a.are_defaults());
}

// ---- resolve_role -----------------------------------------------------------------------------

struct InputsScorer(WorkerInputs);

impl WorkerScorer for InputsScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.0
    }

    fn score(
        &self,
        _context: &WorkerSelectionContext<'_>,
        _candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        costs.fill(0.0);
        Ok(())
    }
}

struct FirstPicker;

impl WorkerPicker for FirstPicker {
    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        Ok(0)
    }
}

fn selection(inputs: WorkerInputs) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "prefill",
        vec![Box::new(InputsScorer(inputs))],
        Box::new(FirstPicker),
    )
}

fn role_policies(policies: Vec<(&str, Recorder)>, defaults: bool) -> RoleHintPolicies {
    let factory = HintPolicyFactory::for_all_roles(
        policies
            .into_iter()
            .map(|(name, policy)| (name.to_string(), HintPolicyConstructor::shared(policy))),
    )
    .unwrap();
    let partition = RoutingPartitionId::new("model", "default");
    let instances = factory.instances(
        &KvRouterConfig::default(),
        WorkerType::Prefill,
        partition.as_ref(),
    );
    if defaults {
        let mut registry = RouterPluginRegistry::default();
        for (name, policy) in instances.policies() {
            let policy = Arc::clone(policy);
            registry
                .register_hint_policy(
                    name.to_string(),
                    Arc::new(move |_| {
                        let policy = Arc::clone(&policy);
                        Ok(HintPolicyConstructor::new(move |_, _, _| {
                            Arc::clone(&policy)
                        }))
                    }),
                    ANY_ON,
                )
                .unwrap();
        }
        return registry
            .resolve_hint_policies(&KvRouterConfig::default())
            .unwrap()
            .instances(
                &KvRouterConfig::default(),
                WorkerType::Prefill,
                partition.as_ref(),
            );
    }
    instances
}

/// A config whose KV index can't keep the request's block chain.
fn approximate_index() -> KvRouterConfig {
    KvRouterConfig {
        use_kv_events: false,
        ..KvRouterConfig::default()
    }
}

#[test]
fn resolve_role_runs_the_copy_consumer_first_then_configured_order() {
    let responsibilities = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(
            vec![
                ("first", Recorder::new("first")),
                ("consumer", Recorder::consumer("consumer")),
                ("last", Recorder::new("last")),
            ],
            false,
        ),
        &KvRouterConfig::default(),
    )
    .unwrap();
    let order: Vec<_> = responsibilities.hint_policy_names().collect();
    assert_eq!(order, ["consumer", "first", "last"]);
    assert!(!responsibilities.commit_kv_copy());
}

#[test]
fn resolve_role_rejects_two_copy_consumers() {
    let error = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(
            vec![
                ("one", Recorder::consumer("one")),
                ("two", Recorder::consumer("two")),
            ],
            false,
        ),
        &KvRouterConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(&error, PluginConfigError::ConflictingCopyConsumers { first, second, .. }
            if first == "one" && second == "two"),
        "{error}"
    );
    assert!(error.to_string().contains("keep one"), "{error}");
}

#[test]
fn resolve_role_rejects_copy_reads_the_index_cannot_serve() {
    let reader = Recorder {
        inputs: WorkerInputs::KV_COPY,
        ..Recorder::new("reader")
    };
    let error = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(vec![("reader", reader.clone())], false),
        &approximate_index(),
    )
    .unwrap_err();
    assert!(
        matches!(&error, PluginConfigError::CopySourcesUnavailable { policy, .. } if policy == "reader"),
        "{error}"
    );
    assert!(error.to_string().contains("enable KV events"), "{error}");

    // The same policy as a default is passive: it does without copy data.
    let responsibilities = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(vec![("reader", reader)], true),
        &approximate_index(),
    )
    .unwrap();
    assert_eq!(
        responsibilities.hint_policy_names().collect::<Vec<_>>(),
        ["reader"]
    );
}

#[test]
fn resolve_role_rejects_hint_inputs_the_context_cannot_serve() {
    let reader = |inputs| Recorder {
        inputs,
        ..Recorder::new("reader")
    };
    // HintContext has no cache accessor, so CACHE is rejected for configured and default
    // policies alike. The error names only the input the policy can't have.
    for defaults in [false, true] {
        let error = resolve_role(
            WorkerType::Prefill,
            &selection(WorkerInputs::LOAD),
            role_policies(
                vec![(
                    "reader",
                    reader(WorkerInputs::CACHE | WorkerInputs::KV_COPY),
                )],
                defaults,
            ),
            &KvRouterConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(&error, PluginConfigError::UnreadableHintInputs { role: "prefill", policy, inputs }
                if policy == "reader" && inputs == "WorkerInputs::CACHE"),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("may declare only WorkerInputs::KV_COPY"),
            "{error}"
        );
    }

    let responsibilities = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(vec![("reader", reader(WorkerInputs::KV_COPY))], false),
        &KvRouterConfig::default(),
    )
    .unwrap();
    assert_eq!(
        responsibilities.hint_policy_names().collect::<Vec<_>>(),
        ["reader"]
    );
}

#[test]
fn configured_policies_add_inputs_and_defaults_add_none() {
    let reader = Recorder {
        inputs: WorkerInputs::KV_COPY,
        ..Recorder::new("reader")
    };
    let configured = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(vec![("reader", reader.clone())], false),
        &KvRouterConfig::default(),
    )
    .unwrap();
    assert_eq!(
        configured.prepare_inputs(),
        WorkerInputs::LOAD | WorkerInputs::KV_COPY
    );

    let defaulted = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        role_policies(vec![("reader", reader)], true),
        &KvRouterConfig::default(),
    )
    .unwrap();
    assert_eq!(defaulted.prepare_inputs(), WorkerInputs::LOAD);

    let none = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        RoleHintPolicies::none(),
        &KvRouterConfig::default(),
    )
    .unwrap();
    assert_eq!(none.hint_policy_names().count(), 0);
}

#[test]
fn a_scorer_that_reads_copy_offers_commits_copies_that_a_consumer_sends() {
    let committing = || selection(WorkerInputs::CACHE | WorkerInputs::KV_COPY);
    let responsibilities = resolve_role(
        WorkerType::Prefill,
        &committing(),
        role_policies(vec![("consumer", Recorder::consumer("consumer"))], false),
        &KvRouterConfig::default(),
    )
    .unwrap();
    assert!(responsibilities.commit_kv_copy());
    // The consumer reads no uncommitted copy source, so bookings compute none.
    assert!(!responsibilities.compute_booked_copy_source());
    assert!(
        responsibilities
            .prepare_inputs()
            .contains(WorkerInputs::KV_COPY)
    );

    // A default consumer pairs too: defaults are passive, not absent.
    resolve_role(
        WorkerType::Prefill,
        &committing(),
        role_policies(vec![("consumer", Recorder::consumer("consumer"))], true),
        &KvRouterConfig::default(),
    )
    .unwrap();

    // Committed copies that nothing sends are rejected, even with other hint policies.
    for policies in [
        RoleHintPolicies::none(),
        role_policies(vec![("other", Recorder::new("other"))], false),
    ] {
        let error = resolve_role(
            WorkerType::Prefill,
            &committing(),
            policies,
            &KvRouterConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, PluginConfigError::UnsentCopies { role: "prefill" }),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("none of the hints in worker_selection.prefill.hints sends them"),
            "{error}"
        );
    }

    // Default hints that send no copies point at listing one instead.
    let error = resolve_role(
        WorkerType::Prefill,
        &committing(),
        role_policies(vec![("other", Recorder::new("other"))], true),
        &KvRouterConfig::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            PluginConfigError::UnsentDefaultCopies { role: "prefill" }
        ),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("none of its default hints sends them"),
        "{error}"
    );

    // Copy offers need an index that keeps the block chain.
    let error = resolve_role(
        WorkerType::Prefill,
        &committing(),
        role_policies(vec![("consumer", Recorder::consumer("consumer"))], false),
        &approximate_index(),
    )
    .unwrap_err();
    assert!(
        matches!(error, PluginConfigError::CopyOffersUnavailable { .. }),
        "{error}"
    );
}

// ---- The hint stage ---------------------------------------------------------------------------

fn slot(name: &str, policy: Recorder) -> HintPolicySlot {
    HintPolicySlot::new(Arc::from(name), Arc::new(policy))
}

fn inputs<'a>() -> HintInputs<'a> {
    HintInputs {
        request_id: "request-1",
        worker: WorkerWithDpRank::new(1, 0),
        isl_tokens: 8,
        block_size: 4,
        worker_accepts_hints: true,
        session_context: None,
        committed_kv_copy: None,
        kv_copy_source: None,
    }
}

fn copy() -> KvCopySource {
    KvCopySource::new(
        WorkerWithDpRank::new(1, 0),
        KvTransferSourceChoice {
            source: WorkerWithDpRank::new(2, 0).into(),
            prefix_blocks: 3,
        },
        1,
    )
}

fn ids_and_types(hint: &crate::kv_hints::KvHint) -> Vec<(String, String)> {
    hint.actions
        .iter()
        .map(|action| (action.action_id.clone(), action.action_type.clone()))
        .collect()
}

#[test]
fn actions_are_numbered_in_hint_order() {
    let policies = [
        slot(
            "two",
            Recorder {
                actions: 2,
                ..Recorder::new("two")
            },
        ),
        slot("one", Recorder::new("one")),
    ];
    let hint = compose_kv_hint(&policies, &inputs())
        .unwrap()
        .expect("hint");
    assert_eq!(hint.message_id, "request-1");
    let pairs: Vec<_> = ids_and_types(&hint)
        .into_iter()
        .map(|(id, kind)| format!("{id}:{kind}"))
        .collect();
    assert_eq!(pairs, ["a1:test.two", "a2:test.two", "a3:test.one"]);
}

#[test]
fn output_that_fails_or_overflows_is_dropped_whole() {
    let failing = Recorder {
        fail: true,
        ..Recorder::new("failing")
    };
    let policies = [
        slot("failing", failing.clone()),
        slot(
            "burst",
            Recorder {
                actions: MAX_KV_HINT_ACTIONS - 1,
                ..Recorder::new("burst")
            },
        ),
        // Would overflow the hint, so it is dropped whole.
        slot(
            "overflow",
            Recorder {
                actions: 2,
                ..Recorder::new("overflow")
            },
        ),
        slot("fits", Recorder::new("fits")),
    ];
    let hint = compose_kv_hint(&policies, &inputs())
        .unwrap()
        .expect("hint");
    assert_eq!(failing.calls(), 1);
    assert_eq!(hint.actions.len(), MAX_KV_HINT_ACTIONS);
    assert_eq!(hint.actions.last().unwrap().action_type, "test.fits");
    assert!(
        hint.actions
            .iter()
            .all(|action| action.action_type != "test.overflow")
    );
}

#[test]
fn workers_without_router_hint_get_no_hint_and_run_no_policy() {
    let recorder = Recorder::new("record");
    let policies = [slot("record", recorder.clone())];
    let inputs = HintInputs {
        worker_accepts_hints: false,
        ..inputs()
    };
    assert!(compose_kv_hint(&policies, &inputs).unwrap().is_none());
    assert_eq!(recorder.calls(), 0);
}

#[test]
fn the_copy_consumer_runs_first_even_when_listed_last() {
    let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    struct Ordered(
        &'static str,
        Commitments,
        Arc<parking_lot::Mutex<Vec<&'static str>>>,
    );
    impl HintPolicy for Ordered {
        fn commitments(&self) -> Commitments {
            self.1
        }
        fn build_hints(
            &self,
            _context: &HintContext<'_>,
        ) -> Result<Vec<KvHintAction>, HintPolicyError> {
            self.2.lock().push(self.0);
            Ok(Vec::new())
        }
    }
    let policies = [
        HintPolicySlot::new(
            Arc::from("other"),
            Arc::new(Ordered("other", Commitments::NONE, Arc::clone(&calls))),
        ),
        HintPolicySlot::new(
            Arc::from("consumer"),
            Arc::new(Ordered(
                "consumer",
                Commitments::KV_COPY,
                Arc::clone(&calls),
            )),
        ),
    ];
    assert!(compose_kv_hint(&policies, &inputs()).unwrap().is_none());
    assert_eq!(*calls.lock(), ["consumer", "other"]);
}

#[test]
fn an_unsent_committed_copy_fails_the_booking() {
    let committed = || HintInputs {
        committed_kv_copy: Some(BookedCopy::new(copy(), None)),
        ..inputs()
    };
    // The consumer returning an error, no action, or too many actions fails the booking.
    for consumer in [
        Recorder {
            fail: true,
            ..Recorder::consumer("consumer")
        },
        Recorder {
            actions: 0,
            ..Recorder::consumer("consumer")
        },
        Recorder {
            actions: MAX_KV_HINT_ACTIONS + 1,
            ..Recorder::consumer("consumer")
        },
    ] {
        let policies = [
            slot("other", Recorder::new("other")),
            slot("consumer", consumer),
        ];
        let error = compose_kv_hint(&policies, &committed()).unwrap_err();
        assert!(error.to_string().contains("\"consumer\""), "{error}");
    }
    // No policy sends committed copies.
    let policies = [slot("other", Recorder::new("other"))];
    assert!(compose_kv_hint(&policies, &committed()).is_err());

    // Any action counts as delivery; the stage never reads payloads.
    let policies = [slot("consumer", Recorder::consumer("consumer"))];
    let hint = compose_kv_hint(&policies, &committed())
        .unwrap()
        .expect("hint");
    assert_eq!(
        ids_and_types(&hint),
        vec![("a1".to_string(), "test.consumer".to_string())]
    );

    // Without a committed copy, the consumer's empty output is fine.
    let policies = [slot(
        "consumer",
        Recorder {
            actions: 0,
            ..Recorder::consumer("consumer")
        },
    )];
    assert!(compose_kv_hint(&policies, &inputs()).unwrap().is_none());
}

/// A test worker config that advertises `router_hint` with one source endpoint.
struct HintWorker(&'static str);

impl WorkerConfigLike for HintWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        None
    }

    fn kv_hint_transfer_metadata_for_dp_rank(
        &self,
        _dp_rank: u32,
    ) -> Option<KvHintTransferWorkerMetadata<'_>> {
        Some(KvHintTransferWorkerMetadata {
            worker_type: "prefill",
            source_control_endpoint: Some(self.0),
        })
    }
}

#[test]
fn copy_accessors_follow_each_policys_declarations() {
    struct Probe(WorkerInputs, Commitments);
    impl HintPolicy for Probe {
        fn required_worker_inputs(&self) -> WorkerInputs {
            self.0
        }
        fn commitments(&self) -> Commitments {
            self.1
        }
        fn build_hints(
            &self,
            context: &HintContext<'_>,
        ) -> Result<Vec<KvHintAction>, HintPolicyError> {
            let copy = context.committed_kv_copy().or(context.kv_copy_source());
            let Some(copy) = copy else {
                return Ok(Vec::new());
            };
            let locations = context
                .kv_copy_locations(copy)
                .ok_or_else(|| HintPolicyError::new("unreachable source"))?;
            Ok(vec![KvHintAction::unnumbered(
                "test.copy",
                "1",
                BTreeMap::from([
                    (
                        "source".to_string(),
                        serde_json::json!(locations.source_address()),
                    ),
                    (
                        "blocks".to_string(),
                        serde_json::json!(&locations.chain()[..locations.source_prefix_blocks()]),
                    ),
                    (
                        "additional".to_string(),
                        serde_json::json!(copy.additional_blocks()),
                    ),
                ]),
            )])
        }
    }

    let workers: HashMap<WorkerId, HintWorker> = HashMap::from([(2, HintWorker("tcp://w2:9000"))]);
    let candidates = KvTransferCandidates {
        block_hashes: [11, 12, 13, 14]
            .into_iter()
            .map(ExternalSequenceBlockHash)
            .collect(),
        owner_prefix_blocks: vec![(
            KvTransferCandidateSource::Worker(WorkerWithDpRank::new(2, 0)),
            3,
        )],
        source_intervals: Vec::new(),
        routing_snapshot: None,
    };
    let source = copy();
    let locations = kv_copy_locations(&workers, &candidates, &source);
    assert_eq!(locations.as_ref().map(|l| l.local_prefix_blocks()), Some(1));
    let inputs = HintInputs {
        kv_copy_source: Some(BookedCopy::new(source, locations)),
        ..inputs()
    };
    let policies = [
        HintPolicySlot::new(
            Arc::from("blind"),
            Arc::new(Probe(WorkerInputs::NONE, Commitments::NONE)),
        ),
        // Declaring a commitment alone doesn't provide the uncommitted source.
        HintPolicySlot::new(
            Arc::from("consumer"),
            Arc::new(Probe(WorkerInputs::NONE, Commitments::KV_COPY)),
        ),
        HintPolicySlot::new(
            Arc::from("reader"),
            Arc::new(Probe(WorkerInputs::KV_COPY, Commitments::NONE)),
        ),
    ];
    let hint = compose_kv_hint(&policies, &inputs)
        .unwrap()
        .expect("the reader's action");
    assert_eq!(hint.actions.len(), 1);
    let payload = &hint.actions[0].payload;
    assert_eq!(payload["source"], serde_json::json!("tcp://w2:9000"));
    assert_eq!(payload["blocks"], serde_json::json!([11, 12, 13]));
    assert_eq!(payload["additional"], serde_json::json!(2));

    // A source that left since selection has no locations.
    let gone: HashMap<WorkerId, HintWorker> = HashMap::new();
    assert!(kv_copy_locations(&gone, &candidates, &source).is_none());
}
