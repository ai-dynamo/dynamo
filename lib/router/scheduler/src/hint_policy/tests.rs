// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::runtime::{BookedCopy, HintInputs, HintPolicySlot, compose_kv_hint};
use super::*;
use crate::kv_hints::{
    KvCopySource, KvTransferCandidateSource, KvTransferCandidates, KvTransferSourceChoice,
    MAX_KV_HINT_ACTIONS, kv_copy_locations,
};
use crate::protocols::WorkerConfigLike;
use crate::resolve::{PluginConfigError, resolve_role};
use dynamo_kv_hashing::ExternalSequenceBlockHash;
use dynamo_kv_hints::KvHintTransferWorkerMetadata;
use dynamo_router_protocol::identity::RoutingPartitionId;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_worker_selection::{
    WorkerCandidates, WorkerInputView, WorkerPicker, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

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

#[test]
fn hosts_cannot_list_a_policy_twice() {
    let constructor = || HintPolicyConstructor::shared(Recorder::new("record"));
    assert!(matches!(
        HintPolicyFactory::for_all_roles([
            ("same".to_string(), constructor()),
            ("same".to_string(), constructor()),
        ]),
        Err(DuplicateHintPolicyName { .. })
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
    let config = SelectionConfig::default();
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
        SelectionConfig::default(),
        "prefill",
        vec![Box::new(InputsScorer(inputs))],
        Box::new(FirstPicker),
    )
}

fn role_policies(policies: Vec<(&str, Recorder)>, defaults: bool) -> RoleHintPolicies {
    let policies = policies
        .into_iter()
        .map(|(name, policy)| (Arc::from(name), HintPolicyConstructor::shared(policy)))
        .collect();
    let partition = RoutingPartitionId::new("model", "default");
    HintPolicyFactory::from_roles([(WorkerType::Prefill, policies, defaults)]).instances(
        &SelectionConfig::default(),
        WorkerType::Prefill,
        partition.as_ref(),
    )
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
        true,
    )
    .unwrap();
    let order: Vec<_> = responsibilities.hint_policy_names().collect();
    assert_eq!(order, ["consumer", "first", "last"]);
    assert!(!responsibilities.commit_kv_copy);
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
        true,
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
        false,
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
        false,
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
            true,
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
        true,
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
        true,
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
        true,
    )
    .unwrap();
    assert_eq!(defaulted.prepare_inputs(), WorkerInputs::LOAD);

    let none = resolve_role(
        WorkerType::Prefill,
        &selection(WorkerInputs::LOAD),
        RoleHintPolicies::none(),
        true,
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
        true,
    )
    .unwrap();
    assert!(responsibilities.commit_kv_copy);
    assert!(responsibilities.compute_booked_copy_source);
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
        true,
    )
    .unwrap();

    // Committed copies that nothing sends are rejected, even with other hint policies.
    for policies in [
        RoleHintPolicies::none(),
        role_policies(vec![("other", Recorder::new("other"))], false),
    ] {
        let error = resolve_role(WorkerType::Prefill, &committing(), policies, true).unwrap_err();
        assert!(
            matches!(error, PluginConfigError::UnsentCopies { role: "prefill" }),
            "{error}"
        );
        assert!(
            error.to_string().contains("Commitments::KV_COPY"),
            "{error}"
        );
    }

    // Copy offers need an index that keeps the block chain.
    let error = resolve_role(
        WorkerType::Prefill,
        &committing(),
        role_policies(vec![("consumer", Recorder::consumer("consumer"))], false),
        false,
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

fn router_fetch() -> KvHintAction {
    KvHintAction::fetch(
        "ignored",
        crate::kv_hints::KvSourceLocationsPayload {
            source_control_endpoint: "tcp://worker-2:9000".to_string(),
            block_hashes: vec![ExternalSequenceBlockHash(7)],
        },
    )
}

fn ids_and_types(hint: &crate::kv_hints::KvHint) -> Vec<(String, String)> {
    hint.actions
        .iter()
        .map(|action| (action.action_id.clone(), action.action_type.clone()))
        .collect()
}

#[test]
fn actions_are_numbered_in_hint_order_after_the_router_fetch() {
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
    let hint = compose_kv_hint(&policies, &inputs(), Some(router_fetch()))
        .unwrap()
        .expect("hint");
    assert_eq!(hint.message_id, "request-1");
    let pairs: Vec<_> = ids_and_types(&hint)
        .into_iter()
        .map(|(id, kind)| format!("{id}:{kind}"))
        .collect();
    assert_eq!(
        pairs,
        ["a1:kv.fetch", "a2:test.two", "a3:test.two", "a4:test.one"]
    );
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
    let hint = compose_kv_hint(&policies, &inputs(), None)
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
    assert!(
        compose_kv_hint(&policies, &inputs, Some(router_fetch()))
            .unwrap()
            .is_none()
    );
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
    assert!(
        compose_kv_hint(&policies, &inputs(), None)
            .unwrap()
            .is_none()
    );
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
        let error = compose_kv_hint(&policies, &committed(), None).unwrap_err();
        assert!(error.to_string().contains("\"consumer\""), "{error}");
    }
    // No policy sends committed copies.
    let policies = [slot("other", Recorder::new("other"))];
    assert!(compose_kv_hint(&policies, &committed(), None).is_err());

    // Any action counts as delivery; the stage never reads payloads.
    let policies = [slot("consumer", Recorder::consumer("consumer"))];
    let hint = compose_kv_hint(&policies, &committed(), None)
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
    assert!(
        compose_kv_hint(&policies, &inputs(), None)
            .unwrap()
            .is_none()
    );
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
    let hint = compose_kv_hint(&policies, &inputs, None)
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
