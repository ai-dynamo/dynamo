// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Property tests for SGLang-backed `/generate` formats.
//!
//! Each law generates requests, ground truths, chunk boundaries, topologies,
//! and failure points, runs them through [`driver`](super::driver), and
//! compares what the client receives. Failures shrink to a minimal scenario.

use std::sync::Arc;

use crate::response_merge::{Assembler, FieldClass, RuleSet};
use dynamo_llm::generate_format::{ClientRequest, Opened, SharedFormat};
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use serde_json::{Value, json};

use super::driver::{self, Outcome, Scenario, Topology};
use super::sim::Truth;
use super::{ClientShape, NativeShape, RequestSpec};

/// Request fields whose engine state a retry cannot rebuild, as
/// `(in sampling_params, name, value)`.
pub fn blocking_fields() -> Vec<(bool, &'static str, Value)> {
    vec![
        (true, "frequency_penalty", json!(0.5)),
        (true, "repetition_penalty", json!(1.2)),
        (true, "sampling_seed", json!(7)),
        (true, "json_schema", json!("{}")),
        (false, "session_params", json!({"id": "s"})),
        (false, "return_hidden_states", json!(true)),
        (false, "return_flat_raw_output_top_logprobs", json!(true)),
        (false, "return_not_yet_classified", json!(true)),
    ]
}

fn budget(n: usize, stop: bool) -> BoxedStrategy<(Option<u32>, Value)> {
    let n = n as u32;
    let preferred = move |max: u32| json!({"preferred_sampling_params": {"max_new_tokens": max}});
    if stop {
        prop_oneof![
            (0u32..4).prop_map(move |extra| (Some(n + extra), Value::Null)),
            Just((None, json!({}))),
            (0u32..4).prop_map(move |extra| (None, preferred(n + extra))),
            // Migration is blocked: the engine default is unknown.
            Just((None, Value::Null)),
        ]
        .boxed()
    } else {
        // A length finish means the budget is exactly the output.
        prop_oneof![
            Just((Some(n), Value::Null)),
            Just((Some(n), json!({}))),
            Just((None, preferred(n))),
            Just((
                None,
                json!({"preferred_sampling_params": format!(r#"{{"max_new_tokens": {n}}}"#)})
            )),
        ]
        .boxed()
    }
}

/// Requests, ground truths, chunking, topology, and failures.
pub fn scenarios() -> impl Strategy<Value = Scenario> {
    (1usize..6, 1usize..10, any::<bool>(), 1usize..4)
        .prop_flat_map(|(p, n, stop, row_bytes)| {
            (
                (
                    vec(1u32..100_000, p),
                    vec(-64i32..0, p),
                    vec((1u32..100_000, -64i32..0), n),
                    vec(any::<u8>(), (p + n + 4) * row_bytes),
                    vec(any::<bool>(), n - 1),
                    0u32..4,
                ),
                (
                    budget(n, stop),
                    proptest::option::of(0..=n as u32),
                    any::<bool>(),
                    any::<bool>(),
                    prop_oneof![Just(None), Just(Some(-1i64)), (0..p as i64).prop_map(Some)],
                    0u32..3,
                    any::<bool>(),
                ),
                prop_oneof![Just(Topology::Aggregated), Just(Topology::Disaggregated)],
                vec(any::<Index>(), 0..4),
                any::<bool>(),
                Just((stop, row_bytes)),
            )
        })
        .prop_map(
            |(
                (prompt, prompt_lps, output, experts, cuts, cached),
                ((max, engine_info), min, stream, logprob, start, top, routed),
                topology,
                failures,
                lose_terminal,
                (stop, row_bytes),
            )| {
                let n = output.len();
                let mut ends: Vec<usize> = (1..n).filter(|&i| cuts[i - 1]).collect();
                ends.push(n);
                let mut failures: Vec<usize> =
                    failures.iter().map(|i| i.index(ends.len())).collect();
                failures.sort_unstable();
                let prompt_logprobs = prompt_lps.iter().map(|&lp| f64::from(lp) / 8.0).collect();
                Scenario {
                    truth: Truth {
                        prompt: prompt.clone(),
                        prompt_logprobs,
                        output: output
                            .into_iter()
                            .map(|(id, lp)| (id, f64::from(lp) / 8.0))
                            .collect(),
                        stop,
                        row_bytes,
                        experts,
                        cuts: ends,
                        cached_tokens: cached,
                    },
                    request: RequestSpec {
                        input_ids: prompt,
                        stream,
                        max_new_tokens: max,
                        min_new_tokens: min,
                        return_logprob: logprob,
                        logprob_start_len: start,
                        top_logprobs_num: top,
                        return_routed_experts: routed,
                        ..Default::default()
                    },
                    topology,
                    engine_info,
                    failures,
                    lose_terminal: lose_terminal && !stop,
                }
            },
        )
}

/// Arbitrary JSON, nested a few levels, biased toward SGLang's field names.
pub fn json_values() -> impl Strategy<Value = Value> {
    let key = prop_oneof![
        Just("output_ids".to_string()),
        Just("text".to_string()),
        Just("meta_info".to_string()),
        Just("finish_reason".to_string()),
        Just("routed_experts".to_string()),
        Just("input_token_logprobs".to_string()),
        Just("completion_tokens".to_string()),
        Just("error".to_string()),
        "[a-z_]{1,8}",
    ];
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        any::<f64>().prop_map(|f| json!(f)),
        "[A-Za-z0-9+/=é]{0,12}".prop_map(Value::from),
    ];
    leaf.prop_recursive(4, 48, 6, move |inner| {
        prop_oneof![
            vec(inner.clone(), 0..6).prop_map(Value::Array),
            proptest::collection::btree_map(key.clone(), inner, 0..6)
                .prop_map(|map| Value::Object(map.into_iter().collect())),
        ]
    })
}

/// Runs the laws for one format and client shape.
pub struct Harness {
    pub format: SharedFormat,
    pub shape: Arc<dyn ClientShape>,
    pub cases: u32,
    pub worker: Arc<dyn super::sim::Worker>,
}

/// The parts of an outcome a client can observe.
fn observed(outcome: &Outcome) -> (Vec<Value>, Option<Value>, Option<&str>) {
    match outcome {
        Outcome::Rejected(reason) => (Vec::new(), None, Some(reason)),
        Outcome::Delivered { chunks, unary } => (chunks.clone(), unary.clone(), None),
        Outcome::Failed { chunks, reason } => (chunks.clone(), None, Some(reason)),
    }
}

/// What a synthetic finish must still get right: the output, its counts,
/// identity, and termination. It ends an attempt the engine never ran, so it
/// has no per-attempt or final-chunk values.
fn essentials(rules: &RuleSet, value: &Value) -> Value {
    let mut kept = json!({});
    for (path, class) in rules.fields() {
        let keep = matches!(class, FieldClass::Delta | FieldClass::OutputOffset)
            || path == "/meta_info/id"
            || path == "/meta_info/finish_reason";
        if let Some(found) = value.pointer(path).filter(|_| keep) {
            let mut target = &mut kept;
            let (parent, key) = path.rsplit_once('/').expect("pointer");
            for segment in parent.split('/').skip(1) {
                target = target
                    .as_object_mut()
                    .expect("object")
                    .entry(segment)
                    .or_insert_with(|| json!({}));
            }
            if let Some(object) = target.as_object_mut() {
                object.insert(key.to_string(), found.clone());
            }
        }
    }
    kept
}

fn join(rules: &Arc<RuleSet>, chunks: &[Value]) -> Option<Value> {
    let mut assembler = Assembler::new(rules.clone(), false);
    chunks
        .iter()
        .cloned()
        .for_each(|chunk| assembler.push(chunk));
    assembler.finish()
}

impl Harness {
    pub fn native(format: SharedFormat) -> Self {
        Self {
            format,
            shape: Arc::new(NativeShape),
            cases: 512,
            worker: Arc::new(super::sim::NativeWorker),
        }
    }

    pub fn check_all(&self) {
        self.stream_join_equals_unary();
        self.disaggregated_equals_aggregated();
        self.migration_is_invisible();
        self.blocked_requests_fail_cleanly();
        self.malformed_responses_never_panic();
    }

    fn law<S: Strategy>(
        &self,
        name: &str,
        strategy: S,
        test: impl Fn(S::Value) -> Result<(), TestCaseError>,
    ) where
        S::Value: std::fmt::Debug,
    {
        let mut runner = TestRunner::new(Config {
            cases: self.cases,
            max_global_rejects: self.cases * 64,
            failure_persistence: None,
            ..Config::default()
        });
        runner
            .run(&strategy, test)
            .unwrap_or_else(|error| panic!("law `{name}`: {error}"));
    }

    fn outcome(&self, scenario: &Scenario) -> Result<Outcome, TestCaseError> {
        match driver::run(
            self.format.as_ref(),
            self.shape.as_ref(),
            scenario,
            self.worker.as_ref(),
        ) {
            None => Err(TestCaseError::reject("the client cannot express it")),
            Some(Err(violation)) => Err(TestCaseError::fail(violation)),
            Some(Ok(outcome)) => Ok(outcome),
        }
    }

    fn migratable(&self, scenario: &Scenario) -> bool {
        let Some(body) = self.shape.body(&scenario.request) else {
            return false;
        };
        self.format
            .open(ClientRequest {
                body: &body,
                headers: &axum::http::HeaderMap::from_iter([(
                    axum::http::header::CONTENT_TYPE,
                    axum::http::HeaderValue::from_static("application/json"),
                )]),
                default_request_id: "dynamo-request",
            })
            .is_ok_and(|opened| {
                let worker = super::worker(&scenario.engine_info);
                opened.validation.is_ok()
                    && (opened.bind)(&worker).is_ok_and(|bound| bound.migration.is_ok())
            })
    }

    /// A client that joins the streamed chunks gets the unary response.
    pub fn stream_join_equals_unary(&self) {
        let Some(rules) = self.shape.client_rules() else {
            return;
        };
        self.law("stream join equals unary", scenarios(), |scenario| {
            let mut streamed = scenario.clone();
            streamed.request.stream = true;
            let mut unary = scenario;
            unary.request.stream = false;
            match (self.outcome(&streamed)?, self.outcome(&unary)?) {
                (Outcome::Delivered { chunks, .. }, Outcome::Delivered { unary, .. }) => {
                    prop_assert_eq!(join(&rules, &chunks), unary);
                }
                (streamed, unary) => {
                    prop_assert_eq!(observed(&streamed).2, observed(&unary).2);
                }
            }
            Ok(())
        });
    }

    /// Prefill/decode disaggregation returns what aggregated serving returns.
    pub fn disaggregated_equals_aggregated(&self) {
        self.law("P/D equals aggregated", scenarios(), |mut scenario| {
            // A synthetic finish carries no final-chunk values in either.
            scenario.lose_terminal = false;
            let mut aggregated = scenario.clone();
            aggregated.topology = Topology::Aggregated;
            let mut disaggregated = scenario;
            disaggregated.topology = Topology::Disaggregated;
            let aggregated = self.outcome(&aggregated)?;
            let disaggregated = self.outcome(&disaggregated)?;
            prop_assert_eq!(observed(&disaggregated), observed(&aggregated));
            Ok(())
        });
    }

    /// A migrated request delivers what the unmigrated one does, or fails
    /// only when its plan said it cannot migrate, after a prefix of it.
    pub fn migration_is_invisible(&self) {
        let rules = self.shape.client_rules();
        self.law("migration is invisible", scenarios(), |scenario| {
            prop_assume!(!scenario.failures.is_empty() || scenario.lose_terminal);
            let baseline = self.outcome(&scenario.without_failures())?;
            let migrated = self.outcome(&scenario)?;
            let (base_chunks, base_unary, base_error) = observed(&baseline);
            prop_assert!(base_error.is_none() || matches!(baseline, Outcome::Rejected(_)));
            match &migrated {
                Outcome::Rejected(_) => prop_assert_eq!(&migrated, &baseline),
                Outcome::Failed { chunks, reason } => {
                    prop_assert!(
                        !self.migratable(&scenario),
                        "a migratable request failed: {}",
                        reason
                    );
                    prop_assert!(base_chunks.starts_with(chunks));
                }
                Outcome::Delivered { chunks, unary } if scenario.lose_terminal => {
                    // The final attempt is a synthetic finish.
                    if let Some(n) = base_chunks.len().checked_sub(1) {
                        prop_assert_eq!(chunks.len(), n + 2);
                        prop_assert_eq!(&chunks[..n], &base_chunks[..n]);
                    }
                    if let Some(rules) = &rules {
                        let essential = |v: Option<Value>| v.map(|v| essentials(rules, &v));
                        prop_assert_eq!(
                            essential(join(rules, chunks)),
                            essential(join(rules, &base_chunks))
                        );
                        prop_assert_eq!(essential(unary.clone()), essential(base_unary));
                    }
                }
                Outcome::Delivered { chunks, unary } => {
                    prop_assert_eq!(chunks, &base_chunks);
                    prop_assert_eq!(unary, &base_unary);
                }
            }
            Ok(())
        });
    }

    /// A request whose engine state a retry cannot rebuild fails after an
    /// attempt fails, instead of continuing with wrong output.
    pub fn blocked_requests_fail_cleanly(&self) {
        // A client that cannot send these fields cannot misuse them.
        let expressible = blocking_fields()
            .into_iter()
            .any(|(sampling, name, value)| {
                let mut spec = RequestSpec {
                    input_ids: vec![1],
                    ..Default::default()
                };
                let target = if sampling {
                    &mut spec.extra_sampling
                } else {
                    &mut spec.extra
                };
                target.insert(name.to_string(), value);
                self.shape.body(&spec).is_some()
            });
        if !expressible {
            return;
        }
        let strategy = (scenarios(), any::<Index>(), any::<Index>(), any::<bool>());
        self.law(
            "blocked requests fail cleanly",
            strategy,
            |(scenario, field, at, engine_default)| {
                let fields = blocking_fields();
                let (sampling, name, value) = fields[field.index(fields.len())].clone();
                let mut scenario = scenario;
                if sampling && engine_default {
                    if !scenario.engine_info.is_object() {
                        scenario.engine_info = json!({});
                    }
                    scenario.engine_info["preferred_sampling_params"] = json!({name: value});
                } else {
                    let target = if sampling {
                        &mut scenario.request.extra_sampling
                    } else {
                        &mut scenario.request.extra
                    };
                    target.insert(name.to_string(), value);
                }
                scenario.failures = vec![at.index(scenario.truth.cuts.len())];
                scenario.lose_terminal = false;
                prop_assume!(self.shape.body(&scenario.request).is_some());
                prop_assert!(!self.migratable(&scenario), "`{}` allows migration", name);
                match self.outcome(&scenario)? {
                    Outcome::Failed { .. } | Outcome::Rejected(_) => Ok(()),
                    outcome => Err(TestCaseError::fail(format!(
                        "`{name}` continued after a failure: {outcome:?}"
                    ))),
                }
            },
        );
    }

    /// Malformed native responses are errors, never panics.
    pub fn malformed_responses_never_panic(&self) {
        let strategy = (scenarios(), vec(json_values(), 1..4), any::<bool>());
        self.law(
            "malformed responses never panic",
            strategy,
            |(scenario, values, prefill)| {
                for is_prefill in [false, true] {
                    for value in &values {
                        let _ = crate::inspect(value.clone(), is_prefill);
                    }
                }
                let Some(body) = self.shape.body(&scenario.request) else {
                    return Ok(());
                };
                let Ok(Opened {
                    bind, validation, ..
                }) = self.format.open(ClientRequest {
                    body: &body,
                    headers: &axum::http::HeaderMap::from_iter([(
                        axum::http::header::CONTENT_TYPE,
                        axum::http::HeaderValue::from_static("application/json"),
                    )]),
                    default_request_id: "dynamo-request",
                })
                else {
                    return Ok(());
                };
                if validation.is_err() {
                    return Ok(());
                }
                let Ok(bound) = bind(&super::worker(&scenario.engine_info)) else {
                    return Ok(());
                };
                let mut request = bound.request;
                let mut exchange = bound.exchange;
                for value in values {
                    let output = LLMEngineOutput {
                        engine_data: Some(json!({ crate::RESPONSE_KEY: value })),
                        ..Default::default()
                    };
                    let _ = exchange.chunk(output, prefill, &mut |_| {});
                }
                let _ = exchange.migrate(0, &mut request);
                let _ = exchange.finish();
                Ok(())
            },
        );
    }
}
