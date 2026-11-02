// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend, prefill router, migration, and worker around one format.
//!
//! The driver does with a format's values what those components do, and checks
//! the contract it keeps with them: the routed request carries the client's
//! prompt, and `inspect` reports exactly the delivered tokens and one terminal
//! per attempt.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::sim::{self, Role, Truth};
use super::{ClientShape, RequestSpec};
use crate::PAYLOAD_KEY;
use dynamo_llm::generate_format::{ClientRequest, GenerateFormat, Opened};
use dynamo_llm::protocols::response_merge::{
    PREFILL_OUTPUT_ANNOTATION, PREFILL_OUTPUT_FIRST_ANNOTATION,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Aggregated,
    Disaggregated,
}

#[derive(Debug, Clone)]
pub struct Scenario {
    pub truth: Truth,
    pub request: RequestSpec,
    pub topology: Topology,
    pub engine_info: Value,
    /// Attempts fail when this many chunks have been delivered in total;
    /// sorted, and a repeat fails a retry before its first chunk.
    pub failures: Vec<usize>,
    /// The final chunk arrives without its terminal, then the attempt fails.
    pub lose_terminal: bool,
}

impl Scenario {
    pub fn without_failures(&self) -> Self {
        Self {
            failures: Vec::new(),
            lose_terminal: false,
            ..self.clone()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The format rejected the request.
    Rejected(String),
    /// The client received `chunks` (streaming) or `unary`.
    Delivered {
        chunks: Vec<Value>,
        unary: Option<Value>,
    },
    /// An attempt failed and the format refused to continue, after `chunks`.
    Failed { chunks: Vec<Value>, reason: String },
}

/// What the worker sends SGLang: the engine body and the routing it owns.
fn worker_body(body: &Map<String, Value>, attempt: usize, prefill: bool) -> Map<String, Value> {
    let mut body = body.clone();
    body.insert("rid".into(), json!(format!("worker-private-{attempt}")));
    if prefill {
        let sampling = body.entry("sampling_params").or_insert(Value::Null);
        if !sampling.is_object() {
            *sampling = json!({});
        }
        let sampling = sampling.as_object_mut().expect("object");
        sampling.extend([("n".into(), json!(1)), ("max_new_tokens".into(), json!(1))]);
        sampling.remove("min_new_tokens");
    }
    body
}

/// `None` when the client shape cannot express the scenario's request; the
/// inner `Err` is a broken contract.
pub fn run(
    format: &dyn GenerateFormat,
    shape: &dyn ClientShape,
    scenario: &Scenario,
    worker: &dyn sim::Worker,
) -> Option<Result<Outcome, String>> {
    let body = shape.body(&scenario.request)?;
    Some(drive(format, &body, scenario, worker))
}

fn drive(
    format: &dyn GenerateFormat,
    body: &[u8],
    scenario: &Scenario,
    worker: &dyn sim::Worker,
) -> Result<Outcome, String> {
    let truth = &scenario.truth;
    let Opened {
        bind, validation, ..
    } = match format.open(ClientRequest {
        body,
        headers: &axum::http::HeaderMap::from_iter([(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        )]),
        default_request_id: "dynamo-request",
    }) {
        Ok(opened) => opened,
        Err(error) => return Ok(Outcome::Rejected(error.message)),
    };
    if let Err(error) = validation {
        return Ok(Outcome::Rejected(error.message));
    }
    let bound = match bind(&super::worker(&scenario.engine_info)) {
        Ok(bound) => bound,
        Err(error) => return Ok(Outcome::Rejected(error.message)),
    };
    let mut request = bound.request;
    let mut exchange = bound.exchange;
    if *request.token_ids != truth.prompt {
        return Err("the routed request does not carry the client's prompt".into());
    }
    let annotated = |name| request.annotations.iter().any(|a| a == name);
    let (wants_prefill, prefill_first) = (
        annotated(PREFILL_OUTPUT_ANNOTATION),
        annotated(PREFILL_OUTPUT_FIRST_ANNOTATION),
    );
    let disaggregated = scenario.topology == Topology::Disaggregated;
    let role = match scenario.topology {
        Topology::Disaggregated => Role::Decode,
        Topology::Aggregated => Role::Aggregated,
    };
    let mut client: Vec<Value> = Vec::new();
    let mut failures = scenario.failures.iter().copied().peekable();
    let mut lose_terminal = scenario.lose_terminal;
    let (mut sent, mut attempt, mut delivered) = (0usize, 0usize, 0usize);
    loop {
        let body = request
            .extra_args
            .as_ref()
            .and_then(|e| e[PAYLOAD_KEY].as_object());
        let body = body
            .ok_or("the routed request carries no engine body")?
            .clone();
        let info = &scenario.engine_info;
        let mut prefill = None;
        if disaggregated {
            let response = worker
                .stream(sim::Request {
                    truth,
                    body: &worker_body(&body, attempt, true),
                    prompt: &request.token_ids,
                    role: Role::Prefill,
                    engine_info: info,
                })
                .pop()
                .ok_or("prefill sent no response")?;
            let inspected = crate::inspect(response, true)
                .map_err(|e| format!("inspect rejected prefill's response: {e}"))?;
            if !inspected.output.token_ids.is_empty() || !inspected.terminal {
                return Err("prefill's final response must report no tokens and a terminal".into());
            }
            prefill = wants_prefill.then_some(inspected.output);
        }
        // A failure before the first chunk loses prefill's output too.
        if prefill_first
            && failures.peek() != Some(&sent)
            && let Some(output) = prefill.take()
        {
            exchange
                .chunk(output, true, &mut |_| {})
                .map_err(|e| e.to_string())?;
        }
        let chunks = worker.stream(sim::Request {
            truth,
            body: &worker_body(&body, attempt, false),
            prompt: &request.token_ids,
            role,
            engine_info: info,
        });
        let count = chunks.len();
        let mut failed = false;
        for (index, mut chunk) in chunks.into_iter().enumerate() {
            if failures.next_if_eq(&sent).is_some() {
                failed = true;
                break;
            }
            let last = index + 1 == count;
            if last && let Some(output) = prefill.take() {
                exchange
                    .chunk(output, true, &mut |_| {})
                    .map_err(|e| e.to_string())?;
            }
            let lost = last && lose_terminal && failures.peek().is_none();
            if lost {
                chunk["meta_info"]["finish_reason"] = Value::Null;
                lose_terminal = false;
            }
            let ids: Vec<u32> =
                serde_json::from_value(chunk["output_ids"].clone()).map_err(|e| e.to_string())?;
            let inspected = crate::inspect(chunk, false)
                .map_err(|e| format!("inspect rejected a response: {e}"))?;
            if inspected.output.token_ids != ids || inspected.terminal != (last && !lost) {
                return Err(format!(
                    "inspect misread chunk {index} of {count}: {:?}",
                    inspected.output
                ));
            }
            Arc::make_mut(&mut request.token_ids).extend(&ids);
            delivered += ids.len();
            exchange
                .chunk(inspected.output, false, &mut |value| {
                    client.push(value.clone())
                })
                .map_err(|e| e.to_string())?;
            sent += 1;
            if lost {
                failed = true;
                break;
            }
        }
        if !failed {
            break;
        }
        attempt += 1;
        if attempt > 16 {
            return Err("too many attempts".to_string());
        }
        match exchange.migrate(delivered, &mut request) {
            Ok(None) => {}
            Ok(Some(output)) => {
                exchange
                    .chunk(output, false, &mut |value| client.push(value.clone()))
                    .map_err(|e| e.to_string())?;
                break;
            }
            Err(reason) => {
                return Ok(Outcome::Failed {
                    chunks: client,
                    reason,
                });
            }
        }
    }
    let unary = exchange.finish().map_err(|e| e.to_string())?;
    Ok(Outcome::Delivered {
        chunks: client,
        unary,
    })
}
