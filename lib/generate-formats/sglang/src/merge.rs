// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang `/generate` as a [`GenerateFormat`].
//!
//! With incremental streaming, each chunk holds only the new part of `text`,
//! `output_ids` and the output-side lists; prompt-side values repeat in full
//! on every chunk, and routed experts arrive on the final one. In P/D decode
//! never runs the prompt: input logprobs come from prefill, and prefill's
//! `routed_experts` rows replace decode's stale prompt rows
//! (sgl-project/sglang#44002).
//!
//! The rules live in `merge_rules.json` and mirror SGLang code. When bumping
//! SGLang, re-check these upstream sources and update the file:
//!
//! - `python/sglang/srt/managers/tokenizer_manager.py`:
//!   `_INCREMENTAL_STREAMING_META_INFO_KEYS` and `_coalesce_streaming_chunks`
//!   (`delta`), `_handle_batch_output` and `add_logprob_to_meta_info` (every
//!   `meta_info` key), `_build_flat_input_top_logprobs_fields*` (flat input
//!   top-logprob keys).
//! - `python/sglang/srt/managers/io_struct.py`: `GenerateReqInput` (the
//!   `return_*` request flags).
//! - `python/sglang/srt/managers/scheduler_components/output_streamer.py`:
//!   decode servers send no input logprobs (`prompt`).
//! - `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
//!   (`_maybe_collect_routed_experts`) and `disaggregation/prefill.py`: which
//!   server reports which `routed_experts` rows (`sequence_final`).
//! - `python/sglang/srt/sampling/sampling_params.py`: the `max_new_tokens`
//!   default a migrated request continues from.
//!
//! `components/src/dynamo/sglang/tests/test_sglang_generate_merge_contract.py`
//! checks the file against the installed SGLang and fails on unclassified
//! fields.

use std::sync::{Arc, LazyLock};

use dynamo_runtime::error::DynamoError;
use serde_json::{Value, json};

use super::generate::SglangGenerateRequest;
use super::migrate::{Continuation, RequestRules};
use super::{CAPABILITY, PAYLOAD_KEY, RESPONSE_KEY};
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::response_merge::{Assembler, RuleSet};
use dynamo_llm::generate_format::{
    Bound, ClientError, ClientRequest, GenerateExchange, GenerateFormat, Opened,
};
use dynamo_llm::protocols::common::extensions::HEADER_REQUEST_PRIORITY;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::protocols::common::preprocessor::RoutingHints;
use dynamo_llm::protocols::common::{FinishReason, OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::protocols::response_merge::prefill_annotations;

/// Shared with the SGLang contract test.
pub(super) const RULES_JSON: &str = include_str!("merge_rules.json");
const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

static RULES: LazyLock<Arc<RuleSet>> =
    LazyLock::new(|| Arc::new(RuleSet::from_json(RULES_JSON).expect("valid merge_rules.json")));
pub(super) static REQUEST_RULES: LazyLock<RequestRules> =
    LazyLock::new(|| serde_json::from_str(RULES_JSON).expect("valid merge_rules.json"));

/// The response join table.
pub fn rules() -> Arc<RuleSet> {
    RULES.clone()
}

/// The client-facing half of the format; the default is SGLang's own body
/// and response. Formats that only reshape what clients send and receive
/// implement this and keep SGLang's joining, P/D, and migration.
pub trait ClientCodec: Send + Sync + 'static {
    /// Response field semantics; extend the native rules to add engine fields.
    fn rules(&self) -> Arc<RuleSet> {
        rules()
    }

    /// Join a sequence value at the P/D seam. Aggregated values pass through.
    /// Return false to omit an unusable field rather than expose stale rows.
    fn overlay_sequence(&self, _path: &str, prefix: &Value, target: &mut Value) -> bool {
        match (prefix, target) {
            (Value::String(prefix), Value::String(target)) => {
                crate::response_merge::overlay_base64_prefix(prefix, target)
            }
            _ => false,
        }
    }

    /// The client's body as an SGLang `/generate` body; `None` when it is one.
    fn decode_request(&self, _body: &[u8]) -> Result<Option<Value>, ClientError> {
        Ok(None)
    }

    /// The client response for one joined SGLang response: a streamed chunk,
    /// or the unary response. Invalid backend data may return an error; the
    /// frontend applies its normal HTTP/SSE error handling.
    fn encode_response(&self, _response: &Value) -> Result<Encoded, DynamoError> {
        Ok(Encoded::Native)
    }

    fn error_body(&self, message: &str) -> Value {
        json!({"error": {"message": message}})
    }

    /// Malformed JSON or request fields; defaults to the codec's error shape.
    fn parse_error_body(&self, message: &str) -> Value {
        self.error_body(message)
    }
}

/// What a [`ClientCodec`] sends for one SGLang response.
pub enum Encoded {
    /// SGLang's response unchanged, without a copy.
    Native,
    Replace(Value),
    /// Nothing, for a streamed chunk.
    Skip,
}

/// SGLang's native client shape.
pub struct Native;

impl ClientCodec for Native {
    fn parse_error_body(&self, message: &str) -> Value {
        json!({
            "object": "error", "message": message, "type": "Bad Request",
            "param": null, "code": 400,
        })
    }
}

pub struct SglangFormat<C: ClientCodec = Native> {
    name: String,
    codec: Arc<C>,
    rules: Arc<RuleSet>,
}

impl SglangFormat {
    pub fn native() -> Self {
        Self::with_codec(super::NAME, Native)
    }
}

impl<C: ClientCodec> SglangFormat<C> {
    fn parse_error(&self, message: String) -> ClientError {
        ClientError {
            status: StatusCode::BAD_REQUEST,
            response: Some(Box::new(
                (
                    StatusCode::BAD_REQUEST,
                    axum::Json(self.codec.parse_error_body(&message)),
                )
                    .into_response(),
            )),
            message,
        }
    }

    pub fn with_codec(name: impl Into<String>, codec: C) -> Self {
        Self {
            name: name.into(),
            rules: codec.rules(),
            codec: Arc::new(codec),
        }
    }
}

impl<C: ClientCodec> GenerateFormat for SglangFormat<C> {
    fn name(&self) -> &str {
        &self.name
    }

    fn capability(&self) -> &'static str {
        CAPABILITY
    }

    fn open(&self, client: ClientRequest<'_>) -> Result<Opened, ClientError> {
        let request: SglangGenerateRequest = match self.codec.decode_request(client.body)? {
            None => client
                .json()
                .map_err(|error| self.parse_error(error.body_text())),
            Some(body) => serde_json::from_value(body).map_err(|error| {
                self.parse_error(format!("Failed to deserialize the JSON body: {error}"))
            }),
        }?;
        let validation = request.validate().map_err(ClientError::new);
        let input_tokens = request.input_ids.len();
        let header = |name| client.headers.get(name).and_then(|v| v.to_str().ok());
        let request_id = request
            .rid
            .clone()
            .or_else(|| header(X_REQUEST_ID_HEADER).map(ToOwned::to_owned))
            .unwrap_or_else(|| client.default_request_id.to_string());
        let dp_rank = header(X_DATA_PARALLEL_RANK_HEADER).and_then(|v| v.trim().parse().ok());
        let limits = (|| {
            Ok::<_, String>(StopConditions {
                max_tokens: request.max_new_tokens()?,
                min_tokens: request.min_new_tokens()?,
                ignore_eos: request.ignore_eos()?,
                ..Default::default()
            })
        })()
        .map_err(ClientError::new);
        // Dynamo's priority header uses the same higher-is-higher scale as
        // SGLang's request field. A valid header wins; malformed values fall
        // back to the body field.
        let priority = header(HEADER_REQUEST_PRIORITY)
            .and_then(|value| value.trim().parse().ok())
            .or(request.priority)
            .unwrap_or_default();
        let cache_namespace = request.cache_namespace().map(str::to_owned);
        let stream = request.stream;
        let (input_ids, body) = request.into_worker_envelope(&request_id);
        let rules = self.rules.clone();
        let prefill_need = rules.prefill_need(&body);
        let prompt_tokens = input_ids.len();
        let codec = self.codec.clone();
        Ok(Opened {
            model: None,
            request_id: request_id.clone(),
            stream,
            input_tokens,
            validation,
            bind: Box::new(move |worker| {
                let continuation = Continuation::new(&REQUEST_RULES, &body, worker.engine_info);
                let request = PreprocessedRequest::builder()
                    .model(worker.model.to_string())
                    .token_ids(input_ids)
                    .stop_conditions(limits?)
                    .sampling_options(SamplingOptions {
                        n: Some(1),
                        ..Default::default()
                    })
                    .output_options(OutputOptions {
                        return_tokens_as_token_ids: Some(true),
                        ..Default::default()
                    })
                    .routing(Some(RoutingHints {
                        dp_rank,
                        priority_jump: Some(priority.max(0) as f64),
                        priority: Some(priority),
                        cache_namespace,
                        ..Default::default()
                    }))
                    .extra_args(Some(json!({ PAYLOAD_KEY: body })))
                    .annotations(prefill_annotations(prefill_need))
                    .build()
                    .map_err(|error| ClientError::new(error.to_string()))?;
                Ok(Bound {
                    request,
                    migration: continuation.as_ref().map(|_| ()).map_err(Clone::clone),
                    exchange: Box::new(Exchange {
                        assembler: Assembler::with_codec(
                            rules,
                            continuation.is_ok(),
                            codec.clone(),
                        ),
                        continuation,
                        codec,
                        delivered: 0,
                        request_id,
                        stream,
                        prompt_tokens,
                    }),
                })
            }),
        })
    }

    fn ambiguous_model_message(&self) -> &'static str {
        "multiple SGLang models are registered; configure a model-specific generate endpoint"
    }

    fn body_error(
        &self,
        error: axum::extract::rejection::BytesRejection,
        headers: &axum::http::HeaderMap,
    ) -> ClientError {
        self.parse_error(dynamo_llm::generate_format::json_body_error(headers, error).body_text())
    }

    fn worker_error(
        &self,
        error: &(dyn std::error::Error + 'static),
        phase: dynamo_llm::generate_format::ErrorPhase,
    ) -> dynamo_llm::generate_format::WorkerError {
        crate::errors::classify(error, phase)
    }

    fn error_body(&self, _status: StatusCode, message: &str) -> Value {
        self.codec.error_body(message)
    }
}

struct Exchange<C: ClientCodec> {
    codec: Arc<C>,
    assembler: Assembler<C>,
    continuation: Result<Continuation, String>,
    /// Generated tokens seen across attempts.
    delivered: usize,
    request_id: String,
    stream: bool,
    prompt_tokens: usize,
}

impl<C: ClientCodec> GenerateExchange for Exchange<C> {
    fn chunk(
        &mut self,
        output: LLMEngineOutput,
        prefill: bool,
        emit: &mut dyn FnMut(&Value),
    ) -> Result<(), DynamoError> {
        let response = output
            .engine_data
            .and_then(|mut data| data.as_object_mut()?.remove(RESPONSE_KEY));
        let mut response = match response {
            Some(response) => response,
            // An older prefill worker sends none; joining nothing drops stale fields.
            None if prefill => Value::Object(Default::default()),
            None => return Err(DynamoError::msg("missing opaque SGLang response")),
        };
        if prefill {
            self.assembler.set_prefill(response);
            return Ok(());
        }
        self.delivered += response
            .get("output_ids")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        // Workers submit a private SGLang request ID; the client sees its own.
        if let Some(meta) = response.get_mut("meta_info").and_then(Value::as_object_mut)
            && meta.contains_key("id")
        {
            meta.insert("id".into(), Value::String(self.request_id.clone()));
        }
        if !self.stream {
            self.assembler.push(response);
            return Ok(());
        }
        let codec = &self.codec;
        self.assembler.render(response, |joined| {
            match codec.encode_response(joined)? {
                Encoded::Native => emit(joined),
                Encoded::Replace(value) => emit(&value),
                Encoded::Skip => {}
            }
            Ok(())
        })
    }

    fn migrate(
        &mut self,
        delivered_tokens: usize,
        request: &mut PreprocessedRequest,
    ) -> Result<Option<LLMEngineOutput>, String> {
        let continuation = self.continuation.as_ref().map_err(Clone::clone)?;
        if delivered_tokens != self.delivered {
            return Err(format!(
                "Dynamo delivered {delivered_tokens} tokens but SGLang reported {}",
                self.delivered
            ));
        }
        let prompt_delivered = self.assembler.prompt_delivered();
        self.assembler.migrate(delivered_tokens as u64);
        let Some(body) = continuation.retry(&REQUEST_RULES, delivered_tokens, prompt_delivered)
        else {
            // The budget is spent: an attempt that generates nothing, whose
            // counts the join rebases onto the delivered tokens.
            let mut response = json!({
                "output_ids": [],
                "meta_info": {
                    "id": self.request_id,
                    "finish_reason": {"type": "length", "length": 0},
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": 0,
                },
            });
            if continuation.returns_logprobs() {
                response["meta_info"]["output_token_logprobs"] = json!([]);
                response["meta_info"]["output_token_logprobs_length"] = json!(0);
            }
            return Ok(Some(LLMEngineOutput {
                finish_reason: Some(FinishReason::Length),
                engine_data: Some(json!({ RESPONSE_KEY: response })),
                ..Default::default()
            }));
        };
        request.extra_args = Some(json!({ PAYLOAD_KEY: body }));
        Ok(None)
    }

    fn finish(&mut self) -> Result<Option<Value>, DynamoError> {
        if self.stream {
            return Ok(None);
        }
        let response = self
            .assembler
            .finish()
            .ok_or_else(|| DynamoError::msg("SGLang stream ended without a response"))?;
        match self.codec.encode_response(&response)? {
            Encoded::Native => Ok(Some(response)),
            Encoded::Replace(value) => Ok(Some(value)),
            Encoded::Skip => Err(DynamoError::msg("the client codec skipped the response")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response_merge::FieldClass;
    use crate::testkit::laws::Harness;

    #[test]
    fn sglang_native_format_passes_every_law() {
        Harness::native(Arc::new(SglangFormat::native())).check_all();
    }

    #[test]
    fn request_priority_header_overrides_body_for_routing() {
        let headers = axum::http::HeaderMap::from_iter([
            (
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            ),
            (
                HEADER_REQUEST_PRIORITY.parse().unwrap(),
                axum::http::HeaderValue::from_static("0"),
            ),
        ]);
        let opened = SglangFormat::native()
            .open(ClientRequest {
                body: br#"{"input_ids":[1],"priority":7}"#,
                headers: &headers,
                default_request_id: "request",
            })
            .unwrap();
        let bound = (opened.bind)(&crate::testkit::worker(&Value::Null)).unwrap();

        assert_eq!(bound.request.routing.as_ref().unwrap().priority, Some(0));
    }

    #[test]
    fn malformed_request_priority_header_falls_back_to_body() {
        let headers = axum::http::HeaderMap::from_iter([
            (
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            ),
            (
                HEADER_REQUEST_PRIORITY.parse().unwrap(),
                axum::http::HeaderValue::from_static("not-an-int"),
            ),
        ]);
        let opened = SglangFormat::native()
            .open(ClientRequest {
                body: br#"{"input_ids":[1],"priority":7}"#,
                headers: &headers,
                default_request_id: "request",
            })
            .unwrap();
        let bound = (opened.bind)(&crate::testkit::worker(&Value::Null)).unwrap();

        assert_eq!(bound.request.routing.as_ref().unwrap().priority, Some(7));
    }

    /// Streams and unary responses recorded from SGLang v0.5.21 (Qwen3-0.6B,
    /// greedy), one request shape per line.
    fn fixtures() -> Vec<Value> {
        include_str!("../tests/data/qwen3-0.6b.jsonl")
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn recorded_fields_are_all_classified() {
        let classified: Vec<&str> = RULES.fields().map(|(path, _)| path).collect();
        for fixture in fixtures() {
            let responses = fixture["stream"].as_array().unwrap().iter();
            for response in responses.chain([&fixture["unary"]]) {
                let meta = response["meta_info"].as_object().unwrap().keys();
                let top = response
                    .as_object()
                    .unwrap()
                    .keys()
                    .filter(|k| *k != "meta_info");
                for path in meta
                    .map(|k| format!("/meta_info/{k}"))
                    .chain(top.map(|k| format!("/{k}")))
                {
                    assert!(
                        classified.contains(&path.as_str()),
                        "{path} is unclassified"
                    );
                }
            }
        }
    }

    #[test]
    fn recorded_streams_join_into_sglangs_unary_response() {
        for fixture in fixtures() {
            let body = Value::Object(fixture["request"].as_object().unwrap().clone());
            let Opened { bind, .. } = SglangFormat::native()
                .open(ClientRequest {
                    body: body.to_string().as_bytes(),
                    headers: &axum::http::HeaderMap::from_iter([(
                        axum::http::header::CONTENT_TYPE,
                        axum::http::HeaderValue::from_static("application/json"),
                    )]),
                    default_request_id: "id",
                })
                .unwrap();
            let mut exchange = bind(&crate::testkit::worker(&Value::Null))
                .unwrap()
                .exchange;
            for chunk in fixture["stream"].as_array().unwrap() {
                let output = LLMEngineOutput {
                    engine_data: Some(json!({ RESPONSE_KEY: chunk })),
                    ..Default::default()
                };
                exchange.chunk(output, false, &mut |_| {}).unwrap();
            }
            let joined = exchange.finish().unwrap().unwrap();
            // Timing, identity, and cache hits differ between the two requests.
            for (path, class) in RULES.fields() {
                if matches!(class, FieldClass::Attempt) || path.contains("cached_tokens") {
                    continue;
                }
                let name = fixture["name"].as_str().unwrap();
                assert_eq!(
                    joined.pointer(path),
                    fixture["unary"].pointer(path),
                    "{name} {path}"
                );
            }
        }
    }

    #[test]
    fn simulator_sends_each_field_the_way_sglang_does() {
        use crate::testkit::sim::{self, Role, Truth};
        for fixture in fixtures() {
            let (request, unary, recorded) =
                (&fixture["request"], &fixture["unary"], &fixture["stream"]);
            let ids: Vec<u32> = serde_json::from_value(unary["output_ids"].clone()).unwrap();
            let lp = |entries: &Value, i: usize| entries[i][0].as_f64().unwrap_or(0.0);
            let prompt: Vec<u32> = serde_json::from_value(request["input_ids"].clone()).unwrap();
            let mut cuts = Vec::new();
            for chunk in recorded.as_array().unwrap() {
                let len = chunk["output_ids"].as_array().unwrap().len();
                cuts.push(cuts.last().copied().unwrap_or(0) + len);
            }
            let truth = Truth {
                prompt_logprobs: (0..prompt.len())
                    .map(|i| lp(&unary["meta_info"]["input_token_logprobs"], i))
                    .collect(),
                output: ids
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| (id, lp(&unary["meta_info"]["output_token_logprobs"], i)))
                    .collect(),
                stop: unary["meta_info"]["finish_reason"]["type"] == "stop",
                row_bytes: 1,
                experts: Vec::new(),
                cuts,
                cached_tokens: 0,
                prompt,
            };
            let simulated = sim::stream(
                &truth,
                request.as_object().unwrap(),
                &truth.prompt,
                Role::Aggregated,
                &Value::Null,
            );
            assert_eq!(simulated.len(), recorded.as_array().unwrap().len());
            for (sim, real) in simulated.iter().zip(recorded.as_array().unwrap()) {
                for (path, class) in RULES.fields() {
                    if matches!(class, FieldClass::Attempt | FieldClass::Cumulative)
                        || path.contains("cached_tokens")
                    {
                        continue;
                    }
                    let (sim, real) = (sim.pointer(path), real.pointer(path));
                    let shape = |v: Option<&Value>| v.map(|v| v.as_array().map(Vec::len));
                    assert_eq!(shape(sim), shape(real), "{} {path}", fixture["name"]);
                }
                assert_eq!(sim["output_ids"], real["output_ids"]);
            }
        }
    }
}
