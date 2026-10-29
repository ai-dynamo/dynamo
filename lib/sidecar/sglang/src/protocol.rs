// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure request lowering and response conversion for SGLang's native
//! `sglang.api.v1` gRPC contract.

use dynamo_backend_common::{
    CompletionUsage, DisaggregationMode, DynamoError, GuidedDecodingOptions, KvHint,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, PromptTokensDetails, StopConditions,
    StopReason, TopLogprob, usage,
};
use serde_json::{Map, Value};

use crate::client;
use crate::proto as pb;
use crate::proto::finish_reason::Kind as FinishKind;

/// The prefill/decode rendezvous of one request: SGLang's bootstrap address
/// and room, and the prefill DP rank that holds the KV when it is named.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Rendezvous {
    pub bootstrap_host: String,
    pub bootstrap_port: i32,
    pub bootstrap_room: i64,
    pub disagg_prefill_dp_rank: Option<u32>,
}

/// Where one request runs, as the node's listeners chose it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target<'a> {
    /// The bootstrap address of a rendezvous prefill builds itself: this
    /// worker's host and the dispatched listener's port.
    pub bootstrap_host: Option<&'a str>,
    pub bootstrap_port: Option<u16>,
    /// The DP rank a prefill request runs on when the deployment has
    /// several; its rendezvous carries it in the port and room.
    pub prefill_rank: Option<u32>,
    /// Every DP rank of the deployment, at least one.
    pub global_dp_size: u32,
}

/// Lower one request to api.v1's single-item Generate form. Also returns the
/// prefill/decode rendezvous the request carries, in the handoff's JSON form.
pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    target: Target<'_>,
) -> Result<(pb::GenerateRequest, Option<Value>), DynamoError> {
    validate_request(request)?;
    let input_ids = request
        .token_ids
        .iter()
        .map(|&token| i64::from(token))
        .collect();
    let (max_new_tokens, min_new_tokens) = if mode.is_prefill() {
        (Some(1), None)
    } else {
        (
            request.stop_conditions.max_tokens.map(i64::from),
            request.stop_conditions.min_tokens.map(i64::from),
        )
    };

    // SGLang keeps a matched stop token in `output_ids` and trims only text,
    // which is never requested, so visible stops (e.g. Harmony's `<|call|>`)
    // reach the frontend decoder.
    let mut stop_token_ids = Vec::new();
    for tokens in [
        request.stop_conditions.stop_token_ids.as_ref(),
        request.stop_conditions.stop_token_ids_hidden.as_ref(),
        request.stop_conditions.stop_token_ids_visible.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for &token in tokens {
            let token = i64::from(token);
            if !stop_token_ids.contains(&token) {
                stop_token_ids.push(token);
            }
        }
    }

    let sampling_options = &request.sampling_options;
    let mut sampling_params = pb::SamplingParams {
        temperature: sampling_options.temperature.map(f64::from),
        top_p: sampling_options.top_p.map(f64::from),
        top_k: sampling_options.top_k.map(i64::from),
        min_p: sampling_options.min_p.map(f64::from),
        frequency_penalty: sampling_options.frequency_penalty.map(f64::from),
        presence_penalty: sampling_options.presence_penalty.map(f64::from),
        repetition_penalty: sampling_options.repetition_penalty.map(f64::from),
        max_new_tokens,
        min_new_tokens,
        stop: request
            .stop_conditions
            .stop
            .clone()
            .filter(|stop| !stop.is_empty())
            .map(|items| pb::StringOrList {
                value: Some(pb::string_or_list::Value::Many(pb::StringList { items })),
            }),
        stop_token_ids: (!stop_token_ids.is_empty()).then_some(pb::Int64List {
            items: stop_token_ids,
        }),
        ignore_eos: request.stop_conditions.ignore_eos,
        n: sampling_options.n.map(i64::from),
        sampling_seed: sampling_options.seed,
        ..Default::default()
    };
    if let Some(guided) = sampling_options.guided_decoding.as_ref() {
        apply_guided_decoding(guided, &mut sampling_params);
    }

    let output_options = &request.output_options;
    let return_logprob = !mode.is_prefill()
        && (output_options.logprobs.is_some() || output_options.prompt_logprobs.is_some());
    let top_logprobs_num = if mode.is_prefill() {
        0
    } else {
        output_options
            .logprobs
            .unwrap_or(0)
            .max(output_options.prompt_logprobs.unwrap_or(0))
    };
    let logprob_start_len = if mode.is_prefill() {
        -1
    } else {
        output_options.prompt_logprobs.map(|_| 0).unwrap_or(-1)
    };
    let routing = request.routing.as_ref();
    let disaggregated = resolve_disaggregated_params(request, mode, target)?;

    // `routed_dp_rank` is not sent: SGLang's Rust listeners run a request on
    // their own rank, so the rank chose the listener.
    let generate = pb::GenerateRequest {
        rid: Some(pb::StringOrList {
            value: Some(pb::string_or_list::Value::One(request_id.to_string())),
        }),
        input_ids: Some(pb::TokenIdsOrList {
            value: Some(pb::token_ids_or_list::Value::One(pb::TokenIds {
                ids: input_ids,
            })),
        }),
        stream: Some(true),
        sampling_params: Some(pb::SamplingParamsOrList {
            value: Some(pb::sampling_params_or_list::Value::One(sampling_params)),
        }),
        return_logprob: Some(pb::BoolOrList {
            value: Some(pb::bool_or_list::Value::One(return_logprob)),
        }),
        logprob_start_len: Some(int64_one(logprob_start_len)),
        top_logprobs_num: Some(int64_one(i64::from(top_logprobs_num))),
        bootstrap_host: disaggregated
            .as_ref()
            .map(|params| optional_string_one(params.bootstrap_host.clone())),
        bootstrap_port: disaggregated
            .as_ref()
            .map(|params| optional_int64_one(i64::from(params.bootstrap_port))),
        bootstrap_room: disaggregated
            .as_ref()
            .map(|params| optional_int64_one(params.bootstrap_room)),
        disagg_prefill_dp_rank: disaggregated
            .as_ref()
            .and_then(|params| params.disagg_prefill_dp_rank)
            .map(i64::from),
        priority: routing.and_then(|routing| routing.priority).map(i64::from),
        require_reasoning: Some(request.require_reasoning),
        lora_path: routing
            .and_then(|routing| routing.lora_name.clone())
            .map(optional_string_one),
        routing_key: request.mdc_sum.clone(),
        max_thinking_tokens: request.stop_conditions.max_thinking_tokens.map(u64::from),
        kv_hints: request
            .kv_hint
            .as_ref()
            .map(|hint| pb::KvHintsEnvelopeOrList {
                value: Some(pb::kv_hints_envelope_or_list::Value::One(kv_hint_to_proto(
                    hint,
                ))),
            }),
        ..Default::default()
    };
    Ok((
        generate,
        disaggregated.as_ref().map(disaggregated_params_to_json),
    ))
}

/// The DP rank the router chose for this request: prefill takes its own rank
/// and falls back to the request's, every other role takes the request's.
fn requested_dp_rank(request: &PreprocessedRequest, mode: DisaggregationMode) -> Option<u32> {
    request.routing.as_ref().and_then(|routing| {
        if mode.is_prefill() {
            routing.prefill_dp_rank.or(routing.dp_rank)
        } else {
            routing.dp_rank
        }
    })
}

/// The DP rank the router chose, which must be one this worker serves.
/// Refuse any other before a stream opens instead of running the request on
/// some listener's rank, as SGLang refused an out-of-range `routed_dp_rank`.
pub(crate) fn validate_dp_rank(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
    ranks: std::ops::Range<u32>,
) -> Result<Option<u32>, DynamoError> {
    match requested_dp_rank(request, mode) {
        Some(rank) if !ranks.contains(&rank) => Err(client::invalid_arg(format!(
            "DP rank {rank} was requested, but this SGLang worker serves DP ranks [{}, {})",
            ranks.start, ranks.end
        ))),
        rank => Ok(rank),
    }
}

/// The prefill handoff in JSON form. It names the prefill's DP rank only when
/// the prefill request named it to SGLang.
pub(crate) fn disaggregated_params_to_json(rendezvous: &Rendezvous) -> Value {
    let mut handoff = serde_json::json!({
        "bootstrap_host": rendezvous.bootstrap_host,
        "bootstrap_port": rendezvous.bootstrap_port,
        "bootstrap_room": rendezvous.bootstrap_room,
    });
    if let Some(rank) = rendezvous.disagg_prefill_dp_rank {
        handoff["disagg_prefill_dp_rank"] = Value::from(rank);
    }
    handoff
}

/// The rendezvous a request carries: the router's bootstrap info first, then
/// the prefill handoff. A prefill request without either gets this worker's
/// bootstrap address and a fresh room, congruent to its rank modulo the DP
/// size (SGLang's `follow_bootstrap_room`); aggregated requests carry none.
/// Prefill names its rank only when the router named it; decode names the
/// router's prefill rank, or the handoff's. For a single-rank prefill that is
/// rank 0, which SGLang resolves anyway.
pub(crate) fn resolve_disaggregated_params(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
    target: Target<'_>,
) -> Result<Option<Rendezvous>, DynamoError> {
    if mode == DisaggregationMode::Aggregated {
        return Ok(None);
    }
    let router_rank = request
        .routing
        .as_ref()
        .and_then(|routing| routing.prefill_dp_rank);
    let named_rank = if mode.is_prefill() {
        target.prefill_rank.filter(|_| router_rank.is_some())
    } else {
        router_rank
    };
    if let Some(info) = request.bootstrap_info.as_ref() {
        return rendezvous(
            &info.bootstrap_host,
            u64::from(info.bootstrap_port),
            info.bootstrap_room,
            named_rank,
        )
        .map(Some);
    }
    if let Some(prefill) = request.prefill_result.as_ref() {
        return handoff_rendezvous(&prefill.disaggregated_params).map(Some);
    }
    if mode.is_prefill() {
        let host = target.bootstrap_host.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap host from discovery")
        })?;
        let port = target.bootstrap_port.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap port from discovery")
        })?;
        let random = rand::random::<u64>() & (i64::MAX as u64);
        let room = match target.prefill_rank {
            Some(rank) => room_for_rank(rank, target.global_dp_size, random),
            None => random,
        };
        return rendezvous(host, u64::from(port), room, named_rank).map(Some);
    }
    Err(client::invalid_arg(
        "decode request has neither bootstrap_info nor prefill_result",
    ))
}

/// A room congruent to `rank` modulo `size`, from `random` in
/// `0..=i64::MAX`, that stays in SGLang's signed 64-bit field: the
/// PrefillRouter's room arithmetic.
fn room_for_rank(rank: u32, size: u32, random: u64) -> u64 {
    let max_room = i64::MAX as u64;
    let (rank, size) = (u64::from(rank), u64::from(size.max(1)));
    let max_quotient = (max_room - rank) / size;
    (random % (max_quotient + 1)) * size + rank
}

fn handoff_rendezvous(value: &Value) -> Result<Rendezvous, DynamoError> {
    let host = value
        .get("bootstrap_host")
        .and_then(Value::as_str)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_host is missing"))?;
    let port = value
        .get("bootstrap_port")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_port is missing"))?;
    let room = value
        .get("bootstrap_room")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_room is missing"))?;
    let rank = value
        .get("disagg_prefill_dp_rank")
        .filter(|rank| !rank.is_null())
        .map(|rank| {
            rank.as_u64()
                .and_then(|rank| u32::try_from(rank).ok())
                .ok_or_else(|| {
                    client::invalid_arg(format!(
                        "disaggregated_params.disagg_prefill_dp_rank must be a DP rank, got {rank}"
                    ))
                })
        })
        .transpose()?;
    rendezvous(host, port, room, rank)
}

fn rendezvous(
    host: &str,
    port: u64,
    room: u64,
    disagg_prefill_dp_rank: Option<u32>,
) -> Result<Rendezvous, DynamoError> {
    if host.trim().is_empty() {
        return Err(client::invalid_arg("bootstrap_host must not be empty"));
    }
    let bootstrap_port = i32::try_from(port)
        .map_err(|_| client::invalid_arg(format!("bootstrap_port is out of range: {port}")))?;
    let bootstrap_room = i64::try_from(room).map_err(|_| {
        client::invalid_arg(format!(
            "bootstrap_room must fit SGLang's signed int64 field: {room}"
        ))
    })?;
    Ok(Rendezvous {
        bootstrap_host: host.to_string(),
        bootstrap_port,
        bootstrap_room,
        disagg_prefill_dp_rank,
    })
}

fn int64_one(value: i64) -> pb::Int64OrList {
    pb::Int64OrList {
        value: Some(pb::int64_or_list::Value::One(value)),
    }
}

fn optional_string_one(value: String) -> pb::OptionalStringOrList {
    pb::OptionalStringOrList {
        value: Some(pb::optional_string_or_list::Value::One(
            pb::OptionalString { value: Some(value) },
        )),
    }
}

fn optional_int64_one(value: i64) -> pb::OptionalInt64OrList {
    pb::OptionalInt64OrList {
        value: Some(pb::optional_int64_or_list::Value::One(pb::OptionalInt64 {
            value: Some(value),
        })),
    }
}

/// Lower the single validated constraint onto SGLang's flat sampling fields.
fn apply_guided_decoding(guided: &GuidedDecodingOptions, sampling: &mut pb::SamplingParams) {
    if let Some(schema) = guided.json.as_ref() {
        sampling.json_schema = Some(json_value_to_string(schema));
    } else if let Some(regex) = guided.regex.as_ref() {
        sampling.regex = Some(regex.clone());
    } else if let Some(grammar) = guided.grammar.as_ref() {
        sampling.ebnf = Some(grammar.clone());
    } else if let Some(choice) = guided.choice.as_ref().filter(|value| !value.is_empty()) {
        sampling.choice = choice.clone();
    } else if let Some(tag) = guided.structural_tag.as_ref() {
        sampling.structural_tag = Some(json_value_to_string(tag));
    }
}

/// The action payload is JSON object text; each action type owns its schema.
fn kv_hint_to_proto(hint: &KvHint) -> pb::KvHintsEnvelope {
    pb::KvHintsEnvelope {
        protocol_version: hint.protocol_version.clone(),
        message_id: hint.message_id.clone(),
        actions: hint
            .actions
            .iter()
            .map(|action| pb::KvHintAction {
                action_id: action.action_id.clone(),
                action_type: action.action_type.clone(),
                action_version: action.action_version.clone(),
                payload: Some(
                    Value::Object(action.payload.clone().into_iter().collect()).to_string(),
                ),
            })
            .collect(),
    }
}

/// Refuse what SGLang's Generate cannot represent. Empty guided-decoding
/// constraints are refused too: SGLang's Rust server treats them as unset and
/// would run unconstrained.
fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    // prompt_embeds requests arrive with empty token_ids, so check them first.
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_request(
            "prompt_embeds are not supported by SGLang's native gRPC proto",
        ));
    }
    if request.token_ids.is_empty() {
        return Err(client::invalid_request("token_ids must not be empty"));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(client::invalid_request(
            "multimodal payloads are not supported by SGLang's native Generate RPC",
        ));
    }
    if request.sampling_options.n.unwrap_or(1) != 1 {
        return Err(client::invalid_request(
            "n must be 1 for the SGLang sidecar",
        ));
    }
    if request.sampling_options.best_of.unwrap_or(1) != 1 {
        return Err(client::invalid_request(
            "best_of is not represented by SGLang's native gRPC proto",
        ));
    }
    if request.sampling_options.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_request(
            "beam search is not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(penalty) = request.sampling_options.length_penalty
        && (penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_request(
            "length_penalty is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .sampling_options
        .include_stop_str_in_output
        .unwrap_or(false)
    {
        return Err(client::invalid_request(
            "include_stop_str_in_output is not represented by SGLang's native gRPC proto",
        ));
    }
    let Some(guided) = request.sampling_options.guided_decoding.as_ref() else {
        return Ok(());
    };
    let constraint_count = [
        guided.json.is_some(),
        guided.regex.is_some(),
        guided
            .choice
            .as_ref()
            .is_some_and(|value| !value.is_empty()),
        guided.grammar.is_some(),
        guided.structural_tag.is_some(),
    ]
    .into_iter()
    .filter(|is_set| *is_set)
    .count();
    if constraint_count > 1 {
        return Err(client::invalid_request(
            "the native SGLang gRPC proto accepts only one guided-decoding constraint",
        ));
    }
    if guided
        .backend
        .as_ref()
        .is_some_and(|value| !value.is_empty())
        || guided.whitespace_pattern.is_some()
    {
        return Err(client::invalid_request(
            "guided-decoding backend and whitespace modifiers are not represented by SGLang's native gRPC proto",
        ));
    }
    let is_empty =
        |value: Option<&Value>| value.is_some_and(|v| json_value_to_string(v).is_empty());
    if is_empty(guided.json.as_ref())
        || is_empty(guided.structural_tag.as_ref())
        || guided.regex.as_ref().is_some_and(String::is_empty)
        || guided.grammar.as_ref().is_some_and(String::is_empty)
    {
        return Err(client::invalid_request(
            "guided decoding constraint must not be empty",
        ));
    }
    if guided
        .choice
        .as_ref()
        .is_some_and(|choice| choice.iter().any(String::is_empty))
    {
        return Err(client::invalid_request(
            "guided choice must contain only non-empty values",
        ));
    }
    Ok(())
}

pub(crate) fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

pub(crate) fn output_ids_to_u32(ids: &[i64]) -> Result<Vec<u32>, DynamoError> {
    ids.iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                client::protocol_error(if id < 0 {
                    format!("SGLang returned a negative token id: {id}")
                } else {
                    format!("SGLang returned a token id that does not fit u32: {id}")
                })
            })
        })
        .collect()
}

/// The outcome of one Generate frame.
#[derive(Debug)]
pub(crate) enum FrameOutput {
    /// Nothing to forward, such as a token-less intermediate frame.
    Skip,
    Delta(LLMEngineOutput),
    /// The last output; the stream ends after it.
    Terminal(LLMEngineOutput),
}

/// Converts one `sglang.api.v1` Generate stream into Dynamo outputs.
pub(crate) struct ResponseConverter {
    is_prefill: bool,
    return_tokens_as_ids: bool,
    stop_conditions: StopConditions,
    prompt_tokens: u32,
    generated: u32,
    prompt_logprobs: PromptLogprobs,
}

impl ResponseConverter {
    pub(crate) fn new(
        is_prefill: bool,
        return_tokens_as_ids: bool,
        stop_conditions: StopConditions,
        prompt_tokens: u32,
    ) -> Self {
        Self {
            is_prefill,
            return_tokens_as_ids,
            stop_conditions,
            prompt_tokens,
            generated: 0,
            prompt_logprobs: PromptLogprobs::default(),
        }
    }

    /// Usage observed so far, for an early cancellation.
    pub(crate) fn usage(&self) -> CompletionUsage {
        usage(self.prompt_tokens, self.generated)
    }

    pub(crate) fn frame(
        &mut self,
        frame: pb::GenerateResponse,
    ) -> Result<FrameOutput, DynamoError> {
        let mut meta = frame.meta_info.unwrap_or_default();
        // SGLang reports the prompt length on every frame; zero means it was
        // not reported, since an empty prompt is rejected before submission.
        if meta.prompt_tokens > 0 {
            self.prompt_tokens = meta.prompt_tokens;
        }
        // Rust incremental streams send input logprobs once, on the first
        // frame, and empty lists afterwards. Keep them for the terminal.
        self.prompt_logprobs.capture(&mut meta);
        let token_ids = output_ids_to_u32(frame.output_ids.as_ref().map_or(&[], |ids| &ids.ids))?;
        let (log_probs, top_logprobs) = extract_logprobs(&meta, self.return_tokens_as_ids)?;
        let is_terminal = meta.finish_reason.is_some();

        if self.is_prefill {
            return if is_terminal {
                terminal_from_meta(&meta, self.prompt_tokens, 0, &self.stop_conditions)
                    .map(FrameOutput::Terminal)
            } else {
                Ok(FrameOutput::Skip)
            };
        }

        self.generated = self.generated.saturating_add(token_ids.len() as u32);
        if is_terminal {
            let mut terminal = terminal_from_meta(
                &meta,
                self.prompt_tokens,
                self.generated,
                &self.stop_conditions,
            )?;
            terminal.token_ids = token_ids;
            terminal.log_probs = log_probs;
            terminal.top_logprobs = top_logprobs;
            terminal.engine_data = self
                .prompt_logprobs
                .to_value()?
                .map(|prompt_logprobs| serde_json::json!({"prompt_logprobs": prompt_logprobs}));
            return Ok(FrameOutput::Terminal(terminal));
        }
        if token_ids.is_empty() {
            return Ok(FrameOutput::Skip);
        }
        Ok(FrameOutput::Delta(LLMEngineOutput {
            token_ids,
            log_probs,
            top_logprobs,
            ..Default::default()
        }))
    }

    /// One request's in-stream failure; the stream ends after it. A 499 after
    /// the request or the sidecar was cancelled ends as a cancelled output.
    pub(crate) fn stream_error(
        &self,
        error: &pb::GenerateStreamError,
        is_cancelled: bool,
    ) -> Result<LLMEngineOutput, DynamoError> {
        if is_cancelled && stream_error_code(error) == CLIENT_CLOSED_REQUEST {
            return Ok(LLMEngineOutput::cancelled().with_usage(self.usage()));
        }
        Err(stream_error_to_dynamo(error))
    }
}

/// SGLang's code for a request cancelled by Abort or a disconnect.
const CLIENT_CLOSED_REQUEST: u32 = 499;

fn stream_error_code(error: &pb::GenerateStreamError) -> u32 {
    error.error.as_ref().map_or(0, |body| body.code)
}

/// Classify an in-stream error by its HTTP status code.
pub(crate) fn stream_error_to_dynamo(error: &pb::GenerateStreamError) -> DynamoError {
    let message = error
        .error
        .as_ref()
        .map_or("SGLang reported an error without a body", |body| {
            body.message.as_str()
        });
    coded_error("Generate", message, stream_error_code(error))
}

/// Classify a failure SGLang's Rust server reports with an HTTP status code,
/// as a gRPC error item or a native HTTP error frame. As in
/// `terminal_failure`, a 4xx is the request's fault and anything else, 503
/// included, the engine's; neither is a transport failure, so migration does
/// not retry it on another worker. A 499 is a cancellation. The engine message
/// stays private here, so the OpenAI surfaces return only the error category.
pub(crate) fn coded_error(operation: &str, message: &str, code: u32) -> DynamoError {
    let detail = format!("{operation}: {message} (code {code})");
    match code {
        CLIENT_CLOSED_REQUEST => client::cancelled(detail),
        400..=499 => client::invalid_arg(detail),
        _ => client::protocol_error(detail),
    }
}

pub(crate) fn terminal_from_meta(
    meta: &pb::GenerateMetaInfo,
    prompt_tokens: u32,
    generated: u32,
    stop_conditions: &StopConditions,
) -> Result<LLMEngineOutput, DynamoError> {
    let kind = meta
        .finish_reason
        .as_ref()
        .ok_or_else(|| client::protocol_error("SGLang terminal is missing finish_reason"))?
        .kind
        .as_ref()
        .ok_or_else(|| client::protocol_error("SGLang finish_reason is missing a type"))?;
    let mut completion_usage = usage(prompt_tokens, generated);
    completion_usage.prompt_tokens_details = Some(PromptTokensDetails {
        audio_tokens: None,
        cached_tokens: Some(meta.cached_tokens.min(prompt_tokens)),
        ..Default::default()
    });
    let (output, stop_reason) = match kind {
        FinishKind::Stop(stop) => (
            LLMEngineOutput::stop(),
            stop.matched
                .as_ref()
                .and_then(|matched| matched_stop_reason(matched, stop_conditions)),
        ),
        FinishKind::Length(_) => (LLMEngineOutput::length(), None),
        FinishKind::Abort(abort) => {
            return Err(terminal_failure(
                "abort",
                &serde_json::json!({
                    "message": abort.message,
                    "status_code": abort.status_code,
                    "err_type": abort.err_type,
                }),
            ));
        }
        FinishKind::Unknown(raw) => (unknown_finish(raw)?, None),
    };
    let mut output = output.with_usage(completion_usage);
    output.stop_reason = stop_reason;
    Ok(output)
}

/// A finish type outside api.v1's typed arms, passed through as JSON text.
fn unknown_finish(raw: &str) -> Result<LLMEngineOutput, DynamoError> {
    let finish: Value = serde_json::from_str(raw).map_err(|error| {
        client::protocol_error(format!("invalid SGLang finish_reason: {error}"))
    })?;
    match finish.get("type").and_then(Value::as_str) {
        Some("cancelled") => Ok(LLMEngineOutput::cancelled()),
        Some(kind @ ("abort" | "error")) => Err(terminal_failure(kind, &finish)),
        Some(other) => Err(client::protocol_error(format!(
            "SGLang returned unsupported finish_reason type `{other}`"
        ))),
        None => Err(client::protocol_error(
            "SGLang finish_reason is missing a type",
        )),
    }
}

/// A failed generation, classified by its HTTP status code: a 4xx is the
/// request's fault, anything else the engine's.
pub(crate) fn terminal_failure(finish_type: &str, finish: &Value) -> DynamoError {
    let message = finish
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("SGLang generation failed");
    let status_code = finish.get("status_code").and_then(Value::as_i64);
    let err_type = finish.get("err_type").and_then(Value::as_str);
    let detail = format!(
        "SGLang generation {finish_type}: {message} (status_code={}, err_type={})",
        status_code
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        err_type.unwrap_or("unknown")
    );
    if matches!(status_code, Some(400..=499)) {
        client::invalid_arg(detail)
    } else {
        client::protocol_error(detail)
    }
}

/// A matched user stop token is reported; system and hidden stop tokens are not.
fn matched_stop_reason(
    matched: &pb::Matched,
    stop_conditions: &StopConditions,
) -> Option<StopReason> {
    match matched.value.as_ref()? {
        pb::matched::Value::Str(value) => Some(StopReason::String(value.clone())),
        pb::matched::Value::Token(id) => u32::try_from(*id)
            .ok()
            .filter(|id| {
                stop_conditions
                    .stop_token_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(id))
            })
            .map(|id| StopReason::Int(i64::from(id))),
        pb::matched::Value::Tokens(_) => None,
    }
}

/// The first non-empty prompt logprobs of a stream. Decode nodes in PD and
/// frames after the first report empty lists, which must not replace them.
#[derive(Default)]
struct PromptLogprobs {
    selected: Vec<pb::LogprobEntry>,
    top: Vec<pb::NullableTopLogprobs>,
}

impl PromptLogprobs {
    fn capture(&mut self, meta: &mut pb::GenerateMetaInfo) {
        if self.selected.is_empty()
            && let Some(selected) = meta.input_token_logprobs.as_mut()
        {
            self.selected = std::mem::take(&mut selected.entries);
        }
        if self.top.is_empty() {
            self.top = std::mem::take(&mut meta.input_top_logprobs);
        }
    }

    fn to_value(&self) -> Result<Option<Value>, DynamoError> {
        prompt_logprobs_value(&self.selected, &self.top)
    }
}

/// `engine_data.prompt_logprobs`: one entry per prompt position, `null` for
/// the first, each mapping token ids to their logprob and decoded text.
pub(crate) fn prompt_logprobs_value(
    selected: &[pb::LogprobEntry],
    top: &[pb::NullableTopLogprobs],
) -> Result<Option<Value>, DynamoError> {
    let Some(first) = selected.first() else {
        return Ok(None);
    };
    let top = (!top.is_empty()).then_some(top);

    // Current SGLang encodes the first requested prompt position with a null
    // logprob and a null top-logprob row; older releases omitted it entirely.
    let has_native_sentinel = first.logprob.is_none();
    if let Some(top) = top {
        let top_has_native_sentinel = top.first().is_some_and(|row| row.row.is_none());
        if top.len() != selected.len() || top_has_native_sentinel != has_native_sentinel {
            return Err(client::protocol_error(
                "input_token_logprobs and input_top_logprobs use inconsistent position encoding",
            ));
        }
    }
    let mut payload = Vec::with_capacity(selected.len() + usize::from(!has_native_sentinel));
    if !has_native_sentinel {
        payload.push(Value::Null);
    }
    for (index, entry) in selected.iter().enumerate() {
        if index == 0 && has_native_sentinel {
            payload.push(Value::Null);
            continue;
        }
        let (token_id, value) = prompt_logprob_entry(entry, "input_token_logprobs")?;
        let mut position = Map::new();
        position.insert(token_id, value);
        if let Some(row) = top
            .and_then(|rows| rows.get(index))
            .and_then(|row| row.row.as_ref())
        {
            for alternative in &row.entries {
                let (token_id, value) = prompt_logprob_entry(alternative, "input_top_logprobs")?;
                position.entry(token_id).or_insert(value);
            }
        }
        payload.push(Value::Object(position));
    }
    Ok(Some(Value::Array(payload)))
}

fn prompt_logprob_entry(
    entry: &pb::LogprobEntry,
    label: &str,
) -> Result<(String, Value), DynamoError> {
    let logprob = entry
        .logprob
        .ok_or_else(|| client::protocol_error(format!("missing logprob in {label}")))?;
    let mut value = Map::new();
    value.insert("logprob".to_string(), Value::from(logprob));
    if let Some(decoded) = entry.text.as_ref() {
        value.insert("decoded_token".to_string(), Value::String(decoded.clone()));
    }
    Ok((entry.token_id.to_string(), Value::Object(value)))
}

pub(crate) type ExtractedLogprobs = (Option<Vec<f64>>, Option<Vec<Vec<TopLogprob>>>);

pub(crate) fn extract_logprobs(
    meta: &pb::GenerateMetaInfo,
    return_tokens_as_ids: bool,
) -> Result<ExtractedLogprobs, DynamoError> {
    let Some(selected) = meta.output_token_logprobs.as_ref() else {
        return Ok((None, None));
    };
    let log_probs = selected
        .entries
        .iter()
        .map(|entry| {
            entry.logprob.ok_or_else(|| {
                client::protocol_error("invalid output_token_logprobs entry from SGLang")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // A repeated field has no presence: an empty list means the request asked
    // for no alternatives, not an empty alternative list per token.
    if meta.output_top_logprobs.is_empty() {
        return Ok((Some(log_probs), None));
    }
    let mut positions = Vec::with_capacity(meta.output_top_logprobs.len());
    for position in &meta.output_top_logprobs {
        let Some(row) = position.row.as_ref() else {
            positions.push(Vec::new());
            continue;
        };
        let mut mapped = Vec::with_capacity(row.entries.len());
        for (index, entry) in row.entries.iter().enumerate() {
            let logprob = entry
                .logprob
                .ok_or_else(|| client::protocol_error("missing top-logprob value from SGLang"))?;
            let token_id = u32::try_from(entry.token_id)
                .map_err(|_| client::protocol_error("top-logprob token id does not fit u32"))?;
            let token = if return_tokens_as_ids {
                Some(format!("token_id:{token_id}"))
            } else {
                entry.text.clone()
            };
            mapped.push(TopLogprob {
                rank: u32::try_from(index + 1).unwrap_or(u32::MAX),
                token_id,
                token,
                logprob,
                bytes: None,
            });
        }
        positions.push(mapped);
    }
    Ok((Some(log_probs), Some(positions)))
}

#[cfg(test)]
mod request_tests;

#[cfg(test)]
mod response_tests;
