// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol types for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! The request follows vLLM's Rust frontend wire contract
//! (`rust/src/server/src/routes/inference/generate/types.rs`). Fields that the
//! Rust frontend types are mirrored here; future top-level fields are captured
//! in `passthrough`. `sampling_params` is validated while its complete JSON
//! object is retained for the version-matched worker.

use std::collections::HashMap;

use anyhow::Result;
use dynamo_runtime::error::{BackendError, DynamoError, ErrorType as DynamoErrorType};
use futures::{Stream, StreamExt, pin_mut};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::{convert_backend_top_logprobs, token_to_utf8_bytes};
use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PromptLogprobs};

/// Token-in/token-out generation request.
///
/// The vLLM Rust frontend's public fields are typed. Future top-level fields
/// remain forward-compatible through [`Self::passthrough`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateRequest {
    /// Client-supplied request id, echoed back. The server generates one if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Pre-tokenized prompt. Required — this is the KV-routing input.
    pub token_ids: Vec<u32>,

    /// vLLM Rust-compatible sampling view backed by the original JSON object.
    pub sampling_params: SamplingParams,

    /// Model / alias for worker selection. Optional (single-model deployments omit it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Streaming vs. unary response.
    #[serde(default)]
    pub stream: bool,

    /// Streaming usage options. The current Dynamo profile rejects streaming,
    /// but typing this field keeps request validation aligned with vLLM Rust.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Prefix-cache isolation salt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,

    /// Engine scheduling priority. vLLM Rust exposes this as a signed 32-bit value.
    #[serde(default)]
    pub priority: i32,

    /// Disaggregated-serving transfer parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<Map<String, Value>>,

    /// Future top-level fields, including Python-frontend-only fields such as
    /// `features`, are retained and forwarded to the worker.
    #[serde(flatten)]
    pub passthrough: Map<String, Value>,
}

impl GenerateRequest {
    pub(crate) fn response_options(&self) -> GenerateResponseOptions {
        GenerateResponseOptions {
            include_logprobs: self.sampling_params.logprobs().is_some(),
            include_prompt_logprobs: self.sampling_params.prompt_logprobs().is_some(),
        }
    }

    /// Validate the request-level rules enforced by vLLM's Rust generate route.
    pub fn validate(&self) -> Result<(), String> {
        if self.token_ids.is_empty() {
            return Err("token_ids must contain at least one token ID.".to_string());
        }

        if self.stream_options.is_some() && !self.stream {
            return Err("stream_options are only supported when stream=true.".to_string());
        }

        if self.sampling_params.max_tokens() == Some(0) {
            return Err("sampling_params.max_tokens must be greater than 0.".to_string());
        }

        if let Some(prompt_logprobs) = self.sampling_params.prompt_logprobs() {
            if prompt_logprobs < 0 && prompt_logprobs != -1 {
                return Err(
                    "sampling_params.prompt_logprobs must be non-negative or -1.".to_string(),
                );
            }
            if self.stream {
                return Err(
                    "sampling_params.prompt_logprobs are not available when stream=true."
                        .to_string(),
                );
            }
        }

        if let (Some(min_tokens), Some(max_tokens)) = (
            self.sampling_params.min_tokens(),
            self.sampling_params.max_tokens(),
        ) && min_tokens > max_tokens
        {
            return Err(format!(
                "sampling_params.min_tokens ({min_tokens}) exceeds max_tokens ({max_tokens})."
            ));
        }

        Ok(())
    }
}

/// vLLM Rust streaming options. Unknown options are retained for forward
/// compatibility even though the current unary Dynamo profile does not consume
/// them.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_usage_stats: Option<bool>,

    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// Sampling parameters with a complete typed vLLM Rust view and the original
/// JSON object retained for backend forwarding.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Original deserialized JSON object and the sole serialization source.
    ///
    /// Keeping this value preserves unknown future fields and the distinction
    /// between omitted fields and fields explicitly set to `null` when the
    /// request is forwarded to a version-matched worker.
    raw: Value,

    // Complete typed view of the fields supported by vLLM Rust. The frontend
    // reads only the controls it needs; `raw` remains authoritative.
    temperature: Option<f32>,
    top_p: Option<f32>,
    // vLLM uses -1 as the sentinel for disabling top-k sampling.
    top_k: Option<i64>,
    seed: Option<i64>,
    max_tokens: Option<u32>,
    min_tokens: Option<u32>,
    thinking_token_budget: Option<i64>,
    logprobs: Option<i32>,
    prompt_logprobs: Option<i32>,
    min_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    stop_token_ids: Option<Vec<u32>>,
    ignore_eos: bool,
    logit_bias: Option<HashMap<u32, f32>>,
    allowed_token_ids: Option<Vec<u32>>,
    bad_words: Option<Vec<String>>,
    logprob_token_ids: Option<Vec<u32>>,
    /// Engine-owned nested shape. Dynamo does not interpret it, so keeping the
    /// typed view opaque avoids duplicating version-specific vLLM validation.
    structured_outputs: Option<Value>,
    skip_reading_prefix_cache: Option<bool>,
    vllm_xargs: Option<HashMap<String, Value>>,
}

impl SamplingParams {
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    pub fn min_tokens(&self) -> Option<u32> {
        self.min_tokens
    }

    pub fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }

    pub fn logprobs(&self) -> Option<i32> {
        self.logprobs
    }

    pub fn prompt_logprobs(&self) -> Option<i32> {
        self.prompt_logprobs
    }

    pub fn as_value(&self) -> &Value {
        &self.raw
    }
}

impl Serialize for SamplingParams {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SamplingParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Value::deserialize(deserializer)?;
        if !raw.is_object() {
            return Err(serde::de::Error::custom(
                "sampling_params must be a JSON object",
            ));
        }

        let object = raw
            .as_object()
            .expect("sampling_params object checked above");
        macro_rules! field {
            ($name:ident) => {
                sampling_field(object, stringify!($name)).map_err(serde::de::Error::custom)?
            };
        }

        Ok(Self {
            temperature: field!(temperature),
            top_p: field!(top_p),
            top_k: field!(top_k),
            seed: field!(seed),
            max_tokens: field!(max_tokens),
            min_tokens: field!(min_tokens),
            thinking_token_budget: field!(thinking_token_budget),
            logprobs: field!(logprobs),
            prompt_logprobs: field!(prompt_logprobs),
            min_p: field!(min_p),
            frequency_penalty: field!(frequency_penalty),
            presence_penalty: field!(presence_penalty),
            repetition_penalty: field!(repetition_penalty),
            stop_token_ids: field!(stop_token_ids),
            ignore_eos: sampling_field_or_default(object, "ignore_eos")
                .map_err(serde::de::Error::custom)?,
            logit_bias: field!(logit_bias),
            allowed_token_ids: field!(allowed_token_ids),
            bad_words: field!(bad_words),
            logprob_token_ids: field!(logprob_token_ids),
            structured_outputs: field!(structured_outputs),
            skip_reading_prefix_cache: field!(skip_reading_prefix_cache),
            vllm_xargs: field!(vllm_xargs),
            raw,
        })
    }
}

fn sampling_field<T>(object: &Map<String, Value>, name: &str) -> Result<Option<T>, String>
where
    T: DeserializeOwned,
{
    object
        .get(name)
        .map(|value| serde_json::from_value::<Option<T>>(value.clone()))
        .transpose()
        .map(Option::flatten)
        .map_err(|error| format!("sampling_params.{name}: {error}"))
}

fn sampling_field_or_default<T>(object: &Map<String, Value>, name: &str) -> Result<T, String>
where
    T: DeserializeOwned + Default,
{
    object
        .get(name)
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()
        .map(Option::unwrap_or_default)
        .map_err(|error| format!("sampling_params.{name}: {error}"))
}

/// A single choice in a `GenerateResponse`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponseChoice {
    pub index: u32,

    pub token_ids: Option<Vec<u32>>,

    pub logprobs: Option<serde_json::Value>,

    pub finish_reason: Option<String>,

    pub routed_experts: Option<String>,
}

/// Token-in/token-out generation response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    pub request_id: String,

    pub choices: Vec<GenerateResponseChoice>,

    pub prompt_logprobs: Option<serde_json::Value>,

    pub kv_transfer_params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GenerateResponseOptions {
    include_logprobs: bool,
    include_prompt_logprobs: bool,
}

/// Per-index accumulation state while folding a stream of
/// [`LLMEngineOutput`] deltas into a single [`GenerateResponse`].
struct GenerateChoiceAcc {
    index: u32,
    token_ids: Vec<crate::protocols::TokenIdType>,
    logprobs: Option<Vec<Value>>,
    finish_reason: Option<String>,
    routed_experts: Option<String>,
}

impl GenerateChoiceAcc {
    fn apply(&mut self, output: &LLMEngineOutput, options: GenerateResponseOptions) -> Result<()> {
        if let Some(finish_reason) = output.finish_reason.as_ref() {
            match finish_reason {
                crate::protocols::common::FinishReason::Error(message) => {
                    return Err(DynamoError::builder()
                        .error_type(DynamoErrorType::Backend(BackendError::Unknown))
                        .message(message)
                        .build()
                        .into());
                }
                crate::protocols::common::FinishReason::Cancelled => {
                    return Err(DynamoError::builder()
                        .error_type(DynamoErrorType::Cancelled)
                        .message("backend cancelled generation")
                        .build()
                        .into());
                }
                _ => self.finish_reason = Some(finish_reason.to_string()),
            }
        }
        if options.include_logprobs {
            // Keep `None` until a logprob-bearing delta arrives: `into_response`
            // distinguishes missing logprobs from an empty content array.
            let has_logprobs = if let Some(content) = self.logprobs.as_mut() {
                append_completion_logprobs_from_output(output, content)?
            } else {
                let mut content = Vec::new();
                let has_logprobs = append_completion_logprobs_from_output(output, &mut content)?;
                if has_logprobs {
                    self.logprobs = Some(content);
                }
                has_logprobs
            };
            if !has_logprobs && !output.token_ids.is_empty() {
                anyhow::bail!(
                    "generate choice {} requested logprobs but a token-bearing delta returned none",
                    self.index
                );
            }
        }
        self.token_ids.extend_from_slice(&output.token_ids);
        Ok(())
    }

    fn into_response(self, options: GenerateResponseOptions) -> Result<GenerateResponseChoice> {
        let Self {
            index,
            token_ids,
            logprobs,
            finish_reason,
            routed_experts,
        } = self;
        let logprobs = if options.include_logprobs {
            let content = logprobs.ok_or_else(|| {
                anyhow::anyhow!("generate choice {index} requested logprobs but returned none")
            })?;
            anyhow::ensure!(
                content.len() == token_ids.len(),
                "generate choice {index} returned {} logprob positions for {} tokens",
                content.len(),
                token_ids.len()
            );
            let mut logprobs = Map::with_capacity(1);
            logprobs.insert("content".to_string(), Value::Array(content));
            Some(Value::Object(logprobs))
        } else {
            None
        };

        Ok(GenerateResponseChoice {
            index,
            token_ids: Some(token_ids),
            logprobs,
            finish_reason,
            routed_experts,
        })
    }
}

fn clamp_vllm_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}

fn append_completion_logprobs_from_output(
    output: &LLMEngineOutput,
    content: &mut Vec<Value>,
) -> Result<bool> {
    let Some(log_probs) = output.log_probs.as_ref() else {
        anyhow::ensure!(
            output.top_logprobs.is_none(),
            "generate output returned top_logprobs without selected-token logprobs"
        );
        return Ok(false);
    };
    anyhow::ensure!(
        log_probs.len() == output.token_ids.len(),
        "generate output returned {} selected-token logprobs for {} tokens",
        log_probs.len(),
        output.token_ids.len()
    );
    if let Some(top_logprobs) = output.top_logprobs.as_ref() {
        anyhow::ensure!(
            top_logprobs.len() == output.token_ids.len(),
            "generate output returned {} top-logprob positions for {} tokens",
            top_logprobs.len(),
            output.token_ids.len()
        );
    }
    if output.token_ids.is_empty() {
        return Ok(false);
    }

    content.reserve(output.token_ids.len());
    for (position, (token_id, logprob)) in output.token_ids.iter().zip(log_probs).enumerate() {
        let token = format!("token_id:{token_id}");
        let backend_top_logprobs = output
            .top_logprobs
            .as_ref()
            .map_or(&[][..], |all| all[position].as_slice());
        let logprob = clamp_vllm_logprob(*logprob as f32);
        let mut top_logprobs =
            convert_backend_top_logprobs(backend_top_logprobs, &token, *token_id, logprob, true);
        for entry in &mut top_logprobs {
            entry.logprob = clamp_vllm_logprob(entry.logprob);
        }
        let bytes = bytes_value(token_to_utf8_bytes(&token));
        let mut token_logprob = Map::with_capacity(4);
        token_logprob.insert("token".to_string(), Value::String(token));
        token_logprob.insert("logprob".to_string(), Value::from(logprob));
        token_logprob.insert("bytes".to_string(), bytes);
        token_logprob.insert(
            "top_logprobs".to_string(),
            Value::Array(top_logprobs.into_iter().map(top_logprob_value).collect()),
        );
        content.push(Value::Object(token_logprob));
    }
    Ok(true)
}

fn top_logprob_value(entry: dynamo_protocols::types::TopLogprobs) -> Value {
    let mut value = Map::with_capacity(3);
    value.insert("token".to_string(), Value::String(entry.token));
    value.insert("logprob".to_string(), Value::from(entry.logprob));
    value.insert("bytes".to_string(), bytes_value(entry.bytes));
    Value::Object(value)
}

fn bytes_value(bytes: Option<Vec<u8>>) -> Value {
    bytes.map_or(Value::Null, |bytes| {
        Value::Array(bytes.into_iter().map(Value::from).collect())
    })
}

fn prompt_logprobs_from_output(output: &LLMEngineOutput) -> Result<Option<PromptLogprobs>> {
    let Some(prompt_logprobs) = output
        .engine_data
        .as_ref()
        .and_then(|engine_data| engine_data.get("prompt_logprobs"))
    else {
        return Ok(None);
    };
    let mut prompt_logprobs: PromptLogprobs = serde_json::from_value(prompt_logprobs.clone())
        .map_err(|error| anyhow::anyhow!("invalid generate prompt_logprobs payload: {error}"))?;
    for entries in prompt_logprobs.iter_mut().flatten() {
        for entry in entries.values_mut() {
            entry.logprob = clamp_vllm_logprob(entry.logprob);
        }
    }
    Ok(Some(prompt_logprobs))
}

/// Folds a stream of [`Annotated<LLMEngineOutput>`] deltas into a single
/// [`GenerateResponse`]. Each chunk carries a delta of newly generated
/// `token_ids` keyed by `index` (default 0).
struct GenerateAggregator {
    request_id: String,
    choices: HashMap<u32, GenerateChoiceAcc>,
    prompt_logprobs: Option<PromptLogprobs>,
    kv_transfer_params: Option<Value>,
    kv_transfer_params_from_engine_data: bool,
}

impl GenerateAggregator {
    fn new(request_id: String) -> Self {
        Self {
            request_id,
            choices: HashMap::new(),
            prompt_logprobs: None,
            kv_transfer_params: None,
            kv_transfer_params_from_engine_data: false,
        }
    }

    fn apply_output(
        &mut self,
        output: LLMEngineOutput,
        options: GenerateResponseOptions,
    ) -> Result<()> {
        if options.include_prompt_logprobs
            && self.prompt_logprobs.is_none()
            && let Some(prompt_logprobs) = prompt_logprobs_from_output(&output)?
        {
            self.prompt_logprobs = Some(prompt_logprobs);
        }
        if !self.kv_transfer_params_from_engine_data
            && output.finish_reason.is_some()
            && output.disaggregated_params.is_some()
        {
            self.kv_transfer_params = output.disaggregated_params.clone();
        }
        let index = output.index.unwrap_or(0);
        let choice = self.choices.entry(index).or_insert(GenerateChoiceAcc {
            index,
            token_ids: Vec::new(),
            logprobs: None,
            finish_reason: None,
            routed_experts: None,
        });
        if let Some(engine_data) = output.engine_data.as_ref() {
            if let Some(routed_experts) = engine_data.get("routed_experts") {
                choice.routed_experts = Some(
                    serde_json::from_value(routed_experts.clone()).map_err(|error| {
                        anyhow::anyhow!("invalid generate routed_experts payload: {error}")
                    })?,
                );
            }
            if let Some(kv_transfer_params) = engine_data.get("kv_transfer_params") {
                self.kv_transfer_params = Some(kv_transfer_params.clone());
                self.kv_transfer_params_from_engine_data = true;
            }
        }
        choice.apply(&output, options)
    }

    async fn apply(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
        options: GenerateResponseOptions,
    ) -> Result<GenerateResponse> {
        let mut aggregator = GenerateAggregator::new(request_id);
        pin_mut!(stream);
        while let Some(delta) = stream.next().await {
            let delta = delta.ok().map_err(anyhow::Error::msg)?;
            if let Some(output) = delta.data {
                aggregator.apply_output(output, options)?;
            }
        }

        let GenerateAggregator {
            request_id,
            choices,
            prompt_logprobs,
            kv_transfer_params,
            kv_transfer_params_from_engine_data: _,
        } = aggregator;

        let mut choices: Vec<GenerateResponseChoice> = choices
            .into_values()
            .map(|choice| choice.into_response(options))
            .collect::<Result<_>>()?;
        choices.sort_by_key(|choice| choice.index);

        let prompt_logprobs = if options.include_prompt_logprobs {
            let prompt_logprobs = prompt_logprobs.ok_or_else(|| {
                anyhow::anyhow!("generate response requested prompt_logprobs but returned none")
            })?;
            Some(serde_json::to_value(prompt_logprobs)?)
        } else {
            None
        };

        Ok(GenerateResponse {
            request_id,
            choices,
            prompt_logprobs,
            kv_transfer_params,
        })
    }
}

impl GenerateResponse {
    /// Aggregate a raw engine stream for the non-streaming endpoint.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> Result<GenerateResponse> {
        Self::from_annotated_stream_with_options(
            stream,
            request_id,
            GenerateResponseOptions::default(),
        )
        .await
    }

    pub(crate) async fn from_annotated_stream_with_options(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
        options: GenerateResponseOptions,
    ) -> Result<GenerateResponse> {
        GenerateAggregator::apply(stream, request_id, options).await
    }

    /// A complete unary response has at least one terminal choice.
    pub fn is_complete_unary(&self) -> bool {
        !self.choices.is_empty()
            && self
                .choices
                .iter()
                .all(|choice| choice.finish_reason.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn generate_request_deserializes_from_vllm_json() {
        let raw = json!({
            "request_id": "req-123",
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {"temperature": 0.7, "max_tokens": 16},
            "model": "test-model",
            "stream": false,
            "cache_salt": "salt",
            "priority": 0,
            "kv_transfer_params": null
        });
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        assert_eq!(req.request_id.as_deref(), Some("req-123"));
        assert_eq!(req.token_ids, vec![1, 2, 3, 4]);
        assert!(!req.stream);
        assert_eq!(req.model.as_deref(), Some("test-model"));
        assert_eq!(req.cache_salt.as_deref(), Some("salt"));
        assert_eq!(req.priority, 0);
        assert!(req.kv_transfer_params.is_none());
        assert_eq!(req.sampling_params.max_tokens(), Some(16));
    }

    #[test]
    fn generate_request_captures_unknown_fields() {
        // Untyped + future fields are CAPTURED in `passthrough` (forwarded
        // verbatim), not dropped — so a field a newer vLLM adds flows through.
        let raw = json!({
            "token_ids": [5, 6],
            "sampling_params": {},
            "priority": 7,
            "future_field": "kept"
        });
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        assert_eq!(req.token_ids, vec![5, 6]);
        assert!(!req.stream);
        assert_eq!(req.request_id, None);
        assert_eq!(req.priority, 7);
        // Future fields remain captured in `passthrough`.
        assert!(!req.passthrough.contains_key("priority"));
        assert_eq!(req.passthrough.get("future_field"), Some(&json!("kept")));
        // Round-trip: flattened fields serialize back at the top level.
        let back = serde_json::to_value(&req).expect("serialize");
        assert_eq!(back.get("priority"), Some(&json!(7)));
        assert_eq!(back.get("future_field"), Some(&json!("kept")));
    }

    #[test]
    fn generate_request_preserves_unknown_sampling_fields() {
        let raw_sampling = json!({
            "temperature": 0.5,
            "top_k": 12,
            "max_tokens": 8,
            "min_tokens": null,
            "bad_words": ["blocked"],
            "future_sampling_field": {"nested": true}
        });
        let req: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [5, 6],
            "sampling_params": raw_sampling.clone()
        }))
        .expect("deserialize");

        assert_eq!(req.sampling_params.max_tokens(), Some(8));
        assert_eq!(req.sampling_params.temperature, Some(0.5));
        assert_eq!(req.sampling_params.top_k, Some(12));
        assert_eq!(
            req.sampling_params.bad_words.as_deref(),
            Some(["blocked".to_string()].as_slice())
        );
        assert_eq!(req.sampling_params.min_tokens, None);
        assert_eq!(req.sampling_params.as_value()["min_tokens"], Value::Null);
        assert_eq!(req.sampling_params.as_value(), &raw_sampling);
        let back = serde_json::to_value(&req).expect("serialize");
        assert_eq!(back["sampling_params"], raw_sampling);
    }

    #[test]
    fn generate_request_preserves_disabled_top_k_sentinel() {
        let raw_sampling = json!({"top_k": -1});
        let req: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [5, 6],
            "sampling_params": raw_sampling.clone()
        }))
        .expect("deserialize");

        assert_eq!(req.sampling_params.top_k, Some(-1));
        assert_eq!(req.sampling_params.as_value(), &raw_sampling);
        let back = serde_json::to_value(&req).expect("serialize");
        assert_eq!(back["sampling_params"], raw_sampling);
    }

    #[test]
    fn generate_request_matches_rust_integer_types() {
        for raw in [
            json!({
                "token_ids": [1],
                "sampling_params": {},
                "priority": i64::from(i32::MAX) + 1
            }),
            json!({
                "token_ids": [1],
                "sampling_params": {"max_tokens": -1}
            }),
            json!({
                "token_ids": [1],
                "sampling_params": {"ignore_eos": null}
            }),
        ] {
            assert!(serde_json::from_value::<GenerateRequest>(raw).is_err());
        }
    }

    #[test]
    fn generate_request_validates_rust_route_rules() {
        let invalid = [
            (
                json!({
                    "token_ids": [1],
                    "sampling_params": {},
                    "stream_options": {"include_usage": true}
                }),
                "stream_options",
            ),
            (
                json!({
                    "token_ids": [1],
                    "sampling_params": {"max_tokens": 0}
                }),
                "max_tokens",
            ),
            (
                json!({
                    "token_ids": [1],
                    "sampling_params": {"prompt_logprobs": -2}
                }),
                "prompt_logprobs",
            ),
            (
                json!({
                    "token_ids": [1],
                    "sampling_params": {"min_tokens": 3, "max_tokens": 2}
                }),
                "min_tokens",
            ),
        ];

        for (raw, expected) in invalid {
            let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
            let error = req.validate().expect_err("must reject");
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn generate_request_preserves_opaque_structured_outputs() {
        let raw = json!({
            "token_ids": [1],
            "sampling_params": {
                "structured_outputs": {
                    "future_constraint": {"version": 2},
                    "future_option": true
                }
            }
        });
        let req: GenerateRequest = serde_json::from_value(raw.clone()).expect("deserialize");

        assert_eq!(
            req.sampling_params.structured_outputs.as_ref(),
            Some(&raw["sampling_params"]["structured_outputs"])
        );
        assert_eq!(
            serde_json::to_value(req).expect("serialize")["sampling_params"],
            raw["sampling_params"]
        );
    }

    #[test]
    fn generate_request_rejects_empty_token_ids() {
        let request = serde_json::from_value::<GenerateRequest>(json!({
            "token_ids": [],
            "sampling_params": {}
        }))
        .expect("vLLM Rust validates empty token IDs after JSON parsing");
        let error = request
            .validate()
            .expect_err("empty token_ids must be rejected");

        assert!(error.contains("token_ids must contain at least one token"));
    }

    #[test]
    fn generate_response_matches_vllm_null_metadata_shape() {
        let resp = GenerateResponse {
            request_id: "req-123".to_string(),
            choices: vec![GenerateResponseChoice {
                index: 0,
                token_ids: None,
                logprobs: None,
                finish_reason: None,
                routed_experts: None,
            }],
            prompt_logprobs: None,
            kv_transfer_params: None,
        };

        let value = serde_json::to_value(&resp).expect("serialize");
        assert!(
            value.get("usage").is_none(),
            "GenerateResponse must not emit a `usage` key"
        );
        assert_eq!(value["prompt_logprobs"], Value::Null);
        assert_eq!(value["kv_transfer_params"], Value::Null);
        assert_eq!(value["choices"][0]["token_ids"], Value::Null);
        assert_eq!(value["choices"][0]["logprobs"], Value::Null);
        assert_eq!(value["choices"][0]["finish_reason"], Value::Null);
        assert_eq!(value["choices"][0]["routed_experts"], Value::Null);

        let round: GenerateResponse =
            serde_json::from_value(value).expect("round-trip deserialize");
        assert_eq!(round.request_id, "req-123");
        assert_eq!(round.choices.len(), 1);
        assert_eq!(round.choices[0].token_ids, None);
        assert_eq!(round.choices[0].finish_reason, None);
    }

    #[test]
    fn completion_logprobs_match_typed_wire_shape() {
        let output = LLMEngineOutput {
            token_ids: vec![100],
            log_probs: Some(vec![-1.0e30]),
            top_logprobs: Some(vec![vec![
                crate::protocols::common::llm_backend::TopLogprob {
                    rank: 1,
                    token_id: 7,
                    token: None,
                    logprob: -1.0e30,
                    bytes: None,
                },
            ]]),
            ..Default::default()
        };
        let mut content = Vec::new();

        assert!(
            append_completion_logprobs_from_output(&output, &mut content)
                .expect("build completion logprobs")
        );

        let expected = dynamo_protocols::types::ChatCompletionTokenLogprob {
            token: "token_id:100".to_string(),
            logprob: -9999.0,
            bytes: Some(b"token_id:100".to_vec()),
            top_logprobs: vec![
                dynamo_protocols::types::TopLogprobs {
                    token: "token_id:7".to_string(),
                    logprob: -9999.0,
                    bytes: Some(b"token_id:7".to_vec()),
                },
                dynamo_protocols::types::TopLogprobs {
                    token: "token_id:100".to_string(),
                    logprob: -9999.0,
                    bytes: Some(b"token_id:100".to_vec()),
                },
            ],
        };

        assert_eq!(
            content,
            vec![serde_json::to_value(expected).expect("serialize typed logprob")]
        );
    }

    #[test]
    fn completion_logprobs_reject_malformed_output() {
        let cases = [
            (
                LLMEngineOutput {
                    token_ids: vec![100],
                    top_logprobs: Some(vec![vec![]]),
                    ..Default::default()
                },
                "top_logprobs without selected-token logprobs",
            ),
            (
                LLMEngineOutput {
                    token_ids: vec![100, 101],
                    log_probs: Some(vec![-0.25]),
                    ..Default::default()
                },
                "1 selected-token logprobs for 2 tokens",
            ),
            (
                LLMEngineOutput {
                    token_ids: vec![100, 101],
                    log_probs: Some(vec![-0.25, -0.5]),
                    top_logprobs: Some(vec![vec![]]),
                    ..Default::default()
                },
                "1 top-logprob positions for 2 tokens",
            ),
        ];

        for (output, expected_error) in cases {
            let mut content = vec![json!("existing")];

            let error = append_completion_logprobs_from_output(&output, &mut content)
                .expect_err("malformed output must fail");

            assert!(
                error.to_string().contains(expected_error),
                "unexpected error: {error}"
            );
            assert_eq!(content, vec![json!("existing")]);
        }
    }

    #[test]
    fn empty_completion_logprobs_do_not_modify_content() {
        let output = LLMEngineOutput {
            log_probs: Some(vec![]),
            top_logprobs: Some(vec![]),
            ..Default::default()
        };
        let mut content = vec![json!("existing")];

        assert!(
            !append_completion_logprobs_from_output(&output, &mut content)
                .expect("empty output is valid")
        );
        assert_eq!(content, vec![json!("existing")]);
    }

    #[tokio::test]
    async fn generate_response_accumulates_engine_deltas() {
        let chunks = vec![
            LLMEngineOutput {
                token_ids: vec![100],
                index: Some(0),
                log_probs: Some(vec![-0.25]),
                top_logprobs: Some(vec![vec![
                    crate::protocols::common::llm_backend::TopLogprob {
                        rank: 1,
                        token_id: 100,
                        token: None,
                        logprob: -0.25,
                        bytes: None,
                    },
                    crate::protocols::common::llm_backend::TopLogprob {
                        rank: 2,
                        token_id: 7,
                        token: None,
                        logprob: -1.5,
                        bytes: None,
                    },
                ]]),
                engine_data: Some(json!({
                    "prompt_logprobs": [
                        null,
                        {
                            "22": {
                                "logprob": -0.125,
                                "rank": 1,
                                "decoded_token": "token_id:22"
                            }
                        }
                    ]
                })),
                ..Default::default()
            },
            LLMEngineOutput {
                token_ids: vec![101],
                index: Some(0),
                log_probs: Some(vec![-0.5]),
                top_logprobs: Some(vec![vec![
                    crate::protocols::common::llm_backend::TopLogprob {
                        rank: 1,
                        token_id: 101,
                        token: None,
                        logprob: -0.5,
                        bytes: None,
                    },
                ]]),
                finish_reason: Some(crate::protocols::common::FinishReason::Length),
                engine_data: Some(json!({
                    "routed_experts": "encoded-experts",
                    "kv_transfer_params": {"connector": "x"}
                })),
                ..Default::default()
            },
        ];
        let stream = futures::stream::iter(chunks.into_iter().map(Annotated::from_data));

        let response = GenerateResponse::from_annotated_stream_with_options(
            stream,
            "req-agg".to_string(),
            GenerateResponseOptions {
                include_logprobs: true,
                include_prompt_logprobs: true,
            },
        )
        .await
        .expect("aggregate engine deltas");

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].token_ids, Some(vec![100, 101]));
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(
            response.choices[0].routed_experts.as_deref(),
            Some("encoded-experts")
        );
        let logprobs = response.choices[0]
            .logprobs
            .as_ref()
            .expect("completion logprobs");
        assert_eq!(logprobs["content"][0]["token"], "token_id:100");
        assert_eq!(logprobs["content"][1]["token"], "token_id:101");
        assert!(logprobs.get("refusal").is_none());
        assert_eq!(
            logprobs["content"][0]["bytes"],
            serde_json::json!(b"token_id:100")
        );
        assert_eq!(
            logprobs["content"][0]["top_logprobs"][0]["bytes"],
            serde_json::json!(b"token_id:100")
        );
        assert_eq!(
            logprobs["content"][0]["top_logprobs"][1]["token"],
            "token_id:7"
        );
        assert_eq!(
            logprobs["content"][0]["top_logprobs"][1]["bytes"],
            serde_json::json!(b"token_id:7")
        );
        assert_eq!(
            response.prompt_logprobs.as_ref().expect("prompt logprobs")[1]["22"]["decoded_token"],
            "token_id:22"
        );
        assert_eq!(
            response
                .kv_transfer_params
                .as_ref()
                .expect("kv transfer params"),
            &json!({"connector": "x"})
        );
        assert!(response.is_complete_unary());
    }

    #[tokio::test]
    async fn generate_response_associates_routed_experts_with_choices() {
        let stream = futures::stream::iter([
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![201],
                index: Some(1),
                finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                engine_data: Some(json!({"routed_experts": "experts-1"})),
                ..Default::default()
            }),
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![100],
                index: Some(0),
                finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                engine_data: Some(json!({"routed_experts": "experts-0"})),
                ..Default::default()
            }),
        ]);

        let response = GenerateResponse::from_annotated_stream(stream, "req-routed".to_string())
            .await
            .expect("aggregate routed experts");

        assert_eq!(response.choices[0].index, 0);
        assert_eq!(
            response.choices[0].routed_experts.as_deref(),
            Some("experts-0")
        );
        assert_eq!(response.choices[1].index, 1);
        assert_eq!(
            response.choices[1].routed_experts.as_deref(),
            Some("experts-1")
        );
    }

    #[tokio::test]
    async fn generate_response_rejects_malformed_routed_experts() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            finish_reason: Some(crate::protocols::common::FinishReason::Stop),
            engine_data: Some(json!({"routed_experts": {"unexpected": "object"}})),
            ..Default::default()
        })]);

        let error = GenerateResponse::from_annotated_stream(stream, "req-routed".to_string())
            .await
            .expect_err("malformed routed experts must fail");

        assert!(
            error
                .to_string()
                .contains("invalid generate routed_experts payload")
        );
    }

    #[tokio::test]
    async fn generate_response_uses_terminal_kv_transfer_fallback() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            finish_reason: Some(crate::protocols::common::FinishReason::Stop),
            disaggregated_params: Some(json!({"source": "terminal-fallback"})),
            ..Default::default()
        })]);

        let response = GenerateResponse::from_annotated_stream(stream, "req-kv".to_string())
            .await
            .expect("aggregate terminal KV metadata");

        assert_eq!(
            response.kv_transfer_params,
            Some(json!({"source": "terminal-fallback"}))
        );
    }

    #[tokio::test]
    async fn generate_response_prefers_engine_kv_metadata_over_later_fallback() {
        let stream = futures::stream::iter([
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![100],
                index: Some(0),
                engine_data: Some(json!({
                    "kv_transfer_params": {"source": "engine-data"}
                })),
                ..Default::default()
            }),
            Annotated::from_data(LLMEngineOutput {
                index: Some(0),
                finish_reason: Some(crate::protocols::common::FinishReason::Stop),
                disaggregated_params: Some(json!({"source": "terminal-fallback"})),
                ..Default::default()
            }),
        ]);

        let response = GenerateResponse::from_annotated_stream(stream, "req-kv".to_string())
            .await
            .expect("aggregate KV metadata with precedence");

        assert_eq!(
            response.kv_transfer_params,
            Some(json!({"source": "engine-data"}))
        );
    }

    #[tokio::test]
    async fn incomplete_engine_streams_are_not_complete_unary_responses() {
        let empty = futures::stream::iter(Vec::<Annotated<LLMEngineOutput>>::new());
        let empty_response =
            GenerateResponse::from_annotated_stream(empty, "req-empty".to_string())
                .await
                .expect("aggregate empty stream");
        assert!(!empty_response.is_complete_unary());

        let partial = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            ..Default::default()
        })]);
        let partial_response =
            GenerateResponse::from_annotated_stream(partial, "req-partial".to_string())
                .await
                .expect("aggregate partial stream");
        assert!(!partial_response.is_complete_unary());
    }

    #[tokio::test]
    async fn generate_response_rejects_error_and_cancelled_finish_reasons() {
        for finish_reason in [
            crate::protocols::common::FinishReason::Error("worker failed".to_string()),
            crate::protocols::common::FinishReason::Cancelled,
        ] {
            let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
                index: Some(0),
                finish_reason: Some(finish_reason),
                ..Default::default()
            })]);

            GenerateResponse::from_annotated_stream(stream, "req-failed".to_string())
                .await
                .expect_err("error-like finish reasons must not produce HTTP-success responses");
        }
    }

    #[tokio::test]
    async fn generate_response_rejects_incomplete_completion_logprobs() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100, 101],
            log_probs: Some(vec![-0.25]),
            index: Some(0),
            finish_reason: Some(crate::protocols::common::FinishReason::Length),
            ..Default::default()
        })]);

        let error = GenerateResponse::from_annotated_stream_with_options(
            stream,
            "req-logprobs".to_string(),
            GenerateResponseOptions {
                include_logprobs: true,
                ..Default::default()
            },
        )
        .await
        .expect_err("misaligned logprobs must fail");

        assert!(
            error
                .to_string()
                .contains("1 selected-token logprobs for 2 tokens")
        );
    }

    #[tokio::test]
    async fn generate_response_rejects_malformed_prompt_logprobs() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            finish_reason: Some(crate::protocols::common::FinishReason::Length),
            engine_data: Some(json!({"prompt_logprobs": "invalid"})),
            ..Default::default()
        })]);

        let error = GenerateResponse::from_annotated_stream_with_options(
            stream,
            "req-prompt".to_string(),
            GenerateResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .await
        .expect_err("malformed prompt logprobs must fail");

        assert!(
            error
                .to_string()
                .contains("invalid generate prompt_logprobs payload")
        );
    }

    #[test]
    fn generate_request_derives_response_metadata_options() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [1],
            "sampling_params": {"logprobs": 0, "prompt_logprobs": 0}
        }))
        .expect("deserialize request");

        let options = request.response_options();

        assert!(options.include_logprobs);
        assert!(options.include_prompt_logprobs);
    }

    #[tokio::test]
    async fn generate_response_rejects_requested_but_missing_metadata() {
        let output = || {
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![100],
                index: Some(0),
                finish_reason: Some(crate::protocols::common::FinishReason::Length),
                ..Default::default()
            })
        };

        let completion_error = GenerateResponse::from_annotated_stream_with_options(
            futures::stream::iter([output()]),
            "req-missing-completion".to_string(),
            GenerateResponseOptions {
                include_logprobs: true,
                ..Default::default()
            },
        )
        .await
        .expect_err("missing requested completion logprobs must fail");
        assert!(completion_error.to_string().contains("returned none"));

        let prompt_error = GenerateResponse::from_annotated_stream_with_options(
            futures::stream::iter([output()]),
            "req-missing-prompt".to_string(),
            GenerateResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .await
        .expect_err("missing requested prompt logprobs must fail");
        assert!(prompt_error.to_string().contains("prompt_logprobs"));
    }

    #[tokio::test]
    async fn generate_response_does_not_leak_unrequested_metadata() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            log_probs: Some(vec![-0.25, -0.5]),
            engine_data: Some(json!({"prompt_logprobs": "invalid"})),
            index: Some(0),
            finish_reason: Some(crate::protocols::common::FinishReason::Length),
            ..Default::default()
        })]);

        let response = GenerateResponse::from_annotated_stream_with_options(
            stream,
            "req-no-metadata".to_string(),
            GenerateResponseOptions::default(),
        )
        .await
        .expect("unrequested metadata must be ignored");

        assert!(response.choices[0].logprobs.is_none());
        assert!(response.prompt_logprobs.is_none());
    }

    #[tokio::test]
    async fn generate_response_clamps_logprobs_to_vllm_floor() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            log_probs: Some(vec![-1.0e30]),
            top_logprobs: Some(vec![vec![
                crate::protocols::common::llm_backend::TopLogprob {
                    rank: 1,
                    token_id: 100,
                    token: None,
                    logprob: -1.0e30,
                    bytes: None,
                },
            ]]),
            engine_data: Some(json!({
                "prompt_logprobs": [
                    null,
                    {
                        "22": {
                            "logprob": -1.0e30,
                            "rank": 1,
                            "decoded_token": "token_id:22"
                        }
                    }
                ]
            })),
            index: Some(0),
            finish_reason: Some(crate::protocols::common::FinishReason::Length),
            ..Default::default()
        })]);

        let response = GenerateResponse::from_annotated_stream_with_options(
            stream,
            "req-clamped".to_string(),
            GenerateResponseOptions {
                include_logprobs: true,
                include_prompt_logprobs: true,
            },
        )
        .await
        .expect("aggregate clamped logprobs");
        let logprobs = response.choices[0].logprobs.as_ref().unwrap();

        assert_eq!(logprobs["content"][0]["logprob"], json!(-9999.0));
        assert_eq!(
            logprobs["content"][0]["top_logprobs"][0]["logprob"],
            json!(-9999.0)
        );
        assert_eq!(
            response.prompt_logprobs.as_ref().unwrap()[1]["22"]["logprob"],
            json!(-9999.0)
        );
    }

    #[tokio::test]
    async fn generate_response_returns_immediately_on_stream_error() {
        let stream = futures::stream::iter([Annotated::from_error("worker failed")])
            .chain(futures::stream::pending::<Annotated<LLMEngineOutput>>());

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            GenerateResponse::from_annotated_stream_with_options(
                stream,
                "req-error".to_string(),
                GenerateResponseOptions::default(),
            ),
        )
        .await
        .expect("stream error must not wait for later items");
        let error = result.expect_err("stream error must be returned");

        assert!(error.to_string().contains("worker failed"));
    }

    #[tokio::test]
    async fn generate_response_rejects_missing_completion_logprobs_before_stream_stalls() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            ..Default::default()
        })])
        .chain(futures::stream::pending::<Annotated<LLMEngineOutput>>());

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            GenerateResponse::from_annotated_stream_with_options(
                stream,
                "req-missing-logprobs".to_string(),
                GenerateResponseOptions {
                    include_logprobs: true,
                    ..Default::default()
                },
            ),
        )
        .await
        .expect("missing delta logprobs must not wait for later items");
        let error = result.expect_err("missing requested completion logprobs must fail");

        assert!(
            error
                .to_string()
                .contains("token-bearing delta returned none")
        );
    }
}
