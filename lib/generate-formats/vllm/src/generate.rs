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
use base64::Engine as _;
use dynamo_protocols::types::ChatCompletionTokenLogprob;
use dynamo_runtime::error::{BackendError, DynamoError, ErrorType as DynamoErrorType};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PromptLogprobs};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use dynamo_llm::protocols::openai::validate;
use dynamo_llm::protocols::openai::{convert_backend_top_logprobs, token_to_utf8_bytes};

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

        if self.sampling_params.thinking_token_budget.is_some() {
            return Err(
                "sampling_params.thinking_token_budget is not supported by vLLM gRPC.".to_string(),
            );
        }

        validate::validate_temperature(self.sampling_params.temperature)
            .map_err(|error| error.to_string())?;
        validate::validate_top_p(self.sampling_params.top_p).map_err(|error| error.to_string())?;
        validate::validate_top_k(self.sampling_params.top_k).map_err(|error| error.to_string())?;
        validate::validate_frequency_penalty(self.sampling_params.frequency_penalty)
            .map_err(|error| error.to_string())?;
        validate::validate_presence_penalty(self.sampling_params.presence_penalty)
            .map_err(|error| error.to_string())?;
        if let Some(value) = self.sampling_params.repetition_penalty
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(format!(
                "sampling_params.repetition_penalty must be a finite positive number, got {value}"
            ));
        }
        validate::validate_min_p(self.sampling_params.min_p).map_err(|error| error.to_string())?;

        project_logprob_count(self.sampling_params.logprobs(), "logprobs")?;
        project_logprob_count(self.sampling_params.prompt_logprobs(), "prompt_logprobs")?;
        if self.sampling_params.prompt_logprobs().is_some() && self.stream {
            return Err(
                "sampling_params.prompt_logprobs are not available when stream=true.".to_string(),
            );
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
    top_k: Option<i32>,
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
    skip_special_tokens: Option<bool>,
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

    pub fn skip_reading_prefix_cache(&self) -> Option<bool> {
        self.skip_reading_prefix_cache
    }

    pub fn as_value(&self) -> &Value {
        &self.raw
    }

    pub(crate) fn project_sampling_options(&self) -> SamplingOptions {
        SamplingOptions {
            n: Some(1),
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed,
            ..Default::default()
        }
    }

    pub(crate) fn project_stop_conditions(&self) -> StopConditions {
        StopConditions {
            max_tokens: self.max_tokens,
            min_tokens: self.min_tokens,
            stop_token_ids_hidden: self.stop_token_ids.clone(),
            ignore_eos: Some(self.ignore_eos),
            ..Default::default()
        }
    }

    pub(crate) fn project_output_options(&self) -> Result<OutputOptions, String> {
        Ok(OutputOptions {
            logprobs: project_logprob_count(self.logprobs, "logprobs")?,
            prompt_logprobs: project_logprob_count(self.prompt_logprobs, "prompt_logprobs")?,
            skip_special_tokens: self.skip_special_tokens,
            ..Default::default()
        })
    }
}

fn project_logprob_count(value: Option<i32>, field: &str) -> Result<Option<u32>, String> {
    match value {
        None => Ok(None),
        Some(-1) => Ok(Some(u32::MAX)),
        Some(value) if value >= 0 => Ok(Some(value as u32)),
        Some(_) => Err(format!(
            "sampling_params.{field} must be non-negative or -1."
        )),
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
            skip_special_tokens: field!(skip_special_tokens),
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum GenerateRoutedExperts {
    Legacy(String),
    Tensor(GenerateRoutedExpertsTensor),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GenerateRoutedExpertsTensor {
    pub data: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub dtype: String,
}

impl GenerateRoutedExperts {
    fn validate(&self) -> Result<()> {
        let Self::Tensor(tensor) = self else {
            return Ok(());
        };
        anyhow::ensure!(
            tensor.shape.len() == 3 && tensor.shape.iter().all(|dimension| *dimension > 0),
            "structured routed_experts shape must contain three positive dimensions"
        );
        let width = match tensor.dtype.as_str() {
            "uint8" | "int8" => 1,
            "uint16" | "int16" => 2,
            "uint32" | "int32" => 4,
            "uint64" | "int64" => 8,
            other => anyhow::bail!("unsupported routed_experts dtype {other:?}"),
        };
        let elements = tensor.shape.iter().try_fold(1usize, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| anyhow::anyhow!("routed_experts shape overflows"))
        })?;
        let expected_bytes = elements
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("routed_experts byte count overflows"))?;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&tensor.data)
            .map_err(|error| anyhow::anyhow!("invalid routed_experts base64 data: {error}"))?;
        anyhow::ensure!(
            decoded.len() == expected_bytes,
            "routed_experts decoded byte count {} does not match shape and dtype ({expected_bytes})",
            decoded.len()
        );
        Ok(())
    }
}

/// A single choice in a `GenerateResponse`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponseChoice {
    pub index: u32,

    pub token_ids: Option<Vec<u32>>,

    pub logprobs: Option<serde_json::Value>,

    pub finish_reason: Option<String>,

    pub routed_experts: Option<GenerateRoutedExperts>,

    pub sampling_mask: Option<Vec<Vec<u32>>>,
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
    token_ids: Vec<dynamo_llm::protocols::TokenIdType>,
    logprobs: Vec<ChatCompletionTokenLogprob>,
    finish_reason: Option<String>,
    routed_experts: Option<GenerateRoutedExperts>,
    sampling_mask: Option<Vec<Vec<u32>>>,
}

impl GenerateChoiceAcc {
    fn apply(&mut self, output: &LLMEngineOutput, options: GenerateResponseOptions) -> Result<()> {
        if let Some(finish_reason) = output.finish_reason.as_ref() {
            match finish_reason {
                dynamo_llm::protocols::common::FinishReason::Error(message) => {
                    return Err(DynamoError::builder()
                        .error_type(DynamoErrorType::Backend(BackendError::Unknown))
                        .message(message)
                        .build()
                        .into());
                }
                dynamo_llm::protocols::common::FinishReason::Cancelled => {
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
            let has_logprobs = append_completion_logprobs_from_output(output, &mut self.logprobs)?;
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
            sampling_mask,
        } = self;
        let logprobs = if options.include_logprobs {
            anyhow::ensure!(
                !logprobs.is_empty(),
                "generate choice {index} requested logprobs but returned none"
            );
            anyhow::ensure!(
                logprobs.len() == token_ids.len(),
                "generate choice {index} returned {} logprob positions for {} tokens",
                logprobs.len(),
                token_ids.len()
            );
            Some(serde_json::json!({"content": logprobs}))
        } else {
            None
        };

        if let Some(rows) = sampling_mask.as_ref() {
            anyhow::ensure!(
                rows.len() == token_ids.len(),
                "generate choice {index} returned {} sampling-mask positions for {} tokens",
                rows.len(),
                token_ids.len()
            );
        }

        Ok(GenerateResponseChoice {
            index,
            token_ids: Some(token_ids),
            logprobs,
            finish_reason,
            routed_experts,
            sampling_mask,
        })
    }
}

fn clamp_vllm_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}

fn append_completion_logprobs_from_output(
    output: &LLMEngineOutput,
    content: &mut Vec<ChatCompletionTokenLogprob>,
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
        content.push(ChatCompletionTokenLogprob {
            bytes: token_to_utf8_bytes(&token),
            token,
            logprob,
            token_id: None,
            top_logprobs,
        });
    }
    Ok(true)
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
        for (token_id, entry) in entries {
            entry.logprob = clamp_vllm_logprob(entry.logprob);
            entry
                .decoded_token
                .get_or_insert_with(|| format!("token_id:{token_id}"));
        }
    }
    Ok(Some(prompt_logprobs))
}

/// Folds worker output deltas into a single
/// [`GenerateResponse`]. Each chunk carries a delta of newly generated
/// `token_ids` keyed by `index` (default 0).
pub(crate) struct GenerateAggregator {
    request_id: String,
    choices: HashMap<u32, GenerateChoiceAcc>,
    prompt_logprobs: Option<PromptLogprobs>,
    kv_transfer_params: Option<Value>,
    kv_transfer_params_from_engine_data: bool,
}

impl GenerateAggregator {
    pub(crate) fn new(request_id: String) -> Self {
        Self {
            request_id,
            choices: HashMap::new(),
            prompt_logprobs: None,
            kv_transfer_params: None,
            kv_transfer_params_from_engine_data: false,
        }
    }

    pub(crate) fn apply_output(
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
            logprobs: Vec::new(),
            finish_reason: None,
            routed_experts: None,
            sampling_mask: None,
        });
        if let Some(engine_data) = output.engine_data.as_ref() {
            if let Some(routed_experts) = engine_data.get("routed_experts") {
                let routed_experts: GenerateRoutedExperts =
                    serde_json::from_value(routed_experts.clone()).map_err(|error| {
                        anyhow::anyhow!("invalid generate routed_experts payload: {error}")
                    })?;
                routed_experts.validate().map_err(|error| {
                    anyhow::anyhow!("invalid generate routed_experts payload: {error}")
                })?;
                choice.routed_experts = Some(routed_experts);
            }
            if let Some(sampling_mask) = engine_data.get("sampling_mask") {
                choice.sampling_mask = Some(
                    serde_json::from_value(sampling_mask.clone()).map_err(|error| {
                        anyhow::anyhow!("invalid generate sampling_mask payload: {error}")
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

    pub(crate) fn finish(self, options: GenerateResponseOptions) -> Result<GenerateResponse> {
        let aggregator = self;
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
mod tests;
