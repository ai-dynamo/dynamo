// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM-compatible token-in/token-out generation protocol.
//!
//! The public request remains typed and lossless from the HTTP boundary to the
//! backend adapter. Dynamo-private routing and cancellation data belongs in a
//! separate envelope and must not be added to these wire types.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use dynamo_protocols::types::CompletionUsage;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const GENERATE_PATH: &str = "/inference/v1/generate";
pub const MAX_GENERATE_CHOICES: u32 = 4_096;

/// Private pipeline-context key for the canonical Dynamo routing headers.
pub(crate) const GENERATE_ROUTING_HINTS_CONTEXT_KEY: &str = "dynamo.llm.generate.routing_hints";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GenerateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<MultiModalFeatures>,
    pub sampling_params: GenerateSamplingParams,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub priority: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
    #[serde(skip)]
    provided_sampling_fields: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
struct GenerateRequestWire {
    request_id: Option<String>,
    model: Option<String>,
    token_ids: Vec<u32>,
    features: Option<MultiModalFeatures>,
    sampling_params: GenerateSamplingParams,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    cache_salt: Option<String>,
    #[serde(default)]
    priority: i32,
    kv_transfer_params: Option<Map<String, Value>>,
    #[serde(flatten)]
    other: Map<String, Value>,
}

impl<'de> Deserialize<'de> for GenerateRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let provided_sampling_fields = value
            .get("sampling_params")
            .and_then(Value::as_object)
            .map(|fields| fields.keys().cloned().collect())
            .unwrap_or_default();
        let wire: GenerateRequestWire =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            request_id: wire.request_id,
            model: wire.model,
            token_ids: wire.token_ids,
            features: wire.features,
            sampling_params: wire.sampling_params,
            stream: wire.stream,
            stream_options: wire.stream_options,
            cache_salt: wire.cache_salt,
            priority: wire.priority,
            kv_transfer_params: wire.kv_transfer_params,
            other: wire.other,
            provided_sampling_fields,
        })
    }
}

/// Owned request payload sent to backend workers.
///
/// Prompt tokens deliberately live only in [`PreprocessedRequest::token_ids`]
/// so routing and backend dispatch share one allocation and one wire field.
/// Sampling parameters use their original field-presence set to preserve the
/// distinction between an omitted value and an explicitly supplied `null`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct GenerateWorkerRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<MultiModalFeatures>,
    pub sampling_params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub priority: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl GenerateRequest {
    pub fn provided_sampling_fields(&self) -> &BTreeSet<String> {
        &self.provided_sampling_fields
    }

    pub fn is_sampling_param_provided(&self, name: &str) -> bool {
        self.provided_sampling_fields.contains(name)
    }

    /// Split the public request into the canonical prompt-token allocation and
    /// the backend-owned envelope that accompanies it.
    pub fn into_worker_parts(self) -> serde_json::Result<(Vec<u32>, GenerateWorkerRequest)> {
        let Self {
            request_id,
            model,
            token_ids,
            features,
            sampling_params,
            stream,
            stream_options,
            cache_salt,
            priority,
            kv_transfer_params,
            other,
            provided_sampling_fields,
        } = self;
        let mut sampling_params = serde_json::to_value(sampling_params)?
            .as_object()
            .cloned()
            .expect("GenerateSamplingParams always serializes as an object");
        for field in provided_sampling_fields {
            sampling_params.entry(field).or_insert(Value::Null);
        }

        Ok((
            token_ids,
            GenerateWorkerRequest {
                request_id,
                model,
                features,
                sampling_params,
                stream,
                stream_options,
                cache_salt,
                priority,
                kv_transfer_params,
                other,
            },
        ))
    }

    pub fn validate(&self) -> Result<(), GenerateProtocolError> {
        if self.token_ids.is_empty() {
            return Err(invalid("`token_ids` must contain at least one token ID"));
        }
        if self
            .model
            .as_ref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(invalid("`model` must be non-empty when supplied"));
        }
        if self
            .request_id
            .as_ref()
            .is_some_and(|request_id| request_id.trim().is_empty())
        {
            return Err(invalid("`request_id` must be non-empty when supplied"));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(invalid("`stream_options` requires `stream=true`"));
        }
        if self.stream && self.sampling_params.prompt_logprobs.is_some() {
            return Err(invalid(
                "`sampling_params.prompt_logprobs` are not available when `stream=true`",
            ));
        }
        self.sampling_params
            .validate_explicit_nulls(&self.provided_sampling_fields)?;
        self.sampling_params.validate()?;
        if let Some(features) = &self.features {
            features.validate(self.token_ids.len())?;
        }
        for internal in [
            "output_kind",
            "skip_clone",
            "output_text_buffer_length",
            "_eos_token_id",
            "_all_stop_token_ids",
            "_bad_words_token_ids",
        ] {
            if self.sampling_params.other.contains_key(internal) {
                return Err(invalid(format!(
                    "`sampling_params.{internal}` is engine-internal"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct GenerateSamplingParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<OneOrManyStrings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flat_logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detokenize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spaces_between_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_stop_str_in_output: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bad_words: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reading_prefix_cache: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_token_budget: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_detection: Option<RepetitionDetectionParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_experts_prompt_start: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vllm_xargs: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl GenerateSamplingParams {
    fn validate_explicit_nulls(
        &self,
        provided_fields: &BTreeSet<String>,
    ) -> Result<(), GenerateProtocolError> {
        // Keep this list aligned with the non-nullable annotations on the
        // target vLLM SamplingParams type. The fields remain Option<T> here
        // only so Dynamo can distinguish omission from an explicit JSON null.
        for (name, is_null) in [
            ("n", self.n.is_none()),
            ("presence_penalty", self.presence_penalty.is_none()),
            ("frequency_penalty", self.frequency_penalty.is_none()),
            ("repetition_penalty", self.repetition_penalty.is_none()),
            ("temperature", self.temperature.is_none()),
            ("top_p", self.top_p.is_none()),
            ("top_k", self.top_k.is_none()),
            ("min_p", self.min_p.is_none()),
            ("ignore_eos", self.ignore_eos.is_none()),
            ("min_tokens", self.min_tokens.is_none()),
            ("flat_logprobs", self.flat_logprobs.is_none()),
            ("detokenize", self.detokenize.is_none()),
            ("skip_special_tokens", self.skip_special_tokens.is_none()),
            (
                "spaces_between_special_tokens",
                self.spaces_between_special_tokens.is_none(),
            ),
            (
                "include_stop_str_in_output",
                self.include_stop_str_in_output.is_none(),
            ),
            (
                "routed_experts_prompt_start",
                self.routed_experts_prompt_start.is_none(),
            ),
        ] {
            if is_null && provided_fields.contains(name) {
                return Err(invalid(format!(
                    "`sampling_params.{name}` must not be null"
                )));
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), GenerateProtocolError> {
        if self.n == Some(0) || self.n.is_some_and(|n| n > MAX_GENERATE_CHOICES) {
            return Err(invalid(format!(
                "`sampling_params.n` must be between 1 and {MAX_GENERATE_CHOICES}"
            )));
        }
        validate_range("presence_penalty", self.presence_penalty, -2.0, 2.0, true)?;
        validate_range("frequency_penalty", self.frequency_penalty, -2.0, 2.0, true)?;
        if self
            .repetition_penalty
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(invalid(
                "`sampling_params.repetition_penalty` must be finite and greater than zero",
            ));
        }
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(invalid(
                "`sampling_params.temperature` must be finite and non-negative",
            ));
        }
        validate_range("top_p", self.top_p, 0.0, 1.0, false)?;
        validate_range("min_p", self.min_p, 0.0, 1.0, true)?;
        if self.top_k.is_some_and(|value| value < -1) {
            return Err(invalid(
                "`sampling_params.top_k` must be -1, 0, or greater than zero",
            ));
        }
        if self.max_tokens == Some(0) {
            return Err(invalid(
                "`sampling_params.max_tokens` must be greater than zero",
            ));
        }
        if let (Some(min_tokens), Some(max_tokens)) = (self.min_tokens, self.max_tokens)
            && min_tokens > max_tokens
        {
            return Err(invalid(
                "`sampling_params.min_tokens` must not exceed `max_tokens`",
            ));
        }
        if self.thinking_token_budget.is_some_and(|value| value < -1) {
            return Err(invalid(
                "`sampling_params.thinking_token_budget` must be non-negative or -1",
            ));
        }
        for (name, value) in [
            ("logprobs", self.logprobs),
            ("prompt_logprobs", self.prompt_logprobs),
        ] {
            if value.is_some_and(|value| value < -1) {
                return Err(invalid(format!(
                    "`sampling_params.{name}` must be non-negative or -1"
                )));
            }
        }
        if let (Some(logprobs), Some(token_ids)) = (self.logprobs, &self.logprob_token_ids)
            && logprobs != -1
            && usize::try_from(logprobs).ok() != Some(token_ids.len())
        {
            return Err(invalid(
                "when both are set, `sampling_params.logprobs` must equal the length of `logprob_token_ids`",
            ));
        }
        if self.temperature.is_some_and(|value| value < 1e-5) && self.n.unwrap_or(1) > 1 {
            return Err(invalid(
                "`sampling_params.n` must be one for greedy sampling",
            ));
        }
        if let Some(stop) = &self.stop {
            if stop.iter().any(str::is_empty) {
                return Err(invalid(
                    "`sampling_params.stop` cannot contain an empty string",
                ));
            }
            if !stop.is_empty() && self.detokenize == Some(false) {
                return Err(invalid("`sampling_params.stop` requires `detokenize=true`"));
            }
        }
        if self
            .bad_words
            .as_ref()
            .is_some_and(|words| words.iter().any(String::is_empty))
        {
            return Err(invalid(
                "`sampling_params.bad_words` cannot contain an empty string",
            ));
        }
        if let Some(params) = &self.repetition_detection {
            params.validate()?;
        }
        if let (Some(extra_args), Some(vllm_xargs)) = (&self.extra_args, &self.vllm_xargs)
            && let Some(key) = extra_args.keys().find(|key| vllm_xargs.contains_key(*key))
        {
            return Err(invalid(format!(
                "backend extension key `{key}` appears in both `extra_args` and `vllm_xargs`"
            )));
        }
        if self.routed_experts_prompt_start.is_some_and(|_| {
            self.extra_args
                .as_ref()
                .is_some_and(|args| args.contains_key("routed_experts_prompt_start"))
                || self
                    .vllm_xargs
                    .as_ref()
                    .is_some_and(|args| args.contains_key("routed_experts_prompt_start"))
        }) {
            return Err(invalid(
                "`routed_experts_prompt_start` must have one unambiguous representation",
            ));
        }
        Ok(())
    }

    pub fn normalized_backend_extensions(
        &self,
    ) -> Result<Map<String, Value>, GenerateProtocolError> {
        self.validate()?;
        let mut extensions = self.extra_args.clone().unwrap_or_default();
        if let Some(vllm_xargs) = &self.vllm_xargs {
            extensions.extend(vllm_xargs.clone());
        }
        if let Some(start) = self.routed_experts_prompt_start {
            extensions.insert(
                "routed_experts_prompt_start".to_string(),
                Value::from(start),
            );
        }
        Ok(extensions)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OneOrManyStrings {
    One(String),
    Many(Vec<String>),
}

impl OneOrManyStrings {
    fn iter(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::One(value) => std::slice::from_ref(value).iter().map(String::as_str),
            Self::Many(values) => values.iter().map(String::as_str),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Many(values) if values.is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RepetitionDetectionParams {
    #[serde(default)]
    pub max_pattern_size: u32,
    #[serde(default)]
    pub min_pattern_size: u32,
    #[serde(default)]
    pub min_count: u32,
}

impl RepetitionDetectionParams {
    fn validate(&self) -> Result<(), GenerateProtocolError> {
        if self.min_pattern_size > self.max_pattern_size {
            return Err(invalid(
                "repetition detection requires min_pattern_size <= max_pattern_size",
            ));
        }
        if self.max_pattern_size > 0 && self.min_count < 2 {
            return Err(invalid(
                "repetition detection requires min_count >= 2 when enabled",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuous_usage_stats: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct MultiModalFeatures {
    pub mm_hashes: HashMap<String, Vec<String>>,
    pub mm_placeholders: HashMap<String, Vec<PlaceholderRangeInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kwargs_data: Option<HashMap<String, Vec<Option<String>>>>,
}

impl MultiModalFeatures {
    fn validate(&self, token_count: usize) -> Result<(), GenerateProtocolError> {
        let hash_modalities: BTreeSet<_> = self.mm_hashes.keys().collect();
        let placeholder_modalities: BTreeSet<_> = self.mm_placeholders.keys().collect();
        if hash_modalities != placeholder_modalities {
            return Err(invalid(
                "multimodal hash and placeholder modality sets differ",
            ));
        }
        if let Some(kwargs_data) = &self.kwargs_data {
            let kwargs_modalities: BTreeSet<_> = kwargs_data.keys().collect();
            if hash_modalities != kwargs_modalities {
                return Err(invalid("multimodal kwargs and hash modality sets differ"));
            }
        }
        for (modality, placeholders) in &self.mm_placeholders {
            let hash_count = self.mm_hashes.get(modality).map_or(0, Vec::len);
            if hash_count != placeholders.len() {
                return Err(invalid(format!(
                    "multimodal `{modality}` hash and placeholder counts differ"
                )));
            }
            if placeholders.iter().any(|range| {
                range
                    .offset
                    .checked_add(range.length)
                    .is_none_or(|end| end as usize > token_count)
            }) {
                return Err(invalid(format!(
                    "multimodal `{modality}` placeholder exceeds `token_ids`"
                )));
            }
            if let Some(items) = self
                .kwargs_data
                .as_ref()
                .and_then(|kwargs| kwargs.get(modality))
                && items.len() != hash_count
            {
                return Err(invalid(format!(
                    "multimodal `{modality}` kwargs and hash counts differ"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PlaceholderRangeInfo {
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub request_id: String,
    pub model: Option<String>,
    pub created: Option<u64>,
    pub choices: Vec<GenerateResponseChoice>,
    pub usage: Option<CompletionUsage>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, GenerateLogprob>>>>,
    pub kv_transfer_params: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateResponseChoice {
    pub index: u32,
    pub logprobs: Option<GenerateChoiceLogprobs>,
    pub finish_reason: Option<String>,
    pub token_ids: Vec<u32>,
    pub routed_experts: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateStreamResponse {
    pub request_id: String,
    pub choices: Vec<GenerateStreamResponseChoice>,
    pub usage: Option<CompletionUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateStreamResponseChoice {
    pub index: u32,
    pub logprobs: Option<GenerateChoiceLogprobs>,
    pub finish_reason: Option<String>,
    pub token_ids: Vec<u32>,
    pub routed_experts: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateLogprob {
    pub logprob: f32,
    pub rank: Option<u32>,
    pub decoded_token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateChoiceLogprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<GenerateTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<GenerateTokenLogprob>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateTokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<GenerateTopLogprob>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

/// Typed worker-to-frontend metadata that does not belong in the generic
/// token delta fields. It stays binary-free except for vLLM's single base64
/// representation of routed-expert NumPy data.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateBackendMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, GenerateLogprob>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_experts: Option<String>,
    /// Prompt-side expert rows captured by an asynchronous bootstrap prefill.
    /// This is an internal worker/frontend field; the HTTP accumulator joins
    /// it with `routed_experts` into the single public NumPy payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_routed_experts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<Value>,
}

/// Prompt-owned metadata captured on the prefill leg of a disaggregated
/// Generate request. Routed-expert tensors are keyed by choice so `n > 1`
/// cannot leak one choice's prompt rows into another choice.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GeneratePrefillMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, GenerateLogprob>>>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub routed_experts_by_choice: BTreeMap<u32, String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GenerateProtocolError {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    InvalidResponse(String),
}

fn invalid(message: impl Into<String>) -> GenerateProtocolError {
    GenerateProtocolError::InvalidRequest(message.into())
}

fn validate_range(
    name: &str,
    value: Option<f32>,
    minimum: f32,
    maximum: f32,
    minimum_inclusive: bool,
) -> Result<(), GenerateProtocolError> {
    if value.is_some_and(|value| {
        !value.is_finite()
            || value > maximum
            || if minimum_inclusive {
                value < minimum
            } else {
                value <= minimum
            }
    }) {
        let lower = if minimum_inclusive { "[" } else { "(" };
        return Err(invalid(format!(
            "`sampling_params.{name}` must be finite and in {lower}{minimum}, {maximum}]"
        )));
    }
    Ok(())
}

const fn is_zero_i32(value: &i32) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_request() -> GenerateRequest {
        serde_json::from_str(include_str!(
            "../../../tests/data/inference_generate/request-full.json"
        ))
        .expect("full request fixture must deserialize")
    }

    #[test]
    fn full_request_round_trips_and_tracks_presence() {
        let request = full_request();
        request.validate().unwrap();
        assert!(request.is_sampling_param_provided("temperature"));
        assert!(request.is_sampling_param_provided("skip_reading_prefix_cache"));
        assert!(request.is_sampling_param_provided("flat_logprobs"));

        let round_trip = serde_json::to_value(&request).unwrap();
        assert_eq!(round_trip["token_ids"], serde_json::json!([11, 22, 33, 44]));
        assert_eq!(round_trip["sampling_params"]["n"], 2);
        assert_eq!(round_trip["future_top_level"]["enabled"], true);
    }

    #[test]
    fn generate_choice_count_has_a_canonical_upper_bound() {
        let request_with_n = |n| {
            serde_json::from_value::<GenerateRequest>(serde_json::json!({
                "token_ids": [1],
                "sampling_params": {"n": n}
            }))
            .unwrap()
        };

        request_with_n(4_096).validate().unwrap();
        let error = request_with_n(4_097)
            .validate()
            .expect_err("oversized n must fail at the shared protocol boundary");
        assert!(error.to_string().contains("4096"));
    }

    #[test]
    fn capability_manifest_covers_every_fixture_field() {
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../tests/data/inference_generate/capabilities.json"
        ))
        .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/data/inference_generate/request-full.json"
        ))
        .unwrap();

        let declared_top: BTreeSet<_> = manifest["request"]["top_level"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        let fixture_top: BTreeSet<_> = fixture
            .as_object()
            .unwrap()
            .keys()
            .filter_map(|key| (key != "future_top_level").then_some(key.as_str()))
            .collect();
        assert_eq!(declared_top, fixture_top);

        let declared_sampling: BTreeSet<_> = manifest["request"]["sampling"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        let fixture_sampling: BTreeSet<_> = fixture["sampling_params"]
            .as_object()
            .unwrap()
            .keys()
            .filter_map(|key| (key != "future_sampling_field").then_some(key.as_str()))
            .collect();
        assert_eq!(declared_sampling, fixture_sampling);
    }

    #[test]
    fn explicit_null_is_tracked_separately_from_absence() {
        let explicit_null: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1],
            "sampling_params": {"max_tokens": null}
        }))
        .unwrap();
        assert!(explicit_null.is_sampling_param_provided("max_tokens"));

        let absent: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1],
            "sampling_params": {}
        }))
        .unwrap();
        assert!(!absent.is_sampling_param_provided("max_tokens"));
    }

    #[test]
    fn explicit_null_matches_target_vllm_sampling_param_nullability() {
        let request_with_null = |field: &str| {
            let mut sampling_params = Map::new();
            sampling_params.insert(field.to_string(), Value::Null);
            serde_json::from_value::<GenerateRequest>(serde_json::json!({
                "token_ids": [1],
                "sampling_params": sampling_params
            }))
            .unwrap()
        };
        let non_nullable = [
            "n",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "ignore_eos",
            "min_tokens",
            "flat_logprobs",
            "detokenize",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "include_stop_str_in_output",
            "routed_experts_prompt_start",
        ];
        for field in non_nullable {
            let request = request_with_null(field);
            let error = request
                .validate()
                .expect_err("target vLLM rejects null for non-nullable fields");
            let message = error.to_string();
            assert!(message.contains(field), "unexpected error: {message}");
            assert!(
                message.contains("must not be null"),
                "unexpected error: {message}"
            );
        }

        let nullable = [
            "seed",
            "stop",
            "stop_token_ids",
            "max_tokens",
            "logprobs",
            "prompt_logprobs",
            "logprob_token_ids",
            "structured_outputs",
            "logit_bias",
            "allowed_token_ids",
            "bad_words",
            "skip_reading_prefix_cache",
            "thinking_token_budget",
            "repetition_detection",
            "extra_args",
        ];
        for field in nullable {
            let request = request_with_null(field);
            request
                .validate()
                .unwrap_or_else(|error| panic!("target vLLM permits `{field}=null`: {error}"));
        }
    }

    #[test]
    fn thinking_token_budget_accepts_unlimited_and_preserves_presence() {
        for (wire_value, expected) in [
            (Value::Null, None),
            (serde_json::json!(-1), Some(-1)),
            (serde_json::json!(0), Some(0)),
            (serde_json::json!(8), Some(8)),
        ] {
            let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                "token_ids": [1],
                "sampling_params": {"thinking_token_budget": wire_value}
            }))
            .unwrap();
            assert_eq!(request.sampling_params.thinking_token_budget, expected);
            assert!(request.validate().is_ok());
            assert!(request.is_sampling_param_provided("thinking_token_budget"));

            let (_, envelope) = request.into_worker_parts().unwrap();
            let worker_wire = serde_json::to_value(envelope).unwrap();
            assert_eq!(
                worker_wire["sampling_params"]["thinking_token_budget"],
                wire_value
            );
        }

        let omitted: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1],
            "sampling_params": {}
        }))
        .unwrap();
        assert_eq!(omitted.sampling_params.thinking_token_budget, None);
        assert!(!omitted.is_sampling_param_provided("thinking_token_budget"));

        let invalid: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1],
            "sampling_params": {"thinking_token_budget": -2}
        }))
        .unwrap();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn worker_envelope_owns_non_token_fields_and_reinserts_explicit_nulls() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "request_id": "request-1",
            "model": "model-a",
            "token_ids": [1, 2, 3],
            "sampling_params": {
                "max_tokens": null,
                "temperature": 0.5,
                "future_sampling": null
            },
            "cache_salt": "tenant-a",
            "priority": i32::MIN,
            "future_top_level": {"enabled": true}
        }))
        .unwrap();

        let (token_ids, envelope) = request.into_worker_parts().unwrap();
        assert_eq!(token_ids, vec![1, 2, 3]);
        let wire = serde_json::to_value(envelope).unwrap();
        assert!(wire.get("token_ids").is_none());
        assert_eq!(wire["sampling_params"]["max_tokens"], Value::Null);
        assert_eq!(wire["sampling_params"]["future_sampling"], Value::Null);
        assert_eq!(wire["priority"], serde_json::json!(i32::MIN));
        assert_eq!(wire["cache_salt"], "tenant-a");
        assert_eq!(wire["future_top_level"]["enabled"], true);
    }

    #[test]
    fn request_validation_rejects_invalid_cross_field_states() {
        let mut request = full_request();
        request.stream = false;
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.max_tokens = Some(0);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.extra_args =
            Some(Map::from_iter([("shared".to_string(), Value::from(1))]));
        request.sampling_params.vllm_xargs =
            Some(Map::from_iter([("shared".to_string(), Value::from(2))]));
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.temperature = Some(0.0);
        request.sampling_params.n = Some(2);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.detokenize = Some(false);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.logprobs = Some(1);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.prompt_logprobs = Some(0);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.prompt_logprobs = Some(1);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.sampling_params.prompt_logprobs = Some(-1);
        assert!(request.validate().is_err());

        let mut request = full_request();
        request.features.as_mut().unwrap().kwargs_data = None;
        request.features.as_mut().unwrap().mm_placeholders.clear();
        assert!(request.validate().is_err());
    }

    #[test]
    fn backend_extensions_normalize_without_data_loss() {
        let request = full_request();
        let extensions = request
            .sampling_params
            .normalized_backend_extensions()
            .unwrap();
        assert_eq!(extensions["python_extension"], "kept");
        assert_eq!(extensions["rust_extension"], 7);
        assert_eq!(extensions["routed_experts_prompt_start"], 3);
    }

    #[test]
    fn response_fixtures_round_trip() {
        let response: GenerateResponse = serde_json::from_str(include_str!(
            "../../../tests/data/inference_generate/response-unary.json"
        ))
        .unwrap();
        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].token_ids, vec![55, 56]);
        assert_eq!(response.model.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(response.created, Some(1_752_000_000));
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, 7);

        let stream: GenerateStreamResponse = serde_json::from_str(include_str!(
            "../../../tests/data/inference_generate/response-stream.json"
        ))
        .unwrap();
        assert_eq!(stream.choices[0].token_ids, vec![55]);

        let missing_token_ids = serde_json::from_value::<GenerateResponseChoice>(
            serde_json::json!({"index": 0, "finish_reason": "stop"}),
        );
        assert!(missing_token_ids.is_err());
    }

    #[test]
    fn unary_response_serializes_optional_fields_as_explicit_null() {
        let response = GenerateResponse {
            request_id: "request-1".to_string(),
            model: None,
            created: None,
            choices: vec![GenerateResponseChoice {
                index: 0,
                logprobs: None,
                finish_reason: None,
                token_ids: vec![],
                routed_experts: None,
            }],
            usage: None,
            prompt_logprobs: None,
            kv_transfer_params: None,
        };

        let value = serde_json::to_value(response).unwrap();
        for field in [
            "model",
            "created",
            "usage",
            "prompt_logprobs",
            "kv_transfer_params",
        ] {
            assert_eq!(value[field], Value::Null, "missing unary field {field}");
        }
        for field in ["logprobs", "finish_reason", "routed_experts"] {
            assert_eq!(
                value["choices"][0][field],
                Value::Null,
                "missing unary choice field {field}"
            );
        }

        let stream = serde_json::to_value(GenerateStreamResponse {
            request_id: "request-1".to_string(),
            choices: vec![GenerateStreamResponseChoice {
                index: 0,
                logprobs: None,
                finish_reason: None,
                token_ids: vec![1],
                routed_experts: None,
            }],
            usage: None,
        })
        .unwrap();
        assert_eq!(stream["usage"], Value::Null);
        for field in ["logprobs", "finish_reason", "routed_experts"] {
            assert_eq!(
                stream["choices"][0][field],
                Value::Null,
                "missing stream choice field {field}"
            );
        }

        let prompt_logprob = serde_json::to_value(GenerateLogprob {
            logprob: -0.25,
            rank: None,
            decoded_token: None,
        })
        .unwrap();
        assert_eq!(prompt_logprob["rank"], Value::Null);
        assert_eq!(prompt_logprob["decoded_token"], Value::Null);
    }
}
