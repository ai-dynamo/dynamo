// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use derive_builder::Builder;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::engines::ValidateRequest;

use super::{
    ContentProvider, OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider,
    OpenAIStopConditionsProvider,
    common::{self, OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
    common_ext::{
        CommonExt, CommonExtProvider, GenerationCommonExt, extract_guided_decoding_options,
    },
    structured_outputs::{StructuredOutputs, invalid},
    validate,
};
use crate::protocols::common::extensions::{
    NvExt, NvExtProvider, validate_completion_token_ids_single_choice,
};

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

/// Completion-specific response constraint, including vLLM's structural tags.
///
/// The HTTP frontend owns shape/selection validation and override precedence;
/// the worker owns grammar compilation and constrained decoding. This is not
/// the chat response_format type, whose existing contract is unchanged.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompletionResponseFormat {
    Text {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        json_schema: Option<CompletionJsonSchema>,
    },
    JsonObject {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        json_schema: Option<CompletionJsonSchema>,
    },
    JsonSchema {
        json_schema: CompletionJsonSchema,
    },
    StructuralTag {
        #[serde(flatten)]
        specification: CompletionStructuralTag,
    },
}

/// JSON-schema metadata accepted on completions. Name is required; a missing or
/// null schema clears the matching nested JSON constraint before validation.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct CompletionJsonSchema {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub schema: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// Modern structural-tag grammar or the legacy trigger/structure representation.
/// Both forms are preserved for the backend compiler, not interpreted as JSON
/// schemas by the frontend. Modern format may be any JSON value, but is required.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum CompletionStructuralTag {
    Modern {
        #[serde(deserialize_with = "serde_json::Value::deserialize")]
        format: serde_json::Value,
        /// Preserve backend grammar members, including overlapping legacy fields.
        #[serde(flatten)]
        extra: serde_json::Map<String, serde_json::Value>,
    },
    Legacy {
        structures: Vec<CompletionLegacyTag>,
        triggers: Vec<String>,
        #[serde(flatten)]
        extra: serde_json::Map<String, serde_json::Value>,
    },
}

impl utoipa::PartialSchema for CompletionStructuralTag {
    /// Describe overlapping input forms with anyOf. Both representations can be
    /// present and reach the backend; the derive's oneOf rejects that valid overlap.
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
        use utoipa::openapi::schema::{AnyOfBuilder, ArrayBuilder, ObjectBuilder};
        AnyOfBuilder::new()
            .item(
                ObjectBuilder::new()
                    .property(
                        "format",
                        <serde_json::Value as utoipa::PartialSchema>::schema(),
                    )
                    .required("format"),
            )
            .item(
                ObjectBuilder::new()
                    .property(
                        "structures",
                        ArrayBuilder::new().items(utoipa::openapi::Ref::from_schema_name(
                            CompletionLegacyTag::name(),
                        )),
                    )
                    .property("triggers", <Vec<String> as utoipa::PartialSchema>::schema())
                    .required("structures")
                    .required("triggers"),
            )
            .into()
    }
}

impl ToSchema for CompletionStructuralTag {
    /// Register the legacy element referenced by the hand-written union schema.
    fn schemas(schemas: &mut Vec<(String, utoipa::openapi::RefOr<utoipa::openapi::Schema>)>) {
        schemas.push((
            CompletionLegacyTag::name().into_owned(),
            <CompletionLegacyTag as utoipa::PartialSchema>::schema(),
        ));
        CompletionLegacyTag::schemas(schemas);
    }
}

/// A legacy structural tag constrains content between begin/end delimiters once
/// its trigger is emitted; accepting the tag does not force the model to emit it.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct CompletionLegacyTag {
    pub begin: String,
    #[serde(default)]
    pub schema: Option<serde_json::Map<String, serde_json::Value>>,
    pub end: String,
    /// Backend grammar extensions must survive the HTTP-to-worker boundary.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateCompletionRequest {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateCompletionRequest,

    #[serde(flatten)]
    #[schema(value_type = GenerationCommonExt)]
    pub common: CommonExt,

    /// Replace the matching structured_outputs constraint. Null/text leave it
    /// unchanged; a different active constraint conflicts. Legacy guided_* fields
    /// cannot be combined with a non-text format. Grammar validity is backend-owned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<CompletionResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub nvext: Option<NvExt>,

    // metadata - passthrough parameter without restrictions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// When true, logprob token fields are returned as `"token_id:<id>"`
    /// instead of the decoded text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,

    /// Preserve matched stop strings and stop/EOS tokens in raw completion text.
    /// Special-token decoding is still controlled by `skip_special_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_stop_trim: Option<bool>,

    /// Catch-all for unsupported fields - checked during validation
    #[serde(
        flatten,
        default,
        skip_serializing,
        deserialize_with = "validate::deserialize_extra_fields"
    )]
    pub unsupported_fields: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateCompletionResponse {
    #[serde(flatten)]
    #[schema(value_type = CompletionResponseSchema)]
    pub inner: dynamo_protocols::types::CreateCompletionResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// Schema-only import for the wholly re-exported async-openai response type.
/// Resolve this explicit slot with the version-pinned OpenAPI composer; it is
/// not a claim that an arbitrary object is the completion response contract.
struct CompletionResponseSchema;

impl utoipa::PartialSchema for CompletionResponseSchema {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
        utoipa::openapi::schema::ObjectBuilder::new()
            .extensions(Some(
                [(
                    "x-dynamo-schema-import",
                    serde_json::json!({
                        "crate": "async-openai",
                        "type": "CreateCompletionResponse"
                    }),
                )]
                .into_iter()
                .collect(),
            ))
            .into()
    }
}

impl ToSchema for CompletionResponseSchema {
    fn name() -> std::borrow::Cow<'static, str> {
        "async_openai.CreateCompletionResponse".into()
    }
}

impl ContentProvider for dynamo_protocols::types::Choice {
    fn content(&self) -> String {
        self.text.clone()
    }
}

pub fn prompt_to_string(prompt: &dynamo_protocols::types::Prompt) -> String {
    match prompt {
        dynamo_protocols::types::Prompt::String(s) => s.clone(),
        dynamo_protocols::types::Prompt::StringArray(arr) => arr.join(" "), // Join strings with spaces
        dynamo_protocols::types::Prompt::IntegerArray(arr) => arr
            .iter()
            .map(|&num| num.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arr) => arr
            .iter()
            .map(|inner| {
                inner
                    .iter()
                    .map(|&num| num.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join(" | "), // Separate arrays with a delimiter
    }
}

/// Get the batch size from a prompt (1 for single prompts, array length for batch prompts)
pub fn get_prompt_batch_size(prompt: &dynamo_protocols::types::Prompt) -> usize {
    match prompt {
        dynamo_protocols::types::Prompt::String(_) => 1,
        dynamo_protocols::types::Prompt::IntegerArray(_) => 1,
        dynamo_protocols::types::Prompt::StringArray(arr) => arr.len(),
        dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arr) => arr.len(),
    }
}

/// Extract a single prompt from a batch at the given index.
/// For single prompts, returns a clone regardless of index.
/// For batch prompts, returns the prompt at the specified index.
pub fn extract_single_prompt(
    prompt: &dynamo_protocols::types::Prompt,
    index: usize,
) -> dynamo_protocols::types::Prompt {
    match prompt {
        dynamo_protocols::types::Prompt::String(s) => {
            dynamo_protocols::types::Prompt::String(s.clone())
        }
        dynamo_protocols::types::Prompt::IntegerArray(arr) => {
            dynamo_protocols::types::Prompt::IntegerArray(arr.clone())
        }
        dynamo_protocols::types::Prompt::StringArray(arr) => {
            dynamo_protocols::types::Prompt::String(arr[index].clone())
        }
        dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arr) => {
            dynamo_protocols::types::Prompt::IntegerArray(arr[index].clone())
        }
    }
}

impl NvExtProvider for NvCreateCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        if let Some(nvext) = self.nvext.as_ref()
            && let Some(use_raw_prompt) = nvext.use_raw_prompt
            && use_raw_prompt
        {
            return Some(prompt_to_string(&self.inner.prompt));
        }
        None
    }

    fn unsupported_fields(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        Some(&self.unsupported_fields)
    }
}

impl AnnotationsProvider for NvCreateCompletionRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

impl OpenAISamplingOptionsProvider for NvCreateCompletionRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn get_seed(&self) -> Option<i64> {
        self.inner.seed
    }

    fn get_n(&self) -> Option<u8> {
        self.inner.n
    }

    fn get_best_of(&self) -> Option<u8> {
        self.inner.best_of
    }
}

impl CommonExtProvider for NvCreateCompletionRequest {
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.common)
    }

    /// Apply completion format precedence using the shared constraint validator.
    ///
    /// Null/text preserve legacy behavior. Other formats replace only their own
    /// nested member, preserving modifiers and rejecting competing constraints.
    /// Legacy guided_* mixtures remain unsupported. See CommonExtProvider for
    /// return/error semantics; this method does not compile backend grammars.
    fn resolve_guided_decoding(&self) -> anyhow::Result<Option<common::GuidedDecodingOptions>> {
        let Some(format) = &self.response_format else {
            return extract_guided_decoding_options(self);
        };
        if matches!(format, CompletionResponseFormat::Text { .. }) {
            return extract_guided_decoding_options(self);
        }
        if self.common.guided_json.is_some()
            || self.common.guided_regex.is_some()
            || self.common.guided_choice.is_some()
            || self.common.guided_grammar.is_some()
            || self.common.guided_decoding_backend.is_some()
            || self.common.guided_whitespace_pattern.is_some()
        {
            return Err(invalid(
                "completion response_format cannot be combined with legacy guided_* fields",
            ));
        }
        let mut effective = if let Some(nested) = &self.common.structured_outputs {
            nested.validate_selection()?;
            nested.clone()
        } else {
            StructuredOutputs::default()
        };
        match format {
            CompletionResponseFormat::Text { .. } => unreachable!("text handled above"),
            CompletionResponseFormat::JsonObject { .. } => effective.json_object = Some(true),
            CompletionResponseFormat::JsonSchema { json_schema } => {
                effective.json = json_schema.schema.clone().map(serde_json::Value::Object);
            }
            CompletionResponseFormat::StructuralTag { .. } => {
                effective.structural_tag = Some(serde_json::to_string(format)?);
            }
        }
        effective.into_guided().map(Some)
    }

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<serde_json::Value> {
        self.common.guided_json.clone()
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.common.guided_regex.clone()
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.common.guided_grammar.clone()
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.common.guided_choice.clone()
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.common.skip_special_tokens
    }

    fn get_prompt_logprobs_count(&self) -> Option<u32> {
        self.common.prompt_logprobs
    }
}

impl OpenAIStopConditionsProvider for NvCreateCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().and_then(|stop| stop.strings())
    }

    fn get_stop_token_ids(&self) -> Option<Vec<crate::types::TokenIdType>> {
        if let Some(ids) = self.inner.stop.as_ref().and_then(|stop| stop.token_ids()) {
            return Some(ids);
        }
        self.unsupported_fields
            .get("stop_token_ids")
            .and_then(|v| serde_json::from_value::<Vec<crate::types::TokenIdType>>(v.clone()).ok())
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }

    /// Get the effective ignore_eos value from CommonExt.
    fn get_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }
}

#[derive(Builder)]
pub struct ResponseFactory {
    #[builder(setter(into))]
    pub model: String,

    #[builder(default)]
    pub system_fingerprint: Option<String>,

    #[builder(default = "format!(\"cmpl-{}\", uuid::Uuid::new_v4())")]
    pub id: String,

    #[builder(default = "\"text_completion\".to_string()")]
    pub object: String,

    #[builder(default = "chrono::Utc::now().timestamp() as u32")]
    pub created: u32,
}

impl ResponseFactory {
    pub fn builder() -> ResponseFactoryBuilder {
        ResponseFactoryBuilder::default()
    }

    pub fn make_response(
        &self,
        choice: dynamo_protocols::types::Choice,
        usage: Option<dynamo_protocols::types::CompletionUsage>,
    ) -> NvCreateCompletionResponse {
        let inner = dynamo_protocols::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            system_fingerprint: self.system_fingerprint.clone(),
            usage,
        };
        NvCreateCompletionResponse { inner, nvext: None }
    }
}

/// Implements TryFrom for converting an OpenAI's CompletionRequest to an Engine's CompletionRequest
impl TryFrom<NvCreateCompletionRequest> for common::CompletionRequest {
    type Error = anyhow::Error;

    fn try_from(request: NvCreateCompletionRequest) -> Result<Self, Self::Error> {
        // openai_api_rs::v1::completion::CompletionRequest {
        // NA  pub model: String,
        //     pub prompt: String,
        // **  pub suffix: Option<String>,
        //     pub max_tokens: Option<i32>,
        //     pub temperature: Option<f32>,
        //     pub top_p: Option<f32>,
        //     pub n: Option<i32>,
        //     pub stream: Option<bool>,
        //     pub logprobs: Option<i32>,
        //     pub echo: Option<bool>,
        //     pub stop: Option<Vec<String, Global>>,
        //     pub presence_penalty: Option<f32>,
        //     pub frequency_penalty: Option<f32>,
        //     pub best_of: Option<i32>,
        //     pub logit_bias: Option<HashMap<String, i32, RandomState>>,
        //     pub user: Option<String>,
        // }
        //
        // ** no supported

        if request.inner.suffix.is_some() {
            return Err(anyhow::anyhow!("suffix is not supported"));
        }

        let stop_conditions = request
            .extract_stop_conditions()
            .context("Failed to extract stop conditions")?;

        let sampling_options = request
            .extract_sampling_options()
            .context("Failed to extract sampling options")?;

        let output_options = request
            .extract_output_options()
            .context("Failed to extract output options")?;

        let prompt = common::PromptType::Completion(common::CompletionContext {
            prompt: prompt_to_string(&request.inner.prompt),
            system_prompt: None,
        });

        Ok(common::CompletionRequest {
            prompt,
            stop_conditions,
            sampling_options,
            output_options,
            mdc_sum: None,
            annotations: None,
        })
    }
}

impl TryFrom<common::StreamingCompletionResponse> for dynamo_protocols::types::Choice {
    type Error = anyhow::Error;

    fn try_from(response: common::StreamingCompletionResponse) -> Result<Self, Self::Error> {
        let text = response
            .delta
            .text
            .ok_or(anyhow::anyhow!("No text in response"))?;

        // SAFETY: we're downcasting from u64 to u32 here but u32::MAX is 4_294_967_295
        // so we're fairly safe knowing we won't generate that many Choices
        let index: u32 = response
            .delta
            .index
            .unwrap_or(0)
            .try_into()
            .expect("index exceeds u32::MAX");

        // TODO handle aggregating logprobs
        let logprobs = None;

        let finish_reason: Option<dynamo_protocols::types::CompletionFinishReason> =
            response.delta.finish_reason.map(Into::into);

        let choice = dynamo_protocols::types::Choice {
            text,
            index,
            logprobs,
            finish_reason,
        };

        Ok(choice)
    }
}

impl OpenAIOutputOptionsProvider for NvCreateCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        self.inner.logprobs.map(|logprobs| logprobs as u32)
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        self.common.prompt_logprobs.or_else(|| {
            self.inner
                .echo
                .and_then(|echo| if echo { Some(1) } else { None })
        })
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        CommonExtProvider::get_skip_special_tokens(self)
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }

    fn get_return_tokens_as_token_ids(&self) -> Option<bool> {
        self.return_tokens_as_token_ids
    }

    fn get_no_stop_trim(&self) -> Option<bool> {
        self.no_stop_trim
    }
}

/// Implements `ValidateRequest` for `NvCreateCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        validate::validate_no_unsupported_fields(&self.unsupported_fields)?;
        validate::validate_guided_decoding(self)?;
        validate::validate_model(&self.inner.model)?;

        // Validate prompt and prompt_embeds together (checks presence, format, and content)
        validate::validate_prompt_or_embeds(
            Some(&self.inner.prompt),
            self.inner.prompt_embeds.as_deref(),
        )?;

        validate::validate_suffix(self.inner.suffix.as_deref())?;
        validate::validate_max_tokens(self.inner.max_tokens)?;
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        validate::validate_n(self.inner.n)?;
        validate_completion_token_ids_single_choice(
            get_prompt_batch_size(&self.inner.prompt) * self.inner.n.unwrap_or(1) as usize,
            self.nvext.as_ref(),
        )?;
        // none for stream
        // none for stream_options
        validate::validate_logprobs(self.inner.logprobs)?;
        // none for echo
        validate::validate_stop(&self.inner.stop)?;
        validate::validate_presence_penalty(self.inner.presence_penalty)?;
        validate::validate_frequency_penalty(self.inner.frequency_penalty)?;
        validate::validate_best_of(self.inner.best_of, self.inner.n)?;
        validate::validate_logit_bias(&self.inner.logit_bias)?;
        validate::validate_user(self.inner.user.as_deref())?;
        // none for seed
        // none for metadata

        // Common Ext
        validate::validate_repetition_penalty(self.get_repetition_penalty())?;
        validate::validate_min_p(self.get_min_p())?;
        validate::validate_top_k(self.get_top_k())?;
        // Cross-field validation
        validate::validate_n_with_temperature(self.inner.n, self.inner.temperature)?;
        // total choices validation for completions batch requests
        validate::validate_total_choices(
            get_prompt_batch_size(&self.inner.prompt),
            self.inner.n.unwrap_or(1),
        )?;
        validate::validate_chat_only_generation_flags(
            self.common.add_generation_prompt,
            self.common.continue_final_message,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod conversion_error_tests {
    use super::*;

    #[test]
    fn no_stop_trim_is_validated_and_carried_to_the_decoder() {
        for no_stop_trim in [None, Some(false), Some(true)] {
            let mut body =
                serde_json::json!({"model": "test", "prompt": "hi", "skip_special_tokens": false});
            if let Some(value) = no_stop_trim {
                body["no_stop_trim"] = value.into();
            }
            let request: NvCreateCompletionRequest = serde_json::from_value(body).unwrap();
            ValidateRequest::validate(&request).unwrap();
            let options = request.extract_output_options().unwrap();
            assert_eq!(options.no_stop_trim, no_stop_trim);
            assert_eq!(options.skip_special_tokens, Some(false));
            let round_trip: common::OutputOptions =
                serde_json::from_value(serde_json::to_value(options).unwrap()).unwrap();
            assert_eq!(round_trip.no_stop_trim, no_stop_trim);
        }
        for value in [serde_json::json!("true"), serde_json::json!(1)] {
            assert!(
                serde_json::from_value::<NvCreateCompletionRequest>(serde_json::json!({
                    "model": "test", "prompt": "hi", "no_stop_trim": value
                }))
                .is_err()
            );
        }
    }

    /// `anyhow!("{e}")` builds a fresh error with no source, which drops the
    /// `DynamoError` and sends a caller error to `ErrorMessage::from_anyhow` as a 500.
    /// `.context()` keeps the chain, so the conflict still maps to 400 here.
    #[test]
    fn guided_decoding_conflict_stays_typed_through_conversion() {
        let request: NvCreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "hi",
            "guided_json": {"type": "object"},
            "guided_regex": "a+",
        }))
        .expect("request should deserialize");

        let error = common::CompletionRequest::try_from(request).unwrap_err();
        let dynamo_error = error
            .chain()
            .find_map(|source| source.downcast_ref::<dynamo_runtime::error::DynamoError>())
            .expect("conversion must preserve the HTTP error type");
        assert_eq!(
            dynamo_error.error_type(),
            dynamo_runtime::error::ErrorType::InvalidArgument
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use crate::protocols::common::OutputOptionsProvider;
    use base64::Engine;
    use serde_json::json;

    /// Parse and validate a completion fragment, then return exactly the worker
    /// constraint produced by sampling extraction. Panics on unexpected rejection.
    fn completion_constraint(fields: serde_json::Value) -> Option<serde_json::Value> {
        let mut value = json!({"model": "test", "prompt": "hello"});
        value
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let request: NvCreateCompletionRequest = serde_json::from_value(value).unwrap();
        ValidateRequest::validate(&request).unwrap();
        request
            .extract_sampling_options()
            .unwrap()
            .guided_decoding
            .map(|options| serde_json::to_value(options).unwrap())
    }

    /// Completion formats must reach worker sampling, not merely leave the
    /// unsupported-field map. Both structural-tag representations survive intact.
    #[test]
    fn completion_response_format_delivers_each_constraint() {
        for (format, member, expected) in [
            (
                json!({"type":"json_schema", "json_schema":{"name":"answer", "schema":{"const":"violet"}}}),
                "json",
                json!({"const":"violet"}),
            ),
            (
                json!({"type":"json_object"}),
                "json",
                json!({"type":"object"}),
            ),
            (
                json!({"type":"structural_tag", "format":{"type":"const_string", "value":"violet"}}),
                "structural_tag",
                serde_json::Value::Null,
            ),
            (
                json!({"type":"structural_tag", "structures":[{"begin":"<a>", "schema":{"const":"violet"}, "end":"</a>"}], "triggers":["<a>"]}),
                "structural_tag",
                serde_json::Value::Null,
            ),
        ] {
            let constraint = completion_constraint(json!({"response_format":format})).unwrap();
            if member == "structural_tag" {
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(constraint[member].as_str().unwrap())
                        .unwrap(),
                    format
                );
            } else {
                assert_eq!(constraint[member], expected);
            }
        }
    }

    /// Same-kind formats replace malformed nested content before content validation;
    /// text/null preserve constraints, while null JSON schemas may erase only JSON.
    #[test]
    fn completion_response_format_overrides_only_its_matching_member() {
        let format = json!({"type":"json_schema", "json_schema":{"name":"answer", "schema":{"const":"violet"}}});
        let options = completion_constraint(json!({
            "response_format":format,
            "structured_outputs":{"json":"{", "disable_any_whitespace":true}
        }))
        .unwrap();
        assert_eq!(options["json"], json!({"const":"violet"}));
        assert_eq!(options["disable_any_whitespace"], true);
        assert_eq!(completion_constraint(json!({"response_format":{"type":"json_object"}, "structured_outputs":{"json_object":false}})).unwrap()["json"], json!({"type":"object"}));
        for format in [
            serde_json::Value::Null,
            json!({"type":"text"}),
            json!({"type":"json_schema", "json_schema":{"name":"answer", "schema":null}}),
        ] {
            assert_eq!(
                completion_constraint(
                    json!({"response_format":format, "structured_outputs":{"regex":"violet"}})
                )
                .unwrap()["regex"],
                "violet"
            );
        }
        for format in [serde_json::Value::Null, json!({"type":"text"})] {
            assert!(completion_constraint(json!({"response_format":format})).is_none());
            assert_eq!(
                completion_constraint(json!({"response_format":format, "guided_regex":"violet"}))
                    .unwrap()["regex"],
                "violet"
            );
        }
    }

    /// Invalid wire shapes fail parsing; missing/competing effective constraints
    /// fail admission and sampling consistently, without invoking an engine.
    #[test]
    fn completion_response_format_rejects_invalid_or_conflicting_constraints() {
        for format in [
            json!({"type":"unknown"}),
            json!({"type":"json_schema"}),
            json!({"type":"json_schema", "json_schema":{"schema":{}}}),
            json!({"type":"json_schema", "json_schema":{"name":"x", "schema":true}}),
            json!({"type":"structural_tag"}),
            json!({"type":"structural_tag", "structures":[], "triggers":1}),
        ] {
            assert!(
                serde_json::from_value::<NvCreateCompletionRequest>(
                    json!({"model":"test", "prompt":"hello", "response_format":format})
                )
                .is_err()
            );
        }
        for fields in [
            json!({"response_format":{"type":"json_schema", "json_schema":{"name":"x"}}}),
            json!({"response_format":{"type":"json_object"}, "structured_outputs":{"regex":"violet"}}),
            json!({"response_format":{"type":"json_object"}, "structured_outputs":{}}),
            json!({"response_format":{"type":"json_object"}, "guided_regex":"violet"}),
            json!({"response_format":{"type":"structural_tag", "format":{}}, "structured_outputs":{"json":{}}}),
        ] {
            let mut value = json!({"model":"test", "prompt":"hello"});
            value
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let request: NvCreateCompletionRequest = serde_json::from_value(value).unwrap();
            assert!(ValidateRequest::validate(&request).is_err());
            assert!(request.extract_sampling_options().is_err());
        }
    }

    #[test]
    fn test_conflicting_guided_decoding_options_fail_request_validation() {
        let request: NvCreateCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": "hello",
            "guided_regex": "a+",
            "guided_choice": ["a"]
        }))
        .expect("request should deserialize");

        let error = ValidateRequest::validate(&request).expect_err("constraints conflict");
        assert!(
            error
                .to_string()
                .contains("Only one guided-decoding constraint")
        );
    }

    #[test]
    fn test_skip_special_tokens_none() {
        let json_str = json!({
            "model": "test-model",
            "prompt": "Hello, world!"
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert_eq!(request.common.skip_special_tokens, None);

        let output_options = request
            .extract_output_options()
            .expect("Failed to extract output options");

        assert_eq!(output_options.skip_special_tokens, None);
    }

    #[test]
    fn test_skip_special_tokens_propagates() {
        for skip_value in [true, false] {
            let json_str = json!({
                "model": "test-model",
                "prompt": "Hello, world!",
                "skip_special_tokens": skip_value
            });

            let request: NvCreateCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");

            let output_options = request
                .extract_output_options()
                .expect("Failed to extract output options");

            assert_eq!(output_options.skip_special_tokens, Some(skip_value));
        }
    }

    #[test]
    fn test_validate_rejects_chat_only_generation_flags() {
        for extra in [
            json!({"add_generation_prompt": false}),
            json!({"continue_final_message": true}),
        ] {
            let mut body = json!({
                "model": "test-model",
                "prompt": "Hello, world!"
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let request: NvCreateCompletionRequest =
                serde_json::from_value(body).expect("Failed to deserialize request");
            let err = ValidateRequest::validate(&request)
                .expect_err("chat-only generation flags must be rejected on completions");
            assert!(
                err.to_string().contains("/v1/chat/completions"),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn test_prompt_logprobs_propagates() {
        let request_json = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "prompt_logprobs": 3
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let output_options = request
            .extract_output_options()
            .expect("Failed to extract output options");
        assert_eq!(output_options.prompt_logprobs, Some(3));
    }

    #[test]
    fn test_prompt_embeds_only() {
        // Create valid embeddings: > 100 bytes (PyTorch format)
        let valid_data = vec![0u8; 256];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&valid_data);

        let json_str = json!({
            "model": "test-model",
            "prompt": "test",
            "prompt_embeds": encoded
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert!(ValidateRequest::validate(&request).is_ok());
        assert!(request.inner.prompt_embeds.is_some());
    }

    #[test]
    fn test_both_prompt_and_embeds() {
        // Both fields are allowed, prompt_embeds takes precedence at worker level
        // Create valid embeddings: > 100 bytes (PyTorch format)
        let valid_data = vec![0u8; 256];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&valid_data);

        let json_str = json!({
            "model": "test-model",
            "prompt": "Hello",
            "prompt_embeds": encoded
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert!(ValidateRequest::validate(&request).is_ok());
    }

    #[test]
    fn test_invalid_base64() {
        // Create invalid base64 that's long enough (>100 bytes) to pass size check
        // Use characters that look like base64 but aren't valid
        let invalid_base64 = "not-valid-base64!!!".repeat(10); // 190 bytes, looks like base64 but invalid

        let json_str = json!({
            "model": "test-model",
            "prompt": "test",
            "prompt_embeds": invalid_base64
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        let result = ValidateRequest::validate(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base64"));
    }

    #[test]
    fn test_embeds_too_large() {
        // Create embeddings with DECODED size larger than 10MB
        // Base64 encoding adds ~33% overhead, so we need 11MB decoded = ~14.7MB encoded
        let large_data = vec![0u8; 11 * 1024 * 1024]; // 11MB decoded
        let large_embeds = base64::engine::general_purpose::STANDARD.encode(&large_data);

        let json_str = json!({
            "model": "test-model",
            "prompt": "test",
            "prompt_embeds": large_embeds
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        let result = ValidateRequest::validate(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("10MB"));
    }

    #[test]
    fn test_embeds_too_small() {
        // Create embeddings with DECODED size smaller than 100 bytes
        let small_data = vec![0u8; 20]; // Only 20 bytes when decoded
        let encoded = base64::engine::general_purpose::STANDARD.encode(&small_data);

        let json_str = json!({
            "model": "test-model",
            "prompt": "test",
            "prompt_embeds": encoded
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        let result = ValidateRequest::validate(&request);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("100 bytes")
                || err_msg.contains("at least")
                || err_msg.contains("decoded")
        );
    }

    #[test]
    fn test_embeddings_with_empty_prompt() {
        // Test that empty prompt is ALLOWED when embeddings provided
        let valid_data = vec![0u8; 256]; // Valid size and aligned
        let encoded = base64::engine::general_purpose::STANDARD.encode(&valid_data);

        let json_str = json!({
            "model": "test-model",
            "prompt": "", // Empty prompt is OK with embeddings
            "prompt_embeds": encoded
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        // Should succeed - embeddings take precedence, prompt can be empty
        assert!(ValidateRequest::validate(&request).is_ok());
    }

    #[test]
    fn test_empty_prompt_without_embeddings_fails() {
        // Empty prompt WITHOUT embeddings should fail
        let json_str = json!({
            "model": "test-model",
            "prompt": "",  // Empty prompt
            // No prompt_embeds
        });

        let request: NvCreateCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        let result = ValidateRequest::validate(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_stop() {
        let null_stop = json!({
            "model": "test-model",
            "prompt": "Hello, world!"
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(null_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), None);
        assert_eq!(request.get_stop_token_ids(), None);

        let one_stop = json!({
            "model": "test-model",
            "prompt": "Hello, world!",
            "stop": "foo"
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(one_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["foo".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let many_stops = json!({
            "model": "test-model",
            "prompt": "Hello, world!",
            "stop": ["foo", "bar"]
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(many_stops).expect("Failed to deserialize request");
        assert_eq!(
            request.get_stop(),
            Some(vec!["foo".to_string(), "bar".to_string()])
        );
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_stop = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "stop": [32, 34]
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(token_id_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), None);
        assert_eq!(request.get_stop_token_ids(), Some(vec![32, 34]));

        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");
        assert_eq!(stop_conditions.stop, None);
        assert_eq!(stop_conditions.stop_token_ids, Some(vec![32, 34]));
        assert_eq!(stop_conditions.stop_token_ids_hidden, None);

        let token_id_display_string_scalar_stop = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "stop": "token_id:576"
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(token_id_display_string_scalar_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_display_string_stop = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "stop": ["token_id:576"]
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(token_id_display_string_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        // Top-level stop_token_ids should be accepted and plumbed.
        let whitelisted_stop_token_ids = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "stop_token_ids": [576]
        });
        let request: NvCreateCompletionRequest = serde_json::from_value(whitelisted_stop_token_ids)
            .expect("Failed to deserialize request");
        assert_eq!(request.get_stop_token_ids(), Some(vec![576]));
        assert!(
            ValidateRequest::validate(&request).is_ok(),
            "stop_token_ids must be accepted via PASSTHROUGH_EXTRA_FIELDS"
        );

        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");
        assert_eq!(stop_conditions.stop_token_ids, Some(vec![576]));

        let invalid_stop_token_ids = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "stop_token_ids": "bad"
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(invalid_stop_token_ids).expect("Failed to deserialize request");
        let err = ValidateRequest::validate(&request).expect_err("invalid stop_token_ids");
        assert!(err.to_string().contains("stop_token_ids"));
    }

    #[test]
    fn test_passthrough_token_constraints_validate() {
        let request_json = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "allowed_token_ids": [10, 11],
            "bad_words_token_ids": [[12, 13]]
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert_eq!(
            request.unsupported_fields.get("allowed_token_ids"),
            Some(&serde_json::json!([10, 11]))
        );
        assert_eq!(
            request.unsupported_fields.get("bad_words_token_ids"),
            Some(&serde_json::json!([[12, 13]]))
        );
        assert!(ValidateRequest::validate(&request).is_ok());
    }

    #[test]
    fn test_completion_token_ids_rejected_for_multi_choice() {
        let request_json = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "n": 2,
            "nvext": {
                "extra_fields": ["completion_token_ids"]
            }
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("multi-choice token ids");
        assert!(err.to_string().contains("completion_token_ids"));
    }

    #[test]
    fn test_completion_token_ids_rejected_for_prompt_batch() {
        let request_json = json!({
            "model": "test-model",
            "prompt": ["first", "second"],
            "n": 1,
            "nvext": {
                "extra_fields": ["completion_token_ids"]
            }
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("prompt batch token ids");
        assert!(err.to_string().contains("completion_token_ids"));
    }

    #[test]
    fn test_truncate_prompt_tokens_rejected_until_supported() {
        let request_json = json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "truncate_prompt_tokens": 2
        });
        let request: NvCreateCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert!(ValidateRequest::validate(&request).is_err());
    }
}
