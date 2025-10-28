// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::engines::ValidateRequest;

use super::{
    ContentProvider, OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider,
    OpenAIStopConditionsProvider,
    common::{self, OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
    common_ext::{CommonExt, CommonExtProvider},
    nvext::{NvExt, NvExtProvider},
    validate,
};

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateCompletionRequest {
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::CreateCompletionRequest,

    #[serde(flatten)]
    pub common: CommonExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,

    // metadata - passthrough parameter without restrictions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateCompletionResponse {
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::CreateCompletionResponse,
}

impl ContentProvider for dynamo_async_openai::types::Choice {
    fn content(&self) -> String {
        self.text.clone()
    }
}

pub fn prompt_to_string(prompt: &dynamo_async_openai::types::Prompt) -> String {
    match prompt {
        dynamo_async_openai::types::Prompt::String(s) => s.clone(),
        dynamo_async_openai::types::Prompt::StringArray(arr) => arr.join(" "), // Join strings with spaces
        dynamo_async_openai::types::Prompt::IntegerArray(arr) => arr
            .iter()
            .map(|&num| num.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        dynamo_async_openai::types::Prompt::ArrayOfIntegerArray(arr) => arr
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

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<&serde_json::Value> {
        self.common.guided_json.as_ref()
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
}

impl OpenAIStopConditionsProvider for NvCreateCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        None
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
        choice: dynamo_async_openai::types::Choice,
        usage: Option<dynamo_async_openai::types::CompletionUsage>,
    ) -> NvCreateCompletionResponse {
        let inner = dynamo_async_openai::types::CreateCompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            system_fingerprint: self.system_fingerprint.clone(),
            usage,
        };
        NvCreateCompletionResponse { inner }
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
            .map_err(|e| anyhow::anyhow!("Failed to extract stop conditions: {}", e))?;

        let sampling_options = request
            .extract_sampling_options()
            .map_err(|e| anyhow::anyhow!("Failed to extract sampling options: {}", e))?;

        let output_options = request
            .extract_output_options()
            .map_err(|e| anyhow::anyhow!("Failed to extract output options: {}", e))?;

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

impl TryFrom<common::StreamingCompletionResponse> for dynamo_async_openai::types::Choice {
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

        let finish_reason: Option<dynamo_async_openai::types::CompletionFinishReason> =
            response.delta.finish_reason.map(Into::into);

        let choice = dynamo_async_openai::types::Choice {
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
        self.inner
            .echo
            .and_then(|echo| if echo { Some(1) } else { None })
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        None
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }
}

/// Implements `ValidateRequest` for `NvCreateCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        validate::validate_model(&self.inner.model)?;
        validate::validate_prompt(&self.inner.prompt)?;
        validate::validate_suffix(self.inner.suffix.as_deref())?;
        validate::validate_max_tokens(self.inner.max_tokens)?;
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        validate::validate_n(self.inner.n)?;
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

        Ok(())
    }
}
