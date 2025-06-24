// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use derive_builder::Builder;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{
    common::{self, SamplingOptionsProvider, StopConditionsProvider},
    nvext::{NvExt, NvExtProvider},
    ContentProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
};

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateCompletionRequest {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateCompletionRequest,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// Legacy OpenAI CompletionResponse
/// Represents a completion response from the API.
/// Note: both the streamed and non-streamed response objects share the same
/// shape (unlike the chat endpoint).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// A unique identifier for the completion.
    pub id: String,

    /// The list of completion choices the model generated for the input prompt.
    pub choices: Vec<CompletionChoice>,

    /// The Unix timestamp (in seconds) of when the completion was created.
    pub created: u64,

    /// The model used for completion.
    pub model: String,

    /// The object type, which is always "text_completion"
    pub object: String,

    /// Usage statistics for the completion request.
    pub usage: Option<async_openai::types::CompletionUsage>,

    /// This fingerprint represents the backend configuration that the model runs with.
    /// Can be used in conjunction with the seed request parameter to understand when backend
    /// changes have been made that might impact determinism.
    ///
    /// NIM Compatibility:
    /// This field is not supported by the NIM; however it will be added in the future.
    /// The optional nature of this field will be relaxed when it is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    // TODO(ryan)
    // pub nvext: Option<NimResponseExt>,
}

/// Legacy OpenAI CompletionResponse Choice component
#[derive(Clone, Debug, Deserialize, Serialize, Builder)]
pub struct CompletionChoice {
    #[builder(setter(into))]
    pub text: String,

    #[builder(default = "0")]
    pub index: u64,

    #[builder(default, setter(into, strip_option))]
    pub finish_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub logprobs: Option<LogprobResult>,
}

impl ContentProvider for CompletionChoice {
    fn content(&self) -> String {
        self.text.clone()
    }
}

impl CompletionChoice {
    pub fn builder() -> CompletionChoiceBuilder {
        CompletionChoiceBuilder::default()
    }
}

// TODO: validate this is the correct format
/// Legacy OpenAI LogprobResult component
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogprobResult {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<HashMap<String, f32>>,
    pub text_offset: Vec<i32>,
}

pub fn prompt_to_string(prompt: &async_openai::types::Prompt) -> String {
    match prompt {
        async_openai::types::Prompt::String(s) => s.clone(),
        async_openai::types::Prompt::StringArray(arr) => arr.join(" "), // Join strings with spaces
        async_openai::types::Prompt::IntegerArray(arr) => arr
            .iter()
            .map(|&num| num.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        async_openai::types::Prompt::ArrayOfIntegerArray(arr) => arr
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
        if let Some(nvext) = self.nvext.as_ref() {
            if let Some(use_raw_prompt) = nvext.use_raw_prompt {
                if use_raw_prompt {
                    return Some(prompt_to_string(&self.inner.prompt));
                }
            }
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
}

impl OpenAIStopConditionsProvider for NvCreateCompletionRequest {
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }

    fn get_min_tokens(&self) -> Option<u32> {
        None
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        None
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
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

    #[builder(default = "chrono::Utc::now().timestamp() as u64")]
    pub created: u64,
}

impl ResponseFactory {
    pub fn builder() -> ResponseFactoryBuilder {
        ResponseFactoryBuilder::default()
    }

    pub fn make_response(
        &self,
        choice: CompletionChoice,
        usage: Option<async_openai::types::CompletionUsage>,
    ) -> CompletionResponse {
        CompletionResponse {
            id: self.id.clone(),
            object: self.object.clone(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            system_fingerprint: self.system_fingerprint.clone(),
            usage,
        }
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

        let prompt = common::PromptType::Completion(common::CompletionContext {
            prompt: prompt_to_string(&request.inner.prompt),
            system_prompt: None,
        });

        Ok(common::CompletionRequest {
            prompt,
            stop_conditions,
            sampling_options,
            mdc_sum: None,
            annotations: None,
        })
    }
}

impl TryFrom<common::StreamingCompletionResponse> for CompletionChoice {
    type Error = anyhow::Error;

    fn try_from(response: common::StreamingCompletionResponse) -> Result<Self, Self::Error> {
        let choice = CompletionChoice {
            text: response
                .delta
                .text
                .ok_or(anyhow::anyhow!("No text in response"))?,
            index: response.delta.index.unwrap_or(0) as u64,
            logprobs: None,
            finish_reason: match &response.delta.finish_reason {
                Some(common::FinishReason::EoS) => Some("stop".to_string()),
                Some(common::FinishReason::Stop) => Some("stop".to_string()),
                Some(common::FinishReason::Length) => Some("length".to_string()),
                Some(common::FinishReason::Error(err_msg)) => {
                    return Err(anyhow::anyhow!("finish_reason::error = {}", err_msg));
                }
                Some(common::FinishReason::Cancelled) => Some("cancelled".to_string()),
                None => None,
            },
        };

        Ok(choice)
    }
}
