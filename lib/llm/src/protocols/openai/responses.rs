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

use async_openai::types::responses::{
    Content, Input, OutputContent, OutputMessage, OutputStatus, OutputText, ReasoningConfig,
    Response, Role as ResponseRole, ServiceTier, Status, Truncation,
};
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse};
use super::nvext::{NvExt, NvExtProvider};
use super::{OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider};

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::responses::CreateResponse,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::responses::Response,
}

/// Implements `NvExtProvider` for `NvCreateResponse`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateResponse {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateResponse`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateResponse {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateResponse`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateResponse {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        None // TODO setting as None for now
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        None // TODO setting as None for now
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateResponse`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateResponse {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_output_tokens
    }

    /// Retrieves the minimum number of tokens required in the response.
    ///
    /// # Note
    /// This method is currently a placeholder and always returns `None`
    /// since `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        None
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        None // TODO returning None for now
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

impl From<NvCreateResponse> for NvCreateChatCompletionRequest {
    fn from(resp: NvCreateResponse) -> Self {
        let input_text = match resp.inner.input {
            Input::Text(text) => text,
            Input::Items(_) => {
                panic!("Input::Items not supported in conversion to NvCreateChatCompletionRequest")
            }
        };

        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(input_text),
                name: None,
            },
        )];

        NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: resp.inner.model,
                messages,
                temperature: resp.inner.temperature,
                top_p: resp.inner.top_p,
                max_completion_tokens: resp.inner.max_output_tokens,
                ..Default::default()
            },
            nvext: resp.nvext,
        }
    }
}

impl From<NvCreateChatCompletionResponse> for NvResponse {
    fn from(nv_resp: NvCreateChatCompletionResponse) -> Self {
        let chat_resp = nv_resp.inner;
        let choice = chat_resp
            .choices
            .into_iter()
            .next()
            .expect("at least one choice expected");
        let content_text = choice.message.content.unwrap_or_default();
        let message_id = format!("msg_{}", Uuid::new_v4().simple());

        let output = vec![OutputContent::Message(OutputMessage {
            id: message_id,
            role: ResponseRole::Assistant,
            status: OutputStatus::Completed,
            content: vec![Content::OutputText(OutputText {
                text: content_text,
                annotations: vec![],
            })],
        })];

        let response = Response {
            id: chat_resp.id,
            object: chat_resp.object,
            created_at: chat_resp.created as u64,
            model: chat_resp.model,
            status: Status::Completed,
            output,
            output_text: None,
            parallel_tool_calls: Some(true),
            reasoning: Some(ReasoningConfig {
                effort: None,
                summary: None,
            }),
            service_tier: Some(ServiceTier::Default),
            store: None,
            truncation: Some(Truncation::Disabled),
            temperature: Some(1.0),
            top_p: Some(1.0),
            tools: Some(vec![]),
            metadata: Some(Default::default()),
            previous_response_id: None,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            text: None,
            tool_choice: None,
            usage: None,
            user: None,
        };

        NvResponse { inner: response }
    }
}
