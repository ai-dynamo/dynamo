// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::responses::{
    Content, Input, OutputContent, OutputMessage, OutputStatus, OutputText, Response,
    Role as ResponseRole, Status,
};
use dynamo_async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
};
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;
use validator::Validate;

use super::chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse};
use super::nvext::{NvExt, NvExtProvider};
use super::{OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider};

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateResponse {
    /// Flattened CreateResponse fields (model, input, temperature, etc.)
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::responses::CreateResponse,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvResponse {
    /// Flattened Response fields.
    #[serde(flatten)]
    pub inner: dynamo_async_openai::types::responses::Response,

    /// NVIDIA extension field for response metadata (worker IDs, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
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

    fn get_seed(&self) -> Option<i64> {
        None // TODO setting as None for now
    }

    fn get_n(&self) -> Option<u8> {
        None // TODO setting as None for now
    }

    fn get_best_of(&self) -> Option<u8> {
        None // TODO setting as None for now
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

impl TryFrom<NvCreateResponse> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(resp: NvCreateResponse) -> Result<Self, Self::Error> {
        // Create messages from input
        let input_text = match resp.inner.input {
            Input::Text(text) => text,
            Input::Items(_) => {
                return Err(anyhow::anyhow!(
                    "Input::Items not supported in conversion to NvCreateChatCompletionRequest"
                ));
            }
        };

        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(input_text),
                name: None,
            },
        )];

        // TODO: See this PR for details: https://github.com/64bit/async-openai/pull/398
        let top_logprobs = convert_top_logprobs(resp.inner.top_logprobs);

        // The below should encompass all of the allowed configurable parameters
        Ok(NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                messages,
                model: resp.inner.model,
                temperature: resp.inner.temperature,
                top_p: resp.inner.top_p,
                max_completion_tokens: resp.inner.max_output_tokens,
                top_logprobs,
                metadata: resp.inner.metadata,
                stream: Some(true), // Set this to Some(True) by default to aggregate stream
                ..Default::default()
            },
            common: Default::default(),
            nvext: resp.nvext,
            chat_template_args: None,
            media_io_kwargs: None,
            unsupported_fields: Default::default(),
        })
    }
}

fn convert_top_logprobs(input: Option<u32>) -> Option<u8> {
    input.map(|x| x.min(20) as u8)
}

impl TryFrom<NvCreateChatCompletionResponse> for NvResponse {
    type Error = anyhow::Error;

    fn try_from(nv_resp: NvCreateChatCompletionResponse) -> Result<Self, Self::Error> {
        let chat_resp = nv_resp;

        // Preserve nvext field from chat completion response
        let nvext = chat_resp.nvext.clone();

        let content_text = chat_resp
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_else(|| {
                tracing::warn!("No choices in chat completion response, using empty content");
                String::new()
            });
        let message_id = format!("msg_{}", Uuid::new_v4().simple());
        let response_id = format!("resp_{}", Uuid::new_v4().simple());

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
            id: response_id,
            object: "response".to_string(),
            created_at: chat_resp.created as u64,
            model: chat_resp.model,
            status: Status::Completed,
            output,
            output_text: None,
            parallel_tool_calls: None,
            reasoning: None,
            service_tier: None,
            store: None,
            truncation: None,
            temperature: None,
            top_p: None,
            tools: None,
            metadata: None,
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

        Ok(NvResponse {
            inner: response,
            nvext,
        })
    }
}

#[cfg(test)]
mod tests {
    use dynamo_async_openai::types::responses::{CreateResponse, Input};
    use dynamo_async_openai::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    };

    use super::*;
    use crate::types::openai::chat_completions::NvCreateChatCompletionResponse;

    fn make_response_with_input(text: &str) -> NvCreateResponse {
        NvCreateResponse {
            inner: CreateResponse {
                input: Input::Text(text.into()),
                model: "test-model".into(),
                max_output_tokens: Some(1024),
                temperature: Some(0.5),
                top_p: Some(0.9),
                top_logprobs: Some(15),
                ..Default::default()
            },
            nvext: Some(NvExt {
                annotations: Some(vec!["debug".into(), "trace".into()]),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn test_annotations_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(
            req.annotations(),
            Some(vec!["debug".to_string(), "trace".to_string()])
        );
        assert!(req.has_annotation("debug"));
        assert!(req.has_annotation("trace"));
        assert!(!req.has_annotation("missing"));
    }

    #[test]
    fn test_openai_sampling_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(req.get_temperature(), Some(0.5));
        assert_eq!(req.get_top_p(), Some(0.9));
        assert_eq!(req.get_frequency_penalty(), None);
        assert_eq!(req.get_presence_penalty(), None);
    }

    #[test]
    fn test_openai_stop_conditions_trait_behavior() {
        let req = make_response_with_input("hello");
        assert_eq!(req.get_max_tokens(), Some(1024));
        assert_eq!(req.get_min_tokens(), None);
        assert_eq!(req.get_stop(), None);
    }

    #[test]
    fn test_into_nvcreate_chat_completion_request() {
        let nv_req: NvCreateChatCompletionRequest =
            make_response_with_input("hi there").try_into().unwrap();

        assert_eq!(nv_req.inner.model, "test-model");
        assert_eq!(nv_req.inner.temperature, Some(0.5));
        assert_eq!(nv_req.inner.top_p, Some(0.9));
        assert_eq!(nv_req.inner.max_completion_tokens, Some(1024));
        assert_eq!(nv_req.inner.top_logprobs, Some(15));
        assert_eq!(nv_req.inner.stream, Some(true));

        let messages = &nv_req.inner.messages;
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatCompletionRequestMessage::User(user_msg) => match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    assert_eq!(t, "hi there");
                }
                _ => panic!("unexpected user content type"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[allow(deprecated)]
    #[test]
    fn test_into_nvresponse_from_chat_response() {
        let now = 1_726_000_000;
        let chat_resp = NvCreateChatCompletionResponse {
            id: "chatcmpl-xyz".into(),
            choices: vec![dynamo_async_openai::types::ChatChoice {
                index: 0,
                message: dynamo_async_openai::types::ChatCompletionResponseMessage {
                    content: Some("This is a reply".into()),
                    refusal: None,
                    tool_calls: None,
                    role: dynamo_async_openai::types::Role::Assistant,
                    function_call: None,
                    audio: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                stop_reason: None,
                logprobs: None,
            }],
            created: now,
            model: "llama-3.1-8b-instruct".into(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: None,
            nvext: None,
        };

        let wrapped: NvResponse = chat_resp.try_into().unwrap();

        assert_eq!(wrapped.inner.model, "llama-3.1-8b-instruct");
        assert_eq!(wrapped.inner.status, Status::Completed);
        assert_eq!(wrapped.inner.object, "response");
        assert!(wrapped.inner.id.starts_with("resp_"));

        let msg = match &wrapped.inner.output[0] {
            OutputContent::Message(m) => m,
            _ => panic!("Expected Message variant"),
        };
        assert_eq!(msg.role, ResponseRole::Assistant);

        match &msg.content[0] {
            Content::OutputText(txt) => {
                assert_eq!(txt.text, "This is a reply");
            }
            _ => panic!("Expected OutputText content"),
        }
    }

    #[test]
    fn test_convert_top_logprobs_clamped() {
        assert_eq!(convert_top_logprobs(Some(5)), Some(5));
        assert_eq!(convert_top_logprobs(Some(21)), Some(20));
        assert_eq!(convert_top_logprobs(Some(1000)), Some(20));
        assert_eq!(convert_top_logprobs(None), None);
    }
}
