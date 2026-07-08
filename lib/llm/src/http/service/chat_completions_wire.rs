// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Borrowed serializers for the configurable Chat Completions reasoning field.

use serde::{
    Serialize, Serializer,
    ser::{SerializeSeq, SerializeStruct},
};

use crate::{
    frontend_config::ChatCompletionsReasoningField,
    protocols::openai::chat_completions::{
        NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_protocols::types::{
    ChatChoice, ChatChoiceStream, ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta,
};

/// Borrowed wire view of a unary Chat Completions response.
pub(super) struct NvCreateChatCompletionResponseWire<'a> {
    response: &'a NvCreateChatCompletionResponse,
    reasoning_field: ChatCompletionsReasoningField,
}

impl<'a> NvCreateChatCompletionResponseWire<'a> {
    pub(super) fn new(
        response: &'a NvCreateChatCompletionResponse,
        reasoning_field: ChatCompletionsReasoningField,
    ) -> Self {
        Self {
            response,
            reasoning_field,
        }
    }
}

impl Serialize for NvCreateChatCompletionResponseWire<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.reasoning_field {
            ChatCompletionsReasoningField::ReasoningContent => self.response.serialize(serializer),
            ChatCompletionsReasoningField::Reasoning => {
                UnaryResponseWithReasoning(self.response).serialize(serializer)
            }
        }
    }
}

/// Borrowed wire view of a streaming Chat Completions response.
pub(super) struct NvCreateChatCompletionStreamResponseWire<'a> {
    response: &'a NvCreateChatCompletionStreamResponse,
    reasoning_field: ChatCompletionsReasoningField,
}

impl<'a> NvCreateChatCompletionStreamResponseWire<'a> {
    pub(super) fn new(
        response: &'a NvCreateChatCompletionStreamResponse,
        reasoning_field: ChatCompletionsReasoningField,
    ) -> Self {
        Self {
            response,
            reasoning_field,
        }
    }
}

impl Serialize for NvCreateChatCompletionStreamResponseWire<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.reasoning_field {
            ChatCompletionsReasoningField::ReasoningContent => self.response.serialize(serializer),
            ChatCompletionsReasoningField::Reasoning => {
                StreamResponseWithReasoning(self.response).serialize(serializer)
            }
        }
    }
}

struct UnaryResponseWithReasoning<'a>(&'a NvCreateChatCompletionResponse);

impl Serialize for UnaryResponseWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let response = self.0;
        let inner = &response.inner;
        let mut state = serializer.serialize_struct(
            "NvCreateChatCompletionResponse",
            8 + usize::from(response.nvext.is_some()),
        )?;
        state.serialize_field("id", &inner.id)?;
        state.serialize_field("choices", &UnaryChoicesWithReasoning(&inner.choices))?;
        state.serialize_field("created", &inner.created)?;
        state.serialize_field("model", &inner.model)?;
        state.serialize_field("service_tier", &inner.service_tier)?;
        state.serialize_field("system_fingerprint", &inner.system_fingerprint)?;
        state.serialize_field("object", &inner.object)?;
        state.serialize_field("usage", &inner.usage)?;
        if let Some(nvext) = &response.nvext {
            state.serialize_field("nvext", nvext)?;
        }
        state.end()
    }
}

struct UnaryChoicesWithReasoning<'a>(&'a [ChatChoice]);

impl Serialize for UnaryChoicesWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for choice in self.0 {
            seq.serialize_element(&UnaryChoiceWithReasoning(choice))?;
        }
        seq.end()
    }
}

struct UnaryChoiceWithReasoning<'a>(&'a ChatChoice);

impl Serialize for UnaryChoiceWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let choice = self.0;
        let mut state = serializer.serialize_struct("ChatChoice", 4)?;
        state.serialize_field("index", &choice.index)?;
        state.serialize_field("message", &UnaryMessageWithReasoning(&choice.message))?;
        state.serialize_field("finish_reason", &choice.finish_reason)?;
        state.serialize_field("logprobs", &choice.logprobs)?;
        state.end()
    }
}

struct UnaryMessageWithReasoning<'a>(&'a ChatCompletionResponseMessage);

#[allow(deprecated)]
impl Serialize for UnaryMessageWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let message = self.0;
        let mut field_count = 3;
        field_count += usize::from(message.refusal.is_some());
        field_count += usize::from(message.tool_calls.is_some());
        field_count += usize::from(message.function_call.is_some());
        field_count += usize::from(message.audio.is_some());

        let mut state =
            serializer.serialize_struct("ChatCompletionResponseMessage", field_count)?;
        state.serialize_field("content", &message.content)?;
        if let Some(refusal) = &message.refusal {
            state.serialize_field("refusal", refusal)?;
        }
        if let Some(tool_calls) = &message.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        state.serialize_field("role", &message.role)?;
        if let Some(function_call) = &message.function_call {
            state.serialize_field("function_call", function_call)?;
        }
        if let Some(audio) = &message.audio {
            state.serialize_field("audio", audio)?;
        }
        state.serialize_field("reasoning", &message.reasoning_content)?;
        state.end()
    }
}

struct StreamResponseWithReasoning<'a>(&'a NvCreateChatCompletionStreamResponse);

impl Serialize for StreamResponseWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let response = self.0;
        let inner = &response.inner;
        let mut state = serializer.serialize_struct(
            "NvCreateChatCompletionStreamResponse",
            8 + usize::from(response.nvext.is_some()),
        )?;
        state.serialize_field("id", &inner.id)?;
        state.serialize_field("choices", &StreamChoicesWithReasoning(&inner.choices))?;
        state.serialize_field("created", &inner.created)?;
        state.serialize_field("model", &inner.model)?;
        state.serialize_field("service_tier", &inner.service_tier)?;
        state.serialize_field("system_fingerprint", &inner.system_fingerprint)?;
        state.serialize_field("object", &inner.object)?;
        state.serialize_field("usage", &inner.usage)?;
        if let Some(nvext) = &response.nvext {
            state.serialize_field("nvext", nvext)?;
        }
        state.end()
    }
}

struct StreamChoicesWithReasoning<'a>(&'a [ChatChoiceStream]);

impl Serialize for StreamChoicesWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for choice in self.0 {
            seq.serialize_element(&StreamChoiceWithReasoning(choice))?;
        }
        seq.end()
    }
}

struct StreamChoiceWithReasoning<'a>(&'a ChatChoiceStream);

impl Serialize for StreamChoiceWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let choice = self.0;
        let mut state = serializer.serialize_struct("ChatChoiceStream", 4)?;
        state.serialize_field("index", &choice.index)?;
        state.serialize_field("delta", &StreamDeltaWithReasoning(&choice.delta))?;
        state.serialize_field("finish_reason", &choice.finish_reason)?;
        state.serialize_field("logprobs", &choice.logprobs)?;
        state.end()
    }
}

struct StreamDeltaWithReasoning<'a>(&'a ChatCompletionStreamResponseDelta);

#[allow(deprecated)]
impl Serialize for StreamDeltaWithReasoning<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let delta = self.0;
        let mut field_count = usize::from(delta.content.is_some());
        field_count += usize::from(delta.function_call.is_some());
        field_count += usize::from(delta.tool_calls.is_some());
        field_count += usize::from(delta.role.is_some());
        field_count += usize::from(delta.refusal.is_some());
        field_count += usize::from(delta.reasoning_content.is_some());

        let mut state =
            serializer.serialize_struct("ChatCompletionStreamResponseDelta", field_count)?;
        if let Some(content) = &delta.content {
            state.serialize_field("content", content)?;
        }
        if let Some(function_call) = &delta.function_call {
            state.serialize_field("function_call", function_call)?;
        }
        if let Some(tool_calls) = &delta.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        if let Some(role) = &delta.role {
            state.serialize_field("role", role)?;
        }
        if let Some(refusal) = &delta.refusal {
            state.serialize_field("refusal", refusal)?;
        }
        if let Some(reasoning) = &delta.reasoning_content {
            state.serialize_field("reasoning", reasoning)?;
        }
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn unary_response() -> NvCreateChatCompletionResponse {
        serde_json::from_value(json!({
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": [
                            {"type": "text", "text": "answer"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"}
                            }
                        ],
                        "role": "assistant",
                        "reasoning_content": "think",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"key\":\"value\"}"
                            }
                        }],
                        "audio": {
                            "id": "audio-1",
                            "expires_at": 2,
                            "data": "ZGF0YQ==",
                            "transcript": "answer"
                        }
                    },
                    "finish_reason": "stop",
                    "logprobs": null
                },
                {
                    "index": 1,
                    "message": {
                        "content": "answer",
                        "role": "assistant",
                        "refusal": "not refused",
                        "reasoning_content": null
                    },
                    "finish_reason": "stop",
                    "logprobs": null
                }
            ],
            "created": 1,
            "model": "test-model",
            "service_tier": null,
            "system_fingerprint": null,
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            },
            "nvext": {
                "reasoning_content": "extension-value"
            }
        }))
        .expect("valid unary response")
    }

    fn stream_response() -> NvCreateChatCompletionStreamResponse {
        serde_json::from_value(json!({
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "think",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"key\":\"value\"}"
                            }
                        }]
                    },
                    "finish_reason": null,
                    "logprobs": null
                },
                {
                    "index": 1,
                    "delta": {
                        "content": [
                            {"type": "text", "text": "answer"},
                            {
                                "type": "audio_url",
                                "audio_url": {"url": "https://example.com/audio.wav"}
                            }
                        ],
                        "refusal": "not refused"
                    },
                    "finish_reason": "stop",
                    "logprobs": null
                }
            ],
            "created": 1,
            "model": "test-model",
            "service_tier": null,
            "system_fingerprint": null,
            "object": "chat.completion.chunk",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            },
            "nvext": {
                "reasoning_content": "extension-value"
            }
        }))
        .expect("valid stream response")
    }

    #[test]
    fn unary_default_delegates_to_canonical_serializer() {
        let response = unary_response();

        let canonical = serde_json::to_string(&response).unwrap();
        let wire = serde_json::to_string(&NvCreateChatCompletionResponseWire::new(
            &response,
            ChatCompletionsReasoningField::ReasoningContent,
        ))
        .unwrap();

        assert_eq!(wire, canonical);
    }

    #[test]
    fn unary_reasoning_field_is_renamed_and_null_is_preserved() {
        let response = unary_response();
        let mut expected = serde_json::to_value(&response).unwrap();
        for choice in expected["choices"].as_array_mut().unwrap() {
            let message = choice["message"].as_object_mut().unwrap();
            let reasoning = message
                .remove("reasoning_content")
                .expect("unary reasoning field is always serialized");
            message.insert("reasoning".to_string(), reasoning);
        }
        let wire: Value = serde_json::to_value(NvCreateChatCompletionResponseWire::new(
            &response,
            ChatCompletionsReasoningField::Reasoning,
        ))
        .unwrap();

        assert_eq!(wire, expected, "only the selected field name may change");
        assert_eq!(wire["choices"][0]["message"]["reasoning"], "think");
        assert!(wire["choices"][1]["message"]["reasoning"].is_null());
        assert!(
            wire["choices"]
                .as_array()
                .unwrap()
                .iter()
                .all(|choice| choice["message"].get("reasoning_content").is_none())
        );
        assert_eq!(
            wire["nvext"]["reasoning_content"], "extension-value",
            "nested extension keys must remain canonical"
        );
    }

    #[test]
    fn stream_default_delegates_to_canonical_serializer() {
        let response = stream_response();

        let canonical = serde_json::to_string(&response).unwrap();
        let wire = serde_json::to_string(&NvCreateChatCompletionStreamResponseWire::new(
            &response,
            ChatCompletionsReasoningField::ReasoningContent,
        ))
        .unwrap();

        assert_eq!(wire, canonical);
    }

    #[test]
    fn stream_reasoning_field_is_renamed_and_none_is_omitted() {
        let response = stream_response();
        let mut expected = serde_json::to_value(&response).unwrap();
        for choice in expected["choices"].as_array_mut().unwrap() {
            let delta = choice["delta"].as_object_mut().unwrap();
            if let Some(reasoning) = delta.remove("reasoning_content") {
                delta.insert("reasoning".to_string(), reasoning);
            }
        }
        let wire: Value = serde_json::to_value(NvCreateChatCompletionStreamResponseWire::new(
            &response,
            ChatCompletionsReasoningField::Reasoning,
        ))
        .unwrap();

        assert_eq!(wire, expected, "only the selected field name may change");
        assert_eq!(wire["choices"][0]["delta"]["reasoning"], "think");
        assert!(
            wire["choices"][0]["delta"]
                .get("reasoning_content")
                .is_none()
        );
        assert!(wire["choices"][1]["delta"].get("reasoning").is_none());
        assert!(
            wire["choices"][1]["delta"]
                .get("reasoning_content")
                .is_none()
        );
        assert_eq!(
            wire["nvext"]["reasoning_content"], "extension-value",
            "nested extension keys must remain canonical"
        );
    }
}
