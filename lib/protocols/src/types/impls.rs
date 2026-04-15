// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Convenience trait implementations for locally-defined types.
// Types re-exported from upstream async-openai already have their own impls.

use std::fmt::Display;

use super::{
    AudioUrl, ChatCompletionNamedToolChoice, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartAudio, ChatCompletionRequestMessageContentPartAudioUrl,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartVideo, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionToolChoiceOption, ChatCompletionToolType, FunctionName, ImageUrl, VideoUrl,
};

use crate::error::OpenAIError;

// --- From impls for locally-defined types ---

impl From<&str> for FunctionName {
    fn from(value: &str) -> Self {
        Self { name: value.into() }
    }
}

impl From<String> for FunctionName {
    fn from(value: String) -> Self {
        Self { name: value }
    }
}

impl From<&str> for ChatCompletionNamedToolChoice {
    fn from(value: &str) -> Self {
        Self {
            r#type: ChatCompletionToolType::Function,
            function: value.into(),
        }
    }
}

impl From<String> for ChatCompletionNamedToolChoice {
    fn from(value: String) -> Self {
        Self {
            r#type: ChatCompletionToolType::Function,
            function: value.into(),
        }
    }
}

impl From<&str> for ChatCompletionToolChoiceOption {
    fn from(value: &str) -> Self {
        match value {
            "auto" => Self::Auto,
            "none" => Self::None,
            _ => Self::Named(value.into()),
        }
    }
}

impl From<String> for ChatCompletionToolChoiceOption {
    fn from(value: String) -> Self {
        match value.as_str() {
            "auto" => Self::Auto,
            "none" => Self::None,
            _ => Self::Named(value.into()),
        }
    }
}

// From message types into ChatCompletionRequestMessage enum
// Note: types from upstream (SystemMessage, ToolMessage, etc.) need From impls
// on our local ChatCompletionRequestMessage enum.

impl From<super::ChatCompletionRequestUserMessage> for ChatCompletionRequestMessage {
    fn from(value: super::ChatCompletionRequestUserMessage) -> Self {
        Self::User(value)
    }
}

impl From<async_openai::types::chat::ChatCompletionRequestSystemMessage>
    for ChatCompletionRequestMessage
{
    fn from(value: async_openai::types::chat::ChatCompletionRequestSystemMessage) -> Self {
        Self::System(value)
    }
}

impl From<async_openai::types::chat::ChatCompletionRequestDeveloperMessage>
    for ChatCompletionRequestMessage
{
    fn from(value: async_openai::types::chat::ChatCompletionRequestDeveloperMessage) -> Self {
        Self::Developer(value)
    }
}

impl From<async_openai::types::chat::ChatCompletionRequestToolMessage>
    for ChatCompletionRequestMessage
{
    fn from(value: async_openai::types::chat::ChatCompletionRequestToolMessage) -> Self {
        Self::Tool(value)
    }
}

impl From<async_openai::types::chat::ChatCompletionRequestFunctionMessage>
    for ChatCompletionRequestMessage
{
    fn from(value: async_openai::types::chat::ChatCompletionRequestFunctionMessage) -> Self {
        Self::Function(value)
    }
}

impl From<ChatCompletionRequestAssistantMessage> for ChatCompletionRequestMessage {
    fn from(value: ChatCompletionRequestAssistantMessage) -> Self {
        Self::Assistant(value)
    }
}

impl From<ChatCompletionRequestAssistantMessageContent> for ChatCompletionRequestAssistantMessage {
    fn from(value: ChatCompletionRequestAssistantMessageContent) -> Self {
        Self {
            content: Some(value),
            ..Default::default()
        }
    }
}

impl From<&str> for ChatCompletionRequestAssistantMessage {
    fn from(value: &str) -> Self {
        ChatCompletionRequestAssistantMessageContent::Text(value.into()).into()
    }
}

impl From<String> for ChatCompletionRequestAssistantMessage {
    fn from(value: String) -> Self {
        value.as_str().into()
    }
}

// From content parts into UserMessageContentPart enum

impl From<ChatCompletionRequestMessageContentPartText>
    for ChatCompletionRequestUserMessageContentPart
{
    fn from(value: ChatCompletionRequestMessageContentPartText) -> Self {
        ChatCompletionRequestUserMessageContentPart::Text(value)
    }
}

impl From<ChatCompletionRequestMessageContentPartImage>
    for ChatCompletionRequestUserMessageContentPart
{
    fn from(value: ChatCompletionRequestMessageContentPartImage) -> Self {
        ChatCompletionRequestUserMessageContentPart::ImageUrl(value)
    }
}

impl From<ChatCompletionRequestMessageContentPartAudio>
    for ChatCompletionRequestUserMessageContentPart
{
    fn from(value: ChatCompletionRequestMessageContentPartAudio) -> Self {
        ChatCompletionRequestUserMessageContentPart::InputAudio(value)
    }
}

impl From<ChatCompletionRequestMessageContentPartVideo>
    for ChatCompletionRequestUserMessageContentPart
{
    fn from(value: ChatCompletionRequestMessageContentPartVideo) -> Self {
        ChatCompletionRequestUserMessageContentPart::VideoUrl(value)
    }
}

impl From<ChatCompletionRequestMessageContentPartAudioUrl>
    for ChatCompletionRequestUserMessageContentPart
{
    fn from(value: ChatCompletionRequestMessageContentPartAudioUrl) -> Self {
        ChatCompletionRequestUserMessageContentPart::AudioUrl(value)
    }
}

// URL type conversions

impl From<&str> for ImageUrl {
    fn from(value: &str) -> Self {
        Self {
            url: value.parse().expect("Invalid URL"),
            detail: Default::default(),
            uuid: None,
        }
    }
}

impl From<String> for ImageUrl {
    fn from(value: String) -> Self {
        Self {
            url: value.parse().expect("Invalid URL"),
            detail: Default::default(),
            uuid: None,
        }
    }
}

impl From<&str> for VideoUrl {
    fn from(value: &str) -> Self {
        Self {
            url: value.parse().expect("Invalid URL"),
            detail: Default::default(),
            uuid: None,
        }
    }
}

impl From<String> for VideoUrl {
    fn from(value: String) -> Self {
        Self {
            url: value.parse().expect("Invalid URL"),
            detail: Default::default(),
            uuid: None,
        }
    }
}

impl From<&str> for AudioUrl {
    fn from(value: &str) -> Self {
        Self {
            url: value.parse().expect("Invalid URL"),
            uuid: None,
        }
    }
}

impl From<String> for AudioUrl {
    fn from(value: String) -> Self {
        Self {
            url: value.parse().expect("Invalid URL"),
            uuid: None,
        }
    }
}
