// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::preprocessor::media::MediaDecoder;
use crate::types::TokenIdType;

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizeCompletionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    #[serde(default = "default_false")]
    pub return_token_strs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizeChatRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default = "default_false")]
    pub return_token_strs: bool,
    #[serde(default = "default_false")]
    pub continue_final_message: bool,
    #[serde(default = "default_false")]
    pub add_special_tokens: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "chat_template_args"
    )]
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_io_kwargs: Option<MediaDecoder>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mm_processor_kwargs: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<dynamo_protocols::types::ChatCompletionTool>>,
}

impl TokenizeChatRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.continue_final_message && self.add_generation_prompt {
            return Err(
                "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
                    .to_string(),
            );
        }

        Ok(())
    }

    pub fn merged_chat_template_kwargs(&self) -> HashMap<String, serde_json::Value> {
        let mut kwargs = self.chat_template_kwargs.clone().unwrap_or_default();
        kwargs.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(self.add_generation_prompt),
        );
        kwargs.insert(
            "continue_final_message".to_string(),
            serde_json::Value::Bool(self.continue_final_message),
        );
        kwargs
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum TokenizeRequest {
    Completion(TokenizeCompletionRequest),
    Chat(TokenizeChatRequest),
}

impl TokenizeRequest {
    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Completion(request) => request.model.as_deref(),
            Self::Chat(request) => request.model.as_deref(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizeResponse {
    pub count: usize,
    pub max_model_len: u32,
    pub tokens: Vec<TokenIdType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_strs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetokenizeRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub tokens: Vec<TokenIdType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DetokenizeResponse {
    pub prompt: String,
}
