// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt formatting (lib/llm side).
//!
//! The reusable chat-template / prompt-formatting engine lives in the
//! standalone, runtime-free [`dynamo_renderer`] crate. This module holds only the
//! lib/llm-local glue that can't live there:
//!   * implements [`OAIChatLikeRequest`] for Dynamo's `Nv*` request wrappers,
//!   * keeps media-IO config off the rendering trait via [`MediaRequestExt`]
//!     (so `dynamo_renderer` need not depend on the media module),
//!   * adapts a [`ModelDeploymentCard`] into a [`PromptFormatter`]
//!     ([`prompt_formatter_from_mdc`]).
//!
//! Everything else imports from `dynamo_renderer` directly.

use anyhow::{Context, Result};
use minijinja::value::Value;

use dynamo_renderer::{
    ChatTemplate, ChatTemplateValue, ContextMixins, OAIChatLikeRequest, PromptFormatter,
    PromptInput, TextInput, TokenInput, deepseek_formatter_for, may_be_fix_tool_schema,
};

use crate::model_card::{ModelDeploymentCard, PromptFormatterArtifact};
use crate::preprocessor::media::MediaDecoder;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};

/// lib/llm-local extension carrying multimodal media-IO config. Kept off
/// [`OAIChatLikeRequest`] so `dynamo_renderer` stays free of the media module;
/// the multimodal preprocessing path bounds on `OAIChatLikeRequest + MediaRequestExt`.
pub trait MediaRequestExt {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder>;
}

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = serde_json::to_value(&self.inner.messages).unwrap();
        Value::from_serialize(&messages_json)
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        Some(self.inner.messages.as_slice())
    }

    fn tools(&self) -> Option<Value> {
        if self.inner.tools.is_none() {
            None
        } else {
            // Try to fix the tool schema if it is missing type and properties
            Some(may_be_fix_tool_schema(
                serde_json::to_value(&self.inner.tools).unwrap(),
            )?)
        }
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.inner.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.inner.tool_choice))
        }
    }

    fn response_format(&self) -> Option<Value> {
        self.inner
            .response_format
            .as_ref()
            .map(Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        // Using vLLM default behavior
        true
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.mm_processor_kwargs.as_ref()
    }
}

impl MediaRequestExt for NvCreateChatCompletionRequest {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        self.media_io_kwargs.as_ref()
    }
}

impl OAIChatLikeRequest for NvCreateCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }
    fn messages(&self) -> minijinja::value::Value {
        let message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
            dynamo_protocols::types::ChatCompletionRequestUserMessage {
                content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                    crate::protocols::openai::completions::prompt_to_string(&self.inner.prompt),
                ),
                name: None,
            },
        );

        minijinja::value::Value::from_serialize(vec![message])
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn prompt_input_type(&self) -> PromptInput {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::IntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Single(vec![]))
            }
            dynamo_protocols::types::Prompt::ArrayOfIntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Batch(vec![]))
            }
            dynamo_protocols::types::Prompt::String(_) => {
                PromptInput::Text(TextInput::Single(String::new()))
            }
            dynamo_protocols::types::Prompt::StringArray(_) => {
                PromptInput::Text(TextInput::Batch(vec![]))
            }
        }
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::IntegerArray(tokens) => {
                Some(TokenInput::Single(tokens.clone()))
            }
            dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arrays) => {
                Some(TokenInput::Batch(arrays.clone()))
            }
            _ => None,
        }
    }

    fn extract_text(&self) -> Option<TextInput> {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::String(text) => {
                Some(TextInput::Single(text.to_string()))
            }
            dynamo_protocols::types::Prompt::StringArray(texts) => {
                Some(TextInput::Batch(texts.to_vec()))
            }
            _ => None,
        }
    }
}

impl MediaRequestExt for NvCreateCompletionRequest {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        None
    }
}

/// Build a [`PromptFormatter`] from a [`ModelDeploymentCard`].
///
/// DeepSeek families whose HF repos ship no Jinja `chat_template` get a native
/// Rust formatter (via [`deepseek_formatter_for`]); everything else loads the
/// HF `tokenizer_config.json` template (and any separate chat-template file)
/// and builds via [`PromptFormatter::from_parts`].
pub fn prompt_formatter_from_mdc(mdc: &ModelDeploymentCard) -> Result<PromptFormatter> {
    // Prefer the authoritative `model_type` from config.json — it's set by the
    // model author and survives any `--served-model-name` rename. An empty
    // `model_type` carries no signal — normalize to `None` so the display-name
    // fallback still runs.
    let model_type_lower = mdc
        .model_info
        .as_ref()
        .and_then(|info| info.get_model_info().ok())
        .map(|info| info.model_type().to_lowercase())
        .filter(|s| !s.is_empty());
    let display_name_lower = mdc.display_name.to_lowercase();

    if let Some(formatter) = deepseek_formatter_for(&model_type_lower, &display_name_lower) {
        return Ok(formatter);
    }

    match mdc
        .prompt_formatter
        .as_ref()
        .ok_or(anyhow::anyhow!("MDC does not contain a prompt formatter"))?
    {
        PromptFormatterArtifact::HfTokenizerConfigJson(checked_file) => {
            let Some(file) = checked_file.path() else {
                anyhow::bail!(
                    "HfTokenizerConfigJson for {} is a URL, cannot load",
                    mdc.display_name
                );
            };
            let contents = std::fs::read_to_string(file).with_context(|| {
                format!(
                    "prompt_formatter_from_mdc fs:read_to_string '{}'",
                    file.display()
                )
            })?;
            let mut config: ChatTemplate = serde_json::from_str(&contents).inspect_err(|err| {
                crate::log_json_err(&file.display().to_string(), &contents, err)
            })?;

            // Some HF models (e.g. Llama-4-Maverick) store the chat template in a
            // separate file, or it may be a custom template provided via CLI flag.
            match mdc.chat_template_file.as_ref() {
                Some(PromptFormatterArtifact::HfChatTemplateJinja {
                    file: checked_file, ..
                }) => {
                    let Some(path) = checked_file.path() else {
                        anyhow::bail!(
                            "HfChatTemplateJinja for {} is a URL, cannot load",
                            mdc.display_name
                        );
                    };
                    let chat_template = std::fs::read_to_string(path)
                        .with_context(|| format!("fs:read_to_string '{}'", path.display()))?;
                    config.chat_template = Some(ChatTemplateValue(either::Left(chat_template)));
                }
                Some(PromptFormatterArtifact::HfChatTemplateJson {
                    file: checked_file, ..
                }) => {
                    let Some(path) = checked_file.path() else {
                        anyhow::bail!(
                            "HfChatTemplateJson for {} is a URL, cannot load",
                            mdc.display_name
                        );
                    };
                    let raw = std::fs::read_to_string(path)
                        .with_context(|| format!("fs:read_to_string '{}'", path.display()))?;
                    let wrapper: serde_json::Value = serde_json::from_str(&raw)
                        .with_context(|| format!("Failed to parse '{}' as JSON", path.display()))?;
                    let field = wrapper.get("chat_template").ok_or_else(|| {
                        anyhow::anyhow!(
                            "'{}' does not contain a 'chat_template' field",
                            path.display()
                        )
                    })?;
                    let value = serde_json::from_value::<ChatTemplateValue>(field.clone())
                        .with_context(|| {
                            format!(
                                "Failed to deserialize 'chat_template' in '{}'",
                                path.display()
                            )
                        })?;
                    config.chat_template = Some(value);
                }
                _ => {}
            }
            PromptFormatter::from_parts(
                config,
                mdc.prompt_context
                    .clone()
                    .map_or(ContextMixins::default(), |x| ContextMixins::new(&x)),
                mdc.runtime_config.exclude_tools_when_tool_choice_none,
            )
        }
        PromptFormatterArtifact::HfChatTemplateJinja { .. }
        | PromptFormatterArtifact::HfChatTemplateJson { .. } => Err(anyhow::anyhow!(
            "prompt_formatter should not have type HfChatTemplate*"
        )),
    }
}
