// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Ok, Result};
use minijinja::Environment;

use crate::model_card::{ModelDeploymentCard, PromptContextMixin, PromptFormatterArtifact};

mod context;
mod formatters;
mod oai;
mod tokcfg;

use super::{OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter};
use tokcfg::{ChatTemplate, ChatTemplateValue};

impl PromptFormatter {
    pub async fn from_mdc(mdc: ModelDeploymentCard) -> Result<PromptFormatter> {
        tracing::info!(
            "Creating PromptFormatter from MDC. Has custom_chat_template: {}, Has chat_template_file: {}",
            mdc.custom_chat_template.is_some(),
            mdc.chat_template_file.is_some()
        );

        match mdc
            .prompt_formatter
            .ok_or(anyhow::anyhow!("MDC does not contain a prompt formatter"))?
        {
            PromptFormatterArtifact::HfTokenizerConfigJson(file) => {
                let content = std::fs::read_to_string(&file)
                    .with_context(|| format!("fs:read_to_string '{file}'"))?;
                let mut config: ChatTemplate = serde_json::from_str(&content)?;

                // Implement template precedence:
                // 1. Custom template (highest priority)
                // 2. chat_template.jinja from model repo
                // 3. chat_template field in tokenizer_config.json (already in config)

                if let Some(PromptFormatterArtifact::HfChatTemplate(custom_template_file)) =
                    mdc.custom_chat_template
                {
                    // Custom template has highest priority
                    tracing::info!("Loading custom chat template from: {}", custom_template_file);
                    let chat_template = std::fs::read_to_string(&custom_template_file)
                        .with_context(|| format!("fs:read_to_string '{}'", custom_template_file))?;
                    // clean up the string to remove newlines
                    let chat_template = chat_template.replace('\n', "");

                    // Log template details for debugging
                    tracing::debug!(
                        "Custom template loaded successfully. Length: {} chars, Preview: {}...",
                        chat_template.len(),
                        &chat_template.chars().take(100).collect::<String>()
                    );

                    config.chat_template = Some(ChatTemplateValue(either::Left(chat_template)));
                    tracing::info!("Using custom chat template from CLI flag: {}", custom_template_file);
                } else if let Some(PromptFormatterArtifact::HfChatTemplate(chat_template_file)) =
                    mdc.chat_template_file
                {
                    // Repository chat template has second priority
                    tracing::debug!("Loading repository chat template from: {}", chat_template_file);
                    let chat_template = std::fs::read_to_string(&chat_template_file)
                        .with_context(|| format!("fs:read_to_string '{}'", chat_template_file))?;
                    // clean up the string to remove newlines
                    let chat_template = chat_template.replace('\n', "");

                    // Log template details for debugging
                    tracing::debug!(
                        "Repository template loaded successfully. Length: {} chars, Preview: {}...",
                        chat_template.len(),
                        &chat_template.chars().take(100).collect::<String>()
                    );

                    config.chat_template = Some(ChatTemplateValue(either::Left(chat_template)));
                    tracing::info!("Using chat template from model repository: {}", chat_template_file);
                } else if config.chat_template.is_some() {
                    // Use the chat_template already in config (from tokenizer_config.json)
                    tracing::info!("Using chat template from tokenizer_config.json");
                    if let Some(ref template_value) = config.chat_template {
                        let template_str = match &template_value.0 {
                            either::Left(s) => s.clone(),
                            either::Right(templates) => {
                                tracing::debug!("Found {} chat templates in tokenizer_config.json", templates.len());
                                format!("Multiple templates: {:?}", templates.iter().take(1).collect::<Vec<_>>())
                            }
                        };
                        tracing::debug!(
                            "Tokenizer config template. Length: {} chars, Preview: {}...",
                            template_str.len(),
                            &template_str.chars().take(100).collect::<String>()
                        );
                    }
                } else {
                    tracing::warn!("No chat template found in any location!");
                }
                // Otherwise use the chat_template already in config (from tokenizer_config.json)

                Self::from_parts(
                    config,
                    mdc.prompt_context
                        .map_or(ContextMixins::default(), |x| ContextMixins::new(&x)),
                )
            }
            PromptFormatterArtifact::HfChatTemplate(_) => Err(anyhow::anyhow!(
                "prompt_formatter should not have type HfChatTemplate"
            )),
            PromptFormatterArtifact::GGUF(gguf_path) => {
                let config = ChatTemplate::from_gguf(&gguf_path)?;
                Self::from_parts(config, ContextMixins::default())
            }
        }
    }

    pub fn from_parts(config: ChatTemplate, context: ContextMixins) -> Result<PromptFormatter> {
        let formatter = HfTokenizerConfigJsonFormatter::new(config, context)?;
        Ok(Self::OAI(Arc::new(formatter)))
    }
}

/// Chat Template Jinja Renderer
///
/// Manages a Jinja environment with registered templates for chat formatting.
/// Handles two types of ChatTemplateValue templates:
///
/// 1. String template: Registered as the 'default' template
/// 2. Map template: Contains 'tool_use' and/or 'default' templates
///    - tool_use: Template for tool-based interactions
///    - default: Template for standard chat interactions
///
///   If the map contains both keys, the `tool_use` template is registered as the `tool_use` template
///   and the `default` template is registered as the `default` template.
struct JinjaEnvironment {
    env: Environment<'static>,
}

/// Formatter for HuggingFace tokenizer config JSON templates
///
/// Implements chat template rendering based on HuggingFace's tokenizer_config.json format.
/// Supports:
/// - Tool usage templates
/// - Generation prompts
/// - Context mixins for template customization
#[derive(Debug)]
struct HfTokenizerConfigJsonFormatter {
    env: Environment<'static>,
    config: ChatTemplate,
    mixins: Arc<ContextMixins>,
    supports_add_generation_prompt: bool,
}

// /// OpenAI Standard Prompt Formatter
// pub trait StandardPromptFormatter {
//     fn render(&self, context: &impl StandardPromptContext) -> Result<String>;
// }

// pub trait StandardPromptContext {
//     fn messages(&self) -> Value;
//     fn tools(&self) -> Option<Value>;
// }

#[derive(Debug, Clone, Default)]
pub struct ContextMixins {
    context_mixins: HashSet<PromptContextMixin>,
}
