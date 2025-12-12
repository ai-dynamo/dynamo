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
    pub fn from_mdc(mdc: &ModelDeploymentCard) -> Result<PromptFormatter> {
        // Special handling for DeepSeek-V3.2(-Speciale) which doesn't provide Jinja chat_template
        let name_lower = mdc.display_name.to_lowercase();
        if name_lower.contains("deepseek")
            && name_lower.contains("v3.2")
            && !name_lower.contains("exp")
        {
            tracing::info!("Detected DeepSeek V3.2 model (non-Exp), using native Rust formatter");
            return Ok(Self::OAI(Arc::new(
                super::deepseek_v32::DeepSeekV32Formatter::new_thinking(),
            )));
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
                let contents = std::fs::read_to_string(file)
                    .with_context(|| format!("fs:read_to_string '{}'", file.display()))?;
                let mut config: ChatTemplate =
                    serde_json::from_str(&contents).inspect_err(|err| {
                        crate::log_json_err(&file.display().to_string(), &contents, err)
                    })?;

                // Some HF model (i.e. meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
                // stores the chat template in a separate file, we check if the file exists and
                // put the chat template into config as normalization.
                // This may also be a custom template provided via CLI flag.
                if let Some(PromptFormatterArtifact::HfChatTemplate(checked_file)) =
                    mdc.chat_template_file.as_ref()
                {
                    let Some(chat_template_file) = checked_file.path() else {
                        anyhow::bail!(
                            "HfChatTemplate for {} is a URL, cannot load",
                            mdc.display_name
                        );
                    };
                    let chat_template =
                        std::fs::read_to_string(chat_template_file).with_context(|| {
                            format!("fs:read_to_string '{}'", chat_template_file.display())
                        })?;
                    // clean up the string to remove newlines
                    let chat_template = chat_template.replace('\n', "");
                    config.chat_template = Some(ChatTemplateValue(either::Left(chat_template)));
                }
                Self::from_parts(
                    config,
                    mdc.prompt_context
                        .clone()
                        .map_or(ContextMixins::default(), |x| ContextMixins::new(&x)),
                )
            }
            PromptFormatterArtifact::HfChatTemplate(_) => Err(anyhow::anyhow!(
                "prompt_formatter should not have type HfChatTemplate"
            )),
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
    requires_content_arrays: bool,
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
