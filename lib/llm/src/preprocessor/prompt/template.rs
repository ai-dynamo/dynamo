// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
pub use tokcfg::ChatTemplate;
use tokcfg::ChatTemplateValue;

/// Returns true when the model display name identifies a DeepSeek V4 model.
/// Matches only when `v4` is adjacent to the `deepseek` series boundary, e.g.
/// `deepseek-v4`, `deepseek_v4`, `deepseekv4`, or `deepseek v4`. Rejects
/// composite/derivative names like `deepseek-llm-coder-v4-tuned` or
/// `deepseek-v3.2-v4-merge` where `v4` is not at the series boundary.
fn is_deepseek_v4_name(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("deepseek-v4")
        || n.contains("deepseek_v4")
        || n.contains("deepseekv4")
        || n.contains("deepseek v4")
}

impl PromptFormatter {
    pub fn from_mdc(mdc: &ModelDeploymentCard) -> Result<PromptFormatter> {
        // Special handling for DeepSeek models whose HF repos don't ship a Jinja chat_template.
        let name_lower = mdc.display_name.to_lowercase();
        if is_deepseek_v4_name(&name_lower) {
            tracing::info!("Detected DeepSeek V4 model, using native Rust formatter");
            return Ok(Self::OAI(Arc::new(
                super::deepseek_v4::DeepSeekV4Formatter::new_thinking(),
            )));
        }
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
                let contents = std::fs::read_to_string(file).with_context(|| {
                    format!(
                        "PromptFormatter.from_mdc fs:read_to_string '{}'",
                        file.display()
                    )
                })?;
                let mut config: ChatTemplate =
                    serde_json::from_str(&contents).inspect_err(|err| {
                        crate::log_json_err(&file.display().to_string(), &contents, err)
                    })?;

                // Some HF model (i.e. meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
                // stores the chat template in a separate file, we check if the file exists and
                // put the chat template into config as normalization.
                // This may also be a custom template provided via CLI flag.
                match mdc.chat_template_file.as_ref() {
                    Some(PromptFormatterArtifact::HfChatTemplateJinja {
                        file: checked_file,
                        ..
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
                        file: checked_file,
                        ..
                    }) => {
                        let Some(path) = checked_file.path() else {
                            anyhow::bail!(
                                "HfChatTemplateJson for {} is a URL, cannot load",
                                mdc.display_name
                            );
                        };
                        let raw = std::fs::read_to_string(path)
                            .with_context(|| format!("fs:read_to_string '{}'", path.display()))?;
                        let wrapper: serde_json::Value =
                            serde_json::from_str(&raw).with_context(|| {
                                format!("Failed to parse '{}' as JSON", path.display())
                            })?;
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
                Self::from_parts(
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

    pub fn from_parts(
        config: ChatTemplate,
        context: ContextMixins,
        exclude_tools_when_tool_choice_none: bool,
    ) -> Result<PromptFormatter> {
        let formatter = HfTokenizerConfigJsonFormatter::with_options(
            config,
            context,
            exclude_tools_when_tool_choice_none,
        )?;
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
    /// When true, strip tool definitions from the chat template when tool_choice is "none".
    /// This prevents models from generating raw XML tool calls in the content field.
    exclude_tools_when_tool_choice_none: bool,
    /// True if the chat template natively references `reasoning_content`.
    /// When true, skip injection — the template handles it.
    template_handles_reasoning: bool,
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

#[cfg(test)]
mod tests {
    use super::is_deepseek_v4_name;

    #[test]
    fn deepseek_v4_name_match_is_tight() {
        assert!(is_deepseek_v4_name("deepseek-v4-pro"));
        assert!(is_deepseek_v4_name("DeepSeek-V4-Pro"));
        assert!(is_deepseek_v4_name("DeepSeek-V4-Chat"));
        assert!(is_deepseek_v4_name("deepseek_v4"));
        assert!(is_deepseek_v4_name("deepseekv4"));
        assert!(is_deepseek_v4_name("deepseek v4"));

        assert!(!is_deepseek_v4_name("deepseek-v3.2-v4-merge"));
        assert!(!is_deepseek_v4_name("deepseek-llm-coder-v4-tuned"));
        assert!(!is_deepseek_v4_name("deepseek-v3"));
    }
}
