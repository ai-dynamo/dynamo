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

impl PromptFormatter {
    pub fn from_mdc(mdc: &ModelDeploymentCard) -> Result<PromptFormatter> {
        // Special handling for DeepSeek models whose HF repos don't ship a Jinja chat_template.
        //
        // Prefer the authoritative `model_type` from config.json — it's set by
        // the model author and survives any `--served-model-name` rename. Fall
        // back to a tight substring match on `display_name` only when config.json
        // is absent (e.g., tokenizer-only MDCs) or unreadable.
        //
        // An empty `model_type` string (rare but legal in the JSON) carries
        // no signal — normalize it to `None` so the display-name fallback
        // still runs instead of being silently suppressed.
        let model_type_lower = mdc
            .model_info
            .as_ref()
            .and_then(|info| info.get_model_info().ok())
            .map(|info| info.model_type().to_lowercase())
            .filter(|s| !s.is_empty());
        let display_name_lower = mdc.display_name.to_lowercase();

        if is_deepseek_v4(&model_type_lower, &display_name_lower) {
            tracing::info!(
                model_type = ?model_type_lower,
                display_name = %mdc.display_name,
                "Detected DeepSeek V4 model, using native Rust formatter",
            );
            return Ok(Self::OAI(Arc::new(
                super::deepseek_v4::DeepSeekV4Formatter::new_thinking(),
            )));
        }
        if is_deepseek_v3_2_non_exp(&model_type_lower, &display_name_lower) {
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

/// Decides whether to activate the DeepSeek-V4 native formatter.
///
/// Primary signal: config.json `model_type`. DeepSeek-V4-Pro and V4-Flash both
/// ship `"model_type": "deepseek_v4"`, set by the model author — this survives
/// any `--served-model-name` rename.
///
/// Fallback: `display_name`, tight-matched against
/// `^deepseek(?:[-_.])?v4(?:[-_.]|$)`. Only consulted when config.json is
/// absent (tokenizer-only MDCs) or unreadable; a concrete config.json value
/// that is *not* `deepseek_v4` is authoritative and suppresses the fallback.
fn is_deepseek_v4(model_type_lower: &Option<String>, display_name_lower: &str) -> bool {
    match model_type_lower.as_deref() {
        Some("deepseek_v4") => true,
        Some(_) => false, // config.json says something else — trust it
        None => is_deepseek_v4_name(display_name_lower),
    }
}

/// Decides whether to activate the DeepSeek-V3.2 (non-Exp) native formatter.
/// Same config-primary / name-fallback rule as V4.
fn is_deepseek_v3_2_non_exp(model_type_lower: &Option<String>, display_name_lower: &str) -> bool {
    let name_match = display_name_lower.contains("deepseek")
        && display_name_lower.contains("v3.2")
        && !display_name_lower.contains("exp");
    match model_type_lower.as_deref() {
        Some("deepseek_v3_2") => !display_name_lower.contains("exp"),
        Some(_) => false,
        None => name_match,
    }
}

/// Tight, anchored match for DeepSeek-V4 display names. Equivalent to the
/// regex `^deepseek(?:[-_.])?v4(?:[-_.]|$)` over an already-lowercased string.
/// Written with string ops to avoid pulling in the `regex` crate.
///
/// Rejects composite names that previously short-circuited the V4 branch:
/// - `deepseek-v3.2-v4-foo` (the `v3.2` variant is the real one)
/// - `deepseek-v40` / `deepseek-v4pro` (no separator after `v4`)
/// - `my-deepseek-v4` (prefix must be at the start)
fn is_deepseek_v4_name(name_lower: &str) -> bool {
    let Some(rest) = name_lower.strip_prefix("deepseek") else {
        return false;
    };
    // Optional single separator between "deepseek" and "v4".
    let rest = rest
        .strip_prefix(|c: char| matches!(c, '-' | '_' | '.'))
        .unwrap_or(rest);
    let Some(after_v4) = rest.strip_prefix("v4") else {
        return false;
    };
    // `v4` must end the name or be followed by a separator — anything else
    // (e.g. `v40`, `v4pro`) is a different model family.
    after_v4.is_empty() || after_v4.starts_with(['-', '_', '.'])
}

#[cfg(test)]
mod detection_tests {
    use super::{is_deepseek_v3_2_non_exp, is_deepseek_v4, is_deepseek_v4_name};

    #[test]
    fn v4_name_matches_canonical_variants() {
        for name in [
            "deepseek-v4",
            "deepseek_v4",
            "deepseek.v4",
            "deepseekv4",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-v4-flash-2507",
            "deepseek-v4.1",
            "deepseek_v4_thinking",
        ] {
            assert!(is_deepseek_v4_name(name), "expected {name} to match V4");
        }
    }

    #[test]
    fn v4_name_rejects_non_v4() {
        // Composite names that previously short-circuited to V4 before the
        // V3.2 branch — now correctly rejected.
        for name in [
            "deepseek-v3.2-v4-foo",
            "my-deepseek-v4",
            "deepseek-v40",
            "deepseek-v4pro",
            "deepseekv40",
            "deepseek-v3",
            "deepseek-v3.2",
            "deepseek-r1",
            "qwen3-v4", // only deepseek-prefixed names qualify
            "dsflash",
            "",
        ] {
            assert!(
                !is_deepseek_v4_name(name),
                "expected {name} to NOT match V4",
            );
        }
    }

    #[test]
    fn v4_detection_prefers_config_model_type() {
        // config.json `model_type = "deepseek_v4"` wins regardless of what
        // the operator calls the model via --served-model-name.
        let v4 = Some("deepseek_v4".to_string());
        for display in ["dsflash", "my-pet-model", "llama-3-8b", ""] {
            assert!(
                is_deepseek_v4(&v4, display),
                "config says deepseek_v4, display {display:?} — expected V4",
            );
        }

        // A concrete non-V4 config.json suppresses the display-name fallback.
        // Even if the operator names the served model "deepseek-v4", a model
        // with `model_type = "llama"` is NOT DeepSeek-V4.
        let llama = Some("llama".to_string());
        for display in ["deepseek-v4", "deepseek-v4-flash", "anything"] {
            assert!(
                !is_deepseek_v4(&llama, display),
                "config says llama, display {display:?} — expected NOT V4",
            );
        }

        // No config.json — fall back to display-name match.
        assert!(is_deepseek_v4(&None, "deepseek-v4-flash"));
        assert!(!is_deepseek_v4(&None, "dsflash"));

        // A config.json with `"model_type": ""` is treated as "no signal" at
        // the call site (normalized to None before is_deepseek_v4 is called),
        // so the display-name fallback still runs — pin that contract.
        let empty: Option<String> = None;
        assert!(is_deepseek_v4(&empty, "deepseek-v4-flash"));
        assert!(!is_deepseek_v4(&empty, "dsflash"));
    }

    #[test]
    fn v3_2_detection_prefers_config_model_type() {
        // config says deepseek_v3_2, any non-"exp" display name triggers.
        let v3_2 = Some("deepseek_v3_2".to_string());
        assert!(is_deepseek_v3_2_non_exp(&v3_2, "whatever"));
        assert!(is_deepseek_v3_2_non_exp(&v3_2, "deepseek-v3.2"));
        // V3.2-Exp is a separate model family; suppress even via config.
        assert!(!is_deepseek_v3_2_non_exp(&v3_2, "deepseek-v3.2-exp"));

        // Other config types lose regardless of display name.
        let other = Some("deepseek_v4".to_string());
        assert!(!is_deepseek_v3_2_non_exp(&other, "deepseek-v3.2"));

        // No config — fall back to the original display-name heuristic.
        assert!(is_deepseek_v3_2_non_exp(&None, "deepseek-v3.2-pro"));
        assert!(!is_deepseek_v3_2_non_exp(&None, "deepseek-v3.2-exp"));
        assert!(!is_deepseek_v3_2_non_exp(&None, "deepseek-v4"));
    }
}
