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

/// How a chat template expects tool_calls[*].function.arguments to be passed.
/// Inferred once from the Jinja source at formatter construction; never from the
/// served model name, which is arbitrary and not reliable for template detection.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ToolArgumentsMode {
    /// Template receives arguments as a JSON-object string (OpenAI wire default).
    #[default]
    JsonString,
    /// Template calls `.items()` on arguments (e.g. GLM-5.2's
    /// `{% for k, v in _args.items() %}`), so arguments must be a parsed object.
    ParsedObject,
}

pub fn detect_tool_arguments_mode(template: &str) -> ToolArgumentsMode {
    // GLM-5.2: `{% set _args = tc.arguments %}{% for k, v in _args.items() %}`
    if template.contains("_args.items()") || template.contains("arguments.items()") {
        ToolArgumentsMode::ParsedObject
    } else {
        ToolArgumentsMode::JsonString
    }
}

thread_local! {
    /// Argument mode for the current formatter.render() call. Set by the preprocessor
    /// immediately before the synchronous `apply_template` invocation; never across an
    /// `.await` point. Using a thread-local avoids adding a field to
    /// `NvCreateChatCompletionRequest` (which would require updating every struct literal).
    static RENDER_TOOL_ARGUMENTS_MODE: std::cell::Cell<ToolArgumentsMode> =
        const { std::cell::Cell::new(ToolArgumentsMode::JsonString) };
}

/// RAII guard that sets the thread-local tool-argument mode for the duration of a
/// synchronous rendering call and restores the previous mode on drop.
///
/// SAFETY: Only use in a *synchronous* (non-`async`) scope with no `.await`
/// between guard creation and drop. Thread-locals are not preserved across
/// async executor boundaries — the task may resume on a different OS thread.
pub(crate) struct ToolArgumentsModeGuard {
    previous: ToolArgumentsMode,
}

impl ToolArgumentsModeGuard {
    pub(crate) fn new(mode: ToolArgumentsMode) -> Self {
        let previous = RENDER_TOOL_ARGUMENTS_MODE.with(|m| m.replace(mode));
        Self { previous }
    }
}

impl Drop for ToolArgumentsModeGuard {
    fn drop(&mut self) {
        RENDER_TOOL_ARGUMENTS_MODE.with(|m| m.set(self.previous));
    }
}

pub(crate) fn get_tool_arguments_mode_for_render() -> ToolArgumentsMode {
    RENDER_TOOL_ARGUMENTS_MODE.with(|m| m.get())
}

/// Extract the Jinja template source from a ModelDeploymentCard for analysis.
///
/// Priority order:
/// 1. `mdc.chat_template_file` — standalone `.jinja` or `chat_template.json` file.
/// 2. `mdc.prompt_formatter` — `tokenizer_config.json` with an embedded
///    `"chat_template"` string (the normal HF layout for most models).
///
/// This covers both layouts so models that ship only a tokenizer config are not
/// silently left in [`ToolArgumentsMode::JsonString`] when their template calls
/// `.items()` on tool-call arguments.
pub fn mdc_jinja_template_text(mdc: &ModelDeploymentCard) -> Option<String> {
    fn read_embedded(checked_file: &crate::common::checked_file::CheckedFile) -> Option<String> {
        let path = checked_file.path()?;
        let contents = std::fs::read_to_string(path).ok()?;
        let config: serde_json::Value = serde_json::from_str(&contents).ok()?;
        let value = config.get("chat_template")?;
        if let Some(s) = value.as_str() {
            return Some(s.to_owned());
        }
        // Some HF configs store templates as [{name, template}, ...]. Concatenate
        // so .items() in any variant is visible to detect_tool_arguments_mode.
        if let Some(arr) = value.as_array() {
            let combined: String = arr
                .iter()
                .filter_map(|v| v.get("template").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            if !combined.is_empty() {
                return Some(combined);
            }
        }
        None
    }

    if let Some(artifact) = mdc.chat_template_file.as_ref() {
        match artifact {
            PromptFormatterArtifact::HfChatTemplateJinja { file, .. } => {
                if let Some(path) = file.path() {
                    if let Ok(s) = std::fs::read_to_string(path) {
                        return Some(s);
                    }
                }
            }
            // HfChatTemplateJson and HfTokenizerConfigJson both embed the template
            // under the "chat_template" JSON key; read_embedded handles both.
            PromptFormatterArtifact::HfChatTemplateJson { file, .. }
            | PromptFormatterArtifact::HfTokenizerConfigJson(file) => {
                if let Some(s) = read_embedded(file) {
                    return Some(s);
                }
            }
        }
    }

    // Fallback: normal HF layout stores tokenizer_config.json in mdc.prompt_formatter;
    // chat_template_file is None unless a separate template file was present.
    if let Some(PromptFormatterArtifact::HfTokenizerConfigJson(f)) = mdc.prompt_formatter.as_ref() {
        if let Some(s) = read_embedded(f) {
            return Some(s);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_mode_glm_items_pattern() {
        let glm_snippet = r#"
            {%- set _args = tc.arguments -%}
            {%- for k, v in _args.items() -%}
        "#;
        assert_eq!(
            detect_tool_arguments_mode(glm_snippet),
            ToolArgumentsMode::ParsedObject
        );
    }

    #[test]
    fn detect_mode_direct_arguments_items() {
        assert_eq!(
            detect_tool_arguments_mode("{% for k, v in arguments.items() %}"),
            ToolArgumentsMode::ParsedObject
        );
    }

    #[test]
    fn detect_mode_standard_template_no_items() {
        let standard = r#"{% for tc in tool_calls %}{{ tc.function.arguments }}{% endfor %}"#;
        assert_eq!(
            detect_tool_arguments_mode(standard),
            ToolArgumentsMode::JsonString
        );
    }

    #[test]
    fn normalize_parses_json_string_to_object() {
        let mut msgs = serde_json::json!([{
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "read",
                    "arguments": r#"{"path": "/tmp/foo"}"#
                }
            }]
        }]);
        normalize_tool_call_arguments(&mut msgs);
        let args = &msgs[0]["tool_calls"][0]["function"]["arguments"];
        assert!(
            args.is_object(),
            "arguments should be an object after normalization"
        );
        assert_eq!(args["path"], "/tmp/foo");
    }

    #[test]
    fn normalize_ignores_non_assistant_messages() {
        let mut msgs = serde_json::json!([{
            "role": "user",
            "content": "hello"
        }]);
        let original = msgs.clone();
        normalize_tool_call_arguments(&mut msgs);
        assert_eq!(msgs, original);
    }

    #[test]
    fn normalize_skips_already_object_arguments() {
        // If somehow arguments is already an object, it should remain unchanged.
        let mut msgs = serde_json::json!([{
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "f",
                    "arguments": {"key": "val"}
                }
            }]
        }]);
        normalize_tool_call_arguments(&mut msgs);
        let args = &msgs[0]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object());
        assert_eq!(args["key"], "val");
    }

    /// Test that mdc_jinja_template_text reads the embedded chat_template
    /// from mdc.prompt_formatter (the HfTokenizerConfigJson / normal HF layout).
    #[test]
    fn mdc_template_text_reads_prompt_formatter_embedded() {
        use crate::model_card::{ModelDeploymentCard, PromptFormatterArtifact};

        // Write a minimal tokenizer_config.json with a chat_template that uses .items()
        let dir = tempfile::tempdir().expect("tempdir");
        let tc_path = dir.path().join("tokenizer_config.json");
        std::fs::write(
            &tc_path,
            r#"{"tokenizer_class":"PreTrainedTokenizer","chat_template":"{% for k, v in _args.items() %}"}"#,
        )
        .expect("write");

        let checked =
            crate::common::checked_file::CheckedFile::from_disk(&tc_path).expect("CheckedFile");

        // Build a minimal MDC with only prompt_formatter set.
        let mut mdc = ModelDeploymentCard::default();
        mdc.prompt_formatter = Some(PromptFormatterArtifact::HfTokenizerConfigJson(checked));

        let text = mdc_jinja_template_text(&mdc).expect("should find template");
        assert!(
            text.contains("_args.items()"),
            "extracted template should contain .items() pattern"
        );
        assert_eq!(
            detect_tool_arguments_mode(&text),
            ToolArgumentsMode::ParsedObject
        );
    }

    /// Test that chat_template.json (HfChatTemplateJson) is also detected.
    #[test]
    fn mdc_template_text_reads_chat_template_json() {
        use crate::model_card::{ModelDeploymentCard, PromptFormatterArtifact};

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("chat_template.json");
        std::fs::write(
            &path,
            r#"{"chat_template":"{% for k, v in arguments.items() %}"}"#,
        )
        .expect("write");

        let checked =
            crate::common::checked_file::CheckedFile::from_disk(&path).expect("CheckedFile");

        let mut mdc = ModelDeploymentCard::default();
        mdc.chat_template_file = Some(PromptFormatterArtifact::HfChatTemplateJson {
            file: checked,
            is_custom: false,
        });

        let text = mdc_jinja_template_text(&mdc).expect("template");
        assert_eq!(
            detect_tool_arguments_mode(&text),
            ToolArgumentsMode::ParsedObject
        );
    }
}

/// Parse `tool_calls[*].function.arguments` from JSON string to object in a
/// serialized messages array before handing it to MiniJinja.
/// Only applied when `ToolArgumentsMode::ParsedObject` is detected from the template.
pub(crate) fn normalize_tool_call_arguments(messages_json: &mut serde_json::Value) {
    let Some(messages) = messages_json.as_array_mut() else {
        return;
    };
    for message in messages {
        let Some(tool_calls) = message
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for tc in tool_calls.iter_mut() {
            let Some(args_str) = tc.pointer("/function/arguments").and_then(|v| v.as_str()) else {
                continue;
            };
            let value = match serde_json::from_str::<serde_json::Value>(args_str) {
                Ok(v) if v.is_object() => v,
                Ok(_) => {
                    // Scalar or array — GLM's .items() would panic at render time.
                    tracing::warn!(
                        args_len = args_str.len(),
                        "tool_call arguments parsed to a non-object; \
                         substituting {{}} for GLM template safety"
                    );
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(_) => {
                    tracing::warn!(
                        args_len = args_str.len(),
                        "tool_call arguments are not valid JSON; \
                         substituting {{}} for GLM template safety"
                    );
                    serde_json::Value::Object(serde_json::Map::new())
                }
            };
            if let Some(obj) = tc
                .get_mut("function")
                .and_then(serde_json::Value::as_object_mut)
            {
                obj.insert("arguments".to_string(), value);
            }
        }
    }
}

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let mut messages_json = serde_json::to_value(&self.inner.messages).unwrap();
        // Normalize tool_calls[*].function.arguments from JSON string to object when
        // the loaded Jinja template requires dict args (e.g. GLM-5.2 .items() call).
        // The mode is written by OpenAIPreprocessor::preprocess before template render.
        if get_tool_arguments_mode_for_render() == ToolArgumentsMode::ParsedObject {
            normalize_tool_call_arguments(&mut messages_json);
        }
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
