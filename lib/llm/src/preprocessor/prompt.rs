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
    PromptInput, RenderedPrompt, TextInput, TokenInput, deepseek_formatter_for,
    kimi_k3_formatter_for, may_be_fix_tool_schema,
};

use crate::model_card::{ModelDeploymentCard, PromptFormatterArtifact};
use crate::preprocessor::media::MediaDecoder;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestMessage,
};

/// lib/llm-local extension carrying multimodal media-IO config. Kept off
/// [`OAIChatLikeRequest`] so `dynamo_renderer` stays free of the media module;
/// the multimodal preprocessing path bounds on `OAIChatLikeRequest + MediaRequestExt`.
pub trait MediaRequestExt {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder>;
}

/// Parse a JSON object string into a serde_json Map, preserving the exact
/// spelling of numbers that cannot be round-tripped through f64 without loss.
/// Numbers that survive f64 conversion unchanged are kept as `Value::Number`;
/// numbers that would be corrupted (e.g. integers with >15 significant digits)
/// are kept as `Value::String` so the original value is available to templates.
/// Returns `None` when the top-level JSON value is not an object.
fn parse_args_object(
    s: &str,
) -> anyhow::Result<Option<serde_json::Map<String, serde_json::Value>>> {
    // First pass: parse the top-level value to check it is an object.
    let top: serde_json::Value = serde_json::from_str(s)?;
    let top_obj = match top {
        serde_json::Value::Object(m) => m,
        _ => return Ok(None),
    };

    // Second pass: re-parse each entry's value via its RawValue to detect
    // numbers that serde_json would corrupt through f64.
    // We use a raw string-to-rawvalue helper to get the exact bytes.
    let mut out = serde_json::Map::with_capacity(top_obj.len());
    for (k, cooked) in top_obj {
        // For non-numbers, use the already-parsed value directly.
        if !cooked.is_number() {
            out.insert(k, cooked);
            continue;
        }
        // For numbers: find the raw token in the source string so we can
        // verify that serde_json's f64 representation matches the original.
        // We do this by re-serializing the cooked number and comparing.
        let cooked_str = cooked.to_string();
        if let Some(raw_token) = extract_value_token(s, &k) {
            let raw_trimmed = raw_token.trim();
            if raw_trimmed != cooked_str.as_str() {
                // Precision was lost (e.g. 30-digit integer → 1.23e+29).
                // Keep the original spelling as a string.
                out.insert(k, serde_json::Value::String(raw_trimmed.to_string()));
                continue;
            }
        }
        out.insert(k, cooked);
    }
    Ok(Some(out))
}

/// Extract the raw JSON token for `key` from a JSON object string `s`.
/// Returns `None` when the key is not found or parsing fails.
fn extract_value_token<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    // Build the escaped key as it appears in JSON.
    let key_json = serde_json::to_string(key).ok()?;
    // Find `"key":` in the source.
    let needle = format!("{}:", key_json);
    let start = s.find(needle.as_str())?;
    let after_colon = s[start + needle.len()..].trim_start();
    // Measure the extent of the JSON token.
    let end = token_end(after_colon)?;
    let source_offset = s.len() - s[start + needle.len()..].len()
        + (after_colon.as_ptr() as usize - s[start + needle.len()..].as_ptr() as usize);
    Some(&s[source_offset..source_offset + end])
}

/// Return the byte length of the first JSON token in `s`.
fn token_end(s: &str) -> Option<usize> {
    let s = s.trim_start();
    let first = s.chars().next()?;
    match first {
        '"' => {
            let mut i = 1;
            let b = s.as_bytes();
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                } else if b[i] == b'"' {
                    return Some(i + 1);
                } else {
                    i += 1;
                }
            }
            None
        }
        '{' | '[' => {
            // Skip balanced braces/brackets — good enough for flat objects.
            let (open, close) = if first == '{' {
                (b'{', b'}')
            } else {
                (b'[', b']')
            };
            let mut depth = 0i32;
            let mut in_str = false;
            let mut escape = false;
            for (i, &b) in s.as_bytes().iter().enumerate() {
                if escape {
                    escape = false;
                    continue;
                }
                if in_str {
                    if b == b'\\' {
                        escape = true;
                    } else if b == b'"' {
                        in_str = false;
                    }
                } else {
                    match b {
                        b'"' => in_str = true,
                        b if b == open => depth += 1,
                        b if b == close => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(i + 1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            None
        }
        _ => {
            // Number, bool, null — ends at first delimiter.
            let end = s
                .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
                .unwrap_or(s.len());
            if end == 0 { None } else { Some(end) }
        }
    }
}

/// Parse `tool_calls[*].function.arguments` from JSON string to object before
/// handing messages to MiniJinja. Some chat templates (e.g. GLM-5.2) call
/// `.items()` on arguments and require a dict; the OpenAI wire format carries
/// arguments as a JSON string. This matches TRT-LLM's chat_utils.py behaviour.
///
/// Large integer values in arguments are preserved exactly because `dynamo-llm`
/// enables the `serde_json/arbitrary_precision` feature.
///
/// Returns an error if any tool_call's arguments string contains invalid JSON,
/// so callers can surface a 400-style error instead of fabricating empty args.
pub(crate) fn normalize_tool_call_arguments(
    messages_json: &mut serde_json::Value,
) -> anyhow::Result<()> {
    let Some(messages) = messages_json.as_array_mut() else {
        return Ok(());
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
            // Empty string is a valid no-argument call; normalise to {}.
            if args_str.is_empty() {
                if let Some(obj) = tc
                    .get_mut("function")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    obj.insert(
                        "arguments".to_string(),
                        serde_json::Value::Object(serde_json::Map::new()),
                    );
                }
                continue;
            }
            let value = match parse_args_object(args_str) {
                Ok(Some(map)) => serde_json::Value::Object(map),
                Ok(None) => {
                    // Scalar or array — GLM's .items() would panic at render time.
                    tracing::warn!(
                        args_len = args_str.len(),
                        "tool_call arguments parsed to a non-object; \
                         substituting {{}} for template safety"
                    );
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => {
                    anyhow::bail!(
                        "tool_call arguments are not valid JSON (len={}): {e}",
                        args_str.len()
                    );
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
    Ok(())
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
        // vLLM / HF: continue_final_message leaves the last assistant turn open,
        // which is incompatible with appending a new generation prompt.
        if self.common.continue_final_message == Some(true) {
            return false;
        }
        self.common.add_generation_prompt.unwrap_or(true)
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
/// Model families whose HF repos ship no Jinja `chat_template` get a native
/// Rust formatter; everything else loads the
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

    if let Some(formatter) = kimi_k3_formatter_for(
        &model_type_lower,
        &display_name_lower,
        mdc.runtime_config.exclude_tools_when_tool_choice_none,
    ) {
        return Ok(formatter);
    }

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

/// HuggingFace `apply_chat_template(continue_final_message=True)`: after render,
/// drop closing tokens so the prompt ends at the last assistant text.
pub(crate) fn apply_continue_final_message(
    prompt: RenderedPrompt,
    messages: Option<&[ChatCompletionRequestMessage]>,
) -> RenderedPrompt {
    let Some(final_text) = last_assistant_text(messages.unwrap_or(&[])) else {
        return prompt;
    };
    let rendered = prompt.as_str();
    let trimmed = final_text.trim();
    let (idx, len) = match rendered.rfind(&final_text) {
        Some(idx) => (idx, final_text.len()),
        None if !trimmed.is_empty() && trimmed != final_text => match rendered.rfind(trimmed) {
            Some(idx) => (idx, trimmed.len()),
            None => {
                tracing::warn!(
                    "continue_final_message: last assistant text not found in rendered prompt"
                );
                return prompt;
            }
        },
        None => {
            tracing::warn!(
                "continue_final_message: last assistant text not found in rendered prompt"
            );
            return prompt;
        }
    };
    RenderedPrompt::text(rendered[..idx + len].to_string())
}

fn last_assistant_text(messages: &[ChatCompletionRequestMessage]) -> Option<String> {
    let ChatCompletionRequestMessage::Assistant(message) = messages.last()? else {
        return None;
    };
    match message.content.as_ref()? {
        ChatCompletionRequestAssistantMessageContent::Text(text) if !text.is_empty() => {
            Some(text.clone())
        }
        ChatCompletionRequestAssistantMessageContent::Array(parts) => {
            parts.iter().rev().find_map(|part| match part {
                ChatCompletionRequestAssistantMessageContentPart::Text(part)
                    if !part.text.is_empty() =>
                {
                    Some(part.text.clone())
                }
                _ => None,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_tool_call_arguments;

    fn make_tool_call_messages(arguments: &str) -> serde_json::Value {
        serde_json::json!([{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "f", "arguments": arguments }
            }]
        }])
    }

    /// Regression: large integers must survive the string→object conversion
    /// without being silently corrupted to an f64 approximation.
    /// parse_args_object detects precision loss and falls back to Value::String,
    /// so the original spelling is preserved for the MiniJinja template.
    #[test]
    fn normalize_preserves_large_integer() {
        let big_int = "123456789012345678901234567890";
        let mut msgs = make_tool_call_messages(&format!("{{\"id\": {big_int}}}"));
        normalize_tool_call_arguments(&mut msgs).unwrap();
        let id_slot = &msgs[0]["tool_calls"][0]["function"]["arguments"]["id"];
        // parse_args_object stores it as Value::String when f64 would lose precision.
        let preserved = id_slot
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| id_slot.to_string());
        assert_eq!(
            preserved, big_int,
            "large integer must not be corrupted by f64 round-trip"
        );
    }

    /// Empty arguments string is a valid no-argument call; must normalise to {}
    /// without emitting a warning or returning an error.
    #[test]
    fn normalize_empty_string_becomes_empty_object() {
        let mut msgs = make_tool_call_messages("");
        normalize_tool_call_arguments(&mut msgs).unwrap();
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!({}),
            "empty arguments string must normalise to an empty object"
        );
    }

    /// Malformed JSON in arguments must return an error, not substitute {}.
    #[test]
    fn normalize_malformed_json_returns_error() {
        let mut msgs = make_tool_call_messages("not-valid-json{");
        let result = normalize_tool_call_arguments(&mut msgs);
        assert!(
            result.is_err(),
            "malformed arguments JSON must return Err, not substitute {{}}"
        );
    }

    /// Well-formed object arguments pass through unchanged.
    #[test]
    fn normalize_valid_object_passes_through() {
        let mut msgs = make_tool_call_messages(r#"{"city": "Paris", "unit": "celsius"}"#);
        normalize_tool_call_arguments(&mut msgs).unwrap();
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["arguments"]["city"],
            serde_json::json!("Paris")
        );
    }

    // --- Multi-turn opt-in behaviour tests ---
    //
    // These guard the boundary between JsonString (default, GPT-OSS / Harmony
    // path) and JsonObject (GLM-5.2 path).  The normalize function is the
    // shared primitive; NormalizedArgsRequest calls it only when the model
    // opts in; other models never call it.

    /// GLM-5.2 (JsonObject mode): historical arguments must be a parsed object
    /// so the chat template can call .items() on them.
    #[test]
    fn glm52_historical_arguments_are_object() {
        let mut msgs =
            make_tool_call_messages(r#"{"location": "San Francisco", "unit": "celsius"}"#);
        normalize_tool_call_arguments(&mut msgs).unwrap();
        let args = &msgs[0]["tool_calls"][0]["function"]["arguments"];
        assert!(
            args.is_object(),
            "GLM-5.2 (JsonObject mode): arguments must be a JSON object, got: {args}"
        );
        assert_eq!(args["location"], serde_json::json!("San Francisco"));
    }

    /// GPT-OSS (JsonString mode, default): historical arguments must stay as
    /// a raw JSON string so the Harmony parser can re-emit a structured call.
    /// This test models the multi-turn scenario where the first tool call is
    /// in conversation history and the second must be parsed correctly.
    #[test]
    fn gptoss_historical_arguments_remain_string() {
        // Without normalization (JsonString default), arguments stay as-is.
        let msgs = make_tool_call_messages(r#"{"location": "San Francisco", "unit": "celsius"}"#);
        let args = &msgs[0]["tool_calls"][0]["function"]["arguments"];
        assert!(
            args.is_string(),
            "GPT-OSS (JsonString mode): arguments must remain a JSON string, got: {args}"
        );
        // The raw string is preserved for the Harmony template to handle.
        assert_eq!(
            args.as_str().unwrap(),
            r#"{"location": "San Francisco", "unit": "celsius"}"#
        );
    }

    #[test]
    fn continue_final_message_truncates_after_last_assistant_text() {
        use dynamo_protocols::types::ChatCompletionRequestMessage;
        use dynamo_renderer::RenderedPrompt;

        let prompt = RenderedPrompt::text(
            "user text<|im_end|>LLM-Native Interaction<|im_end|><|im_start|>assistant".to_string(),
        );
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "user text"},
                {"role": "assistant", "content": "LLM-Native Interaction"}
            ]))
            .unwrap();

        let rendered = super::apply_continue_final_message(prompt, Some(&messages));
        assert_eq!(
            rendered.as_str(),
            "user text<|im_end|>LLM-Native Interaction"
        );
    }

    #[test]
    fn continue_final_message_uses_trimmed_content_when_template_trims() {
        use dynamo_protocols::types::ChatCompletionRequestMessage;
        use dynamo_renderer::RenderedPrompt;

        let prompt = RenderedPrompt::text("hello world extra".to_string());
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "assistant", "content": "  world  "}
            ]))
            .unwrap();

        let rendered = super::apply_continue_final_message(prompt, Some(&messages));
        assert_eq!(rendered.as_str(), "hello world");
    }

    #[test]
    fn continue_final_message_uses_last_occurrence_when_text_repeats() {
        use dynamo_protocols::types::ChatCompletionRequestMessage;
        use dynamo_renderer::RenderedPrompt;

        let prompt = RenderedPrompt::text(
            "LLM-Native Interaction in the user turn. LLM-Native Interaction<|im_end|>".to_string(),
        );
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "LLM-Native Interaction in the user turn."},
                {"role": "assistant", "content": "LLM-Native Interaction"}
            ]))
            .unwrap();

        let rendered = super::apply_continue_final_message(prompt, Some(&messages));
        assert_eq!(
            rendered.as_str(),
            "LLM-Native Interaction in the user turn. LLM-Native Interaction"
        );
    }

    #[test]
    fn continue_final_message_reads_last_text_part_from_array_content() {
        use dynamo_protocols::types::ChatCompletionRequestMessage;
        use dynamo_renderer::RenderedPrompt;

        let prompt = RenderedPrompt::text("prefix Design extra".to_string());
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "assistant", "content": [
                    {"type": "text", "text": "ignored"},
                    {"type": "text", "text": "Design"}
                ]}
            ]))
            .unwrap();

        let rendered = super::apply_continue_final_message(prompt, Some(&messages));
        assert_eq!(rendered.as_str(), "prefix Design");
    }

    #[test]
    fn continue_final_message_keeps_prompt_when_assistant_text_is_missing() {
        use dynamo_protocols::types::ChatCompletionRequestMessage;
        use dynamo_renderer::RenderedPrompt;

        let prompt = RenderedPrompt::text("unchanged".to_string());
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "assistant", "content": "not-in-the-prompt"}
            ]))
            .unwrap();

        let rendered = super::apply_continue_final_message(prompt, Some(&messages));
        assert_eq!(rendered.as_str(), "unchanged");
    }
}
