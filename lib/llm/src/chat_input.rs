// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model input adaptation, separate from output parsing and API conversion.

use anyhow::{Context, Result};
use dynamo_protocols::types::ReasoningEffort;

use crate::model_card::{ModelDeploymentCard, TekkenMetadata, TokenizerKind};
use crate::protocols::anthropic::types::{
    AnthropicCreateMessageRequest, AnthropicRole, SystemContent, ThinkingConfig,
    system_message_content,
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::unified::UnifiedRequest;

#[cfg(test)]
#[path = "../tests/common/mistral_model.rs"]
mod mistral_model;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum ChatInputPolicy {
    #[default]
    Standard,
    Mistral {
        effort_when_disabled: ReasoningEffort,
    },
}

impl ChatInputPolicy {
    /// Resolve once from downloaded artifacts, including factory-owned renderers.
    pub(crate) fn from_card(card: &ModelDeploymentCard) -> Result<Self> {
        if card
            .chat_template_file
            .as_ref()
            .is_some_and(|template| template.is_custom())
        {
            return Ok(Self::Standard);
        }
        let Some(file) = card.tokenizer.as_ref().and_then(TokenizerKind::tekken_file) else {
            return Ok(Self::Standard);
        };
        let metadata = TekkenMetadata::from_checked_file(file)?;
        let effort_when_disabled = Self::disabled_effort(&metadata)?;
        Ok(Self::Mistral {
            effort_when_disabled,
        })
    }

    fn disabled_effort(metadata: &TekkenMetadata) -> Result<ReasoningEffort> {
        let Some(setting) = metadata
            .model_settings_builder()
            .and_then(|builder| builder.get("reasoning_effort"))
            .filter(|setting| !setting.is_null())
        else {
            // Without a builder the formatter ignores the effort.
            return Ok(ReasoningEffort::Low);
        };
        let values = setting
            .get("values")
            .and_then(serde_json::Value::as_array)
            .context("Mistral reasoning_effort builder must list supported values")?;
        for (name, effort) in [
            ("none", ReasoningEffort::None),
            ("minimal", ReasoningEffort::Minimal),
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            ("xhigh", ReasoningEffort::Xhigh),
            ("max", ReasoningEffort::Max),
        ] {
            if values.iter().any(|value| value == name) {
                return Ok(effort);
            }
        }
        anyhow::bail!("Mistral reasoning_effort builder has no supported effort")
    }

    pub(crate) fn prepare_anthropic(&self, request: &mut AnthropicCreateMessageRequest) {
        if matches!(self, Self::Mistral { .. }) && merge_inline_system_messages(request) {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    model = request.model,
                    "Inline system messages in /v1/messages are merged into the leading \
                     system prompt for the native Mistral renderer; each new one changes the \
                     prompt prefix, so the conversation misses the prefix cache"
                );
            });
        }
    }

    pub(crate) fn normalize(&self, request: &mut UnifiedRequest) -> Result<()> {
        let Self::Mistral {
            effort_when_disabled,
        } = self
        else {
            return Ok(());
        };
        let Some(context) = request.anthropic_context() else {
            return Ok(());
        };
        let effort = reasoning_effort_from_anthropic(
            context.thinking.as_ref(),
            context.output_config.as_ref(),
            effort_when_disabled.clone(),
        )?;
        if let Some(effort) = effort {
            Self::set_effort(&mut request.inner, effort)?;
        }
        Ok(())
    }

    pub(crate) fn normalize_chat(&self, request: &mut NvCreateChatCompletionRequest) -> Result<()> {
        // Validate controls and resolve top-level/nested precedence before adapting.
        request.normalize_reasoning_template_args()?;
        if let Self::Mistral {
            effort_when_disabled,
        } = self
            && request
                .chat_template_args
                .as_ref()
                .and_then(|args| args.get("reasoning_effort"))
                .is_some_and(|effort| effort == "none")
        {
            Self::set_effort(request, effort_when_disabled.clone())?;
        }
        Ok(())
    }

    fn set_effort(
        request: &mut NvCreateChatCompletionRequest,
        effort: ReasoningEffort,
    ) -> Result<()> {
        let enabled = effort != ReasoningEffort::None;
        let args = request
            .chat_template_args
            .get_or_insert_with(Default::default);
        args.insert("reasoning_effort".into(), serde_json::to_value(&effort)?);
        for key in ["thinking", "enable_thinking"] {
            args.insert(key.into(), enabled.into());
        }
        args.insert(
            "thinking_mode".into(),
            if enabled { "enabled" } else { "disabled" }.into(),
        );
        request.inner.reasoning_effort = Some(effort);
        Ok(())
    }
}

/// Fold `role: "system"` messages from `messages[]` into the leading system prompt,
/// and report whether any were folded.
///
/// mistral-common's message order rejects a system message after an assistant or tool
/// turn, and Claude Code sends one after each tool result while its `<total_tokens>`
/// reminder is on. Upstream vLLM merges them similarly for templates that reject them.
/// The leading system prompt precedes the tools and the history, so each new inline
/// message changes the whole prompt prefix and the next turn misses the prefix cache.
fn merge_inline_system_messages(req: &mut AnthropicCreateMessageRequest) -> bool {
    let mut inline = Vec::new();
    req.messages.retain(|msg| {
        if !matches!(msg.role, AnthropicRole::System) {
            return true;
        }
        let text = system_message_content(&msg.content);
        if !text.is_empty() {
            inline.push(text);
        }
        false
    });
    if inline.is_empty() {
        return false;
    }
    let (leading, cache_control) = match req.system.take() {
        Some(SystemContent {
            text,
            cache_control,
        }) => (Some(text), cache_control),
        None => (None, None),
    };
    let text = leading
        .into_iter()
        .chain(inline)
        .collect::<Vec<_>>()
        .join("\n");
    req.system = Some(SystemContent {
        text,
        cache_control,
    });
    true
}

/// The `reasoning_effort` for a model whose only reasoning control is `reasoning_effort`,
/// as upstream vLLM's Anthropic endpoint sets it (`_handle_output_config`, then
/// `_handle_thinking` in `vllm/entrypoints/anthropic/serving.py`).
///
/// `output_config.effort` passes through unchanged, so the renderer rejects a level the
/// model does not support, as native `vllm serve` does. `thinking.type` `disabled`
/// selects `disabled_effort` over any valid effort: `none`, or `low` for a model whose
/// renderer rejects `none`, as upstream's `DisabledThinkingEffortResolver` probes it.
/// `adaptive` and `enabled` set nothing, so a request without an effort keeps the
/// model's default.
fn reasoning_effort_from_anthropic(
    thinking: Option<&ThinkingConfig>,
    output_config: Option<&serde_json::Value>,
    disabled_effort: ReasoningEffort,
) -> anyhow::Result<Option<ReasoningEffort>> {
    let effort = match output_config.and_then(|config| config.get("effort")) {
        None | Some(serde_json::Value::Null) => None,
        Some(effort) => Some(match effort.as_str() {
            Some("low") => ReasoningEffort::Low,
            Some("medium") => ReasoningEffort::Medium,
            Some("high") => ReasoningEffort::High,
            Some("xhigh") => ReasoningEffort::Xhigh,
            Some("max") => ReasoningEffort::Max,
            _ => anyhow::bail!(
                "output_config.effort must be one of low, medium, high, xhigh, max; got {effort}"
            ),
        }),
    };
    if thinking.is_some_and(|thinking| thinking.thinking_type == "disabled") {
        return Ok(Some(disabled_effort));
    }
    Ok(effort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::ChatCompletionRequestMessage;
    fn low_policy() -> ChatInputPolicy {
        ChatInputPolicy::Mistral {
            effort_when_disabled: ReasoningEffort::Low,
        }
    }

    fn chat(extra: serde_json::Value) -> NvCreateChatCompletionRequest {
        let mut request =
            serde_json::json!({"model":"test", "messages":[{"role":"user","content":"hi"}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    fn assert_controls(request: &NvCreateChatCompletionRequest, effort: &str, enabled: bool) {
        assert_eq!(
            serde_json::to_value(&request.chat_template_args).unwrap(),
            serde_json::json!({
                "reasoning_effort": effort, "thinking": enabled, "enable_thinking": enabled,
                "thinking_mode": if enabled {"enabled"} else {"disabled"}
            })
        );
    }

    #[test]
    fn native_policy_follows_the_card_for_token_and_text_input() {
        let mut card = ModelDeploymentCard::load_from_disk(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-mistral-tekken"),
            None,
        )
        .unwrap();
        for input in [
            crate::model_type::ModelInput::Tokens,
            crate::model_type::ModelInput::Text,
        ] {
            card.model_input = input;
            assert!(matches!(
                ChatInputPolicy::from_card(&card).unwrap(),
                ChatInputPolicy::Mistral { .. }
            ));
        }
    }

    #[test]
    fn v13_chat_none_renders_like_unset_effort() {
        for with_hf_config in [false, true] {
            let dir = mistral_model::model_dir(with_hf_config, "v13", &[]);
            let card = ModelDeploymentCard::load_from_disk(dir.path(), None).unwrap();
            let policy = ChatInputPolicy::from_card(&card).unwrap();
            let dynamo_renderer::PromptFormatter::OAI(formatter) =
                crate::preprocessor::prompt::prompt_formatter_from_mdc(&card).unwrap();
            let unset = chat(serde_json::json!({}));
            let expected = formatter.render(&unset).unwrap();
            let mut request = chat(serde_json::json!({"reasoning_effort":"none"}));
            policy.normalize_chat(&mut request).unwrap();
            assert_controls(&request, "low", true);
            assert_eq!(formatter.render(&request).unwrap(), expected);
        }
    }

    #[test]
    fn chat_none_uses_the_model_effort_with_canonical_controls_and_precedence() {
        for (extra, expected) in [
            (serde_json::json!({"reasoning_effort":"none"}), "low"),
            (
                serde_json::json!({"chat_template_kwargs":{"reasoning_effort":"none"}}),
                "low",
            ),
            (
                serde_json::json!({"reasoning_effort":"high", "chat_template_kwargs":{"reasoning_effort":"none"}}),
                "high",
            ),
            (
                serde_json::json!({"reasoning_effort":"none", "chat_template_kwargs":{"reasoning_effort":"high"}}),
                "low",
            ),
        ] {
            let mut request = chat(extra);
            // The HTTP boundary may already have canonicalized none to false.
            request.normalize_reasoning_template_args().unwrap();
            low_policy().normalize_chat(&mut request).unwrap();
            assert_controls(&request, expected, true);
        }
        let mut request = chat(serde_json::json!({"reasoning_effort":"none"}));
        let none_policy = ChatInputPolicy::Mistral {
            effort_when_disabled: ReasoningEffort::None,
        };
        none_policy.normalize_chat(&mut request).unwrap();
        assert_controls(&request, "none", false);
        let mut standard = chat(serde_json::json!({"reasoning_effort":"none"}));
        ChatInputPolicy::Standard
            .normalize_chat(&mut standard)
            .unwrap();
        assert_controls(&standard, "none", false);
    }

    #[test]
    fn anthropic_disabled_normalizes_controls_without_rewriting_api_context() {
        let raw = serde_json::from_value::<AnthropicCreateMessageRequest>(serde_json::json!({
            "model":"test", "max_tokens":32, "thinking":{"type":"disabled"},
            "output_config":{"effort":"high"}, "messages":[{"role":"user","content":"hi"}]
        }))
        .unwrap();
        let mut request: UnifiedRequest = raw.try_into().unwrap();
        low_policy().normalize(&mut request).unwrap();
        assert_controls(&request.inner, "low", true);
        let context = request.anthropic_context().unwrap();
        assert_eq!(context.thinking.as_ref().unwrap().thinking_type, "disabled");
        assert_eq!(context.output_config.as_ref().unwrap()["effort"], "high");
    }

    #[test]
    fn disabled_effort_is_supported_and_null_is_not_the_none_grade() {
        for (values, expected) in [
            (
                serde_json::json!(["low", "high"]),
                Some(ReasoningEffort::Low),
            ),
            (
                serde_json::json!(["medium", "high"]),
                Some(ReasoningEffort::Medium),
            ),
            (
                serde_json::json!(["none", "high"]),
                Some(ReasoningEffort::None),
            ),
            (serde_json::json!(["unknown"]), None),
        ] {
            let metadata: TekkenMetadata = serde_json::from_value(serde_json::json!({
                "config":{"default_vocab_size":1,"version":"v15"}, "special_tokens":[],
                "model_settings_builder":{"reasoning_effort":{
                    "type":"enum", "accepts_none":true, "default":"high", "values":values
                }}
            }))
            .unwrap();
            assert_eq!(ChatInputPolicy::disabled_effort(&metadata).ok(), expected);
        }
        let metadata: TekkenMetadata = serde_json::from_value(serde_json::json!({
            "config":{"default_vocab_size":1,"version":"v13"}, "special_tokens":[]
        }))
        .unwrap();
        assert_eq!(
            ChatInputPolicy::disabled_effort(&metadata).unwrap(),
            ReasoningEffort::Low
        );
        for extra in [
            serde_json::json!({}),
            serde_json::json!({"reasoning_effort":null}),
            serde_json::json!({"chat_template_kwargs":{"reasoning_effort":null}}),
        ] {
            let mut request = chat(extra);
            low_policy().normalize_chat(&mut request).unwrap();
            assert!(request.inner.reasoning_effort.is_none());
        }
    }

    #[test]
    fn response_and_toggle_only_controls_keep_their_existing_scope() {
        let mut request = UnifiedRequest {
            inner: chat(serde_json::json!({"reasoning_effort":"none"})),
            api_context: crate::protocols::unified::ApiContext::Responses(Default::default()),
        };
        low_policy().normalize(&mut request).unwrap();
        assert_eq!(
            request.inner.inner.reasoning_effort,
            Some(ReasoningEffort::None)
        );
        let mut request = chat(serde_json::json!({"thinking":{"type":"disabled"}}));
        low_policy().normalize_chat(&mut request).unwrap();
        assert!(request.inner.reasoning_effort.is_none());
        assert_eq!(
            request.chat_template_args.unwrap()["enable_thinking"],
            false
        );
        let mut invalid =
            chat(serde_json::json!({"reasoning_effort":"none", "thinking":{"type":"bogus"}}));
        assert!(low_policy().normalize_chat(&mut invalid).is_err());
    }

    #[test]
    fn chat_deployment_none_default_maps_after_existing_default_precedence() {
        let mut defaults = crate::frontend_config::ChatTemplateDefaults::default();
        defaults.set_default_chat_template_kwargs(Some(
            serde_json::from_value(serde_json::json!({"reasoning_effort":"none"})).unwrap(),
        ));
        let mut request = chat(serde_json::json!({}));
        defaults.apply_to_request(&mut request);
        low_policy().normalize_chat(&mut request).unwrap();
        assert_controls(&request, "low", true);
        let mut request = chat(serde_json::json!({"reasoning_effort":"high"}));
        defaults.apply_to_request(&mut request);
        low_policy().normalize_chat(&mut request).unwrap();
        assert_controls(&request, "high", true);
    }

    #[test]
    fn test_merge_inline_system_messages_after_tool_result() {
        let json = r#"{
            "model": "test-model",
            "max_tokens": 100,
            "system": "You are a coding agent.",
            "messages": [
                {"role": "user", "content": "Run cat notes.txt"},
                {"role": "system", "content": "Environment: macOS"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "cat notes.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "hello"}
                ]},
                {"role": "system", "content": [{"type": "text", "text": "1000 tokens left"}]}
            ]
        }"#;
        let mut req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(merge_inline_system_messages(&mut req));

        assert_eq!(
            req.system.as_ref().unwrap().text,
            "You are a coding agent.\nEnvironment: macOS\n1000 tokens left"
        );
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let roles: Vec<_> = chat_req
            .inner
            .messages
            .iter()
            .map(|message| match message {
                ChatCompletionRequestMessage::System(_) => "system",
                ChatCompletionRequestMessage::User(_) => "user",
                ChatCompletionRequestMessage::Assistant(_) => "assistant",
                ChatCompletionRequestMessage::Tool(_) => "tool",
                _ => "other",
            })
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    }

    #[test]
    fn test_merge_inline_system_messages_without_inline_keeps_system() {
        let json = r#"{
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let mut req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(!merge_inline_system_messages(&mut req));
        assert!(req.system.is_none());
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_reasoning_effort_from_anthropic() {
        let thinking = |kind: &str| ThinkingConfig {
            thinking_type: kind.to_string(),
            budget_tokens: None,
            display: None,
        };
        let effort = |level: serde_json::Value| serde_json::json!({ "effort": level });
        let resolve = |kind: Option<&str>, config: Option<serde_json::Value>| {
            reasoning_effort_from_anthropic(
                kind.map(thinking).as_ref(),
                config.as_ref(),
                ReasoningEffort::None,
            )
        };

        // A level the model lacks passes through; the renderer rejects it.
        assert_eq!(
            resolve(Some("adaptive"), Some(effort("medium".into()))).unwrap(),
            Some(ReasoningEffort::Medium)
        );
        // Without an effort, adaptive keeps the model's default.
        assert_eq!(resolve(Some("adaptive"), None).unwrap(), None);
        assert_eq!(
            resolve(Some("adaptive"), Some(effort(serde_json::Value::Null))).unwrap(),
            None
        );
        // Disabled thinking selects `none` over a valid effort, not over an invalid one.
        assert_eq!(
            resolve(Some("disabled"), Some(effort("high".into()))).unwrap(),
            Some(ReasoningEffort::None)
        );
        assert!(resolve(Some("disabled"), Some(effort("bogus".into()))).is_err());
        assert!(resolve(Some("adaptive"), Some(effort("none".into()))).is_err());
        // A model without `none` turns thinking down to `low` instead.
        assert_eq!(
            reasoning_effort_from_anthropic(
                Some(&thinking("disabled")),
                Some(&effort("high".into())),
                ReasoningEffort::Low,
            )
            .unwrap(),
            Some(ReasoningEffort::Low)
        );
    }
}
