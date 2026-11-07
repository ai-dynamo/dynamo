// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned configuration groups shared by the Python entrypoint,
//! `LocalModel`, and HTTP/gRPC service setup.
//!
//! Python may expose these as flat CLI flags for compatibility, but Rust stores
//! them by domain so each service consumes an explicit typed contract. Defaults
//! read the legacy environment variables only for direct Rust/non-Python callers.

use std::sync::LazyLock;

use dynamo_runtime::config::{
    env_is_truthy,
    environment_names::llm::{self as env_llm, metrics as env_metrics},
    is_truthy,
};

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, is_thinking_control_key,
};
use crate::reasoning_field::ReasoningField;

/// Process-wide strict-schema policy, initialized before the HTTP service accepts requests.
pub(crate) static DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy(env_llm::DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION));

/// Metrics naming controls for frontend-owned services.
///
/// Contains the optional metric name prefix resolved from `--metrics-prefix` or
/// `DYN_METRICS_PREFIX`. HTTP services use it when constructing
/// `http::service::metrics::Metrics`; gRPC mode also exposes the prefix through
/// the existing `LocalModel::metrics_prefix()` compatibility accessor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsConfig {
    prefix: Option<String>,
}

impl MetricsConfig {
    pub fn new(prefix: Option<String>) -> Self {
        Self { prefix }
    }

    pub fn prefix(&self) -> Option<String> {
        self.prefix.clone()
    }

    pub fn set_prefix(&mut self, prefix: Option<String>) {
        self.prefix = prefix;
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prefix: std::env::var(env_metrics::DYN_METRICS_PREFIX).ok(),
        }
    }
}

/// Anthropic API surface controls.
///
/// Contains whether the experimental Anthropic Messages API routes are exposed
/// and whether Anthropic billing preambles are stripped from requests. The HTTP
/// service uses these values to choose Anthropic vs OpenAI model routes, enable
/// `/v1/messages`, and drive Anthropic request handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicApiConfig {
    enabled: bool,
    strip_preamble: bool,
}

impl AnthropicApiConfig {
    pub fn new(enabled: bool, strip_preamble: bool) -> Self {
        Self {
            enabled,
            strip_preamble,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn strip_preamble(&self) -> bool {
        self.strip_preamble
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_strip_preamble(&mut self, strip_preamble: bool) {
        self.strip_preamble = strip_preamble;
    }
}

impl Default for AnthropicApiConfig {
    fn default() -> Self {
        Self {
            enabled: env_is_truthy(env_llm::DYN_ENABLE_ANTHROPIC_API),
            strip_preamble: std::env::var(env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE)
                .map(|value| is_truthy(&value))
                .unwrap_or(true),
        }
    }
}

/// Streaming-specific response dispatch controls.
///
/// Contains the OpenAI-compatible streaming toggles for tool-call dispatch and
/// reasoning dispatch events. HTTP request handlers read these values from
/// shared service state when deciding whether to emit the extra SSE events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingDispatchConfig {
    tool_dispatch: bool,
    reasoning_dispatch: bool,
}

impl StreamingDispatchConfig {
    pub fn new(tool_dispatch: bool, reasoning_dispatch: bool) -> Self {
        Self {
            tool_dispatch,
            reasoning_dispatch,
        }
    }

    pub fn tool_dispatch(&self) -> bool {
        self.tool_dispatch
    }

    pub fn reasoning_dispatch(&self) -> bool {
        self.reasoning_dispatch
    }

    pub fn set_tool_dispatch(&mut self, tool_dispatch: bool) {
        self.tool_dispatch = tool_dispatch;
    }

    pub fn set_reasoning_dispatch(&mut self, reasoning_dispatch: bool) {
        self.reasoning_dispatch = reasoning_dispatch;
    }
}

impl Default for StreamingDispatchConfig {
    fn default() -> Self {
        Self {
            tool_dispatch: env_is_truthy(env_llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH),
            reasoning_dispatch: env_is_truthy(env_llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH),
        }
    }
}

fn default_reasoning_field() -> ReasoningField {
    match std::env::var(env_llm::DYN_REASONING_FIELD_NAME) {
        Ok(value) => match value.parse::<ReasoningField>() {
            Ok(field) => field,
            Err(_) => {
                tracing::warn!(
                    "{}={value:?} is not \"reasoning\" or \"reasoning_content\"; defaulting to {}",
                    env_llm::DYN_REASONING_FIELD_NAME,
                    ReasoningField::default()
                );
                ReasoningField::default()
            }
        },
        Err(_) => ReasoningField::default(),
    }
}

/// Frontend API behavior consumed by the HTTP service.
///
/// Groups endpoint-surface and streaming-behavior settings that originate from
/// the frontend CLI/env contract. `EntrypointArgs` builds this from flat Python
/// kwargs, `LocalModel` carries it, and `HttpServiceConfig` installs it into
/// request-handler state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontendApiConfig {
    anthropic: AnthropicApiConfig,
    systemone_enabled: bool,
    streaming_dispatch: StreamingDispatchConfig,
    reasoning_field: ReasoningField,
    chat_template_defaults: ChatTemplateDefaults,
}

impl FrontendApiConfig {
    pub fn new(
        anthropic: AnthropicApiConfig,
        systemone_enabled: bool,
        streaming_dispatch: StreamingDispatchConfig,
        reasoning_field: ReasoningField,
    ) -> Self {
        Self {
            anthropic,
            systemone_enabled,
            streaming_dispatch,
            reasoning_field,
            chat_template_defaults: ChatTemplateDefaults::default(),
        }
    }

    pub fn from_flags(
        enable_anthropic_api: bool,
        enable_systemone_api: bool,
        strip_anthropic_preamble: bool,
        enable_streaming_tool_dispatch: bool,
        enable_streaming_reasoning_dispatch: bool,
        reasoning_field: ReasoningField,
    ) -> Self {
        Self {
            anthropic: AnthropicApiConfig::new(enable_anthropic_api, strip_anthropic_preamble),
            systemone_enabled: enable_systemone_api,
            streaming_dispatch: StreamingDispatchConfig::new(
                enable_streaming_tool_dispatch,
                enable_streaming_reasoning_dispatch,
            ),
            reasoning_field,
            chat_template_defaults: ChatTemplateDefaults::default(),
        }
    }

    pub fn from_optional_flags(
        enable_anthropic_api: Option<bool>,
        enable_systemone_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
        reasoning_field: Option<ReasoningField>,
    ) -> Option<Self> {
        if enable_anthropic_api.is_none()
            && enable_systemone_api.is_none()
            && strip_anthropic_preamble.is_none()
            && enable_streaming_tool_dispatch.is_none()
            && enable_streaming_reasoning_dispatch.is_none()
            && reasoning_field.is_none()
        {
            return None;
        }

        let defaults = Self::default();
        Some(Self::from_flags(
            enable_anthropic_api.unwrap_or_else(|| defaults.anthropic().enabled()),
            enable_systemone_api.unwrap_or_else(|| defaults.systemone_enabled()),
            strip_anthropic_preamble.unwrap_or_else(|| defaults.anthropic().strip_preamble()),
            enable_streaming_tool_dispatch
                .unwrap_or_else(|| defaults.streaming_dispatch().tool_dispatch()),
            enable_streaming_reasoning_dispatch
                .unwrap_or_else(|| defaults.streaming_dispatch().reasoning_dispatch()),
            reasoning_field.unwrap_or_else(|| defaults.reasoning_field()),
        ))
    }

    pub fn anthropic(&self) -> &AnthropicApiConfig {
        &self.anthropic
    }

    pub fn anthropic_mut(&mut self) -> &mut AnthropicApiConfig {
        &mut self.anthropic
    }

    pub fn systemone_enabled(&self) -> bool {
        self.systemone_enabled
    }

    pub fn set_systemone_enabled(&mut self, enabled: bool) {
        self.systemone_enabled = enabled;
    }

    pub fn streaming_dispatch(&self) -> &StreamingDispatchConfig {
        &self.streaming_dispatch
    }

    pub fn streaming_dispatch_mut(&mut self) -> &mut StreamingDispatchConfig {
        &mut self.streaming_dispatch
    }

    pub fn reasoning_field(&self) -> ReasoningField {
        self.reasoning_field
    }

    pub fn set_reasoning_field(&mut self, reasoning_field: ReasoningField) {
        self.reasoning_field = reasoning_field;
    }

    pub fn chat_template_defaults(&self) -> &ChatTemplateDefaults {
        &self.chat_template_defaults
    }

    pub fn set_chat_template_defaults(&mut self, chat_template_defaults: ChatTemplateDefaults) {
        self.chat_template_defaults = chat_template_defaults;
    }
}

impl Default for FrontendApiConfig {
    fn default() -> Self {
        Self {
            anthropic: Default::default(),
            systemone_enabled: env_is_truthy(env_llm::DYN_ENABLE_SYSTEMONE_API),
            streaming_dispatch: Default::default(),
            reasoning_field: default_reasoning_field(),
            chat_template_defaults: ChatTemplateDefaults::default(),
        }
    }
}

/// Chat template kwargs that the frontend applies to every model. The chat handlers apply them
/// before they resolve thinking controls, so every later reader sees the merged request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatTemplateDefaults {
    default_chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ChatTemplateDefaults {
    pub fn set_default_chat_template_kwargs(
        &mut self,
        kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.default_chat_template_kwargs = kwargs.filter(|kwargs| !kwargs.is_empty());
    }

    pub fn apply_to_request(&self, request: &mut NvCreateChatCompletionRequest) {
        let Some(defaults) = &self.default_chat_template_kwargs else {
            return;
        };
        // As in vLLM `merge_kwargs`: a null or "auto" request value leaves the key unset.
        if let Some(args) = request.chat_template_args.as_mut() {
            args.retain(|_, value| !(value.is_null() || value.as_str() == Some("auto")));
        }
        // The thinking keys are dialects of one control, so a request control in any
        // dialect replaces all of them.
        let request_controls_thinking = request.thinking.is_some()
            || request.inner.reasoning_effort.is_some()
            || request
                .chat_template_args
                .as_ref()
                .is_some_and(|args| args.keys().any(|key| is_thinking_control_key(key)));
        let mut defaults = defaults
            .iter()
            .filter(|(key, _)| !(request_controls_thinking && is_thinking_control_key(key)))
            .peekable();
        if defaults.peek().is_none() {
            return;
        }
        let args = request
            .chat_template_args
            .get_or_insert_with(Default::default);
        for (key, value) in defaults {
            args.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_flags_return_none_when_all_values_are_unspecified() {
        let config = FrontendApiConfig::from_optional_flags(None, None, None, None, None, None);

        assert_eq!(config, None);
    }

    #[test]
    fn optional_flags_preserve_explicit_false_values() {
        let config = FrontendApiConfig::from_optional_flags(
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(ReasoningField::Reasoning),
        )
        .expect("explicit flags should produce a config");

        assert!(!config.anthropic().enabled());
        assert!(!config.systemone_enabled());
        assert!(config.anthropic().strip_preamble());
        assert!(!config.streaming_dispatch().tool_dispatch());
        assert!(config.streaming_dispatch().reasoning_dispatch());
        assert_eq!(config.reasoning_field(), ReasoningField::Reasoning);
    }

    #[test]
    fn optional_flags_use_env_defaults_for_unspecified_values() {
        temp_env::with_vars(
            [
                (env_llm::DYN_ENABLE_ANTHROPIC_API, Some("1")),
                (env_llm::DYN_ENABLE_SYSTEMONE_API, Some("1")),
                (env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH, Some("1")),
                (env_llm::DYN_REASONING_FIELD_NAME, Some("reasoning")),
            ],
            || {
                let config = FrontendApiConfig::from_optional_flags(
                    Some(false),
                    Some(false),
                    None,
                    None,
                    Some(false),
                    None,
                )
                .expect("partial flags should produce a config");

                assert!(!config.anthropic().enabled());
                assert!(!config.systemone_enabled());
                assert!(config.anthropic().strip_preamble());
                assert!(config.streaming_dispatch().tool_dispatch());
                assert!(!config.streaming_dispatch().reasoning_dispatch());
                assert_eq!(config.reasoning_field(), ReasoningField::Reasoning);
            },
        );
    }

    fn template_defaults(kwargs: serde_json::Value) -> ChatTemplateDefaults {
        let mut defaults = ChatTemplateDefaults::default();
        defaults.set_default_chat_template_kwargs(kwargs.as_object().cloned());
        defaults
    }

    fn request(extra: serde_json::Value) -> NvCreateChatCompletionRequest {
        let mut body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn chat_template_defaults_layers_defaults_under_request_kwargs() {
        let defaults = template_defaults(
            serde_json::json!({"clear_thinking": false, "preserve_thinking": true}),
        );
        let mut req = request(serde_json::json!({
            "chat_template_kwargs": {"preserve_thinking": false}
        }));
        defaults.apply_to_request(&mut req);
        assert_eq!(
            serde_json::to_value(req.chat_template_args).unwrap(),
            serde_json::json!({"clear_thinking": false, "preserve_thinking": false})
        );

        let mut req = request(serde_json::json!({}));
        ChatTemplateDefaults::default().apply_to_request(&mut req);
        assert!(req.chat_template_args.is_none());
    }

    #[test]
    fn chat_template_defaults_thinking_defaults_yield_to_request_thinking_controls() {
        let defaults = template_defaults(
            serde_json::json!({"enable_thinking": false, "clear_thinking": false}),
        );

        let mut req = request(serde_json::json!({}));
        defaults.apply_to_request(&mut req);
        req.normalize_reasoning_template_args().unwrap();
        let args = req.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));

        let mut req = request(serde_json::json!({"reasoning_effort": "high"}));
        defaults.apply_to_request(&mut req);
        req.normalize_reasoning_template_args().unwrap();
        let args = req.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("clear_thinking"), Some(&serde_json::json!(false)));
    }

    #[test]
    fn chat_template_defaults_treats_null_and_auto_request_values_as_unset() {
        let defaults = template_defaults(
            serde_json::json!({"enable_thinking": false, "clear_thinking": false}),
        );
        for unset in [serde_json::Value::Null, serde_json::json!("auto")] {
            let mut req = request(serde_json::json!({
                "chat_template_kwargs": {"enable_thinking": unset, "reasoning_effort": unset, "x": unset}
            }));
            defaults.apply_to_request(&mut req);
            assert_eq!(
                serde_json::to_value(req.chat_template_args).unwrap(),
                serde_json::json!({"enable_thinking": false, "clear_thinking": false})
            );
        }

        let mut req = request(serde_json::json!({
            "chat_template_kwargs": {"enable_thinking": true}
        }));
        defaults.apply_to_request(&mut req);
        assert_eq!(
            serde_json::to_value(req.chat_template_args).unwrap(),
            serde_json::json!({"enable_thinking": true, "clear_thinking": false})
        );

        let mut req = request(serde_json::json!({
            "chat_template_kwargs": {"enable_thinking": null}
        }));
        ChatTemplateDefaults::default().apply_to_request(&mut req);
        assert_eq!(
            serde_json::to_value(req.chat_template_args).unwrap(),
            serde_json::json!({"enable_thinking": null})
        );
    }

    #[test]
    fn invalid_reasoning_field_env_defaults_to_reasoning_content() {
        temp_env::with_var(env_llm::DYN_REASONING_FIELD_NAME, Some("invalid"), || {
            let config = FrontendApiConfig::default();

            assert_eq!(config.reasoning_field(), ReasoningField::ReasoningContent);
        });
    }
}
