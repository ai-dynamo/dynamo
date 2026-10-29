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

use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
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
    streaming_dispatch: StreamingDispatchConfig,
    reasoning_field: ReasoningField,
    serving_policy: ServingPolicy,
}

impl FrontendApiConfig {
    pub fn new(
        anthropic: AnthropicApiConfig,
        streaming_dispatch: StreamingDispatchConfig,
        reasoning_field: ReasoningField,
    ) -> Self {
        Self {
            anthropic,
            streaming_dispatch,
            reasoning_field,
            serving_policy: ServingPolicy::default(),
        }
    }

    pub fn from_flags(
        enable_anthropic_api: bool,
        strip_anthropic_preamble: bool,
        enable_streaming_tool_dispatch: bool,
        enable_streaming_reasoning_dispatch: bool,
        reasoning_field: ReasoningField,
    ) -> Self {
        Self {
            anthropic: AnthropicApiConfig::new(enable_anthropic_api, strip_anthropic_preamble),
            streaming_dispatch: StreamingDispatchConfig::new(
                enable_streaming_tool_dispatch,
                enable_streaming_reasoning_dispatch,
            ),
            reasoning_field,
            serving_policy: ServingPolicy::default(),
        }
    }

    pub fn from_optional_flags(
        enable_anthropic_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
        reasoning_field: Option<ReasoningField>,
    ) -> Option<Self> {
        if enable_anthropic_api.is_none()
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

    pub fn serving_policy(&self) -> &ServingPolicy {
        &self.serving_policy
    }

    pub fn set_serving_policy(&mut self, serving_policy: ServingPolicy) {
        self.serving_policy = serving_policy;
    }
}

impl Default for FrontendApiConfig {
    fn default() -> Self {
        Self {
            anthropic: Default::default(),
            streaming_dispatch: Default::default(),
            reasoning_field: default_reasoning_field(),
            serving_policy: ServingPolicy::default(),
        }
    }
}

/// Serving policy that the frontend owns.
///
/// The model card carries engine facts. Policy that only the frontend applies comes from
/// frontend flags, so one flag works the same for every worker and sidecar. A lever that
/// shapes the request applies in [`ServingPolicy::apply_to_request`], which each chat
/// handler calls at the HTTP boundary, before it resolves thinking controls, so every
/// later reader sees the same request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServingPolicy {
    default_chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

impl ServingPolicy {
    /// Chat template kwargs that apply under each request's own kwargs
    /// (`--default-chat-template-kwargs`). An empty map is the same as no defaults.
    pub fn set_default_chat_template_kwargs(
        &mut self,
        kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.default_chat_template_kwargs = kwargs.filter(|kwargs| !kwargs.is_empty());
    }

    /// Merge the default chat template kwargs under the request's own; request keys win.
    /// An explicit request thinking control replaces every thinking key in the defaults, so
    /// the defaults rank below the request and above `--dyn-default-thinking-mode`.
    pub fn apply_to_request(&self, request: &mut NvCreateChatCompletionRequest) {
        const THINKING_CONTROL_KEYS: [&str; 4] = [
            "thinking",
            "enable_thinking",
            "thinking_mode",
            "reasoning_effort",
        ];
        let Some(defaults) = &self.default_chat_template_kwargs else {
            return;
        };
        let request_controls_thinking = request.thinking.is_some()
            || request.inner.reasoning_effort.is_some()
            || request.chat_template_args.as_ref().is_some_and(|args| {
                THINKING_CONTROL_KEYS
                    .iter()
                    .any(|key| args.contains_key(*key))
            });
        let mut defaults = defaults
            .iter()
            .filter(|(key, _)| {
                !(request_controls_thinking && THINKING_CONTROL_KEYS.contains(&key.as_str()))
            })
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
        let config = FrontendApiConfig::from_optional_flags(None, None, None, None, None);

        assert_eq!(config, None);
    }

    #[test]
    fn optional_flags_preserve_explicit_false_values() {
        let config = FrontendApiConfig::from_optional_flags(
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(ReasoningField::Reasoning),
        )
        .expect("explicit flags should produce a config");

        assert!(!config.anthropic().enabled());
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
                (env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH, Some("1")),
                (env_llm::DYN_REASONING_FIELD_NAME, Some("reasoning")),
            ],
            || {
                let config = FrontendApiConfig::from_optional_flags(
                    Some(false),
                    None,
                    None,
                    Some(false),
                    None,
                )
                .expect("partial flags should produce a config");

                assert!(!config.anthropic().enabled());
                assert!(config.anthropic().strip_preamble());
                assert!(config.streaming_dispatch().tool_dispatch());
                assert!(!config.streaming_dispatch().reasoning_dispatch());
                assert_eq!(config.reasoning_field(), ReasoningField::Reasoning);
            },
        );
    }

    fn policy(defaults: serde_json::Value) -> ServingPolicy {
        let mut policy = ServingPolicy::default();
        policy.set_default_chat_template_kwargs(defaults.as_object().cloned());
        policy
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
    fn serving_policy_layers_defaults_under_request_kwargs() {
        let policy =
            policy(serde_json::json!({"clear_thinking": false, "preserve_thinking": true}));
        let mut req = request(serde_json::json!({
            "chat_template_kwargs": {"preserve_thinking": false}
        }));
        policy.apply_to_request(&mut req);
        assert_eq!(
            serde_json::to_value(req.chat_template_args).unwrap(),
            serde_json::json!({"clear_thinking": false, "preserve_thinking": false})
        );

        // No defaults leave the request untouched.
        let mut req = request(serde_json::json!({}));
        ServingPolicy::default().apply_to_request(&mut req);
        assert!(req.chat_template_args.is_none());
    }

    #[test]
    fn serving_policy_thinking_defaults_yield_to_request_thinking_controls() {
        let policy = policy(serde_json::json!({"enable_thinking": false, "clear_thinking": false}));

        // With no request control, the default decides, in every dialect.
        let mut req = request(serde_json::json!({}));
        policy.apply_to_request(&mut req);
        req.normalize_reasoning_template_args().unwrap();
        let args = req.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));

        // A top-level reasoning_effort replaces the default's thinking keys.
        let mut req = request(serde_json::json!({"reasoning_effort": "high"}));
        policy.apply_to_request(&mut req);
        req.normalize_reasoning_template_args().unwrap();
        let args = req.chat_template_args.unwrap();
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("clear_thinking"), Some(&serde_json::json!(false)));
    }

    #[test]
    fn invalid_reasoning_field_env_defaults_to_reasoning_content() {
        temp_env::with_var(env_llm::DYN_REASONING_FIELD_NAME, Some("invalid"), || {
            let config = FrontendApiConfig::default();

            assert_eq!(config.reasoning_field(), ReasoningField::ReasoningContent);
        });
    }
}
