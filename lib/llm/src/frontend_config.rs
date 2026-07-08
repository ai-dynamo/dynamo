// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned configuration groups shared by the Python entrypoint,
//! `LocalModel`, and HTTP/gRPC service setup.
//!
//! Python may expose these as flat CLI flags for compatibility, but Rust stores
//! them by domain so each service consumes an explicit typed contract. Defaults
//! read the legacy environment variables only for direct Rust/non-Python callers.

use dynamo_runtime::config::{
    env_is_truthy,
    environment_names::llm::{self as env_llm, metrics as env_metrics},
};
use std::fmt;
use std::str::FromStr;

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
            strip_preamble: env_is_truthy(env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE),
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

/// Response field used for reasoning output on the Chat Completions API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChatCompletionsReasoningField {
    /// OpenAI-compatible Dynamo default.
    #[default]
    ReasoningContent,
    /// vLLM-compatible field name.
    Reasoning,
}

impl ChatCompletionsReasoningField {
    pub const REASONING_CONTENT: &'static str = "reasoning_content";
    pub const REASONING: &'static str = "reasoning";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReasoningContent => Self::REASONING_CONTENT,
            Self::Reasoning => Self::REASONING,
        }
    }

    /// Resolve the direct-Rust fallback from the environment.
    ///
    /// Python callers pass an explicit value after parsing CLI/environment
    /// configuration, so this fallback is only used when no value was supplied
    /// through [`FrontendApiConfig`].
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(env_llm::DYN_CHAT_COMPLETIONS_REASONING_FIELD) {
            Ok(value) => value.parse(),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(std::env::VarError::NotUnicode(_)) => Err(format!(
                "{} must contain valid UTF-8",
                env_llm::DYN_CHAT_COMPLETIONS_REASONING_FIELD
            )),
        }
    }
}

impl fmt::Display for ChatCompletionsReasoningField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ChatCompletionsReasoningField {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            Self::REASONING_CONTENT => Ok(Self::ReasoningContent),
            Self::REASONING => Ok(Self::Reasoning),
            _ => Err(format!(
                "invalid Chat Completions reasoning field {value:?}; expected {:?} or {:?}",
                Self::REASONING_CONTENT,
                Self::REASONING
            )),
        }
    }
}

/// Frontend API behavior consumed by the HTTP service.
///
/// Groups endpoint-surface and streaming-behavior settings that originate from
/// the frontend CLI/env contract. `EntrypointArgs` builds this from flat Python
/// kwargs, `LocalModel` carries it, and `HttpServiceConfig` installs it into
/// request-handler state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrontendApiConfig {
    anthropic: AnthropicApiConfig,
    streaming_dispatch: StreamingDispatchConfig,
    chat_completions_reasoning_field: Option<ChatCompletionsReasoningField>,
}

impl FrontendApiConfig {
    pub fn new(anthropic: AnthropicApiConfig, streaming_dispatch: StreamingDispatchConfig) -> Self {
        Self {
            anthropic,
            streaming_dispatch,
            chat_completions_reasoning_field: None,
        }
    }

    pub fn from_flags(
        enable_anthropic_api: bool,
        strip_anthropic_preamble: bool,
        enable_streaming_tool_dispatch: bool,
        enable_streaming_reasoning_dispatch: bool,
    ) -> Self {
        Self {
            anthropic: AnthropicApiConfig::new(enable_anthropic_api, strip_anthropic_preamble),
            streaming_dispatch: StreamingDispatchConfig::new(
                enable_streaming_tool_dispatch,
                enable_streaming_reasoning_dispatch,
            ),
            chat_completions_reasoning_field: None,
        }
    }

    pub fn from_optional_flags(
        enable_anthropic_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
    ) -> Option<Self> {
        Self::from_optional_flags_with_reasoning_field(
            enable_anthropic_api,
            strip_anthropic_preamble,
            enable_streaming_tool_dispatch,
            enable_streaming_reasoning_dispatch,
            None,
        )
    }

    pub fn from_optional_flags_with_reasoning_field(
        enable_anthropic_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
        chat_completions_reasoning_field: Option<ChatCompletionsReasoningField>,
    ) -> Option<Self> {
        if enable_anthropic_api.is_none()
            && strip_anthropic_preamble.is_none()
            && enable_streaming_tool_dispatch.is_none()
            && enable_streaming_reasoning_dispatch.is_none()
            && chat_completions_reasoning_field.is_none()
        {
            return None;
        }

        let defaults = Self::default();
        Some(Self {
            anthropic: AnthropicApiConfig::new(
                enable_anthropic_api.unwrap_or_else(|| defaults.anthropic().enabled()),
                strip_anthropic_preamble.unwrap_or_else(|| defaults.anthropic().strip_preamble()),
            ),
            streaming_dispatch: StreamingDispatchConfig::new(
                enable_streaming_tool_dispatch
                    .unwrap_or_else(|| defaults.streaming_dispatch().tool_dispatch()),
                enable_streaming_reasoning_dispatch
                    .unwrap_or_else(|| defaults.streaming_dispatch().reasoning_dispatch()),
            ),
            chat_completions_reasoning_field,
        })
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

    /// Return an explicitly configured reasoning field, if one was supplied.
    pub fn configured_chat_completions_reasoning_field(
        &self,
    ) -> Option<ChatCompletionsReasoningField> {
        self.chat_completions_reasoning_field
    }

    pub fn set_chat_completions_reasoning_field(&mut self, field: ChatCompletionsReasoningField) {
        self.chat_completions_reasoning_field = Some(field);
    }

    /// Resolve and retain the field name exactly once while building the HTTP service.
    pub fn resolve_chat_completions_reasoning_field(
        &mut self,
    ) -> Result<ChatCompletionsReasoningField, String> {
        let field = match self.chat_completions_reasoning_field {
            Some(field) => field,
            None => ChatCompletionsReasoningField::from_env()?,
        };
        self.chat_completions_reasoning_field = Some(field);
        Ok(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_flags_return_none_when_all_values_are_unspecified() {
        let config = FrontendApiConfig::from_optional_flags(None, None, None, None);

        assert_eq!(config, None);
    }

    #[test]
    fn optional_flags_preserve_explicit_false_values() {
        let config = FrontendApiConfig::from_optional_flags_with_reasoning_field(
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(ChatCompletionsReasoningField::Reasoning),
        )
        .expect("explicit flags should produce a config");

        assert!(!config.anthropic().enabled());
        assert!(config.anthropic().strip_preamble());
        assert!(!config.streaming_dispatch().tool_dispatch());
        assert!(config.streaming_dispatch().reasoning_dispatch());
        assert_eq!(
            config.configured_chat_completions_reasoning_field(),
            Some(ChatCompletionsReasoningField::Reasoning)
        );
    }

    #[test]
    fn optional_flags_use_env_defaults_for_unspecified_values() {
        temp_env::with_vars(
            [
                (env_llm::DYN_ENABLE_ANTHROPIC_API, Some("1")),
                (env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH, Some("1")),
                (env_llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH, Some("1")),
            ],
            || {
                let config =
                    FrontendApiConfig::from_optional_flags(Some(false), None, None, Some(false))
                        .expect("partial flags should produce a config");

                assert!(!config.anthropic().enabled());
                assert!(config.anthropic().strip_preamble());
                assert!(config.streaming_dispatch().tool_dispatch());
                assert!(!config.streaming_dispatch().reasoning_dispatch());
            },
        );
    }

    #[test]
    fn reasoning_field_parsing_is_exact_and_case_sensitive() {
        assert_eq!(
            "reasoning_content".parse(),
            Ok(ChatCompletionsReasoningField::ReasoningContent)
        );
        assert_eq!(
            "reasoning".parse(),
            Ok(ChatCompletionsReasoningField::Reasoning)
        );

        for invalid in ["", "Reasoning", " reasoning", "reasoning_content "] {
            assert!(
                invalid.parse::<ChatCompletionsReasoningField>().is_err(),
                "unexpectedly accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn explicit_reasoning_field_wins_over_invalid_environment() {
        temp_env::with_var(
            env_llm::DYN_CHAT_COMPLETIONS_REASONING_FIELD,
            Some("invalid"),
            || {
                let mut config = FrontendApiConfig::default();
                config
                    .set_chat_completions_reasoning_field(ChatCompletionsReasoningField::Reasoning);

                assert_eq!(
                    config.resolve_chat_completions_reasoning_field(),
                    Ok(ChatCompletionsReasoningField::Reasoning)
                );
            },
        );
    }
}
