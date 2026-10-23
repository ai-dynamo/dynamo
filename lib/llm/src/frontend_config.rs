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
    is_truthy,
};

use crate::reasoning_field::ReasoningField;

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

/// Status used for a media refusal when none is configured: the status the
/// `UnsupportedMedia` error class maps to.
pub const DEFAULT_REJECT_MEDIA_STATUS_CODE: u16 = 415;

/// Parse a media-refusal status. Only 4xx and 5xx are accepted: the refusal
/// is sent before any response body, and a 1xx/2xx/3xx would read as success.
pub fn parse_reject_media_status_code(value: &str) -> Option<u16> {
    value
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|code| (400..=599).contains(code))
}

/// Media refusal for frontends that serve text-only models.
///
/// When enabled, chat requests (OpenAI Chat Completions, Responses, Anthropic
/// Messages) that carry an image, video, or audio part in any message are
/// refused before routing, so no worker sees them and a streaming client gets
/// an HTTP error instead of a 200 followed by an error event. The status is
/// configurable so an upstream gateway can fail over on it (for example 429).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaRejectionConfig {
    enabled: bool,
    status_code: u16,
}

impl MediaRejectionConfig {
    /// Fails unless `status_code` is 4xx or 5xx.
    pub fn new(enabled: bool, status_code: u16) -> anyhow::Result<Self> {
        if !(400..=599).contains(&status_code) {
            anyhow::bail!("reject_media status code must be 4xx or 5xx, got {status_code}");
        }
        Ok(Self {
            enabled,
            status_code,
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn status_code(&self) -> u16 {
        self.status_code
    }
}

impl Default for MediaRejectionConfig {
    fn default() -> Self {
        let status_code = match std::env::var(env_llm::DYN_REJECT_MEDIA_STATUS_CODE) {
            Ok(value) => parse_reject_media_status_code(&value).unwrap_or_else(|| {
                tracing::warn!(
                    "{}={value:?} is not a 4xx or 5xx status; defaulting to {}",
                    env_llm::DYN_REJECT_MEDIA_STATUS_CODE,
                    DEFAULT_REJECT_MEDIA_STATUS_CODE
                );
                DEFAULT_REJECT_MEDIA_STATUS_CODE
            }),
            Err(_) => DEFAULT_REJECT_MEDIA_STATUS_CODE,
        };
        Self {
            enabled: env_is_truthy(env_llm::DYN_REJECT_MEDIA),
            status_code,
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
    media_rejection: MediaRejectionConfig,
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
            media_rejection: MediaRejectionConfig::default(),
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
            media_rejection: MediaRejectionConfig::default(),
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

    pub fn media_rejection(&self) -> MediaRejectionConfig {
        self.media_rejection
    }

    pub fn set_media_rejection(&mut self, media_rejection: MediaRejectionConfig) {
        self.media_rejection = media_rejection;
    }
}

impl Default for FrontendApiConfig {
    fn default() -> Self {
        Self {
            anthropic: Default::default(),
            streaming_dispatch: Default::default(),
            reasoning_field: default_reasoning_field(),
            media_rejection: Default::default(),
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
    fn media_rejection_new_requires_error_status() {
        assert!(MediaRejectionConfig::new(true, 200).is_err());
        assert!(MediaRejectionConfig::new(true, 600).is_err());
        assert_eq!(
            MediaRejectionConfig::new(true, 429).unwrap().status_code(),
            429
        );
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

    #[test]
    fn invalid_reasoning_field_env_defaults_to_reasoning_content() {
        temp_env::with_var(env_llm::DYN_REASONING_FIELD_NAME, Some("invalid"), || {
            let config = FrontendApiConfig::default();

            assert_eq!(config.reasoning_field(), ReasoningField::ReasoningContent);
        });
    }

    #[test]
    fn media_rejection_defaults_to_disabled_with_unsupported_media_status() {
        temp_env::with_vars(
            [
                (env_llm::DYN_REJECT_MEDIA, None::<&str>),
                (env_llm::DYN_REJECT_MEDIA_STATUS_CODE, None),
            ],
            || {
                let config = FrontendApiConfig::default().media_rejection();

                assert!(!config.enabled());
                assert_eq!(config.status_code(), DEFAULT_REJECT_MEDIA_STATUS_CODE);
            },
        );
    }

    #[test]
    fn media_rejection_reads_env_and_ignores_invalid_status() {
        temp_env::with_vars(
            [
                (env_llm::DYN_REJECT_MEDIA, Some("1")),
                (env_llm::DYN_REJECT_MEDIA_STATUS_CODE, Some("429")),
            ],
            || {
                let config = MediaRejectionConfig::default();

                assert!(config.enabled());
                assert_eq!(config.status_code(), 429);
            },
        );
        for invalid in ["200", "302", "600", "abc"] {
            temp_env::with_var(env_llm::DYN_REJECT_MEDIA_STATUS_CODE, Some(invalid), || {
                assert_eq!(
                    MediaRejectionConfig::default().status_code(),
                    DEFAULT_REJECT_MEDIA_STATUS_CODE,
                    "{invalid}"
                );
            });
        }
    }
}
