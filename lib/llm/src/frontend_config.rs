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

/// Frontend admission gate limits (DEP: Request Admission and Rejection Controls).
///
/// Each gate is disabled when its limit is `None` and active only when
/// explicitly configured. The HTTP service evaluates these before dispatching
/// a request to the engine and rejects with HTTP 503 when admitting it would
/// exceed a configured limit.
///
/// - `request_concurrency_limit` is enforced separately for each served model.
/// - `runtime_task_limit` and `request_plane_connection_limit` are
///   frontend-local self-protection gates and are not per-model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionGateConfig {
    request_concurrency_limit: Option<u64>,
    runtime_task_limit: Option<u64>,
    request_plane_connection_limit: Option<u64>,
}

impl AdmissionGateConfig {
    pub fn new(
        request_concurrency_limit: Option<u64>,
        runtime_task_limit: Option<u64>,
        request_plane_connection_limit: Option<u64>,
    ) -> Result<Self, String> {
        for (name, limit) in [
            ("request_concurrency_limit", request_concurrency_limit),
            ("runtime_task_limit", runtime_task_limit),
            (
                "request_plane_connection_limit",
                request_plane_connection_limit,
            ),
        ] {
            if limit == Some(0) {
                return Err(format!("{name} must be >= 1 (omit it to disable the gate)"));
            }
        }

        Ok(Self {
            request_concurrency_limit,
            runtime_task_limit,
            request_plane_connection_limit,
        })
    }

    /// Build from optional Python kwargs. Returns `None` when every limit is
    /// unspecified so env-backed defaults win for direct Rust callers.
    pub fn from_optional_limits(
        request_concurrency_limit: Option<u64>,
        runtime_task_limit: Option<u64>,
        request_plane_connection_limit: Option<u64>,
    ) -> Result<Option<Self>, String> {
        if request_concurrency_limit.is_none()
            && runtime_task_limit.is_none()
            && request_plane_connection_limit.is_none()
        {
            return Ok(None);
        }
        Self::new(
            request_concurrency_limit,
            runtime_task_limit,
            request_plane_connection_limit,
        )
        .map(Some)
    }

    pub fn request_concurrency_limit(&self) -> Option<u64> {
        self.request_concurrency_limit
    }

    pub fn runtime_task_limit(&self) -> Option<u64> {
        self.runtime_task_limit
    }

    pub fn request_plane_connection_limit(&self) -> Option<u64> {
        self.request_plane_connection_limit
    }

    /// True when no gate is configured, i.e. admission control is fully off.
    pub fn is_disabled(&self) -> bool {
        self.request_concurrency_limit.is_none()
            && self.runtime_task_limit.is_none()
            && self.request_plane_connection_limit.is_none()
    }

    /// Read a gate limit from the environment for direct Rust callers.
    /// Unset means disabled; unparsable or zero values disable the gate with
    /// a warning because every gate must be enabled explicitly and `0` would
    /// reject all traffic.
    fn limit_from_env(var: &str) -> Option<u64> {
        let raw = std::env::var(var).ok()?;
        match raw.parse::<u64>() {
            Ok(0) | Err(_) => {
                tracing::warn!(
                    env = var,
                    value = raw,
                    "invalid admission gate limit (must be an integer >= 1); gate stays disabled"
                );
                None
            }
            Ok(limit) => Some(limit),
        }
    }
}

impl Default for AdmissionGateConfig {
    fn default() -> Self {
        Self {
            request_concurrency_limit: Self::limit_from_env(
                env_llm::DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT,
            ),
            runtime_task_limit: Self::limit_from_env(
                env_llm::DYN_REJECTION_FRONTEND_RUNTIME_TASK_LIMIT,
            ),
            request_plane_connection_limit: Self::limit_from_env(
                env_llm::DYN_REJECTION_FRONTEND_REQUEST_PLANE_CONNECTION_LIMIT,
            ),
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
}

impl FrontendApiConfig {
    pub fn new(anthropic: AnthropicApiConfig, streaming_dispatch: StreamingDispatchConfig) -> Self {
        Self {
            anthropic,
            streaming_dispatch,
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
        }
    }

    pub fn from_optional_flags(
        enable_anthropic_api: Option<bool>,
        strip_anthropic_preamble: Option<bool>,
        enable_streaming_tool_dispatch: Option<bool>,
        enable_streaming_reasoning_dispatch: Option<bool>,
    ) -> Option<Self> {
        if enable_anthropic_api.is_none()
            && strip_anthropic_preamble.is_none()
            && enable_streaming_tool_dispatch.is_none()
            && enable_streaming_reasoning_dispatch.is_none()
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
        let config = FrontendApiConfig::from_optional_flags(
            Some(false),
            Some(true),
            Some(false),
            Some(true),
        )
        .expect("explicit flags should produce a config");

        assert!(!config.anthropic().enabled());
        assert!(config.anthropic().strip_preamble());
        assert!(!config.streaming_dispatch().tool_dispatch());
        assert!(config.streaming_dispatch().reasoning_dispatch());
    }

    #[test]
    fn admission_gate_optional_limits_return_none_when_all_unspecified() {
        assert_eq!(
            AdmissionGateConfig::from_optional_limits(None, None, None)
                .expect("unspecified limits should be valid"),
            None
        );
    }

    #[test]
    fn admission_gate_optional_limits_preserve_partial_values() {
        let config = AdmissionGateConfig::from_optional_limits(Some(8), None, Some(512))
            .expect("positive limits should be valid")
            .expect("partial limits should produce a config");

        assert_eq!(config.request_concurrency_limit(), Some(8));
        assert_eq!(config.runtime_task_limit(), None);
        assert_eq!(config.request_plane_connection_limit(), Some(512));
        assert!(!config.is_disabled());
    }

    #[test]
    fn admission_gate_programmatic_limits_reject_zero() {
        for (request, runtime, request_plane, field) in [
            (Some(0), None, None, "request_concurrency_limit"),
            (None, Some(0), None, "runtime_task_limit"),
            (None, None, Some(0), "request_plane_connection_limit"),
        ] {
            let error = AdmissionGateConfig::from_optional_limits(request, runtime, request_plane)
                .expect_err("zero must not enable a reject-all gate");
            assert!(error.contains(field), "unexpected error: {error}");
        }
    }

    #[test]
    fn admission_gate_default_is_disabled_without_env() {
        temp_env::with_vars(
            [
                (
                    env_llm::DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT,
                    None::<&str>,
                ),
                (env_llm::DYN_REJECTION_FRONTEND_RUNTIME_TASK_LIMIT, None),
                (
                    env_llm::DYN_REJECTION_FRONTEND_REQUEST_PLANE_CONNECTION_LIMIT,
                    None,
                ),
            ],
            || {
                assert!(AdmissionGateConfig::default().is_disabled());
            },
        );
    }

    #[test]
    fn admission_gate_default_reads_env_for_direct_rust_callers() {
        temp_env::with_vars(
            [
                (
                    env_llm::DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT,
                    Some("4"),
                ),
                (
                    env_llm::DYN_REJECTION_FRONTEND_RUNTIME_TASK_LIMIT,
                    Some("20000"),
                ),
                (
                    env_llm::DYN_REJECTION_FRONTEND_REQUEST_PLANE_CONNECTION_LIMIT,
                    Some("1024"),
                ),
            ],
            || {
                let config = AdmissionGateConfig::default();
                assert_eq!(config.request_concurrency_limit(), Some(4));
                assert_eq!(config.runtime_task_limit(), Some(20000));
                assert_eq!(config.request_plane_connection_limit(), Some(1024));
            },
        );
    }

    #[test]
    fn admission_gate_env_zero_or_garbage_stays_disabled() {
        for bad in ["0", "-1", "lots"] {
            temp_env::with_var(
                env_llm::DYN_REJECTION_FRONTEND_REQUEST_CONCURRENCY_LIMIT,
                Some(bad),
                || {
                    assert_eq!(
                        AdmissionGateConfig::default().request_concurrency_limit(),
                        None,
                        "env value {bad:?} must leave the gate disabled"
                    );
                },
            );
        }
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
}
