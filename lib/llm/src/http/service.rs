// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP Service for Dynamo LLM
//!
//! The primary purpose of this crate is to service the dynamo-llm protocols via OpenAI compatible HTTP endpoints. This component
//! is meant to be a gateway/ingress into the Dynamo LLM Distributed Runtime.
//!
//! In order to create a common pattern, the HttpService forwards the incoming OAI Chat Request or OAI Completion Request to the
//! to a model-specific engines.  The engines can be attached and detached dynamically using the [`ModelManager`](crate::discovery::ModelManager).
//!
//! Note: All requests, whether the client requests `stream=true` or `stream=false`, are propagated downstream as `stream=true`.
//! This enables use to handle only 1 pattern of request-response in the downstream services. Non-streaming user requests are
//! aggregated by the HttpService and returned as a single response.
//!
//! TODO(): Add support for model-specific metadata and status. Status will allow us to return a 503 when the model is supposed
//! to be ready, but there is a problem with the model.
//!
//! The [`service_v2::HttpService`] can be further extended to host any [`axum::Router`] using the [`service_v2::HttpServiceConfigBuilder`].

mod anthropic;
pub mod metadata;
mod openai;

pub mod busy_threshold;
pub mod disconnect;
pub mod error;
pub mod frontend_extension;
pub mod generate;
pub mod health;
pub mod metrics;
pub mod openapi_docs;
pub mod realtime;
pub mod service_v2;
pub mod sglang_generate;

pub use axum;
pub use frontend_extension::{
    FrontendExtensionContext, FrontendRouteExtension, FrontendRouteSet,
    validate_extension_route_path,
};
pub use metrics::Metrics;

use crate::{
    preprocessor::OpenAIPreprocessor,
    protocols::openai::{ParsingOptions, chat_completions::NvCreateChatCompletionRequest},
};

use dynamo_protocols::types::ChatCompletionToolChoiceOption;
use dynamo_runtime::error::DynamoError;

/// Apply the request-level tool-call gates shared by the HTTP protocol handlers.
fn apply_request_tool_call_parsing_options(
    parsing_options: ParsingOptions,
    request: &NvCreateChatCompletionRequest,
) -> Result<ParsingOptions, DynamoError> {
    let tool_call_parsing_enabled = OpenAIPreprocessor::tool_call_parsing_enabled(request);
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    let converted_tool_choice = crate::preprocessor::tool_choice::convert_tool_choice(tool_choice);
    let tools = request.inner.tools.as_deref().unwrap_or(&[]);
    let converted_tools = crate::preprocessor::tool_choice::convert_tools(tools);
    let uses_structural_tag = crate::preprocessor::structural_tag::structural_tag_decision(
        parsing_options.tool_call_parser.as_deref(),
        &converted_tool_choice,
        &converted_tools,
        request.inner.parallel_tool_calls,
        parsing_options.structural_tag_mode,
        parsing_options.structural_tag_scope,
        parsing_options.exclude_tools_when_tool_choice_none,
    )?
    .is_required();
    let guided_tool_constraint = crate::preprocessor::tool_choice::guided_tool_constraint(
        request,
        parsing_options.tool_call_parser.as_deref(),
        parsing_options.reasoning_parser.as_deref(),
        uses_structural_tag,
    )?;
    Ok(parsing_options
        .with_guided_tool_constraint(guided_tool_constraint)
        .with_tool_call_parsing_enabled(tool_call_parsing_enabled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::GuidedToolConstraint;
    use serde_json::{Value, json};

    fn request(tool_choice: Value) -> NvCreateChatCompletionRequest {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": tool_choice,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }]
        });
        serde_json::from_value(value).expect("request must deserialize")
    }

    // A Kimi K2 pair with a forced/named tool_choice must resolve to the real
    // structural-tag decision computed by the single shared owner,
    // `structural_tag::structural_tag_decision`, not to a second, independently
    // reconstructed decision. `apply_tool_choice_structural_tag` is the in-process
    // preprocessing owner that consults the same shared function; this HTTP-layer
    // helper must agree with it, including when the parser registry does or does not
    // actually have a builder for the parser (see `structural_tag_decision`'s own
    // doc comment for why eligibility and registry-availability are bundled into one
    // value rather than checked independently).
    #[test]
    fn kimi_k2_required_resolves_to_the_real_structural_tag_decision() {
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("kimi_k2".to_string()),
            ..Default::default()
        };
        let result =
            apply_request_tool_call_parsing_options(parsing_options, &request(json!("required")))
                .expect("kimi_k2 + required must resolve a constraint");
        assert_eq!(
            result.guided_tool_constraint,
            GuidedToolConstraint::StructuralTag,
            "kimi_k2 + required must use the intrinsic structural tag, not a reconstructed JSON schema"
        );
    }

    #[test]
    fn kimi_k2_named_resolves_to_the_real_structural_tag_decision() {
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("kimi_k2".to_string()),
            ..Default::default()
        };
        let named = json!({"type": "function", "function": {"name": "get_weather"}});
        let result = apply_request_tool_call_parsing_options(parsing_options, &request(named))
            .expect("kimi_k2 + named must resolve a constraint");
        assert_eq!(
            result.guided_tool_constraint,
            GuidedToolConstraint::StructuralTag,
            "kimi_k2 + a named tool choice must use the intrinsic structural tag, not a reconstructed JSON schema"
        );
    }

    // Non-structural-tag parsers must be unaffected by this fix: they still get a
    // reconstructed JSON-schema constraint for a forced tool_choice.
    #[test]
    fn non_structural_tag_parser_still_gets_guided_json_required() {
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("hermes".to_string()),
            ..Default::default()
        };
        let result =
            apply_request_tool_call_parsing_options(parsing_options, &request(json!("required")))
                .expect("hermes + required must resolve a constraint");
        assert_eq!(
            result.guided_tool_constraint,
            GuidedToolConstraint::GuidedJsonRequired
        );
    }

    // The registry-builder gap this test suite exists to close: a non-Kimi parser
    // (qwen3_coder, which the parser registry does register a structural-tag builder
    // for) with a forced tool_choice must still resolve to `StructuralTag` once the
    // operator has globally enabled structural-tag mode — matching the real
    // preprocessing path (`apply_tool_choice_structural_tag`'s
    // `structural_tag_mode != Off` branch), not the narrower Kimi-only reconstruction
    // this helper used before this fix.
    #[test]
    fn operator_enabled_mode_resolves_structural_tag_for_a_non_kimi_parser() {
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("qwen3_coder".to_string()),
            structural_tag_mode: crate::local_model::runtime_config::StructuralTagMode::On,
            ..Default::default()
        };
        let result =
            apply_request_tool_call_parsing_options(parsing_options, &request(json!("required")))
                .expect("qwen3_coder + required must resolve a constraint");
        assert_eq!(
            result.guided_tool_constraint,
            GuidedToolConstraint::StructuralTag,
            "an operator-enabled structural_tag_mode must apply to any registry-supported \
             parser, not only the Kimi-intrinsic case"
        );
    }

    // With the operator default (`structural_tag_mode = Off`), the same non-Kimi
    // parser must NOT get a structural tag — confirms the mode gate above is real,
    // not a permanently-on regression.
    #[test]
    fn operator_default_mode_off_does_not_resolve_structural_tag_for_a_non_kimi_parser() {
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("qwen3_coder".to_string()),
            ..Default::default()
        };
        let result =
            apply_request_tool_call_parsing_options(parsing_options, &request(json!("required")))
                .expect("qwen3_coder + required (mode off) must resolve a constraint");
        assert_ne!(
            result.guided_tool_constraint,
            GuidedToolConstraint::StructuralTag
        );
    }

    // `tool_choice: none` must never leave the client able to observe a tool call or
    // a tool-call finish reason, regardless of the operator's structural-tag mode or
    // exclusion setting. The real preprocessing path may install a ban tag at
    // generation time for `none` (which is a request-time generation constraint on
    // the engine, tracked separately from this field), but that must not surface as
    // a guided-tool-output-parsing constraint here: `with_tool_call_parsing_enabled`
    // (called unconditionally by this function, below) always suppresses tool-call
    // output and resets `guided_tool_constraint` to `None` whenever tool-call parsing
    // is disabled, which `tool_choice: none` always is. Covering both the ban-enabled
    // and default-exclusion configurations proves this safety invariant holds either
    // way, not just in the configuration that happens not to trigger the ban path.
    #[test]
    fn tool_choice_none_never_exposes_tool_calls_regardless_of_ban_tag_configuration() {
        for exclude_tools_when_tool_choice_none in [true, false] {
            let parsing_options = ParsingOptions {
                tool_call_parser: Some("qwen3_coder".to_string()),
                structural_tag_mode: crate::local_model::runtime_config::StructuralTagMode::On,
                exclude_tools_when_tool_choice_none,
                ..Default::default()
            };
            let result =
                apply_request_tool_call_parsing_options(parsing_options, &request(json!("none")))
                    .expect("tool_choice=none must resolve a constraint");
            assert_eq!(
                result.guided_tool_constraint,
                GuidedToolConstraint::None,
                "tool_choice=none must never report a guided-tool-output constraint \
                 (exclude_tools_when_tool_choice_none={exclude_tools_when_tool_choice_none})"
            );
            assert!(
                result.suppress_tool_calls,
                "tool_choice=none must suppress tool calls \
                 (exclude_tools_when_tool_choice_none={exclude_tools_when_tool_choice_none})"
            );
        }
    }

    // `guided_tool_constraint` must validate the forced choice against the real
    // `tools` list via `get_json_schema_from_tools` (the same function
    // `apply_tool_choice_guided_decoding` calls at request-preprocessing time)
    // instead of unconditionally installing a JSON-schema constraint. A
    // `tool_choice: "required"` request with an empty `tools` list has nothing to
    // constrain against and must be rejected, not silently resolved to
    // `GuidedJsonRequired`.
    #[test]
    fn required_tool_choice_with_empty_tools_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": "required",
            "tools": []
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("hermes".to_string()),
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "tool_choice=required with no tools must not resolve to a bogus GuidedJsonRequired"
        );
    }

    // A named tool_choice that references a tool absent from `tools` is equally
    // unvalidatable and must also be rejected rather than resolved to a
    // constraint for a tool the request never declared.
    #[test]
    fn named_tool_choice_for_absent_tool_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": {"type": "function", "function": {"name": "does_not_exist"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("hermes".to_string()),
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "a named tool_choice for a tool absent from `tools` must not resolve to a \
             bogus constraint"
        );
    }

    // Regression for the CodeRabbit finding on PR #12576 (review comment 3836534587,
    // filed against this file): `structural_tag_decision` returned `Required`
    // without checking `tools`, so `guided_tool_constraint` skipped
    // `get_json_schema_from_tools` entirely for the STRUCTURAL-TAG branch (it
    // short-circuits to `Ok(StructuralTag)` whenever `uses_structural_tag` is true,
    // without building or validating anything). The two `hermes` tests above only
    // ever exercised the non-structural-tag JSON-schema fallback, which already
    // validated correctly — they never covered the actual bug. Kimi K2 is the
    // structural-tag-intrinsic parser, so this reproduces the real gap.
    #[test]
    fn kimi_k2_required_with_empty_tools_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": "required",
            "tools": []
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("kimi_k2".to_string()),
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "kimi_k2 + required with no tools must not resolve to StructuralTag"
        );
    }

    #[test]
    fn kimi_k2_named_tool_choice_for_absent_tool_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": {"type": "function", "function": {"name": "does_not_exist"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("kimi_k2".to_string()),
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "kimi_k2 + a named tool_choice for a tool absent from `tools` must not \
             resolve to StructuralTag"
        );
    }

    #[test]
    fn kimi_k3_named_tool_choice_for_absent_tool_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": {"type": "function", "function": {"name": "does_not_exist"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("kimi_k3".to_string()),
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "kimi_k3 + a named tool_choice for a tool absent from `tools` must not \
             resolve to StructuralTag"
        );
    }

    #[test]
    fn kimi_k3_required_with_empty_tools_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": "required",
            "tools": []
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("kimi_k3".to_string()),
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "kimi_k3 + required with no tools must be rejected before the XTML path"
        );
    }

    // Same gap, operator-enabled path: qwen3_coder only gets a structural tag when
    // `structural_tag_mode = On` (it is not Kimi-intrinsic). Confirms the fix isn't
    // narrowly scoped to the two Kimi-intrinsic parsers.
    #[test]
    fn operator_enabled_qwen3_coder_required_with_empty_tools_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": "required",
            "tools": []
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("qwen3_coder".to_string()),
            structural_tag_mode: crate::local_model::runtime_config::StructuralTagMode::On,
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "operator-enabled qwen3_coder + required with no tools must not resolve \
             to StructuralTag"
        );
    }

    #[test]
    fn operator_enabled_qwen3_coder_named_tool_choice_for_absent_tool_is_rejected() {
        let value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tool_choice": {"type": "function", "function": {"name": "does_not_exist"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        let parsing_options = ParsingOptions {
            tool_call_parser: Some("qwen3_coder".to_string()),
            structural_tag_mode: crate::local_model::runtime_config::StructuralTagMode::On,
            ..Default::default()
        };
        let result = apply_request_tool_call_parsing_options(parsing_options, &request);
        assert!(
            result.is_err(),
            "operator-enabled qwen3_coder + a named tool_choice for a tool absent from \
             `tools` must not resolve to StructuralTag"
        );
    }
}

/// Documentation for a route
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RouteDoc {
    method: axum::http::Method,
    path: String,
}

impl std::fmt::Display for RouteDoc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {}", self.method, self.path)
    }
}

impl RouteDoc {
    pub fn new<T: Into<String>>(method: axum::http::Method, path: T) -> Self {
        RouteDoc {
            method,
            path: path.into(),
        }
    }

    pub fn method(&self) -> &axum::http::Method {
        &self.method
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}
