// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static approach: the parser is linked in and registered by the catalog crate.

mod parser_plugin_support;

#[tokio::test]
async fn statically_linked_dummy_parser_streams_reasoning_and_tool_calls() {
    dynamo_parser_plugin_catalog::register();
    let seen = parser_plugin_support::run_dummy_stream("dummy").await;
    assert_eq!(seen, parser_plugin_support::expected());
}

/// `tool_choice=required` makes the model emit bare JSON; the plugin must handle that
/// mode in `initialize_request` or Dynamo silently falls back to plain text.
#[tokio::test]
async fn tool_choice_required_guided_json_becomes_tool_calls() {
    dynamo_parser_plugin_catalog::register();
    let seen = parser_plugin_support::run_stream(
        "dummy-alias",
        serde_json::json!({"tool_choice": "required"}),
        &[
            r#"[{"name":"get_weather","#,
            r#""parameters":{"location":"SF"}}]"#,
        ],
    )
    .await;
    assert_eq!(seen.calls, parser_plugin_support::expected().calls);
    assert_eq!(seen.finish, parser_plugin_support::expected().finish);
}

/// The dummy family allows requests to turn thinking off.
#[tokio::test]
async fn thinking_disabled_folds_plugin_reasoning_into_content() {
    dynamo_parser_plugin_catalog::register();
    let seen = parser_plugin_support::run_stream(
        "dummy",
        serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}}),
        parser_plugin_support::CHUNKS,
    )
    .await;
    assert_eq!(seen.reasoning, "");
    assert_eq!(seen.content, "user wants weatherChecking. ");
    assert_eq!(seen.calls, parser_plugin_support::expected().calls);
}

#[tokio::test]
async fn statically_linked_qwen_parser_streams_xml_tool_calls() {
    dynamo_parser_plugin_catalog::register();
    assert!(
        dynamo_parsers_v2::vendor_unified_families()
            .iter()
            .any(|family| family == "qwen3")
    );
    let seen = parser_plugin_support::run_stream(
        "qwen3_coder",
        serde_json::json!({}),
        &[
            "<thi",
            "nk>user wants weather</think>Checking. <tool_ca",
            "ll><function=get_weather><parameter=loc",
            "ation>SF</parameter></function></tool_call>",
        ],
    )
    .await;
    assert_eq!(seen, parser_plugin_support::expected());
}
