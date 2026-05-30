// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::TOOL_NAME_PLACEHOLDER;
use super::builder::{StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext};
use super::dsml::DsmlToolCallsConfig;
use super::format::JsonSchemaStyle;
use super::triggered_tags::TriggeredTagsConfig;
use crate::tool_calling::{ToolCallConfig, ToolChoice, ToolDefinition};

fn sample_config() -> TriggeredTagsConfig {
    TriggeredTagsConfig {
        begin_template: format!("<function={}>", TOOL_NAME_PLACEHOLDER),
        end_template: "</function>".to_string(),
        triggers: vec!["<function=".to_string()],
        content_style: JsonSchemaStyle::Json,
        tool_call_ban_tokens: vec!["<tool_call>".to_string()],
        reasoning_end: Some("</think>".to_string()),
    }
}

fn sample_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "add_numbers".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            })),
            strict: None,
        },
        ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            })),
            strict: None,
        },
    ]
}

fn builder() -> StructuralTagBuilder {
    StructuralTagBuilder::TriggeredTags(sample_config())
}

fn ctx<'a>(
    tool_choice: &'a ToolChoice,
    tools: &'a [ToolDefinition],
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
) -> ToolCallFormatBuildContext<'a> {
    ToolCallFormatBuildContext {
        tool_choice,
        tools,
        parallel_tool_calls,
        schema_mode,
        starts_in_reasoning: false,
    }
}

fn ctx_starting_in_reasoning<'a>(
    tool_choice: &'a ToolChoice,
    tools: &'a [ToolDefinition],
) -> ToolCallFormatBuildContext<'a> {
    ToolCallFormatBuildContext {
        tool_choice,
        tools,
        parallel_tool_calls: None,
        schema_mode: StructuralTagSchemaMode::Auto,
        starts_in_reasoning: true,
    }
}

/// Helper: build and unwrap both Result and Option layers.
fn build_unwrap(
    builder: &StructuralTagBuilder,
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
) -> Value {
    let c = ctx(tool_choice, tools, parallel_tool_calls, schema_mode);
    builder
        .build_tool_call_format(&c)
        .expect("build should not error")
        .expect("build should return Some")
}

#[test]
fn required_builds_all_tools_with_at_least_one() {
    let tools = sample_tools();
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Required,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["type"], "structural_tag");
    assert_eq!(parsed["format"]["type"], "triggered_tags");
    assert_eq!(parsed["format"]["at_least_one"], true);
    let tags = parsed["format"]["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 2);
    // Each tag element must have "type": "tag" per xgrammar spec.
    assert_eq!(tags[0]["type"], "tag");
    assert_eq!(tags[0]["begin"], "<function=add_numbers>");
    assert_eq!(tags[1]["type"], "tag");
    assert_eq!(tags[1]["begin"], "<function=get_weather>");
    assert_eq!(tags[0]["end"], "</function>");
}

#[test]
fn auto_builds_all_tools_without_at_least_one() {
    let tools = sample_tools();
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Auto,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["format"]["at_least_one"], false);
    let tags = parsed["format"]["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 2);
    assert_eq!(tags[0]["type"], "tag");
    assert_eq!(tags[1]["type"], "tag");
}

#[test]
fn named_builds_single_tool() {
    let tools = sample_tools();
    let named = ToolChoice::Named("get_weather".to_string());
    let parsed = build_unwrap(
        &builder(),
        &named,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["format"]["at_least_one"], true);
    let tags = parsed["format"]["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["type"], "tag");
    assert_eq!(tags[0]["begin"], "<function=get_weather>");
}

#[test]
fn named_unknown_tool_returns_error() {
    let tools = sample_tools();
    let named = ToolChoice::Named("nonexistent".to_string());
    let c = ctx(&named, &tools, None, StructuralTagSchemaMode::Auto);
    let err = builder()
        .build_tool_call_format(&c)
        .expect_err("should error for unknown tool");
    assert!(
        err.to_string().contains("nonexistent"),
        "error should mention the missing tool name: {err}"
    );
}

#[test]
fn required_with_empty_tools_returns_error() {
    let c = ctx(
        &ToolChoice::Required,
        &[],
        None,
        StructuralTagSchemaMode::Auto,
    );
    let err = builder()
        .build_tool_call_format(&c)
        .expect_err("should error for empty tools");
    assert!(
        err.to_string().contains("empty"),
        "error should mention empty tools: {err}"
    );
}

#[test]
fn none_returns_ok_none() {
    let tools = sample_tools();
    let c = ctx(
        &ToolChoice::None,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );
    let result = builder()
        .build_tool_call_format(&c)
        .expect("should not error");
    assert!(result.is_none());
}

#[test]
fn auto_with_empty_tools_returns_ok_none() {
    let c = ctx(&ToolChoice::Auto, &[], None, StructuralTagSchemaMode::Auto);
    let result = builder()
        .build_tool_call_format(&c)
        .expect("should not error");
    assert!(result.is_none());
}

#[test]
fn parallel_false_sets_stop_after_first() {
    let tools = sample_tools();
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Required,
        &tools,
        Some(false),
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["format"]["stop_after_first"], true);
}

#[test]
fn parallel_true_does_not_stop_after_first() {
    let tools = sample_tools();
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Required,
        &tools,
        Some(true),
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["format"]["stop_after_first"], false);
}

#[test]
fn non_strict_tools_get_unconstrained_schema() {
    let tools = sample_tools();
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Required,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    let tags = parsed["format"]["tags"].as_array().unwrap();
    assert_eq!(tags[0]["content"]["json_schema"], json!(true));
}

#[test]
fn strict_tool_uses_own_schema() {
    let mut tools = sample_tools();
    tools[0].strict = Some(true);
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Required,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    let tags = parsed["format"]["tags"].as_array().unwrap();
    assert!(tags[0]["content"]["json_schema"]["properties"]["a"].is_object());
    assert_eq!(tags[1]["content"]["json_schema"], json!(true));
}

#[test]
fn strict_schema_mode_uses_real_schema_for_all() {
    let tools = sample_tools();
    let parsed = build_unwrap(
        &builder(),
        &ToolChoice::Required,
        &tools,
        None,
        StructuralTagSchemaMode::Strict,
    );

    let tags = parsed["format"]["tags"].as_array().unwrap();
    assert!(tags[0]["content"]["json_schema"]["properties"]["a"].is_object());
    assert!(tags[1]["content"]["json_schema"]["properties"]["location"].is_object());
}

#[test]
fn tool_call_ban_tag_bans_tokens() {
    let result = builder()
        .build_tool_call_ban()
        .expect("should serialize")
        .expect("should build");

    assert_eq!(result["type"], "structural_tag");
    assert_eq!(result["format"]["type"], "tag");
    assert_eq!(result["format"]["content"]["type"], "any_tokens");
    assert_eq!(
        result["format"]["content"]["exclude_tokens"][0],
        "<tool_call>"
    );
}

#[test]
fn tool_call_ban_empty_tokens_returns_none() {
    let mut config = sample_config();
    config.tool_call_ban_tokens = vec![];
    let b = StructuralTagBuilder::TriggeredTags(config);
    assert!(b.build_tool_call_ban().expect("should serialize").is_none());
}

#[test]
fn required_starting_in_reasoning_wraps_tool_calls_in_sequence() {
    let tools = sample_tools();
    let c = ctx_starting_in_reasoning(&ToolChoice::Required, &tools);

    let parsed = builder()
        .build_tool_call_format(&c)
        .expect("build should not error")
        .expect("build should return Some");

    assert_eq!(parsed["format"]["type"], "sequence");
    assert_eq!(parsed["format"]["elements"][0]["type"], "tag");
    assert_eq!(parsed["format"]["elements"][0]["begin"], "");
    assert_eq!(
        parsed["format"]["elements"][0]["content"]["type"],
        "any_text"
    );
    assert_eq!(parsed["format"]["elements"][0]["end"], "</think>");
    assert_eq!(parsed["format"]["elements"][1]["type"], "triggered_tags");
}

#[test]
fn auto_starting_in_reasoning_keeps_plain_tool_call_format() {
    let tools = sample_tools();
    let c = ctx_starting_in_reasoning(&ToolChoice::Auto, &tools);

    let parsed = builder()
        .build_tool_call_format(&c)
        .expect("build should not error")
        .expect("build should return Some");

    assert_eq!(parsed["format"]["type"], "triggered_tags");
}

// ---- DSML builder tests ----

fn dsml_config() -> DsmlToolCallsConfig {
    DsmlToolCallsConfig {
        trigger: "<｜DSML｜function_calls>".to_string(),
        block_begin: "<｜DSML｜function_calls>\n".to_string(),
        block_end: "</｜DSML｜function_calls>".to_string(),
        invoke_begin_template: format!("<｜DSML｜invoke name=\"{}\">\n", TOOL_NAME_PLACEHOLDER),
        invoke_end: "</｜DSML｜invoke>\n".to_string(),
        separator: "".to_string(),
        tool_call_ban_tokens: vec![],
        reasoning_end: Some("</think>".to_string()),
    }
}

fn dsml_builder() -> StructuralTagBuilder {
    StructuralTagBuilder::DsmlToolCalls(dsml_config())
}

fn dsml_build_unwrap(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
) -> Value {
    let c = ctx(tool_choice, tools, parallel_tool_calls, schema_mode);
    dsml_builder()
        .build_tool_call_format(&c)
        .expect("build should not error")
        .expect("build should return Some")
}

#[test]
fn dsml_required_builds_nested_structure() {
    let tools = sample_tools();
    let parsed = dsml_build_unwrap(
        &ToolChoice::Required,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    // Top level: structural_tag
    assert_eq!(parsed["type"], "structural_tag");

    // Outer: triggered_tags with one block tag
    let fmt = &parsed["format"];
    assert_eq!(fmt["type"], "triggered_tags");
    assert_eq!(fmt["triggers"][0], "<｜DSML｜function_calls>");
    assert_eq!(fmt["at_least_one"], true);
    assert_eq!(fmt["stop_after_first"], false);
    assert!(fmt.get("excludes").is_none());

    let outer_tags = fmt["tags"].as_array().unwrap();
    assert_eq!(outer_tags.len(), 1);

    // Block tag wrapping the invokes
    let block = &outer_tags[0];
    assert_eq!(block["type"], "tag");
    assert_eq!(block["begin"], "<｜DSML｜function_calls>\n");
    assert_eq!(block["end"], "</｜DSML｜function_calls>");

    // Inner: tags_with_separator
    let inner = &block["content"];
    assert_eq!(inner["type"], "tags_with_separator");
    assert_eq!(inner["separator"], "");
    assert_eq!(inner["at_least_one"], true);
    assert_eq!(inner["stop_after_first"], false);

    // Per-tool invoke tags
    let invoke_tags = inner["tags"].as_array().unwrap();
    assert_eq!(invoke_tags.len(), 2);
    assert_eq!(invoke_tags[0]["type"], "tag");
    assert_eq!(
        invoke_tags[0]["begin"],
        "<｜DSML｜invoke name=\"add_numbers\">\n"
    );
    assert_eq!(invoke_tags[0]["end"], "</｜DSML｜invoke>\n");
    assert_eq!(invoke_tags[0]["content"]["type"], "json_schema");
    assert_eq!(invoke_tags[0]["content"]["style"], "deepseek_xml");

    assert_eq!(
        invoke_tags[1]["begin"],
        "<｜DSML｜invoke name=\"get_weather\">\n"
    );
}

#[test]
fn dsml_auto_sets_inner_at_least_one() {
    let tools = sample_tools();
    let parsed = dsml_build_unwrap(
        &ToolChoice::Auto,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["format"]["at_least_one"], false);
    assert_eq!(parsed["format"]["stop_after_first"], false);
    let inner = &parsed["format"]["tags"][0]["content"];
    assert_eq!(inner["at_least_one"], true);
}

#[test]
fn dsml_named_builds_single_invoke() {
    let tools = sample_tools();
    let named = ToolChoice::Named("get_weather".to_string());
    let parsed = dsml_build_unwrap(&named, &tools, None, StructuralTagSchemaMode::Auto);

    let invoke_tags = parsed["format"]["tags"][0]["content"]["tags"]
        .as_array()
        .unwrap();
    assert_eq!(invoke_tags.len(), 1);
    assert_eq!(
        invoke_tags[0]["begin"],
        "<｜DSML｜invoke name=\"get_weather\">\n"
    );
}

#[test]
fn dsml_parallel_false_sets_stop_after_first() {
    let tools = sample_tools();
    let parsed = dsml_build_unwrap(
        &ToolChoice::Required,
        &tools,
        Some(false),
        StructuralTagSchemaMode::Auto,
    );

    assert_eq!(parsed["format"]["stop_after_first"], true);
    let inner = &parsed["format"]["tags"][0]["content"];
    assert_eq!(inner["stop_after_first"], true);
}

#[test]
fn dsml_none_returns_ok_none() {
    let tools = sample_tools();
    let c = ctx(
        &ToolChoice::None,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );
    let result = dsml_builder()
        .build_tool_call_format(&c)
        .expect("should not error");
    assert!(result.is_none());
}

#[test]
fn dsml_required_starting_in_reasoning_wraps_tool_calls_in_sequence() {
    let tools = sample_tools();
    let c = ctx_starting_in_reasoning(&ToolChoice::Required, &tools);

    let parsed = dsml_builder()
        .build_tool_call_format(&c)
        .expect("build should not error")
        .expect("build should return Some");

    assert_eq!(parsed["format"]["type"], "sequence");
    assert_eq!(parsed["format"]["elements"][0]["end"], "</think>");
    assert_eq!(parsed["format"]["elements"][1]["type"], "triggered_tags");
    assert_eq!(
        parsed["format"]["elements"][1]["triggers"][0],
        "<｜DSML｜function_calls>"
    );
}

#[test]
fn deepseek_v4_config_builds_tool_calls_block() {
    let tools = sample_tools();
    let config = ToolCallConfig::deepseek_v4();
    let builder = config
        .structural_tag_builder
        .as_ref()
        .expect("deepseek_v4 should provide a structural tag builder");
    let c = ctx(
        &ToolChoice::Required,
        &tools,
        None,
        StructuralTagSchemaMode::Auto,
    );
    let parsed = builder
        .build_tool_call_format(&c)
        .expect("build should not error")
        .expect("build should return Some");

    let fmt = &parsed["format"];
    assert_eq!(fmt["type"], "triggered_tags");
    assert_eq!(fmt["triggers"][0], "<｜DSML｜tool_calls>");
    assert_eq!(fmt["at_least_one"], true);
    assert!(fmt.get("excludes").is_none());

    let block = &fmt["tags"][0];
    assert_eq!(block["type"], "tag");
    assert_eq!(block["begin"], "<｜DSML｜tool_calls>\n");
    assert_eq!(block["end"], "</｜DSML｜tool_calls>");

    let inner = &block["content"];
    assert_eq!(inner["type"], "tags_with_separator");
    assert_eq!(inner["at_least_one"], true);
    assert_eq!(inner["separator"], "");

    let invoke_tags = inner["tags"].as_array().unwrap();
    assert_eq!(invoke_tags.len(), 2);
    assert_eq!(
        invoke_tags[0]["begin"],
        "<｜DSML｜invoke name=\"add_numbers\">\n"
    );
    assert_eq!(invoke_tags[0]["end"], "</｜DSML｜invoke>\n");
    assert_eq!(invoke_tags[0]["content"]["style"], "deepseek_xml");
}

/// Round-trip every supported parser through the real parser map.
#[test]
fn parser_map_structural_tag_smoke() {
    let map = crate::tool_calling::parsers::get_tool_parser_map();
    let tools = sample_tools();
    let names = ["hermes", "qwen3_coder", "deepseek_v3_2", "deepseek_v4"];

    for name in names {
        let builder = map
            .get(name)
            .unwrap_or_else(|| panic!("'{name}' not in parser map"))
            .structural_tag_builder
            .as_ref()
            .unwrap_or_else(|| panic!("'{name}' has no structural_tag_builder"));

        let c = ctx(
            &ToolChoice::Required,
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
        );
        let tag = builder
            .build_tool_call_format(&c)
            .unwrap_or_else(|e| panic!("'{name}' failed: {e}"))
            .unwrap_or_else(|| panic!("'{name}' returned None for Required"));

        assert_eq!(tag["type"], "structural_tag", "'{name}'");
    }
}
