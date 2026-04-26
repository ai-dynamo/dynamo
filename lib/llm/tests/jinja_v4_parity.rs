// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-identical parity check: minijinja + `deepseek_v4_inline.jinja` against
//! the same `tests/data/deepseek-v4/` fixtures the existing native-Rust port
//! already validates against. If these tests pass, the Jinja path is a drop-in
//! replacement for `deepseek_v4::encode_messages` and the native port can be
//! retired.

use dynamo_llm::preprocessor::prompt::jinja_chat::{ThinkingMode, render_v4};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/deepseek-v4")
}

/// V4 fixtures come in two shapes:
///   1. `{"tools": [...], "messages": [...]}` — tools injected on first message
///   2. bare `[...]` — just the messages array
fn load_messages(input_file: &str) -> Vec<JsonValue> {
    let path = fixture_dir().join(input_file);
    let raw: JsonValue = serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|_| panic!("read {input_file}")),
    )
    .unwrap_or_else(|_| panic!("parse {input_file}"));

    if let Some(messages) = raw.get("messages").and_then(|m| m.as_array()) {
        let mut messages = messages.clone();
        if let Some(tools) = raw.get("tools")
            && let Some(first) = messages.get_mut(0)
            && let Some(obj) = first.as_object_mut()
        {
            obj.insert("tools".to_string(), tools.clone());
        }
        messages
    } else if let Some(arr) = raw.as_array() {
        arr.clone()
    } else {
        panic!("Unexpected input shape in {input_file}");
    }
}

fn run_parity(input_file: &str, output_file: &str, thinking_mode: ThinkingMode) {
    let messages = load_messages(input_file);
    let expected = fs::read_to_string(fixture_dir().join(output_file))
        .unwrap_or_else(|_| panic!("read {output_file}"));

    let actual = render_v4(&messages, thinking_mode, true, true, None)
        .unwrap_or_else(|e| panic!("render_v4 failed for {input_file}: {e:?}"));

    let exp = expected.trim_end();
    let act = actual.trim_end();

    if exp != act {
        let exp_lines: Vec<&str> = exp.lines().collect();
        let act_lines: Vec<&str> = act.lines().collect();
        for (i, (el, al)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
            if el != al {
                eprintln!("=== {input_file} ===");
                eprintln!("Line {} differs:", i + 1);
                eprintln!("  Expected: {el:?}");
                eprintln!("  Actual:   {al:?}");
                break;
            }
        }
        if exp_lines.len() != act_lines.len() {
            eprintln!(
                "Line count: expected {} vs actual {}",
                exp_lines.len(),
                act_lines.len()
            );
        }
        panic!("Jinja V4 output diverges from fixture for {input_file}");
    }
}

#[test]
fn jinja_v4_fixture_1_thinking_with_tools() {
    run_parity(
        "test_input_1.json",
        "test_output_1.txt",
        ThinkingMode::Thinking,
    );
}

#[test]
fn jinja_v4_fixture_2_thinking_no_tools_multiturn() {
    run_parity(
        "test_input_2.json",
        "test_output_2.txt",
        ThinkingMode::Thinking,
    );
}

#[test]
fn jinja_v4_fixture_3_developer_with_tools_and_reminder() {
    run_parity(
        "test_input_3.json",
        "test_output_3.txt",
        ThinkingMode::Thinking,
    );
}

#[test]
fn jinja_v4_fixture_4_chat_mode_action_task() {
    run_parity(
        "test_input_4.json",
        "test_output_4.txt",
        ThinkingMode::Chat,
    );
}
