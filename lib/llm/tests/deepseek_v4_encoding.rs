// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for DeepSeek V4 encoding against official test data
//!
//! These tests use the official test files from:
//! https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/encoding

use dynamo_llm::preprocessor::prompt::deepseek_v4::{ThinkingMode, encode_messages};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::PathBuf;

fn get_test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/deepseek-v4")
}

/// Load an input fixture. V4 fixtures come in two shapes:
///   1. `{"tools": [...], "messages": [...]}` — tools injected on first (system) message
///   2. bare `[...]` — just the messages array
fn load_messages(path: &PathBuf) -> Vec<JsonValue> {
    let raw: JsonValue = serde_json::from_str(
        &fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {:?}", path)),
    )
    .unwrap_or_else(|_| panic!("Failed to parse {:?}", path));

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
        panic!("Unexpected input shape in {:?}", path);
    }
}

fn run_official_test(input_file: &str, output_file: &str, thinking_mode: ThinkingMode) {
    let test_dir = get_test_data_path();
    let messages = load_messages(&test_dir.join(input_file));
    let expected = fs::read_to_string(test_dir.join(output_file))
        .unwrap_or_else(|_| panic!("Failed to read {}", output_file));

    let actual = encode_messages(&messages, thinking_mode, true)
        .unwrap_or_else(|e| panic!("encode_messages failed for {}: {:?}", input_file, e));

    let exp = expected.trim_end();
    let act = actual.trim_end();

    if exp != act {
        println!("=== Test: {} ===", input_file);
        let exp_lines: Vec<&str> = exp.lines().collect();
        let act_lines: Vec<&str> = act.lines().collect();
        for (i, (el, al)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
            if el != al {
                println!("Line {} differs:", i + 1);
                println!("  Expected: {:?}", el);
                println!("  Actual:   {:?}", al);
                break;
            }
        }
        if exp_lines.len() != act_lines.len() {
            println!(
                "\nLine count mismatch: expected {} lines, got {} lines",
                exp_lines.len(),
                act_lines.len()
            );
        }
        panic!("Output does not match expected for {}", input_file);
    }
}

/// Case 1 — thinking mode, single tool, tool result round-trip.
#[test]
fn test_official_thinking_with_tools() {
    run_official_test(
        "test_input_1.json",
        "test_output_1.txt",
        ThinkingMode::Thinking,
    );
}

/// Case 2 — thinking mode, no tools, multi-turn (drop_thinking strips earlier reasoning).
#[test]
fn test_official_thinking_no_tools_multiturn() {
    run_official_test(
        "test_input_2.json",
        "test_output_2.txt",
        ThinkingMode::Thinking,
    );
}

/// Case 3 — thinking mode, developer role with tools + latest_reminder + tool result.
#[test]
fn test_official_developer_with_tools_and_reminder() {
    run_official_test(
        "test_input_3.json",
        "test_output_3.txt",
        ThinkingMode::Thinking,
    );
}

/// Case 4 — chat mode, latest_reminder + task="action" + mask preservation.
#[test]
fn test_official_chat_mode_action_task() {
    run_official_test("test_input_4.json", "test_output_4.txt", ThinkingMode::Chat);
}
