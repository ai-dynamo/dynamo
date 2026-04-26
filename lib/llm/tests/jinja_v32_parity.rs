// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-identical parity check: minijinja + `deepseek_v32_inline.jinja` against
//! the same `tests/data/deepseek-v3.2/` fixtures the existing native-Rust port
//! already validates against.

use dynamo_llm::preprocessor::prompt::jinja_chat::{ThinkingMode, render_v32};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/deepseek-v3.2")
}

fn load_messages(input_file: &str) -> Vec<JsonValue> {
    let path = fixture_dir().join(input_file);
    let raw: JsonValue = serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|_| panic!("read {input_file}")),
    )
    .unwrap_or_else(|_| panic!("parse {input_file}"));

    let mut messages = raw["messages"]
        .as_array()
        .unwrap_or_else(|| panic!("Missing messages in {input_file}"))
        .clone();

    if let Some(tools) = raw.get("tools")
        && let Some(first) = messages.get_mut(0)
        && let Some(obj) = first.as_object_mut()
    {
        obj.insert("tools".to_string(), tools.clone());
    }
    messages
}

fn run_parity(input_file: &str, output_file: &str) {
    let messages = load_messages(input_file);
    let expected = fs::read_to_string(fixture_dir().join(output_file))
        .unwrap_or_else(|_| panic!("read {output_file}"));

    let actual = render_v32(&messages, ThinkingMode::Thinking, true, true)
        .unwrap_or_else(|e| panic!("render_v32 failed for {input_file}: {e:?}"));

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
        panic!("Jinja V3.2 output diverges from fixture for {input_file}");
    }
}

#[test]
fn jinja_v32_basic_example() {
    run_parity("test_input.json", "test_output.txt");
}

#[test]
fn jinja_v32_search_without_date() {
    run_parity(
        "test_input_search_wo_date.json",
        "test_output_search_wo_date.txt",
    );
}

#[test]
fn jinja_v32_search_with_date() {
    run_parity(
        "test_input_search_w_date.json",
        "test_output_search_w_date.txt",
    );
}
