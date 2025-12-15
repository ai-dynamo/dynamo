// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for DeepSeek V3.2 encoding against official test data
//!
//! These tests use the official test files from:
//! https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding

use dynamo_llm::preprocessor::prompt::deepseek_v32::{ThinkingMode, encode_messages};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::PathBuf;

fn get_test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/deepseek-v3.2")
}

fn run_official_test(input_file: &str, output_file: &str) {
    let test_dir = get_test_data_path();

    // Load test input
    let input_path = test_dir.join(input_file);
    let input_data: JsonValue = serde_json::from_str(
        &fs::read_to_string(&input_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", input_file)),
    )
    .unwrap_or_else(|_| panic!("Failed to parse {}", input_file));

    // Load expected output
    let output_path = test_dir.join(output_file);
    let expected_output = fs::read_to_string(&output_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", output_file));

    // Extract messages and tools
    let mut messages = input_data["messages"]
        .as_array()
        .expect("Missing messages")
        .clone();

    // Add tools to first message (system) if present
    if let Some(tools) = input_data.get("tools")
        && let Some(first_msg) = messages.get_mut(0)
    {
        first_msg
            .as_object_mut()
            .unwrap()
            .insert("tools".to_string(), tools.clone());
    }

    // Encode messages
    let result = encode_messages(
        &messages,
        ThinkingMode::Thinking,
        true, // add_bos_token
    )
    .expect("Failed to encode messages");

    // Compare outputs
    let expected = expected_output.trim();
    let actual = result.trim();

    if expected != actual {
        println!("=== Test: {} ===", input_file);

        // Show first difference
        let exp_lines: Vec<&str> = expected.lines().collect();
        let act_lines: Vec<&str> = actual.lines().collect();

        for (i, (exp_line, act_line)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
            if exp_line != act_line {
                println!("Line {} differs:", i + 1);
                println!("  Expected: {}", exp_line);
                println!("  Actual:   {}", act_line);
                break;
            }
        }

        if exp_lines.len() != act_lines.len() {
            println!("\nLine count mismatch:");
            println!("  Expected: {} lines", exp_lines.len());
            println!("  Actual:   {} lines", act_lines.len());
        }

        panic!("Output does not match expected for {}", input_file);
    }
}

#[test]
fn test_official_basic_example() {
    run_official_test("test_input.json", "test_output.txt");
}

#[test]
fn test_official_search_without_date() {
    run_official_test(
        "test_input_search_wo_date.json",
        "test_output_search_wo_date.txt",
    );
}

#[test]
fn test_official_search_with_date() {
    run_official_test(
        "test_input_search_w_date.json",
        "test_output_search_w_date.txt",
    );
}

#[test]
fn test_simple_conversation_no_tools() {
    // Simple test without tools
    let messages = serde_json::json!([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "What is 2+2?"}
    ]);

    let result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
        .expect("Failed to encode");

    // Check basic structure
    assert!(result.starts_with("<｜begin▁of▁sentence｜>"));
    assert!(result.contains("<｜User｜>Hello!<｜Assistant｜>"));
    assert!(result.contains("Hi! How can I help you today?"));
    assert!(result.contains("<｜end▁of▁sentence｜>"));
}

#[test]
fn test_comprehensive_conversation_with_tools() {
    // Comprehensive test covering all features with English text
    let messages = serde_json::json!([
        {
            "role": "system",
            "content": "You are a helpful weather assistant.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_datetime",
                        "description": "Get the current date and time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {
                                    "type": "string",
                                    "description": "The timezone, e.g. America/New_York, UTC"
                                }
                            },
                            "required": ["timezone"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for a specific date and location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name, e.g. New York, San Francisco"
                                },
                                "date": {
                                    "type": "string",
                                    "description": "The date in YYYY-MM-DD format"
                                }
                            },
                            "required": ["location", "date"]
                        }
                    }
                }
            ]
        },
        {"role": "user", "content": "What's the weather tomorrow in San Francisco and New York?"},
        {
            "role": "assistant",
            "reasoning_content": "User is asking about tomorrow's weather. I need to first get the current date to calculate tomorrow's date.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_datetime",
                    "arguments": "{\"timezone\": \"America/New_York\"}"
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{\"current_date\": \"2024-01-15\", \"current_time\": \"14:30:00\", \"timezone\": \"America/New_York\"}"
        },
        {
            "role": "assistant",
            "reasoning_content": "Now I know today is 2024-01-15, so tomorrow is 2024-01-16. Let me query the weather for both cities.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"San Francisco\", \"date\": \"2024-01-16\"}"
                    }
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"New York\", \"date\": \"2024-01-16\"}"
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": "{\"location\": \"San Francisco\", \"date\": \"2024-01-16\", \"temperature_high\": \"65\", \"temperature_low\": \"55\", \"weather\": \"sunny\", \"humidity\": \"70%\"}"
        },
        {
            "role": "tool",
            "tool_call_id": "call_3",
            "content": "{\"location\": \"New York\", \"date\": \"2024-01-16\", \"temperature_high\": \"30\", \"temperature_low\": \"20\", \"weather\": \"cloudy\", \"humidity\": \"45%\"}"
        },
        {
            "role": "assistant",
            "reasoning_content": "Got the weather data for both cities. Let me format a nice response for the user.",
            "content": "Here's the weather forecast for tomorrow (January 16, 2024):\n\n**San Francisco**:\n- Weather: Sunny\n- High: 65°F\n- Low: 55°F\n- Humidity: 70%\n\n**New York**:\n- Weather: Cloudy\n- High: 30°F\n- Low: 20°F\n- Humidity: 45%\n\nSan Francisco will be warm and sunny, while New York will be cold and cloudy. Dress warmly if you're in New York!"
        }
    ]);

    let result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
        .expect("Failed to encode");

    // Check all major components are present
    assert!(result.starts_with("<｜begin▁of▁sentence｜>"));
    assert!(result.contains("## Tools"));
    assert!(result.contains("get_datetime"));
    assert!(result.contains("get_weather"));
    assert!(result.contains("<｜User｜>What's the weather tomorrow"));
    assert!(result.contains("<｜Assistant｜><think>"));
    assert!(result.contains("User is asking about tomorrow's weather"));
    assert!(result.contains("</think>"));
    assert!(result.contains("<｜DSML｜function_calls>"));
    assert!(result.contains("<｜DSML｜invoke name=\"get_datetime\">"));
    assert!(result.contains(
        "<｜DSML｜parameter name=\"timezone\" string=\"true\">America/New_York</｜DSML｜parameter>"
    ));
    assert!(result.contains("</｜DSML｜function_calls>"));
    assert!(result.contains("<function_results>"));
    assert!(result.contains("<result>"));
    assert!(result.contains("</function_results>"));
    assert!(result.contains("San Francisco"));
    assert!(result.contains("New York"));
    assert!(result.contains("<｜end▁of▁sentence｜>"));
}

#[test]
fn test_with_reasoning_content() {
    let messages = serde_json::json!([
        {"role": "user", "content": "Calculate 15 * 23"},
        {
            "role": "assistant",
            "content": "The answer is 345.",
            "reasoning_content": "Let me compute this step by step: 15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345"
        }
    ]);

    let result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
        .expect("Failed to encode");

    // Should contain thinking tags with reasoning
    assert!(result.contains("<think>"));
    assert!(result.contains("</think>"));
    assert!(result.contains("Let me compute this step by step"));
}

#[test]
fn test_tool_call_formatting() {
    let messages = serde_json::json!([
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Beijing\", \"unit\": \"celsius\"}"
                }
            }]
        }
    ]);

    let result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
        .expect("Failed to encode");

    // Check DSML format
    assert!(result.contains("<｜DSML｜function_calls>"));
    assert!(result.contains("<｜DSML｜invoke name=\"get_weather\">"));
    assert!(result.contains(
        "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>"
    ));
    assert!(
        result.contains(
            "<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>"
        )
    );
    assert!(result.contains("</｜DSML｜invoke>"));
    assert!(result.contains("</｜DSML｜function_calls>"));
}

#[test]
fn test_tool_results() {
    let messages = serde_json::json!([
        {"role": "user", "content": "Check weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Beijing\"}"
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "{\"temperature\": \"20C\", \"condition\": \"sunny\"}"
        }
    ]);

    let result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
        .expect("Failed to encode");

    // Check function_results wrapper
    assert!(result.contains("<function_results>"));
    assert!(result.contains("<result>"));
    assert!(result.contains("{\"temperature\": \"20C\", \"condition\": \"sunny\"}"));
    assert!(result.contains("</result>"));
    assert!(result.contains("</function_results>"));
}

#[test]
fn test_multiple_tool_calls() {
    let messages = serde_json::json!([
        {"role": "user", "content": "Weather in Beijing and Shanghai"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"Beijing\"}"
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"Shanghai\"}"
                    }
                }
            ]
        }
    ]);

    let result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
        .expect("Failed to encode");

    // Should contain both tool calls
    assert!(result.contains("Beijing"));
    assert!(result.contains("Shanghai"));
    // Should be in same function_calls block
    assert_eq!(result.matches("<｜DSML｜function_calls>").count(), 1);
    assert_eq!(result.matches("</｜DSML｜function_calls>").count(), 1);
    // But two invocations
    assert_eq!(result.matches("<｜DSML｜invoke").count(), 2);
}

#[test]
fn test_chat_mode_vs_thinking_mode() {
    let messages = serde_json::json!([
        {"role": "user", "content": "Hello"}
    ]);

    let chat_result = encode_messages(messages.as_array().unwrap(), ThinkingMode::Chat, true)
        .expect("Failed to encode");

    let thinking_result =
        encode_messages(messages.as_array().unwrap(), ThinkingMode::Thinking, true)
            .expect("Failed to encode");

    // Chat mode should have </think>, thinking mode should have <think>
    assert!(chat_result.contains("</think>"));
    assert!(!chat_result.contains("<think>"));

    assert!(thinking_result.contains("<think>"));
}
