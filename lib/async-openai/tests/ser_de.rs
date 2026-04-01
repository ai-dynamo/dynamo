// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use dynamo_async_openai::types::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, FunctionCall,
    FunctionCallStream,
};

#[tokio::test]
async fn chat_types_serde() {
    let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("your are a calculator")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("what is the result of 1+1")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    // serialize the request
    let serialized = serde_json::to_string(&request).unwrap();
    // deserialize the request
    let deserialized: CreateChatCompletionRequest = serde_json::from_str(&serialized).unwrap();
    assert_eq!(request, deserialized);
}

// ---------------------------------------------------------------------------
// Tool-call arguments: string vs dict normalisation
// ---------------------------------------------------------------------------

/// String-encoded arguments round-trip unchanged.
#[test]
fn tool_call_arguments_string_passthrough() {
    let json = r#"{"name":"get_weather","arguments":"{\"location\":\"SF\",\"unit\":\"celsius\"}"}"#;
    let fc: FunctionCall = serde_json::from_str(json).unwrap();
    assert_eq!(fc.name, "get_weather");
    assert_eq!(fc.arguments, r#"{"location":"SF","unit":"celsius"}"#);
    // Re-serialise — arguments must remain a JSON string.
    let re = serde_json::to_string(&fc).unwrap();
    let fc2: FunctionCall = serde_json::from_str(&re).unwrap();
    assert_eq!(fc, fc2);
}

/// Dict-encoded arguments are normalised to a JSON string.
#[test]
fn tool_call_arguments_dict_normalised_to_string() {
    let json = r#"{"name":"get_weather","arguments":{"location":"SF","unit":"celsius"}}"#;
    let fc: FunctionCall = serde_json::from_str(json).unwrap();
    assert_eq!(fc.name, "get_weather");
    // After normalisation the value must be a valid JSON string.
    let parsed: serde_json::Value = serde_json::from_str(&fc.arguments).unwrap();
    assert_eq!(parsed["location"], "SF");
    assert_eq!(parsed["unit"], "celsius");
}

/// Dict with nested objects is normalised correctly.
#[test]
fn tool_call_arguments_nested_dict() {
    let json = r#"{"name":"create_event","arguments":{"title":"standup","time":{"hour":9,"minute":0}}}"#;
    let fc: FunctionCall = serde_json::from_str(json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&fc.arguments).unwrap();
    assert_eq!(parsed["time"]["hour"], 9);
}

/// Streaming variant: dict arguments are normalised.
#[test]
fn function_call_stream_dict_arguments() {
    let json = r#"{"name":"search","arguments":{"query":"rust serde","limit":5}}"#;
    let fcs: FunctionCallStream = serde_json::from_str(json).unwrap();
    let args = fcs.arguments.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&args).unwrap();
    assert_eq!(parsed["query"], "rust serde");
    assert_eq!(parsed["limit"], 5);
}

/// Streaming variant: absent arguments deserialise to None.
#[test]
fn function_call_stream_missing_arguments() {
    let json = r#"{"name":"ping"}"#;
    let fcs: FunctionCallStream = serde_json::from_str(json).unwrap();
    assert_eq!(fcs.name.as_deref(), Some("ping"));
    assert!(fcs.arguments.is_none());
}
