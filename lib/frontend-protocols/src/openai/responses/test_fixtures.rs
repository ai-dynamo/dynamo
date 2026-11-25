// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixtures the Responses tests share with the preprocessor's tests.

use dynamo_renderer::{ChatTemplate, ContextMixins, OAIPromptFormatter, PromptFormatter};
use serde_json::{Value, json};

use super::NvCreateResponse;

pub const TEMPLATE: &str =
    include_str!("../../../../llm/tests/data/muse_spark_open_deferred_tools.jinja");

pub fn formatter(template: &str) -> std::sync::Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({"chat_template":template})).unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), true).unwrap();
    formatter
}

pub fn request(tools: Value, input: Value) -> NvCreateResponse {
    serde_json::from_value(json!({"model":"/model", "input":input, "tools":tools,
        "chat_template_args":{"current_date":"2026-10-06"}}))
    .unwrap()
}

pub fn namespace(defer: bool) -> Value {
    json!([{"type":"namespace", "name":"fs", "description":"Files", "defer_loading":defer,
        "tools":[{"type":"function", "name":"read", "description":"Read files", "parameters":{"type":"object", "properties":{"path":{"type":"string"}},"required":["path"]}}]}])
}
