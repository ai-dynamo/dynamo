// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
use serde_json::{Value, json};

fn render(fields: Value) -> anyhow::Result<String> {
    let mut payload =
        json!({"model":"deepseek-v4.1", "messages":[{"role":"user", "content":"Hello"}]});
    payload
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let mut request: NvCreateChatCompletionRequest = serde_json::from_value(payload)?;
    request.normalize_reasoning_template_args()?;
    DeepSeekV41Formatter.render(&request)
}

#[test]
fn normalized_effort_matches_vllm_for_both_request_fields() {
    for field in ["chat_template_args", "chat_template_kwargs"] {
        for (effort, budget) in [("low", 25), ("high", 50), ("xhigh", 75), ("max", 100)] {
            for fields in [
                json!({"reasoning_effort":effort}),
                json!({(field):{"reasoning_effort":effort}}),
            ] {
                let output = render(fields).unwrap();
                assert!(output.contains(&format!("Reasoning Effort: {budget} ")));
                assert!(output.ends_with("<think>"));
            }
        }
        assert!(
            render(json!({(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 37 ")
        );
        assert!(
            render(json!({"reasoning_effort":"low",(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 25 ")
        );
        for fields in [
            json!({"reasoning_effort":"none"}),
            json!({(field):{"reasoning_effort":"none"}}),
            json!({(field):{"thinking":false}}),
        ] {
            let output = render(fields).unwrap();
            assert!(!output.contains("Reasoning Effort:"));
            assert!(output.ends_with("</think>"));
        }
        for invalid in [json!(0), json!(101), json!(true), json!(1.5)] {
            assert!(render(json!({(field):{"reasoning_effort":invalid}})).is_err());
        }
    }
}
