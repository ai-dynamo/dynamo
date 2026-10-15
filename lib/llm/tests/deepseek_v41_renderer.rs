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
fn normalized_effort_matches_the_reference_encoder_for_both_request_fields() {
    assert!(render(json!({})).unwrap().contains("Reasoning Effort: 75 "));
    assert!(render(json!({"reasoning_effort":"xhigh"})).is_err());
    for field in ["chat_template_args", "chat_template_kwargs"] {
        for (effort, budget) in [("low", 50), ("high", 75), ("max", 100)] {
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
                .contains("Reasoning Effort: 50 ")
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
        for invalid in [
            json!(0),
            json!(101),
            json!(true),
            json!(1.5),
            json!("xhigh"),
        ] {
            assert!(render(json!({(field):{"reasoning_effort":invalid}})).is_err());
        }
    }
}

#[test]
fn image_blocks_preserve_text_order_and_reasoning_mode() {
    for effort in ["high", "none"] {
        let output = render(json!({
            "reasoning_effort": effort,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "first:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/first.png"}},
                {"type": "text", "text": "second:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "compare"}
            ]}]
        }))
        .unwrap();
        assert!(output.contains(
            "first:\n\n<｜deepseek_image｜>\n\nsecond:\n\n<｜deepseek_image｜>\n\ncompare"
        ));
        assert_eq!(output.matches("<｜deepseek_image｜>").count(), 2);
        assert!(!output.contains("https://example.com/first.png"));
        assert!(!output.contains("data:image/png;base64,AAAA"));
        assert!(output.ends_with(if effort == "none" {
            "</think>"
        } else {
            "<think>"
        }));
    }
}

#[test]
fn cached_images_render_in_user_and_tool_messages() {
    for role in ["user", "tool"] {
        let output = render(json!({
            "reasoning_effort": "none",
            "messages": [{"role": role, "tool_call_id": "screenshot", "content": [
                {"type": "text", "text": "cached:"},
                {"type": "image_url", "image_url": null, "uuid": "cached-screenshot"}
            ]}]
        }))
        .unwrap();
        assert!(output.contains("cached:\n\n<｜deepseek_image｜>"));
        assert_eq!(output.matches("<｜deepseek_image｜>").count(), 1);
        assert!(!output.contains("cached-screenshot"));
    }
}
