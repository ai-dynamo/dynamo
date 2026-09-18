// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::ChatCompletionRequestMessage as Message;
use std::collections::HashMap;

/// Match the V4.1 encoder's tool-result sorting before collecting image slots.
/// Public user/tool messages merge into one user turn. Only tool-result slots
/// are sorted; interleaved user images keep their positions. Return indices to
/// avoid copying image payloads or changing the caller's conversation.
pub(super) fn media_message_order(messages: &[Message]) -> Vec<usize> {
    let mut order: Vec<_> = (0..messages.len()).collect();
    let mut calls = HashMap::new();
    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            Message::Assistant(assistant) => {
                if let Some(tool_calls) = &assistant.tool_calls {
                    calls.clear();
                    for (rank, call) in tool_calls.iter().enumerate() {
                        if !call.id.is_empty() {
                            calls.insert(call.id.as_str(), rank);
                        }
                    }
                }
            }
            Message::User(_) | Message::Tool(_) => {
                let start = index;
                while index < messages.len()
                    && matches!(&messages[index], Message::User(_) | Message::Tool(_))
                {
                    index += 1;
                }
                if !calls.is_empty() {
                    let slots: Vec<_> = (start..index)
                        .filter(|&i| matches!(&messages[i], Message::Tool(_)))
                        .collect();
                    let mut sorted = slots.clone();
                    sorted.sort_by_key(|&i| match &messages[i] {
                        Message::Tool(tool) => *calls.get(tool.tool_call_id.as_str()).unwrap_or(&0),
                        _ => unreachable!(),
                    });
                    for (slot, source) in slots.into_iter().zip(sorted) {
                        order[slot] = source;
                    }
                }
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn assistant(ids: &[&str]) -> Value {
        json!({"role":"assistant","tool_calls":ids.iter().map(|id| json!({
            "id":id,"type":"function","function":{"name":"image","arguments":"{}"}
        })).collect::<Vec<_>>()})
    }

    fn tool(id: &str, parts: usize) -> Value {
        json!({"role":"tool","tool_call_id":id,"content":(0..parts).map(|n|json!({
            "type":"image_url","image_url":{"url":format!("https://example.com/{id}/{n}")},"uuid":format!("{id}-{n}")
        })).collect::<Vec<_>>()})
    }

    fn check(messages: Vec<Value>, expected: &[usize]) {
        let messages: Vec<Message> = serde_json::from_value(json!(messages)).unwrap();
        let original = serde_json::to_value(&messages).unwrap();
        assert_eq!(media_message_order(&messages), expected);
        assert_eq!(serde_json::to_value(&messages).unwrap(), original);
    }

    #[test]
    fn reversed_results_keep_each_image_group_and_uuid_together() {
        check(
            vec![assistant(&["a", "b"]), tool("b", 2), tool("a", 1)],
            &[0, 2, 1],
        );
        check(
            vec![assistant(&["a", "b"]), tool("a", 1), tool("b", 2)],
            &[0, 1, 2],
        );
    }

    #[test]
    fn user_images_stay_between_sorted_tool_result_slots() {
        let user = json!({"role":"user","content":[{"type":"image_url","uuid":"user-image"}]});
        check(
            vec![assistant(&["a", "b"]), tool("b", 2), user, tool("a", 1)],
            &[0, 3, 2, 1],
        );
    }

    #[test]
    fn separate_turns_use_their_own_call_order() {
        check(
            vec![
                assistant(&["a", "b"]),
                tool("b", 1),
                tool("a", 1),
                assistant(&["d", "c"]),
                tool("c", 1),
                tool("d", 1),
            ],
            &[0, 2, 1, 3, 5, 4],
        );
    }

    #[test]
    fn system_boundaries_do_not_merge_tool_result_groups() {
        check(
            vec![
                assistant(&["a", "b"]),
                tool("b", 1),
                json!({"role":"system","content":"New instruction"}),
                tool("a", 1),
            ],
            &[0, 1, 2, 3],
        );
    }

    #[test]
    fn missing_and_unknown_call_ids_keep_reference_stable_order() {
        check(vec![tool("b", 1), tool("a", 1)], &[0, 1]);
        check(
            vec![
                assistant(&["a", "b"]),
                tool("b", 1),
                tool("unknown", 1),
                tool("a", 1),
            ],
            &[0, 2, 3, 1],
        );
    }
    #[test]
    fn collection_order_matches_actual_renderer_for_all_tool_permutations() {
        use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
        for ids in [
            ["a", "b", "c"],
            ["a", "c", "b"],
            ["b", "a", "c"],
            ["b", "c", "a"],
            ["c", "a", "b"],
            ["c", "b", "a"],
        ] {
            let mut raw = vec![
                assistant(&["a", "b", "c"]),
                tool(ids[0], 2),
                json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/user"}}]}),
                tool(ids[1], 1),
                tool(ids[2], 2),
            ];
            let typed: Vec<Message> = serde_json::from_value(json!(raw)).unwrap();
            let mut slot = 0;
            let mut message_slots = vec![Vec::new(); raw.len()];
            for (i, message) in raw.iter_mut().enumerate() {
                if let Some(parts) = message["content"].as_array_mut() {
                    for part in parts {
                        if part["type"] == "image_url" {
                            let label = format!("IMAGE_SLOT[{slot}]");
                            message_slots[i].push(label.clone());
                            *part = json!({"type":"text","text":label});
                            slot += 1;
                        }
                    }
                }
            }
            // Label images as text so the real formatter exposes its slot order.
            let req: dynamo_protocols::types::CreateChatCompletionRequest = serde_json::from_value(
                json!({"model":"alias","messages":raw,"reasoning_effort":"none"}),
            )
            .unwrap();
            let prompt = DeepSeekV41Formatter.render(&req).unwrap();
            let actual: Vec<_> = media_message_order(&typed)
                .into_iter()
                .flat_map(|i| message_slots[i].iter())
                .map(|label| prompt.find(label).unwrap())
                .collect();
            assert!(
                actual.windows(2).all(|pair| pair[0] < pair[1]),
                "{ids:?}: {actual:?}"
            );
        }
    }
}
