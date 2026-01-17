// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests request with the token_data in nvext used for the EPP-aware Gateway integration. When the backend_instance_id is in the request along with the token_data the tokenization will be skipped in the preprocesor.rs

use anyhow::Result;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

#[test]
fn test_request_json_structure() -> Result<()> {
    // Test that the JSON structure matches what clients will send

    let json_input = r#"{
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hello"}],
        "nvext": {
            "backend_instance_id": 12345,
            "token_data": [15496, 1917, 264]
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_input)?;

    // Verify parsing
    assert!(request.nvext.is_some());
    let nvext = request.nvext.as_ref().unwrap();
    assert_eq!(nvext.backend_instance_id, Some(12345));
    assert_eq!(nvext.token_data, Some(vec![15496, 1917, 264]));

    Ok(())
}
