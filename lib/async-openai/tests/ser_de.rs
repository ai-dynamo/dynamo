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
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
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
