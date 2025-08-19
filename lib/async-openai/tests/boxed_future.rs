// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use futures::future::{BoxFuture, FutureExt};
use futures::StreamExt;

use dynamo_async_openai::types::{CompletionResponseStream, CreateCompletionRequestArgs};
use dynamo_async_openai::Client;

#[tokio::test]
async fn boxed_future_test() {
    fn interpret_bool(token_stream: &mut CompletionResponseStream) -> BoxFuture<'_, bool> {
        async move {
            while let Some(response) = token_stream.next().await {
                match response {
                    Ok(response) => {
                        let token_str = &response.choices[0].text.trim();
                        if !token_str.is_empty() {
                            return token_str.contains("yes") || token_str.contains("Yes");
                        }
                    }
                    Err(e) => eprintln!("Error: {e}"),
                }
            }
            false
        }
        .boxed()
    }

    let client = Client::new();

    let request = CreateCompletionRequestArgs::default()
        .model("gpt-3.5-turbo-instruct")
        .n(1)
        .prompt("does 2 and 2 add to four? (yes/no):\n")
        .stream(true)
        .logprobs(3)
        .max_tokens(64_u32)
        .build()
        .unwrap();

    let mut stream = client.completions().create_stream(request).await.unwrap();

    let result = interpret_bool(&mut stream).await;
    assert!(result);
}
