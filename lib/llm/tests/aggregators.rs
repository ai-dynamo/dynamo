// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use dynamo_llm::protocols::{
    codec::{create_message_stream, Message, SseCodecError},
    openai::{
        chat_completions::{aggregator::ChatCompletionAggregator, NvCreateChatCompletionResponse},
        completions::NvCreateCompletionResponse,
        ParsingOptions,
    },
    ContentProvider, DataStream,
};
use futures::StreamExt;

const CMPL_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/completions";
const CHAT_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions";

fn create_stream(root_path: &str, file_name: &str) -> DataStream<Result<Message, SseCodecError>> {
    let data = std::fs::read_to_string(format!("{}/{}", root_path, file_name)).unwrap();
    create_message_stream(&data)
}

#[tokio::test]
async fn test_openai_chat_stream() {
    let data = std::fs::read_to_string("tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions/chat-completion.streaming.1").unwrap();

    // note: we are only taking the first 16 messages to keep the size of the response small
    let stream = create_message_stream(&data).take(16);
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    // todo: provide a cleaner way to extract the content from choices
    assert_eq!(
        result
            .choices
            .first()
            .unwrap()
            .message
            .content
            .clone()
            .expect("there to be content"),
        "Deep learning is a subfield of machine learning that involves the use of artificial"
            .to_string()
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_multi_line_data() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-multi-line-data");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        result
            .choices
            .first()
            .unwrap()
            .message
            .content
            .clone()
            .expect("there to be content"),
        "Deep learning".to_string()
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_comments_per_response() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-comments_per_response");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        result
            .choices
            .first()
            .unwrap()
            .message
            .content
            .clone()
            .expect("there to be content"),
        "Deep learning".to_string()
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_invalid_deserialize_error() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/invalid-deserialize_error");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await;

    assert!(result.is_err());
    // insta::assert_debug_snapshot!(result);
}

// =============================
// Completions (/v1/completions)
// =============================

#[tokio::test]
async fn test_openai_cmpl_stream() {
    let stream = create_stream(CMPL_ROOT_PATH, "completion.streaming.1").take(16);
    let result =
        NvCreateCompletionResponse::from_sse_stream(Box::pin(stream), ParsingOptions::default())
            .await
            .unwrap();

    // todo: provide a cleaner way to extract the content from choices
    assert_eq!(
        result.inner.choices.first().unwrap().content(),
        " This is a question that is often asked by those outside of AI research and development"
    );
}
