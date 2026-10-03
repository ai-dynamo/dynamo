// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of the once-per-stream end-of-stream flush envelope clone, measured
//! both in isolation and as a fraction of a whole reasoning stream.
//!
//! `cargo bench -p dynamo-llm --features reasoning-flush-bench --bench reasoning_flush_envelope`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason, Role,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use futures::stream;

/// Chunk counts spanning a short answer through a long one. gpt-oss-class
/// reasoning turns routinely run into the hundreds of tokens.
const STREAM_CHUNKS: &[usize] = &[16, 64, 256, 1024];

/// A chunk shaped like a real first content-bearing response: realistic `id` and
/// `model` lengths, one choice, one token of text. This is what retention
/// clones, so understating its size would understate the cost.
fn content_chunk(
    text: &str,
    role: Option<Role>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role,
            content: Some(ChatCompletionMessageContent::Text(text.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "chatcmpl-9f2c1b7a4e6d4f8fae1b0c3d5e7f9a1b".to_string(),
            choices: vec![choice],
            created: 1_726_500_000,
            model: "openai/gpt-oss-120b".to_string(),
            system_fingerprint: Some("fp_44709d6fcb".to_string()),
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    })
}

fn reasoning_stream_chunks(chunks: usize) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let mut out = Vec::with_capacity(chunks + 2);
    out.push(content_chunk("<think>", Some(Role::Assistant)));
    let half = chunks / 2;
    for i in 0..half {
        out.push(content_chunk(
            if i % 7 == 0 { " step" } else { " tok" },
            None,
        ));
    }
    out.push(content_chunk("</think>", None));
    for i in 0..(chunks - half) {
        out.push(content_chunk(
            if i % 5 == 0 { " word" } else { " ans" },
            None,
        ));
    }
    let mut terminal = content_chunk("", None);
    terminal.data.as_mut().unwrap().inner.choices[0].finish_reason = Some(FinishReason::Stop);
    out.push(terminal);
    out
}

fn bench_envelope_clone(c: &mut Criterion) {
    let chunk = content_chunk("Hello", Some(Role::Assistant));
    c.bench_function("envelope_clone", |b| {
        b.iter(|| black_box(black_box(&chunk).clone()));
    });
}

fn bench_reasoning_stream(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime");
    let mut group = c.benchmark_group("reasoning_stream");
    for &chunks in STREAM_CHUNKS {
        let source = reasoning_stream_chunks(chunks);
        group.bench_with_input(BenchmarkId::from_parameter(chunks), &chunks, |b, _| {
            b.iter(|| {
                runtime.block_on(async {
                    let out = OpenAIPreprocessor::parse_reasoning_content_from_stream(
                        stream::iter(source.clone()),
                        "deepseek_r1".to_string(),
                        false,
                    )
                    .collect::<Vec<_>>()
                    .await;
                    black_box(out.len())
                })
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_envelope_clone, bench_reasoning_stream);
criterion_main!(benches);
