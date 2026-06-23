// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::llm_backend::{
    BackendOutput, FinishReason as BackendFinishReason,
};
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::request_trace::SharedFinishReasonMetadata;
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
};
use dynamo_runtime::engine::{AsyncEngineContext, AsyncEngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use futures::stream;

const STREAM_CHUNKS: usize = 256;
const TOOL_CALL_SIZES: &[usize] = &[1, 8, 64, 512, 4096];

#[derive(Debug)]
struct MockContext {
    id: String,
    stopped: AtomicBool,
    killed: AtomicBool,
}

impl MockContext {
    fn new() -> Self {
        Self {
            id: "request-trace-finish-metadata-bench".to_string(),
            stopped: AtomicBool::new(false),
            killed: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl AsyncEngineContext for MockContext {
    fn id(&self) -> &str {
        &self.id
    }

    fn stop_generating(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    fn is_killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst)
    }

    async fn stopped(&self) {}

    async fn killed(&self) {}

    fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn kill(&self) {
        self.killed.store(true, Ordering::SeqCst);
    }

    fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {}
}

fn chat_request() -> NvCreateChatCompletionRequest {
    NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            model: "bench-model".to_string(),
            messages: vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                    name: None,
                },
            )],
            stream: Some(true),
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        media_io_kwargs: None,
        return_tokens_as_token_ids: None,
        unsupported_fields: Default::default(),
    }
}

fn backend_outputs(count: usize) -> Vec<BackendOutput> {
    (0..count)
        .map(|index| BackendOutput {
            token_ids: vec![index as u32],
            tokens: vec![Some("x".to_string())],
            text: Some("x".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: (index + 1 == count).then_some(BackendFinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            worker_trace_link: None,
            engine_data: None,
            routing_data: None,
        })
        .collect()
}

fn backend_stream(
    ctx: Arc<dyn AsyncEngineContext>,
    count: usize,
) -> Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>> {
    let stream = stream::iter(backend_outputs(count).into_iter().map(Annotated::from_data));
    ResponseStream::new(Box::pin(stream), ctx)
}

async fn consume_postprocessor_stream(trace_finish_metadata: bool) -> usize {
    let request = chat_request();
    let generator = Box::new(request.response_generator("bench-request".to_string()));
    let ctx = Arc::new(MockContext::new());
    let stream = OpenAIPreprocessor::transform_postprocessor_stream(
        backend_stream(ctx.clone(), STREAM_CHUNKS),
        generator,
        ctx,
        false,
        false,
        trace_finish_metadata.then(SharedFinishReasonMetadata::default),
    );

    stream.collect::<Vec<_>>().await.len()
}

fn tool_call_chunks(count: usize) -> Vec<(u32, Option<String>, Option<String>)> {
    (0..count)
        .map(|index| {
            (
                index as u32,
                Some(format!("call-{index}")),
                Some("lookup".to_string()),
            )
        })
        .collect()
}

fn argument_only_chunks(count: usize) -> Vec<(u32, Option<String>, Option<String>)> {
    (0..count)
        .map(|index| {
            (
                0,
                (index == 0).then(|| "call-0".to_string()),
                (index == 0).then(|| "lookup".to_string()),
            )
        })
        .collect()
}

fn bench_postprocessor_finish_metadata(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime should build");
    let mut group = c.benchmark_group("request_trace_finish_metadata/postprocessor");
    group.throughput(Throughput::Elements(STREAM_CHUNKS as u64));

    group.bench_function("finish_metadata_off", |b| {
        b.iter(|| {
            black_box(runtime.block_on(consume_postprocessor_stream(false)));
        });
    });

    group.bench_function("finish_metadata_on", |b| {
        b.iter(|| {
            black_box(runtime.block_on(consume_postprocessor_stream(true)));
        });
    });

    group.finish();
}

fn bench_tool_call_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_trace_finish_metadata/tool_calls");

    for &count in TOOL_CALL_SIZES {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("record_unique", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || tool_call_chunks(count),
                    |chunks| {
                        let metadata = SharedFinishReasonMetadata::default();
                        for (index, id, name) in chunks {
                            metadata.record_tool_call_chunk_for_bench(
                                0,
                                index,
                                id.as_deref(),
                                name.as_deref(),
                            );
                        }
                        black_box(metadata.snapshot_for_bench().unwrap().tool_calls.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("skip_argument_only_chunks", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || argument_only_chunks(count),
                    |chunks| {
                        let metadata = SharedFinishReasonMetadata::default();
                        for (index, id, name) in chunks {
                            metadata.record_tool_call_chunk_for_bench(
                                0,
                                index,
                                id.as_deref(),
                                name.as_deref(),
                            );
                        }
                        black_box(metadata.snapshot_for_bench().unwrap().tool_calls.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_postprocessor_finish_metadata,
    bench_tool_call_metadata
);
criterion_main!(benches);
