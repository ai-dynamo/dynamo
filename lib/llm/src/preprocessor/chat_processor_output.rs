// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building the response for a chat processor that parses the engine's output itself.
//!
//! Such a processor detokenizes and runs its own tool-call and reasoning parsers, so it hands
//! Dynamo message deltas instead of text. The rest of a response chunk (its id, usage, finish
//! reason, metrics and `nvext`) is built here, by the code Dynamo's own preprocessor uses.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatCompletionStreamResponseDelta, CompletionUsage,
    FinishReason as ChoiceFinishReason, StopReason,
};
use dynamo_runtime::pipeline::AsyncEngineContext;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};
use serde::Deserialize;

use super::request_envelope::RequestScope;
use super::{MultimodalCounts, OpenAIPreprocessor};
use crate::protocols::TokenIdType;
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::BackendOutput;
use crate::protocols::openai::chat_completions::{
    DeltaGenerator, NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};

type Chunk = NvCreateChatCompletionStreamResponse;

/// One step of a chat processor's output: what it parsed from the engine tokens it consumed.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ParsedOutput {
    /// The choice this step belongs to.
    #[serde(default)]
    pub index: Option<u32>,
    /// The content, reasoning and tool calls parsed in this step. The role is set here.
    #[serde(default)]
    pub delta: Option<ChatCompletionStreamResponseDelta>,
    /// Log probabilities of the tokens behind `delta`, with the processor's token text.
    #[serde(default)]
    pub logprobs: Option<ChatChoiceLogprobs>,
    /// The generated tokens this step accounts for. A token is reported once, and not at all
    /// when it follows a stop the processor matched.
    #[serde(default)]
    pub token_ids: Vec<TokenIdType>,
    /// Why the choice ended: the engine's reason, or `stop` for a stop the processor matched.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    /// The stop string or token that ended the choice.
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// The engine's usage report, as it sent it.
    #[serde(default)]
    pub completion_usage: Option<CompletionUsage>,
    /// The engine's `engine_data`, as it sent it.
    #[serde(default)]
    pub engine_data: Option<Box<serde_json::Value>>,
    /// Reasoning tokens this choice has generated so far, when the processor's parser counts
    /// them. Used where the engine reports none.
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

/// What a chat processor yields.
#[derive(Debug, Clone)]
pub enum ChatProcessorItem {
    /// A finished response chunk, sent on as it is.
    Chunk(Box<Chunk>),
    /// A parsed step, for Dynamo to build the chunk from.
    Parsed(Box<ParsedOutput>),
}

/// The parts of a parsed step that the chunk built from its [`BackendOutput`] does not carry.
struct MessageParts {
    delta: Option<ChatCompletionStreamResponseDelta>,
    logprobs: Option<ChatChoiceLogprobs>,
    reasoning_tokens: Option<u32>,
}

impl ParsedOutput {
    fn split(self) -> (BackendOutput, MessageParts) {
        let output = BackendOutput {
            token_ids: self.token_ids,
            tokens: Vec::new(),
            text: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: self.finish_reason,
            stop_reason: self.stop_reason,
            index: self.index,
            completion_usage: self.completion_usage,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            engine_data: self.engine_data,
            routing_data: None,
            jailed_text: None,
        };
        let parts = MessageParts {
            delta: self.delta,
            logprobs: self.logprobs,
            reasoning_tokens: self.reasoning_tokens,
        };
        (output, parts)
    }
}

/// Puts each step's message parts on the chunk built for it.
#[derive(Default)]
struct MessageWriter {
    tool_call_choices: HashSet<u32>,
    reasoning_tokens: HashMap<u32, u32>,
}

impl MessageWriter {
    fn write(&mut self, chunk: &mut Chunk, parts: Option<MessageParts>) {
        if let Some(parts) = parts
            && let Some(choice) = chunk.inner.choices.first_mut()
        {
            if let Some(delta) = parts.delta {
                choice.delta.content = delta.content;
                choice.delta.reasoning_content = delta.reasoning_content;
                choice.delta.refusal = delta.refusal;
                if delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                {
                    self.tool_call_choices.insert(choice.index);
                    choice.delta.tool_calls = delta.tool_calls;
                }
            }
            choice.logprobs = parts.logprobs;
            // A choice that called tools and then stopped ended because of the calls.
            if choice.finish_reason == Some(ChoiceFinishReason::Stop)
                && self.tool_call_choices.contains(&choice.index)
            {
                choice.finish_reason = Some(ChoiceFinishReason::ToolCalls);
            }
            if let Some(reasoning_tokens) = parts.reasoning_tokens {
                self.reasoning_tokens.insert(choice.index, reasoning_tokens);
            }
        }

        // A processor that counts reasoning tokens reports zero for a reply without any.
        if !self.reasoning_tokens.is_empty()
            && let Some(usage) = chunk.inner.usage.as_mut()
        {
            let details = usage.completion_tokens_details.get_or_insert_default();
            if details.reasoning_tokens.unwrap_or(0) == 0 {
                details.reasoning_tokens = Some(self.reasoning_tokens.values().sum());
            }
        }
    }
}

/// What the response to one chat request is built from, when a chat processor serves it.
pub struct ChatResponseStage {
    generator: DeltaGenerator,
    scope: RequestScope,
    mm_counts: MultimodalCounts,
}

impl ChatResponseStage {
    /// Start the response to `request`. The request `scope` routes with shares this
    /// response's tracker, so the worker and timing the router records reach the response.
    pub fn new(
        request: &NvCreateChatCompletionRequest,
        request_id: String,
        scope: &mut RequestScope,
    ) -> Self {
        let generator = request.response_generator_for_client_request(request_id);
        scope.share_tracker(generator.tracker());
        Self {
            generator,
            scope: scope.clone(),
            mm_counts: MultimodalCounts::from_messages(&request.inner.messages),
        }
    }

    /// The response chunks for what a chat processor yields: its finished chunks as they are,
    /// and a chunk built for each parsed step from the first one on.
    pub fn stream<S>(
        self,
        items: S,
        context: Arc<dyn AsyncEngineContext>,
    ) -> impl Stream<Item = Annotated<Chunk>> + Send
    where
        S: Stream<Item = Annotated<ChatProcessorItem>> + Send + Unpin + 'static,
    {
        async_stream::stream! {
            let mut items = items;
            while let Some(item) = items.next().await {
                match item.data {
                    Some(ChatProcessorItem::Parsed(_)) => {
                        let steps = futures::stream::once(std::future::ready(item)).chain(items);
                        let chunks = self.build(steps, context);
                        futures::pin_mut!(chunks);
                        while let Some(chunk) = chunks.next().await {
                            yield chunk;
                        }
                        return;
                    }
                    Some(ChatProcessorItem::Chunk(_)) | None => {
                        yield item.map_data(|data| match data {
                            ChatProcessorItem::Chunk(chunk) => Ok(*chunk),
                            ChatProcessorItem::Parsed(_) => unreachable!("matched above"),
                        });
                    }
                }
            }
        }
    }

    fn build<S>(
        mut self,
        steps: S,
        context: Arc<dyn AsyncEngineContext>,
    ) -> impl Stream<Item = Annotated<Chunk>> + Send
    where
        S: Stream<Item = Annotated<ChatProcessorItem>> + Send + 'static,
    {
        // The processor has routed its request by the time it yields a step.
        if let Some(prompt_tokens) = self.scope.prompt_tokens() {
            self.generator.update_isl(prompt_tokens);
        }

        // The chunk for a step is built in the poll that reads the step, so this holds at
        // most one entry: the parts of the step whose chunk is being built.
        let pending = Arc::new(Mutex::new(VecDeque::<Option<MessageParts>>::new()));
        let outputs = {
            let pending = pending.clone();
            steps.map(move |step| {
                let mut parts = None;
                let output = step.map_data(|data| match data {
                    ChatProcessorItem::Parsed(parsed) => {
                        let (output, message) = parsed.split();
                        parts = Some(message);
                        Ok(output)
                    }
                    ChatProcessorItem::Chunk(_) => Err(
                        "chat processor yielded a finished chunk after a parsed step".to_string(),
                    ),
                });
                pending.lock().unwrap().push_back(parts);
                output
            })
        };

        let mut writer = MessageWriter::default();
        OpenAIPreprocessor::transform_postprocessor_stream(
            outputs,
            Box::new(self.generator),
            context,
            false,
            false,
            None,
            self.mm_counts,
        )
        .map(move |mut chunk| {
            let parts = pending.lock().unwrap().pop_front().flatten();
            if let Some(data) = chunk.data.as_mut() {
                writer.write(data, parts);
            }
            chunk
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;
    use crate::preprocessor::request_envelope::RequestEnvelope;
    use crate::protocols::common::preprocessor::PreprocessedRequest;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};

    fn step(value: serde_json::Value) -> Annotated<ChatProcessorItem> {
        Annotated::from_data(ChatProcessorItem::Parsed(Box::new(
            serde_json::from_value(value).unwrap(),
        )))
    }

    async fn respond(
        request: serde_json::Value,
        items: Vec<Annotated<ChatProcessorItem>>,
    ) -> Vec<serde_json::Value> {
        let mut body = request;
        body["model"] = "m".into();
        body["messages"] = serde_json::json!([{"role": "user", "content": "hi"}]);
        let request: Context<NvCreateChatCompletionRequest> =
            Context::new(serde_json::from_value(body).unwrap());
        let envelope = RequestEnvelope::from_card(&ModelDeploymentCard::with_name_only("m"));
        let mut scope = envelope.begin_chat(&request);
        let stage = ChatResponseStage::new(&request, "req-1".to_string(), &mut scope);

        // The processor routes a three-token prompt before it yields anything.
        let mut routed = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap();
        scope.complete_request(&mut routed);
        scope.record_prompt(&routed);

        let chunks = stage.stream(futures::stream::iter(items), request.context());
        futures::pin_mut!(chunks);
        let mut out = Vec::new();
        while let Some(chunk) = chunks.next().await {
            assert!(chunk.error.is_none(), "{:?}", chunk.error);
            if let Some(data) = chunk.data {
                out.push(serde_json::to_value(&data).unwrap());
            }
        }
        out
    }

    /// Parsed steps become chunks with the envelope Dynamo's preprocessor builds: one id,
    /// the role once, tool calls ending the choice, and a trailing usage chunk.
    #[tokio::test]
    async fn parsed_steps_get_the_shared_envelope() {
        let chunks = respond(
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true}}),
            vec![
                step(serde_json::json!({
                    "delta": {"reasoning_content": "think"},
                    "token_ids": [10, 11],
                    "reasoning_tokens": 2,
                })),
                // Tokens held back by the processor's parser still count.
                step(serde_json::json!({"token_ids": [12]})),
                step(serde_json::json!({
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }]},
                    "token_ids": [13],
                    "finish_reason": "stop",
                })),
            ],
        )
        .await;

        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|chunk| chunk["id"] == "chatcmpl-req-1"));
        let first = &chunks[0]["choices"][0]["delta"];
        assert_eq!(first["role"], "assistant");
        assert_eq!(first["reasoning_content"], "think");
        assert!(chunks[1]["choices"][0]["delta"].get("role").is_none());
        let last = &chunks[2]["choices"][0];
        assert_eq!(last["delta"]["tool_calls"][0]["function"]["name"], "search");
        assert_eq!(last["finish_reason"], "tool_calls");

        let usage = &chunks[3]["usage"];
        assert_eq!(chunks[3]["choices"], serde_json::json!([]));
        assert_eq!(usage["prompt_tokens"], 3);
        assert_eq!(usage["completion_tokens"], 4);
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 2);
    }

    /// A processor that builds its own chunks is passed through untouched.
    #[tokio::test]
    async fn finished_chunks_pass_through() {
        let chunk: Chunk = serde_json::from_value(serde_json::json!({
            "id": "from-processor",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        }))
        .unwrap();
        let chunks = respond(
            serde_json::json!({"stream": true}),
            vec![Annotated::from_data(ChatProcessorItem::Chunk(Box::new(
                chunk,
            )))],
        )
        .await;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0]["id"], "from-processor");
    }
}
