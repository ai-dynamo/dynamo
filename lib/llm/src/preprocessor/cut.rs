// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The chat and completions preprocessors' closing output after a request is killed.
//!
//! A killed request can end its response stream cleanly, as if it had finished. The
//! preprocessor's response transform decides once, where its input ends, whether the request
//! was killed, and then writes no usage chunk. The response parsers above it (reasoning, tool calls, the jail of the published
//! `dynamo-parsers`) write on their own input's end and cannot be asked, so the decision reaches
//! them as a [`CutFlag`], and [`UntilCut`] ends their output where it is set.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};

use futures::Stream;

/// The payload capture's drop reason for a killed request.
pub(super) const DROP_CUT_SHORT: &str = "request_cut_short";

/// The decision that a request was killed, shared by the stage that makes it and the stages
/// that write on end. One allocation per request, made only where such a stage runs.
#[derive(Clone, Debug, Default)]
pub(super) struct CutFlag(Arc<AtomicBool>);

impl CutFlag {
    pub(super) fn set(&self) {
        // Relaxed: it is set and read inside one poll of one task, or read after the task
        // that set it has ended (the payload capture), which orders it.
        self.0.store(true, Ordering::Relaxed);
    }

    pub(super) fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// `S` until `cut` is set, then ended: what `S` writes after the stage below it saw a killed
/// request is closing output, and is dropped.
///
/// Per item it reads the flag once, a load. It runs only on a request with response parsers,
/// which cost far more per item.
pub(super) struct UntilCut<S> {
    stream: Option<S>,
    cut: CutFlag,
}

impl<S> UntilCut<S> {
    pub(super) fn new(stream: S, cut: CutFlag) -> Self {
        Self {
            stream: Some(stream),
            cut,
        }
    }
}

impl<S: Unpin> Unpin for UntilCut<S> {}

impl<S: Stream + Unpin> Stream for UntilCut<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let Some(stream) = self.stream.as_mut() else {
            return Poll::Ready(None);
        };
        match ready!(Pin::new(stream).poll_next(cx)) {
            Some(item) if !self.cut.is_set() => Poll::Ready(Some(item)),
            _ => {
                self.stream = None;
                Poll::Ready(None)
            }
        }
    }
}

/// What the response transform does when its input ends after the request was killed.
pub(super) enum OnKill {
    /// Nothing: the caller drives the transform with a stream of its own, and no request is
    /// behind it.
    Ignore,
    /// End with no usage chunk, and set the flag, when given, so the stages above that write on
    /// end stop too ([`UntilCut`]).
    End(Option<CutFlag>),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
    use dynamo_runtime::pipeline::{
        AsyncEngineContextProvider, Context, Error, ManyOut, Operator, SingleIn, async_trait,
    };
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::{StreamExt, stream};
    use serde::Serialize;

    use crate::model_card::ModelDeploymentCard;
    use crate::preprocessor::OpenAIPreprocessor;
    use crate::protocols::common::llm_backend::{BackendOutput, FinishReason, PreprocessedRequest};
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;

    const TINY_LLAMA: &str = "tests/data/sample-models/TinyLlama_v1.1";
    const TOOL_MODEL: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn output(text: &str, finish_reason: Option<FinishReason>) -> BackendOutput {
        BackendOutput {
            token_ids: vec![1],
            tokens: vec![Some(text.to_string())],
            text: Some(text.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason,
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            engine_data: None,
            routing_data: None,
            jailed_text: None,
        }
    }

    /// An answer the engine finished: some text, then a chunk with its `finish_reason`.
    fn finished_answer() -> Vec<BackendOutput> {
        vec![
            output("Hello", None),
            output(" world", Some(FinishReason::Stop)),
        ]
    }

    /// A tool call the engine had not closed when its stream ended.
    fn open_tool_call() -> Vec<BackendOutput> {
        vec![
            output("<tool_call>\n{\"name\": \"get_weather\", ", None),
            output("\"arguments\": {\"city\": \"Par", None),
        ]
    }

    /// The stages below a preprocessor: sends `outputs`, then does `at_end` to the request's
    /// context, then ends cleanly. A stop or kill that lands before the stream ends.
    struct SignalAtEnd {
        outputs: Vec<BackendOutput>,
        at_end: fn(&dyn AsyncEngineContext),
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
        for SignalAtEnd
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
            let context = request.context();
            let signalled = context.clone();
            let at_end = self.at_end;
            let end = stream::poll_fn(move |_| {
                at_end(signalled.as_ref());
                std::task::Poll::Ready(None)
            });
            let outputs = self.outputs.clone().into_iter().map(Annotated::from_data);
            Ok(ResponseStream::new(
                Box::pin(stream::iter(outputs).chain(end)),
                context,
            ))
        }
    }

    fn kill(context: &dyn AsyncEngineContext) {
        context.kill();
    }

    fn stop(context: &dyn AsyncEngineContext) {
        context.stop_generating();
    }

    fn preprocessor(model: &str, tool_call_parser: Option<&str>) -> Arc<OpenAIPreprocessor> {
        let mut card = ModelDeploymentCard::load_from_disk(model, None).unwrap();
        card.runtime_config.tool_call_parser = tool_call_parser.map(str::to_string);
        OpenAIPreprocessor::new(card).unwrap()
    }

    fn chat_request() -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .unwrap()
    }

    fn tool_request() -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            }],
            "tool_choice": "auto",
        }))
        .unwrap()
    }

    fn completion_request() -> NvCreateCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "Hello",
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .unwrap()
    }

    /// The completion markers in a response: each choice's `finish_reason`, the usage chunks,
    /// and the tool calls.
    #[derive(Debug, Default, PartialEq)]
    struct Markers {
        finish_reasons: Vec<String>,
        usage_chunks: usize,
        tool_calls: usize,
    }

    fn markers<T: Serialize>(items: &[Annotated<T>]) -> Markers {
        let mut markers = Markers::default();
        for item in items {
            let Some(data) = &item.data else { continue };
            let data = serde_json::to_value(data).unwrap();
            if !data["usage"].is_null() {
                markers.usage_chunks += 1;
            }
            for choice in data["choices"].as_array().into_iter().flatten() {
                if let Some(reason) = choice["finish_reason"].as_str() {
                    markers.finish_reasons.push(reason.to_string());
                }
                if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
                    markers.tool_calls += calls.len();
                }
            }
        }
        markers
    }

    /// The request's response through the preprocessor, as the frontend composes it, over
    /// stages that send `outputs` and then do `at_end`.
    async fn through_pipeline<Req, Resp>(
        preprocessor: Arc<OpenAIPreprocessor>,
        request: Req,
        outputs: Vec<BackendOutput>,
        at_end: fn(&dyn AsyncEngineContext),
    ) -> Vec<Annotated<Resp>>
    where
        Req: dynamo_runtime::engine::Data,
        Resp: dynamo_runtime::engine::Data,
        OpenAIPreprocessor: Operator<
                SingleIn<Req>,
                ManyOut<Annotated<Resp>>,
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
            >,
    {
        let below = Arc::new(SignalAtEnd { outputs, at_end });
        Operator::generate(preprocessor.as_ref(), Context::new(request), below)
            .await
            .expect("the preprocessor accepts the request")
            .collect()
            .await
    }

    fn finished(usage_chunks: usize) -> Markers {
        Markers {
            finish_reasons: vec!["stop".to_string()],
            usage_chunks,
            tool_calls: 0,
        }
    }

    /// The finish the engine sent before the kill passes through; the usage chunk the
    /// preprocessor writes when the stream ends does not.
    #[tokio::test]
    async fn chat_writes_no_usage_chunk_after_a_kill() {
        let items = through_pipeline::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TINY_LLAMA, None),
            chat_request(),
            finished_answer(),
            kill,
        )
        .await;
        assert_eq!(markers(&items), finished(0), "{items:?}");
    }

    #[tokio::test]
    async fn chat_still_writes_its_usage_chunk_after_a_stop() {
        let items = through_pipeline::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TINY_LLAMA, None),
            chat_request(),
            finished_answer(),
            stop,
        )
        .await;
        assert_eq!(markers(&items), finished(1), "{items:?}");
    }

    #[tokio::test]
    async fn completion_writes_no_usage_chunk_after_a_kill() {
        let items =
            through_pipeline::<_, crate::types::openai::completions::NvCreateCompletionResponse>(
                preprocessor(TINY_LLAMA, None),
                completion_request(),
                finished_answer(),
                kill,
            )
            .await;
        assert_eq!(markers(&items), finished(0), "{items:?}");
    }

    #[tokio::test]
    async fn completion_still_writes_its_usage_chunk_after_a_stop() {
        let items =
            through_pipeline::<_, crate::types::openai::completions::NvCreateCompletionResponse>(
                preprocessor(TINY_LLAMA, None),
                completion_request(),
                finished_answer(),
                stop,
            )
            .await;
        assert_eq!(markers(&items), finished(1), "{items:?}");
    }

    /// What the tool-call jail writes when the stream ends with a call still open.
    async fn jail_after(at_end: fn(&dyn AsyncEngineContext)) -> Markers {
        let items = through_pipeline::<
            _,
            crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        >(
            preprocessor(TOOL_MODEL, Some("hermes")),
            tool_request(),
            open_tool_call(),
            at_end,
        )
        .await;
        markers(&items)
    }

    /// On a clean end the jail closes the open call and finishes the answer. After a kill
    /// it writes nothing of that.
    #[tokio::test]
    async fn the_jail_writes_no_tool_call_or_finish_after_a_kill() {
        let stopped = jail_after(stop).await;
        assert_ne!(stopped, Markers::default(), "the control must write on end");
        assert_eq!(jail_after(kill).await, Markers::default());
    }

    #[tokio::test]
    async fn custom_tool_stream_ends_quietly_after_a_kill() {
        use crate::protocols::openai::chat_completions::{
            NvCreateChatCompletionStreamResponse, custom_tools::ChatCustomToolAdapter,
        };

        async fn response(
            at_end: fn(&dyn AsyncEngineContext),
        ) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Calculate"}],
                    "stream": true,
                    "tools": [{
                        "type": "custom",
                        "custom": {"name": "run_python", "format": {"type": "text"}}
                    }],
                    "tool_choice": {"type": "custom", "custom": {"name": "run_python"}}
                }))
                .unwrap();
            let custom_tools = ChatCustomToolAdapter::prepare(&mut request).unwrap();
            let below = Arc::new(SignalAtEnd {
                outputs: vec![output(r#"{"input":"print(1"#, None)],
                at_end,
            });
            let preprocessor = preprocessor(TOOL_MODEL, None);
            let stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>> =
                Operator::generate(preprocessor.as_ref(), Context::new(request), below)
                    .await
                    .unwrap();
            let context = stream.context();
            custom_tools.stream(stream, context).collect().await
        }

        let stopped = response(stop).await;
        assert!(stopped.iter().all(|item| !item.is_error()), "{stopped:?}");
        assert!(markers(&stopped).tool_calls > 0, "{stopped:?}");

        let killed = response(kill).await;
        assert!(
            killed.iter().all(|item| !item.is_error()),
            "a killed custom-tool stream must end quietly: {killed:?}"
        );
    }
}
