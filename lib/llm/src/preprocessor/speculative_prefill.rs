// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Speculative next-turn prefill for reasoning models.
//!
//! After an assistant turn completes, we know what the next turn's prompt prefix
//! will look like: the full conversation history (with thinking content stripped by
//! the Jinja template for non-last assistant turns). We render it, tokenize it,
//! and send a `max_tokens=1` request through the pipeline to warm the KV cache.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
};
use futures::Stream;
use futures::stream::StreamExt;
use minijinja::value::Value;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider};
use dynamo_runtime::pipeline::{Context as PipelineContext, Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::tokenizers::traits::Tokenizer;
use dynamo_renderer::{OAIChatLikeRequest, OAIPromptFormatter};

/// Upper bound on the lifetime of one detached speculative-prefill task.
///
/// The task is a best-effort KV-cache warmup for a turn the user may never
/// send, and it is dispatched on a context nobody else holds, so nothing
/// upstream can ever stop it. A backend that returns a stream which stays
/// pending instead of reaching EOF would otherwise pin one task — and
/// everything it captured — for the life of the process, once per request
/// that used the hint. The value is generous relative to a `max_tokens=1`
/// request on a healthy backend and short relative to a human turn: a warmup
/// that has not finished by now has already lost the race with the next turn
/// it was warming for.
const PREFILL_TASK_TIMEOUT: Duration = Duration::from_secs(60);

/// Publication slot for the warmup stream's context.
///
/// [`prefill_task`] fills it before draining so that the bounding wrapper in
/// [`maybe_wrap_stream`] can reach the downstream context on the timeout and
/// cancellation paths.
type PrefillContextSlot = Arc<Mutex<Option<Arc<dyn AsyncEngineContext>>>>;

/// A minimal `OAIChatLikeRequest` for speculative next-turn prefill.
/// Holds the full conversation (including a new assistant message) and
/// renders with `add_generation_prompt = false` so the result is the
/// exact prefix the next user turn will see.
pub struct SpeculativePrefillRequest {
    messages: Vec<ChatCompletionRequestMessage>,
}

impl SpeculativePrefillRequest {
    pub fn new(messages: Vec<ChatCompletionRequestMessage>) -> Self {
        Self { messages }
    }
}

impl OAIChatLikeRequest for SpeculativePrefillRequest {
    fn model(&self) -> String {
        "speculative_prefill".to_string()
    }

    fn messages(&self) -> Value {
        let json = serde_json::to_value(&self.messages).unwrap();
        Value::from_serialize(&json)
    }

    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        Some(&self.messages)
    }

    fn should_add_generation_prompt(&self) -> bool {
        false
    }
}

/// Optionally wraps a chat completion response stream to enable speculative
/// next-turn prefill. When `nvext.speculative_prefill` is set, the returned
/// stream accumulates the assistant response text and, on completion, spawns
/// a background task that renders the next-turn prefix and fires a
/// `max_tokens=1` request through the pipeline to warm the KV cache.
///
/// The spawned task is bounded by [`PREFILL_TASK_TIMEOUT`] and, when `cancel`
/// is supplied, also ends when that token is cancelled; nothing else can stop
/// it, because it runs on a context of its own and no owner keeps its handle.
///
/// When the flag is not set, returns the stream unmodified with zero overhead.
pub fn maybe_wrap_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    request: &NvCreateChatCompletionRequest,
    next: &Arc<
        dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
    >,
    formatter: &Arc<dyn OAIPromptFormatter>,
    tokenizer: &Arc<dyn Tokenizer>,
    cancel: Option<&CancellationToken>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let enabled = request
        .nvext
        .as_ref()
        .and_then(|ext| ext.agent_hints.as_ref())
        .and_then(|hints| hints.speculative_prefill)
        .unwrap_or(false);

    if !enabled {
        return stream;
    }

    let (tx, rx) = tokio::sync::oneshot::channel::<String>();

    let next = next.clone();
    let formatter = formatter.clone();
    let tokenizer = tokenizer.clone();
    let messages = request.inner.messages.clone();
    let cancel = cancel.cloned();
    tokio::spawn(async move {
        let context_slot: PrefillContextSlot = Arc::new(Mutex::new(None));

        let warmup = {
            let context_slot = context_slot.clone();
            async move {
                let Ok(response_text) = rx.await else {
                    return;
                };
                if let Err(e) = prefill_task(
                    next,
                    formatter,
                    tokenizer,
                    messages,
                    response_text,
                    &context_slot,
                )
                .await
                {
                    tracing::warn!(error = %e, "Speculative prefill failed");
                }
            }
        };
        // Pinned and polled by reference so that neither `select!` arm consumes
        // it: on the bail-out arms the warmup future — and with it the
        // downstream stream — is still alive while `stop_downstream` runs, and
        // is dropped only when this block ends.
        let mut warmup = std::pin::pin!(warmup);

        tokio::select! {
            biased;

            outcome = tokio::time::timeout(PREFILL_TASK_TIMEOUT, &mut warmup) => {
                if outcome.is_err() {
                    tracing::warn!(
                        timeout_secs = PREFILL_TASK_TIMEOUT.as_secs(),
                        "Speculative prefill exceeded its lifetime bound; abandoning warmup"
                    );
                    stop_downstream(&context_slot);
                }
            }

            () = cancelled(cancel) => {
                tracing::debug!("Speculative prefill cancelled by runtime shutdown");
                stop_downstream(&context_slot);
            }
        }
    });

    let mut accumulated_text = String::new();
    let mut prefill_tx = Some(tx);
    Box::pin(stream.map(move |item| {
        if let Some(ref resp) = item.data {
            for choice in &resp.inner.choices {
                if let Some(ChatCompletionMessageContent::Text(ref text)) = choice.delta.content {
                    accumulated_text.push_str(text);
                }
                // Send accumulated text once we see finish_reason (works
                // regardless of whether usage reporting is enabled).
                if choice.finish_reason.is_some()
                    && let Some(tx) = prefill_tx.take()
                {
                    let _ = tx.send(accumulated_text.clone());
                }
            }
        }

        item
    }))
}

/// Resolves when `token` is cancelled, and never when there is no token, so
/// the caller's timeout remains the only bound in that case.
async fn cancelled(token: Option<CancellationToken>) {
    match token {
        Some(token) => token.cancelled().await,
        None => std::future::pending().await,
    }
}

/// Asks the warmup's downstream to wind down before the stream is dropped, so
/// the KV router's `RequestGuard` still gets a chance at its normal lifecycle.
/// A downstream that ignores the request is not a problem: the caller drops
/// the stream regardless, so the task ends either way.
fn stop_downstream(slot: &PrefillContextSlot) {
    if let Some(context) = slot.lock().take() {
        context.stop_generating();
    }
}

/// Fire-and-forget task that renders the next-turn prefix and sends it
/// through the pipeline as a `max_tokens=1` request to warm the KV cache.
async fn prefill_task(
    next: Arc<
        dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
    >,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    original_messages: Vec<ChatCompletionRequestMessage>,
    response_text: String,
    context_slot: &PrefillContextSlot,
) -> Result<()> {
    let assistant_msg =
        ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                response_text,
            )),
            ..Default::default()
        });

    let mut messages = original_messages;
    messages.push(assistant_msg);

    let prefill_request = SpeculativePrefillRequest::new(messages);
    let formatted_prompt = formatter.render(&prefill_request)?;
    let encoding = tokenizer.encode(&formatted_prompt)?;
    let token_ids = encoding.token_ids().to_vec();

    tracing::info!(
        num_tokens = token_ids.len(),
        "Speculative prefill: sending next-turn prefix"
    );

    let preprocessed = PreprocessedRequest::builder()
        .model("speculative_prefill".to_string())
        .token_ids(token_ids)
        .stop_conditions(StopConditions {
            max_tokens: Some(1),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![])
        .annotations(vec![])
        .build()?;

    let context = PipelineContext::with_id_and_metadata(
        preprocessed,
        uuid::Uuid::new_v4().to_string(),
        Default::default(),
    );
    // Drain the stream so the KV router's RequestGuard runs its full lifecycle
    // (mark_prefill_completed, block tracking, free) instead of relying on drop.
    if let Ok(mut stream) = next.generate(context).await {
        // Published before the drain, because a drain that never returns is
        // exactly the case in which the caller needs this handle.
        *context_slot.lock() = Some(stream.context());
        while stream.next().await.is_some() {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::Poll;

    use async_trait::async_trait;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionStreamResponseDelta,
        CreateChatCompletionRequest, CreateChatCompletionStreamResponse, FinishReason,
    };
    use dynamo_renderer::PromptFormatter;
    use dynamo_runtime::pipeline::ResponseStream;
    use tokio::time::Instant;

    use crate::model_card::ModelDeploymentCard;
    use crate::preprocessor::prompt::prompt_formatter_from_mdc;
    use crate::protocols::common::extensions::{AgentHints, NvExt};

    use super::*;

    type BackendEngine =
        dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>;

    /// Flips a shared flag when dropped, so a test can observe the exact moment
    /// the detached task lets go of the downstream stream it was draining.
    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// The failure this module has to survive: a backend that accepts the
    /// warmup request and hands back a stream which never yields and never
    /// reaches EOF. Draining it is an await that by itself never returns.
    #[derive(Debug, Default)]
    struct StallingBackend {
        generate_calls: AtomicUsize,
        stream_dropped: Arc<AtomicBool>,
        context: Mutex<Option<Arc<dyn AsyncEngineContext>>>,
    }

    impl StallingBackend {
        fn generate_calls(&self) -> usize {
            self.generate_calls.load(Ordering::SeqCst)
        }

        fn stream_dropped(&self) -> bool {
            self.stream_dropped.load(Ordering::SeqCst)
        }

        /// Whether the downstream was asked to wind down, i.e. whether
        /// `stop_generating` reached the context the backend published.
        fn was_asked_to_stop(&self) -> bool {
            self.context
                .lock()
                .as_ref()
                .map(|ctx| ctx.is_stopped())
                .unwrap_or(false)
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
        for StallingBackend
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
            self.generate_calls.fetch_add(1, Ordering::SeqCst);
            let (_request, context) = request.transfer(());
            let ctx = context.context();
            *self.context.lock() = Some(ctx.clone());

            let probe = DropProbe(self.stream_dropped.clone());
            let stalled = futures::stream::poll_fn(move |_cx| {
                // Touched so the probe is owned by the stream and released
                // only when the stream itself is dropped.
                let _ = &probe;
                Poll::Pending
            });

            Ok(ResponseStream::new(Box::pin(stalled), ctx))
        }
    }

    fn sample_model_parts() -> (Arc<dyn OAIPromptFormatter>, Arc<dyn Tokenizer>) {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&mdc).unwrap();
        let tokenizer = mdc.tokenizer().unwrap();
        let tokenizer: Arc<dyn Tokenizer> = (*tokenizer).clone();
        (formatter, tokenizer)
    }

    fn chat_request(speculative_prefill: bool) -> NvCreateChatCompletionRequest {
        let messages = vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(
                    "how tall is everest?".to_string(),
                ),
                name: None,
            },
        )];

        let nvext = speculative_prefill.then(|| NvExt {
            agent_hints: Some(AgentHints {
                speculative_prefill: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        });

        NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                model: "mock-llama".to_string(),
                messages,
                stream: Some(true),
                ..Default::default()
            },
            common: Default::default(),
            nvext,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    /// One upstream chunk carrying `text`, terminal when `finish` is set: the
    /// terminal chunk is what releases the warmup.
    fn chunk(text: &str, finish: bool) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: None,
                content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: finish.then_some(FinishReason::Stop),
            logprobs: None,
        };

        Annotated::from_data(NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "chatcmpl-test".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 0,
                model: "mock-llama".to_string(),
                system_fingerprint: None,
                service_tier: None,
                choices: vec![choice],
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
        })
    }

    fn upstream()
    -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
        Box::pin(futures::stream::iter(vec![
            chunk("everest is ", false),
            chunk("8849 m tall.", true),
        ]))
    }

    fn text_of(items: &[Annotated<NvCreateChatCompletionStreamResponse>]) -> String {
        items
            .iter()
            .filter_map(|item| item.data.as_ref())
            .flat_map(|resp| resp.inner.choices.iter())
            .filter_map(|choice| match choice.delta.content {
                Some(ChatCompletionMessageContent::Text(ref text)) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Lets the detached task run up to its next suspension point without
    /// advancing the paused clock.
    async fn settle() {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_warmup_is_released_when_the_lifetime_bound_elapses() {
        let (formatter, tokenizer) = sample_model_parts();
        let backend = Arc::new(StallingBackend::default());
        let engine: Arc<BackendEngine> = backend.clone();

        let request = chat_request(true);
        let wrapped =
            maybe_wrap_stream(upstream(), &request, &engine, &formatter, &tokenizer, None);
        // Only the test's own handle and the detached task's clone remain, so
        // the strong count is a faithful release probe.
        drop(engine);

        let items: Vec<_> = wrapped.collect().await;
        assert_eq!(
            items.len(),
            2,
            "client stream must be passed through intact"
        );

        // The warmup is genuinely in flight and genuinely stuck: it reached the
        // backend, and well past the point where a healthy max_tokens=1 request
        // would have finished it is still holding the stream.
        settle().await;
        assert_eq!(backend.generate_calls(), 1, "warmup should have dispatched");
        tokio::time::sleep(PREFILL_TASK_TIMEOUT / 2).await;
        settle().await;
        assert!(
            !backend.stream_dropped(),
            "warmup abandoned before its bound elapsed"
        );
        assert_eq!(Arc::strong_count(&backend), 2);

        tokio::time::sleep(PREFILL_TASK_TIMEOUT).await;
        settle().await;

        assert!(
            backend.was_asked_to_stop(),
            "downstream context should be told to stop before the stream is dropped"
        );
        assert!(
            backend.stream_dropped(),
            "stalled downstream stream should be released once the bound elapses"
        );
        assert_eq!(
            Arc::strong_count(&backend),
            1,
            "detached task should have released the engine it captured"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_warmup_is_released_when_the_runtime_token_is_cancelled() {
        let (formatter, tokenizer) = sample_model_parts();
        let backend = Arc::new(StallingBackend::default());
        let engine: Arc<BackendEngine> = backend.clone();
        let cancel = CancellationToken::new();

        let request = chat_request(true);
        let wrapped = maybe_wrap_stream(
            upstream(),
            &request,
            &engine,
            &formatter,
            &tokenizer,
            Some(&cancel),
        );
        drop(engine);

        let started = Instant::now();
        let items: Vec<_> = wrapped.collect().await;
        assert_eq!(items.len(), 2);

        settle().await;
        assert_eq!(backend.generate_calls(), 1, "warmup should have dispatched");
        assert!(!backend.stream_dropped());

        cancel.cancel();
        settle().await;

        assert!(
            backend.was_asked_to_stop(),
            "cancellation should reach the downstream context"
        );
        assert!(
            backend.stream_dropped(),
            "cancellation should release the stalled downstream stream"
        );
        assert_eq!(Arc::strong_count(&backend), 1);
        // Proves the release came from cancellation and not from the timeout.
        assert!(started.elapsed() < PREFILL_TASK_TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn without_the_hint_the_stream_is_untouched_and_nothing_is_dispatched() {
        let (formatter, tokenizer) = sample_model_parts();
        let backend = Arc::new(StallingBackend::default());
        let engine: Arc<BackendEngine> = backend.clone();

        let request = chat_request(false);
        let wrapped =
            maybe_wrap_stream(upstream(), &request, &engine, &formatter, &tokenizer, None);
        drop(engine);

        let items: Vec<_> = wrapped.collect().await;
        assert_eq!(items.len(), 2);
        assert_eq!(text_of(&items), "everest is 8849 m tall.");

        settle().await;
        tokio::time::sleep(PREFILL_TASK_TIMEOUT * 2).await;
        settle().await;

        assert_eq!(
            backend.generate_calls(),
            0,
            "the disabled path must never dispatch a warmup"
        );
        assert_eq!(
            Arc::strong_count(&backend),
            1,
            "the disabled path must not capture the engine at all"
        );
    }
}
