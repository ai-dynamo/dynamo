// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn, async_trait};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

/// Publishes the request payload record for chat pipelines that bring their own
/// pre/post processor (`--dyn-chat-processor vllm|sglang`). Those pipelines come
/// from `PreprocessedRouting::build_preprocessed_pipeline`, which links no
/// `OpenAIPreprocessor`, so the capture site in `OpenAIPreprocessor::generate`
/// never runs for them. This wrapper reuses the same `payload::create_handle` /
/// `RequestPayloadHandle::emit` pair, so the emitted row is indistinguishable
/// from the one the preprocessor produces.
struct RequestPayloadChatEngine {
    inner: OpenAIChatCompletionsStreamingEngine,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for RequestPayloadChatEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // Mirrors preprocessor.rs: the handle snapshots the pristine request and
        // its arrival time here, together with the headers the HTTP layer stashed
        // on the context; the single record is published once at stream
        // completion (or with an empty response on cancel/timeout).
        let request_id = request.id().to_string();
        let payload_http_headers = if super::payload::http_header_capture_active() {
            request
                .get_optional::<BTreeMap<String, String>>(super::payload::HTTP_HEADERS_CONTEXT_KEY)
                .ok()
                .flatten()
        } else {
            None
        };
        let payload_handle =
            super::payload::create_handle(request.content(), &request_id, payload_http_headers);
        let Some(payload) = payload_handle else {
            return self.inner.generate(request).await;
        };

        let response_stream = self.inner.generate(request).await?;
        let context = response_stream.context();

        // Pass-through aggregation for both streaming and non-streaming: this
        // pipeline owns no postprocessor, so the chunks the chat processor
        // produced must reach the client unchanged. (The preprocessor can use
        // the `fold` variant for `stream=false` because it owns the chunk shape.)
        let (stream, agg_fut) = super::payload_stream::scan_aggregate_with_future(response_stream);

        // Spawn the payload emit off the request path. `agg_fut` resolves to None
        // on client cancel / gateway timeout / aggregation failure; we still emit
        // the payload record with an empty response so those cases remain
        // inspectable.
        tokio::spawn(async move {
            match agg_fut.await {
                Some(final_resp) => payload.emit(Some(Arc::new(final_resp))),
                None => {
                    tracing::debug!(
                        request_id = %payload.request_id(),
                        "request payload: response aggregation incomplete (client cancel / timeout); emitting request-only record"
                    );
                    payload.emit(None);
                }
            }
        });

        Ok(ResponseStream::new(stream, context))
    }
}

/// Wrap a chat engine built by a `--dyn-chat-processor` factory so request
/// payload records are emitted for it. Returns the engine untouched unless the
/// policy selects `request_payload`, and is only applied on the factory path, so
/// `OpenAIPreprocessor` remains the single producer on the default path.
pub(crate) fn wrap_chat_request_payload_engine(
    engine: OpenAIChatCompletionsStreamingEngine,
) -> OpenAIChatCompletionsStreamingEngine {
    if !super::policy().emit_request_payload_records() {
        return engine;
    }
    Arc::new(RequestPayloadChatEngine { inner: engine })
}
