// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Audit middleware for the factory-provided chat engine.
//!
//! Background (v1.2.0 adaptation):
//! When the Python chat processor is selected (`--dyn-chat-processor vllm/sglang`),
//! `discovery::watcher` installs the Python `chat_engine_factory` engine *directly*
//! as `WorkerSet::chat_engine`. On that path, requests never flow through
//! [`crate::preprocessor::OpenAIPreprocessor::generate`], which is where the audit-bus
//! capture point lives (`create_handle` / `emit_request` / `emit_response`). As a
//! result audit records were never published for the python-chat-processor path even
//! though the sinks initialized correctly.
//!
//! This module re-homes the PR's capture point as a thin `AsyncEngine` wrapper so the
//! factory chat engine is audited too. It mirrors the exact idioms from
//! [`crate::preprocessor::OpenAIPreprocessor::generate`] (request-id derivation, OTEL
//! header lookup, `create_handle`, detached `emit_request`, and the
//! scan/fold aggregation + detached `emit_response`).
//!
//! Only the factory path is wrapped (see `watcher.rs`). The Rust-preprocessor path
//! already audits inside `generate`; wrapping it too would double-emit records.

use std::sync::Arc;

use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait,
};
use futures::StreamExt;

use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;

/// Wrap a factory-provided chat engine so its requests/responses are published on
/// the audit bus, mirroring the Rust-preprocessor capture point.
///
/// Returns the same [`OpenAIChatCompletionsStreamingEngine`] alias so the call site
/// in `watcher.rs` stays byte-identical apart from the wrap.
pub fn wrap_chat_engine(
    inner: OpenAIChatCompletionsStreamingEngine,
) -> OpenAIChatCompletionsStreamingEngine {
    Arc::new(AuditedChatEngine { inner })
}

/// `AsyncEngine` adapter that captures audit records around an inner chat engine.
///
/// Implements the same trait shape as the factory engine
/// (`AsyncEngine<SingleIn<NvCreateChatCompletionRequest>,
/// ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error>` =
/// [`OpenAIChatCompletionsStreamingEngine`]) and delegates `generate` to `inner`.
struct AuditedChatEngine {
    inner: OpenAIChatCompletionsStreamingEngine,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for AuditedChatEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // Derive request_id exactly as OpenAIPreprocessor::generate does.
        // `SingleIn<T>` is `Context<T>`; its inherent `id()` yields the stream id.
        let request_id = request.id().to_string();

        // Build audit handle (None if no DYN_AUDIT_SINKS). Mirror the preprocessor:
        // only look up OTEL HTTP headers when the otel sink is active. `Context<T>`
        // exposes `get::<V>(key)` directly (same accessor the preprocessor calls).
        let audit_http_headers = if crate::audit::config::otel_sink_capture_enabled() {
            request
                .get::<crate::audit::handle::AuditHttpRequestHeaders>(
                    crate::audit::handle::OTEL_HTTP_HEADERS_CONTEXT_KEY,
                )
                .ok()
        } else {
            None
        };

        // `create_handle` borrows the inner `NvCreateChatCompletionRequest`. Unlike
        // the preprocessor we do NOT `into_parts()` (that would consume the request
        // we must forward intact); instead we borrow the inner value via `Context`'s
        // `Deref<Target = T>`.
        let audit_handle =
            crate::audit::handle::create_handle(&*request, &request_id, audit_http_headers);

        // Publish the request-side record on a detached task before dispatch, so
        // hung / canceled requests are still observable. `emit_request` performs the
        // text-only media redaction internally; we do not duplicate it here.
        if let Some(h) = audit_handle.clone() {
            // Clone the inner request out of the Context via Deref.
            let audit_request = Arc::new((*request).clone());
            tokio::spawn(async move {
                h.emit_request(audit_request);
            });
        }

        // Forward the request to the inner factory engine untouched.
        let stream = self.inner.generate(request).await?;

        // No audit handle: return the inner stream unchanged.
        let Some(audit) = audit_handle else {
            return Ok(stream);
        };

        // Capture the engine context off the inner stream before we move it, then
        // re-wrap the aggregated pass-through stream with that same context — the
        // exact `ResponseStream::new(stream, ctx)` idiom the preprocessor uses.
        let ctx = stream.context();

        // `ManyOut` (= `Pin<Box<dyn AsyncEngineStream>>`) is `Unpin + Send`; box it
        // into a plain `Send` stream so it satisfies the scan/fold bounds, matching
        // the preprocessor which always boxes before passing to the aggregators.
        let boxed = stream.boxed();
        let (wrapped, agg_fut) = if audit.streaming() {
            crate::audit::stream::scan_aggregate_with_future(boxed)
        } else {
            crate::audit::stream::fold_aggregate_with_future(boxed)
        };

        // Spawn the response-side audit task. The future resolves to `None` on client
        // cancel or aggregation failure; in that case the (already published) request
        // record stands alone and we skip the response emit.
        //
        // v1.2.0 nuance: on client disconnect the HTTP layer's monitor_for_disconnects
        // calls ctx.stop_generating() and the engine stream ENDS (rather than being
        // dropped mid-flight), so the aggregator can resolve Some(partial). Guard on
        // the context cancellation state to preserve the design contract: a cancelled
        // request leaves a request record only.
        let audit_ctx = ctx.clone();
        tokio::spawn(async move {
            match agg_fut.await {
                Some(_) if audit_ctx.is_stopped() || audit_ctx.is_killed() => tracing::debug!(
                    request_id = %audit.request_id(),
                    "audit: response record skipped (client cancelled; partial aggregation discarded)"
                ),
                Some(final_resp) => audit.emit_response(Arc::new(final_resp)),
                None => tracing::debug!(
                    request_id = %audit.request_id(),
                    "audit: response record skipped (client cancel or aggregation error)"
                ),
            }
        });

        Ok(ResponseStream::new(wrapped, ctx))
    }
}
