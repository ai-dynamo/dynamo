// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-EPP routing through the frontend's own routing pipeline.
//!
//! With `DYN_EPP_ROUTING_HOST=true` the runtime EPP builds the same
//! [`PreprocessedRequest`] the frontend would and hands it to the frontend's
//! `RoutingHost`, whose final hop is a [`GatewayDispatch`]: instead of sending
//! the request, it gives the selected worker back to the EPP, and the EPP feeds
//! the response progress Envoy relays into the host's response stream. Session
//! affinity, request classifiers, load booking, output tracking and rejection
//! statuses therefore come from the frontend's code rather than an EPP copy.
//!
//! Aggregated routing only: while prefill workers are registered the EPP keeps
//! its existing disaggregated path.

use std::collections::HashSet;
use std::sync::Arc;

use dashmap::DashMap;
use dynamo_llm::http::service::{
    axum::http::{HeaderMap, HeaderName, HeaderValue},
    client_error_response, preprocessed_request_context,
};
use dynamo_llm::kv_router::RoutingHost;
use dynamo_llm::kv_router::gateway_dispatch::{GatewayDispatch, GatewayProgress};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::common::FinishReason;
use dynamo_llm::protocols::common::extensions::{
    HEADER_DP_RANK, HEADER_WORKER_INSTANCE_ID, apply_frontend_nvext_policy,
};
use dynamo_llm::protocols::common::input_trigger::{
    classify_chat_request, classify_completion_request,
};
use dynamo_llm::protocols::common::llm_backend::PreprocessedRequest;
use dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::types::openai::completions::NvCreateCompletionRequest;
use dynamo_protocols::types::{CompletionUsage, Prompt};
use dynamo_runtime::pipeline::{AsyncEngine, Context};
use futures::StreamExt;

use crate::picker::{PickError, PickResult};

/// Enables this path for the runtime EPP.
pub const ROUTING_HOST_ENV: &str = "DYN_EPP_ROUTING_HOST";

/// An SSE line longer than this is not a token delta; drop it rather than
/// buffer it (the delta it might have carried counts as one token).
const MAX_PENDING_SSE_BYTES: usize = 1024 * 1024;

pub(crate) struct GatewayRouting {
    host: Arc<RoutingHost>,
    dispatch: Arc<GatewayDispatch>,
    responses: DashMap<String, ResponseProgress>,
}

/// A request ready for the routing host.
pub(crate) struct RoutableRequest {
    context: Context<PreprocessedRequest>,
}

impl GatewayRouting {
    pub(crate) fn new(host: Arc<RoutingHost>, dispatch: Arc<GatewayDispatch>) -> Self {
        Self {
            host,
            dispatch,
            responses: DashMap::new(),
        }
    }

    /// Whether `booking_id` was routed through this path.
    pub(crate) fn owns(&self, booking_id: &str) -> bool {
        self.responses.contains_key(booking_id)
    }

    /// Build the request the frontend's routing host would receive: the
    /// frontend's header policy (priority, pins, cache salt), its preprocessor,
    /// and its request context (session affinity, agent context, metadata).
    ///
    /// `Ok(None)` when the body is not a chat or completions request this path
    /// handles; the caller falls back to the existing path.
    pub(crate) async fn prepare(
        preprocessor: &OpenAIPreprocessor,
        body: &str,
        headers: &[(String, String)],
        request_id: &str,
        allowed_worker_ids: Option<HashSet<u64>>,
    ) -> Result<Option<RoutableRequest>, PickError> {
        let headers = header_map(headers);
        let value: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| PickError::InvalidRequest(format!("invalid JSON body: {e}")))?;
        let is_chat = value
            .get("messages")
            .and_then(|messages| messages.as_array())
            .is_some_and(|messages| !messages.is_empty());

        // Classify user vs tool-result input on the typed request, as the
        // frontend's routes do, so trigger-aware policies see it.
        let (mut preprocessed, input_trigger) = if is_chat {
            let mut request: NvCreateChatCompletionRequest = serde_json::from_value(value)
                .map_err(|e| PickError::InvalidRequest(e.to_string()))?;
            request.nvext = apply_frontend_nvext_policy(request.nvext.take(), &headers, true);
            let (preprocessed, _, _) = preprocessor
                .preprocess_request(&request, None)
                .await
                .map_err(|e| PickError::InvalidRequest(format!("{e:#}")))?;
            (preprocessed, classify_chat_request(&request))
        } else if value.get("prompt").is_some() {
            let mut request: NvCreateCompletionRequest = serde_json::from_value(value)
                .map_err(|e| PickError::InvalidRequest(e.to_string()))?;
            // A multi-prompt batch is split into one engine request per prompt
            // downstream; it keeps the existing path, which routes on the first.
            if is_multi_prompt_batch(&request.inner.prompt) {
                return Ok(None);
            }
            request.nvext = apply_frontend_nvext_policy(request.nvext.take(), &headers, true);
            let (preprocessed, _, _) = preprocessor
                .preprocess_request(&request, None)
                .await
                .map_err(|e| PickError::InvalidRequest(format!("{e:#}")))?;
            (preprocessed, classify_completion_request(&request))
        } else {
            return Ok(None);
        };

        if allowed_worker_ids.is_some() {
            preprocessed.routing_mut().allowed_worker_ids = allowed_worker_ids;
        }
        let context = preprocessed_request_context(
            preprocessed,
            request_id.to_string(),
            &headers,
            Some(input_trigger),
        )
        .map_err(|(status, message)| PickError::Rejected { status, message })?;
        Ok(Some(RoutableRequest { context }))
    }

    /// Route through the host and return where Envoy should send the request.
    /// Pre-dispatch failures carry the status the frontend would return.
    pub(crate) async fn pick(&self, request: RoutableRequest) -> Result<PickResult, PickError> {
        let request_id = request.context.id().to_string();
        // If this future is dropped (the ext_proc stream closed mid-pick) after
        // the host dispatched, release the dispatch entry it left behind.
        let mut pending = AbortOnDrop {
            dispatch: &self.dispatch,
            request_id: &request_id,
            armed: true,
        };

        let stream = self.host.generate(request.context).await.map_err(|error| {
            let (status, message) = client_error_response(error);
            PickError::Rejected { status, message }
        })?;
        let route = self.dispatch.take_route(&request_id).ok_or_else(|| {
            PickError::RoutingFailed("routing host returned without dispatching".to_string())
        })?;
        // The host's request lifecycle runs as its response stream is polled:
        // keep polling until the EPP completes or aborts the stream.
        tokio::spawn(stream.for_each(|_| async {}));

        self.responses
            .insert(request_id.clone(), ResponseProgress::default());
        pending.armed = false;

        let dp_rank = route.dp_rank.unwrap_or(0);
        tracing::info!(
            worker_id = route.worker_id,
            dp_rank,
            endpoint = %route.endpoint,
            token_count = route.token_ids.len(),
            "Picked endpoint through the routing host"
        );
        Ok(PickResult {
            endpoint: route.endpoint,
            headers: vec![
                (
                    HEADER_WORKER_INSTANCE_ID.to_string(),
                    route.worker_id.to_string(),
                ),
                (HEADER_DP_RANK.to_string(), dp_rank.to_string()),
                (
                    "x-dynamo-routing-mode".to_string(),
                    "aggregated".to_string(),
                ),
            ],
            token_ids: Some(route.token_ids),
            reservation_id: Some(request_id.clone()),
            ..Default::default()
        })
    }

    /// Relay response progress Envoy delivered for `booking_id`.
    pub(crate) fn on_response_body(
        &self,
        booking_id: &str,
        chunk: &[u8],
        streaming: bool,
        end_of_stream: bool,
    ) {
        let Some(mut response) = self.responses.get_mut(booking_id) else {
            return;
        };
        let progress = if streaming {
            response.ingest_sse(chunk, end_of_stream)
        } else if end_of_stream {
            response.ingest_unary(chunk)
        } else {
            Vec::new()
        };
        drop(response);
        for progress in progress {
            self.dispatch.progress(booking_id, progress);
        }
    }

    /// The ext_proc stream for `booking_id` ended: complete the host's stream
    /// when the response finished, and abort it otherwise.
    pub(crate) fn on_request_complete(&self, booking_id: &str) {
        let Some((_, response)) = self.responses.remove(booking_id) else {
            return;
        };
        if response.finished {
            self.dispatch.complete(booking_id);
        } else {
            self.dispatch.abort(
                booking_id,
                "gateway stream ended before the response finished",
            );
        }
    }
}

struct AbortOnDrop<'a> {
    dispatch: &'a GatewayDispatch,
    request_id: &'a str,
    armed: bool,
}

impl Drop for AbortOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.dispatch
                .abort(self.request_id, "pick did not complete");
        }
    }
}

fn is_multi_prompt_batch(prompt: &Prompt) -> bool {
    match prompt {
        Prompt::StringArray(prompts) => prompts.len() > 1,
        Prompt::ArrayOfIntegerArray(prompts) => prompts.len() > 1,
        Prompt::String(_) | Prompt::IntegerArray(_) => false,
    }
}

fn header_map(headers: &[(String, String)]) -> HeaderMap {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            map.append(name, value);
        }
    }
    map
}

/// Response progress parsed from the gateway's copy of an OpenAI response.
#[derive(Default)]
struct ResponseProgress {
    pending: Vec<u8>,
    reason: Option<FinishReason>,
    usage: Option<CompletionUsage>,
    finished: bool,
}

impl ResponseProgress {
    /// Count one token per content-bearing choice in each complete SSE event.
    /// Envoy may split events across chunks, so incomplete lines are held.
    fn ingest_sse(&mut self, chunk: &[u8], end_of_stream: bool) -> Vec<GatewayProgress> {
        self.pending.extend_from_slice(chunk);
        let complete = if end_of_stream {
            self.pending.len()
        } else {
            match self.pending.iter().rposition(|&b| b == b'\n') {
                Some(index) => index + 1,
                None => {
                    if self.pending.len() > MAX_PENDING_SSE_BYTES {
                        self.pending.clear();
                        return vec![GatewayProgress::Tokens(1)];
                    }
                    return Vec::new();
                }
            }
        };
        let lines: Vec<u8> = self.pending.drain(..complete).collect();

        let mut tokens = 0;
        for line in lines.split(|&b| b == b'\n') {
            let Some(data) = line.trim_ascii().strip_prefix(b"data:") else {
                continue;
            };
            let data = data.trim_ascii();
            if data == b"[DONE]" {
                continue;
            }
            if let Ok(event) = serde_json::from_slice::<serde_json::Value>(data) {
                tokens += self.observe_event(&event);
            }
        }

        let mut progress = Vec::new();
        if tokens > 0 {
            progress.push(GatewayProgress::Tokens(tokens));
        }
        if end_of_stream {
            progress.push(self.finish());
        }
        progress
    }

    /// A buffered (non-streaming) response reports its output only through
    /// `usage.completion_tokens`.
    fn ingest_unary(&mut self, body: &[u8]) -> Vec<GatewayProgress> {
        let mut progress = Vec::new();
        if let Ok(response) = serde_json::from_slice::<serde_json::Value>(body) {
            self.observe_event(&response);
            let tokens = self
                .usage
                .as_ref()
                .map_or(0, |usage| usage.completion_tokens as usize);
            if tokens > 0 {
                progress.push(GatewayProgress::Tokens(tokens));
            }
        }
        progress.push(self.finish());
        progress
    }

    /// Record finish reason and usage; return the number of content deltas.
    fn observe_event(&mut self, event: &serde_json::Value) -> usize {
        if let Some(usage) = event.get("usage").filter(|usage| !usage.is_null())
            && let Ok(usage) = serde_json::from_value(usage.clone())
        {
            self.usage = Some(usage);
        }
        let Some(choices) = event.get("choices").and_then(|c| c.as_array()) else {
            return 0;
        };
        let mut tokens = 0;
        for choice in choices {
            if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
                self.reason = Some(finish_reason(reason));
            }
            let delta = choice.get("delta");
            let has_content = ["content", "reasoning_content", "tool_calls"]
                .iter()
                .any(|field| delta.and_then(|d| d.get(field)).is_some_and(non_empty))
                || choice.get("text").is_some_and(non_empty);
            if has_content {
                tokens += 1;
            }
        }
        tokens
    }

    fn finish(&mut self) -> GatewayProgress {
        self.finished = true;
        GatewayProgress::Finish {
            reason: self.reason.take(),
            usage: self.usage.take(),
        }
    }
}

fn non_empty(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Array(items) => !items.is_empty(),
        _ => true,
    }
}

fn finish_reason(reason: &str) -> FinishReason {
    match reason {
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests;
