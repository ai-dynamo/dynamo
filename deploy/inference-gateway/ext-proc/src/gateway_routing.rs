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

use crate::picker::{PickError, PickResult, ResponseEvent, ResponseUsage};

/// Enables this path for the runtime EPP.
pub const ROUTING_HOST_ENV: &str = "DYN_EPP_ROUTING_HOST";

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

    /// A non-2xx upstream status marks the response failed.
    pub(crate) fn on_response_headers(&self, booking_id: &str, status: u16) {
        if !(200..300).contains(&status)
            && let Some(mut response) = self.responses.get_mut(booking_id)
        {
            response.failed = true;
        }
    }

    /// Relay the output progress one response event reports.
    pub(crate) fn on_response_event(&self, booking_id: &str, event: &ResponseEvent) {
        let Some(mut response) = self.responses.get_mut(booking_id) else {
            return;
        };
        let tokens = response.observe(event);
        drop(response);
        if tokens > 0 {
            self.dispatch
                .progress(booking_id, GatewayProgress::Tokens(tokens));
        }
    }

    /// The response finished: relay its finish unless it failed.
    pub(crate) fn on_response_end(&self, booking_id: &str) {
        let Some(mut response) = self.responses.get_mut(booking_id) else {
            return;
        };
        let progress = response.finish();
        drop(response);
        for progress in progress {
            self.dispatch.progress(booking_id, progress);
        }
    }

    /// The ext_proc stream for `booking_id` ended. Complete the host's stream
    /// for a response that finished cleanly; abort it for one that failed or
    /// ended early, so the request is recorded as aborted, not completed.
    pub(crate) fn on_request_complete(&self, booking_id: &str) {
        let Some((_, response)) = self.responses.remove(booking_id) else {
            return;
        };
        match response.outcome() {
            Ok(()) => self.dispatch.complete(booking_id),
            Err(reason) => self.dispatch.abort(booking_id, reason),
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

/// What the response has reported so far for one routed request.
#[derive(Default)]
struct ResponseProgress {
    tokens: usize,
    reason: Option<FinishReason>,
    usage: Option<ResponseUsage>,
    failed: bool,
    finished: bool,
}

impl ResponseProgress {
    /// Record one event; returns the output tokens it reports.
    fn observe(&mut self, event: &ResponseEvent) -> usize {
        self.failed |= event.error;
        if let Some(reason) = &event.finish_reason {
            self.reason = Some(finish_reason(reason));
        }
        if let Some(usage) = &event.usage {
            self.usage = Some(usage.clone());
        }
        self.tokens += event.content_deltas;
        event.content_deltas
    }

    /// The progress that ends a successful response: output the events did not
    /// show (all of it for a buffered body, per `usage`), then the finish.
    /// How the request ends: completed only when the response finished
    /// without a failure status or error event.
    fn outcome(&self) -> Result<(), &'static str> {
        match (self.finished, self.failed) {
            (_, true) => Err("the upstream response failed"),
            (false, false) => Err("gateway stream ended before the response finished"),
            (true, false) => Ok(()),
        }
    }

    fn finish(&mut self) -> Vec<GatewayProgress> {
        self.finished = true;
        if self.failed {
            return Vec::new();
        }
        let mut progress = Vec::with_capacity(2);
        let reported = self
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens)
            .map_or(0, |tokens| usize::try_from(tokens).unwrap_or(usize::MAX));
        if reported > self.tokens {
            progress.push(GatewayProgress::Tokens(reported - self.tokens));
            self.tokens = reported;
        }
        progress.push(GatewayProgress::Finish {
            reason: self.reason.take(),
            usage: self.usage.as_ref().map(completion_usage),
        });
        progress
    }
}

fn completion_usage(usage: &ResponseUsage) -> CompletionUsage {
    let count = |tokens: Option<u64>| u32::try_from(tokens.unwrap_or(0)).unwrap_or(u32::MAX);
    let prompt_tokens = count(usage.prompt_tokens);
    let completion_tokens = count(usage.completion_tokens);
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: usage
            .total_tokens
            .map_or(prompt_tokens.saturating_add(completion_tokens), |total| {
                count(Some(total))
            }),
        ..Default::default()
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
