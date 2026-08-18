// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned request counters exposed as a canonical engine stats stream.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, header};
use axum::response::Response;
use axum::routing::get;
use dynamo_runtime::pipeline::Context;
use serde::Serialize;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use super::{RouteDoc, service_v2};

const STATS_PATH: &str = "/v1/stats/stream";
const CORRELATION_ID_HEADER: &str = "x-dynamo-stats-correlation-id";
const CORRELATION_ID_CONTEXT_KEY: &str = "engine_stats_correlation_id";
const DEFAULT_CHANNEL_CAPACITY: usize = 1024;
const PING_INTERVAL: Duration = Duration::from_secs(15);
const PING_LINE: &[u8] = b"{\"v\":1,\"type\":\"ping\"}\n";

pub(super) fn attach_correlation_id<T: Send + Sync + 'static>(
    request: &mut Context<T>,
    headers: &HeaderMap,
) {
    let Some(correlation_id) = headers
        .get(CORRELATION_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return;
    };
    request.insert(CORRELATION_ID_CONTEXT_KEY, correlation_id.to_owned());
}

pub(super) fn correlation_id<T: Send + Sync + 'static>(request: &Context<T>) -> Option<String> {
    request
        .get::<String>(CORRELATION_ID_CONTEXT_KEY)
        .ok()
        .map(|value| value.as_ref().clone())
}

#[derive(Clone)]
pub(super) struct EngineStats {
    tx: broadcast::Sender<Bytes>,
}

impl Default for EngineStats {
    fn default() -> Self {
        let (tx, _) = broadcast::channel(DEFAULT_CHANNEL_CAPACITY);
        Self { tx }
    }
}

impl EngineStats {
    fn publish(&self, event: RequestStatsEvent<'_>) -> serde_json::Result<()> {
        if self.tx.receiver_count() == 0 {
            return Ok(());
        }

        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');

        // Disconnecting between receiver_count and send is expected telemetry loss.
        let _ = self.tx.send(Bytes::from(line));
        Ok(())
    }

    fn subscribe(&self) -> broadcast::Receiver<Bytes> {
        self.tx.subscribe()
    }
}

#[derive(Serialize)]
struct RequestStatsEvent<'a> {
    v: u8,
    #[serde(rename = "type")]
    event_type: &'static str,
    request_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    correlation_id: Option<&'a str>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_generated: Option<u64>,
    finished: bool,
}

pub(super) struct RequestStats {
    stats: EngineStats,
    request_id: String,
    correlation_id: Option<String>,
    model: String,
    tokens_processed: Option<u64>,
    tokens_generated: Option<u64>,
}

impl RequestStats {
    pub(super) fn new(
        stats: EngineStats,
        request_id: &str,
        correlation_id: Option<String>,
        model: &str,
    ) -> Self {
        Self {
            stats,
            request_id: request_id.to_owned(),
            correlation_id,
            model: model.to_owned(),
            tokens_processed: None,
            tokens_generated: None,
        }
    }

    pub(super) fn observe(&mut self, input_tokens: usize, generated_tokens: usize) {
        let input_tokens = u64::try_from(input_tokens).unwrap_or(u64::MAX);
        let generated_tokens = u64::try_from(generated_tokens).unwrap_or(u64::MAX);

        // Prompt-embedding paths can report zero until backend usage arrives.
        // Publish only positive, monotonic counts so a later exact value can win.
        let previous = self.tokens_processed.unwrap_or_default();
        let tokens_processed = (input_tokens > previous).then(|| {
            self.tokens_processed = Some(input_tokens);
            input_tokens
        });
        let tokens_generated = if generated_tokens > 0 {
            let previous = self.tokens_generated.unwrap_or_default();
            let total = previous.saturating_add(generated_tokens);
            self.tokens_generated = Some(total);
            // Equality means the counter saturated; do not publish a non-increasing total.
            (total > previous).then_some(total)
        } else {
            None
        };

        if tokens_processed.is_some() || tokens_generated.is_some() {
            self.publish(tokens_processed, tokens_generated, false);
        }
    }

    fn publish(
        &self,
        tokens_processed: Option<u64>,
        tokens_generated: Option<u64>,
        finished: bool,
    ) {
        let result = self.stats.publish(RequestStatsEvent {
            v: 1,
            event_type: "stats",
            request_id: &self.request_id,
            correlation_id: self.correlation_id.as_deref(),
            model: &self.model,
            tokens_processed,
            tokens_generated,
            finished,
        });
        if let Err(error) = result {
            tracing::debug!(
                request_id = %self.request_id,
                %error,
                "failed to serialize request stats"
            );
        }
    }
}

impl Drop for RequestStats {
    fn drop(&mut self) {
        self.publish(self.tokens_processed, self.tokens_generated, true);
    }
}

pub(super) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let docs = vec![RouteDoc::new(Method::GET, STATS_PATH)];
    let router = Router::new()
        .route(STATS_PATH, get(stats_stream_handler))
        .with_state(state);
    (docs, router)
}

async fn stats_stream_handler(State(state): State<Arc<service_v2::State>>) -> Response {
    stats_stream_response(state.engine_stats().clone(), state.cancel_token().clone())
}

fn stats_stream_response(stats: EngineStats, shutdown: CancellationToken) -> Response {
    let mut receiver = stats.subscribe();
    let stream = async_stream::stream! {
        let mut ping = tokio::time::interval_at(
            tokio::time::Instant::now() + PING_INTERVAL,
            PING_INTERVAL,
        );
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        yield Ok::<Bytes, Infallible>(Bytes::from_static(PING_LINE));

        loop {
            tokio::select! {
                event = receiver.recv() => match event {
                    Ok(line) => yield Ok(line),
                    Err(broadcast::error::RecvError::Lagged(dropped)) => {
                        tracing::debug!(dropped, "engine stats stream subscriber lagged");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                _ = ping.tick() => yield Ok(Bytes::from_static(PING_LINE)),
                _ = shutdown.cancelled() => break,
            }
        }
    };

    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::Value;
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;

    fn json(line: &Bytes) -> Value {
        serde_json::from_slice(line).unwrap()
    }

    #[test]
    fn correlation_id_is_separate_from_the_native_request_id() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CORRELATION_ID_HEADER,
            HeaderValue::from_static("dynamo-correlation"),
        );
        let mut context =
            Context::with_id_and_metadata((), "dynamo-request".to_string(), Default::default());

        attach_correlation_id(&mut context, &headers);

        assert_eq!(context.id(), "dynamo-request");
        assert_eq!(
            correlation_id(&context).as_deref(),
            Some("dynamo-correlation")
        );
    }

    #[tokio::test]
    async fn request_stats_are_cumulative_and_finish_once() {
        let stats = EngineStats::default();
        let mut receiver = stats.subscribe();
        {
            let mut request = RequestStats::new(
                stats.clone(),
                "dynamo-id",
                Some("dynamo-correlation".to_string()),
                "dynamo-model",
            );
            request.observe(3, 0);
            request.observe(3, 2);
            request.observe(3, 3);
            request.observe(3, 0);
        }

        let processed = json(&receiver.recv().await.unwrap());
        assert_eq!(processed["request_id"], "dynamo-id");
        assert_eq!(processed["correlation_id"], "dynamo-correlation");
        assert_eq!(processed["model"], "dynamo-model");
        assert_eq!(processed["tokens_processed"], 3);
        assert!(processed.get("tokens_generated").is_none());
        assert_eq!(processed["finished"], false);

        let generated = json(&receiver.recv().await.unwrap());
        assert_eq!(generated["tokens_generated"], 2);
        assert!(generated.get("tokens_processed").is_none());
        assert_eq!(generated["finished"], false);

        let generated = json(&receiver.recv().await.unwrap());
        assert_eq!(generated["tokens_generated"], 5);
        assert_eq!(generated["finished"], false);

        let finished = json(&receiver.recv().await.unwrap());
        assert_eq!(finished["tokens_processed"], 3);
        assert_eq!(finished["tokens_generated"], 5);
        assert_eq!(finished["finished"], true);
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn delayed_input_count_is_published_when_known() {
        let stats = EngineStats::default();
        let mut receiver = stats.subscribe();
        {
            let mut request = RequestStats::new(stats.clone(), "dynamo-id", None, "dynamo-model");
            request.observe(0, 2);
            request.observe(12, 3);
        }

        let generated = json(&receiver.recv().await.unwrap());
        assert!(generated.get("tokens_processed").is_none());
        assert_eq!(generated["tokens_generated"], 2);

        let processed = json(&receiver.recv().await.unwrap());
        assert_eq!(processed["tokens_processed"], 12);
        assert_eq!(processed["tokens_generated"], 5);

        let finished = json(&receiver.recv().await.unwrap());
        assert_eq!(finished["tokens_processed"], 12);
        assert_eq!(finished["tokens_generated"], 5);
        assert_eq!(finished["finished"], true);
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn unobserved_request_finishes_without_counters() {
        let stats = EngineStats::default();
        let mut receiver = stats.subscribe();
        let request = RequestStats::new(stats.clone(), "dynamo-id", None, "dynamo-model");

        drop(request);

        let finished = json(&receiver.recv().await.unwrap());
        assert_eq!(finished["finished"], true);
        assert!(finished.get("tokens_processed").is_none());
        assert!(finished.get("tokens_generated").is_none());
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn stream_is_ready_immediately_and_closes_on_shutdown() {
        let shutdown = CancellationToken::new();
        let response = stats_stream_response(EngineStats::default(), shutdown.clone());
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/x-ndjson"
        );

        let mut body = response.into_body().into_data_stream();
        assert_eq!(
            body.next().await.unwrap().unwrap(),
            Bytes::from_static(PING_LINE)
        );
        shutdown.cancel();
        assert!(body.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn idle_stream_emits_periodic_pings() {
        let stats = EngineStats::default();
        let response = stats_stream_response(stats.clone(), CancellationToken::new());
        let mut body = response.into_body().into_data_stream();
        assert_eq!(
            body.next().await.unwrap().unwrap(),
            Bytes::from_static(PING_LINE)
        );

        let next = tokio::time::timeout(Duration::from_secs(16), body.next())
            .await
            .expect("periodic ping should arrive")
            .unwrap()
            .unwrap();
        assert_eq!(next, Bytes::from_static(PING_LINE));
    }

    #[tokio::test]
    async fn frontend_serves_models_and_stats_on_one_listener_before_readiness() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let service = service_v2::HttpService::builder()
            .port(port)
            .build()
            .unwrap();
        let shutdown = CancellationToken::new();
        let handle = service
            .spawn_with_listener(shutdown.clone(), listener)
            .await;

        let client = reqwest::Client::new();
        let models = client
            .get(format!("http://127.0.0.1:{port}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(models.status(), reqwest::StatusCode::OK);

        let mut stats = client
            .get(format!("http://127.0.0.1:{port}{STATS_PATH}"))
            .send()
            .await
            .unwrap();
        assert_eq!(stats.status(), reqwest::StatusCode::OK);
        assert_eq!(
            stats.headers().get(reqwest::header::CONTENT_TYPE).unwrap(),
            "application/x-ndjson"
        );
        assert_eq!(stats.chunk().await.unwrap().unwrap(), PING_LINE);

        drop(stats);
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("frontend should stop after cancellation")
            .unwrap()
            .unwrap();
    }
}
