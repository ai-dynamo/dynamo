// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-owned request counters exposed as a canonical engine stats stream.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, Method, header};
use axum::response::Response;
use axum::routing::get;
use futures::Stream;
use serde::Serialize;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::discovery::ModelManager;

use super::{RouteDoc, service_v2};

const STATS_PATH: &str = "/v1/stats/stream";
const DEFAULT_CHANNEL_CAPACITY: usize = 1024;
const KV_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone)]
pub(crate) struct EngineStats {
    tx: broadcast::Sender<RequestStatsEvent>,
}

impl Default for EngineStats {
    fn default() -> Self {
        let (tx, _) = broadcast::channel(DEFAULT_CHANNEL_CAPACITY);
        Self { tx }
    }
}

impl EngineStats {
    fn publish(
        &self,
        request_id: &str,
        model: &str,
        tokens_processed: Option<u64>,
        tokens_generated: Option<u64>,
        finished: bool,
    ) {
        if self.tx.receiver_count() == 0 {
            return;
        }

        // Disconnecting between receiver_count and send is expected telemetry loss.
        let _ = self.tx.send(RequestStatsEvent {
            v: 1,
            event_type: "stats",
            request_id: request_id.to_owned(),
            model: model.to_owned(),
            tokens_processed,
            tokens_generated,
            finished,
        });
    }

    fn subscribe(&self) -> broadcast::Receiver<RequestStatsEvent> {
        self.tx.subscribe()
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RequestStatsEvent {
    pub(crate) v: u8,
    #[serde(rename = "type")]
    pub(crate) event_type: &'static str,
    pub(crate) request_id: String,
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tokens_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tokens_generated: Option<u64>,
    pub(crate) finished: bool,
}

pub(crate) enum StatsUpdate {
    Request(RequestStatsEvent),
    Kv(super::kv_stats::KvStatsSnapshot),
}

pub(super) struct RequestStats {
    stats: EngineStats,
    request_id: String,
    model: String,
    tokens_processed: Option<u64>,
    tokens_generated: Option<u64>,
}

impl RequestStats {
    pub(super) fn new(stats: EngineStats, request_id: &str, model: &str) -> Self {
        Self {
            stats,
            request_id: request_id.to_owned(),
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
        self.stats.publish(
            &self.request_id,
            &self.model,
            tokens_processed,
            tokens_generated,
            finished,
        );
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
    stats_stream_response(
        state.engine_stats().clone(),
        state.manager_clone(),
        state.cancel_token().clone(),
    )
}

pub(crate) type StatsStream = Pin<Box<dyn Stream<Item = StatsUpdate> + Send>>;

pub(crate) fn stats_stream(
    stats: EngineStats,
    manager: Arc<ModelManager>,
    shutdown: CancellationToken,
) -> StatsStream {
    Box::pin(async_stream::stream! {
        let mut receiver = stats.subscribe();
        let mut snapshots = tokio::time::interval(KV_SNAPSHOT_INTERVAL);
        snapshots.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                event = receiver.recv() => match event {
                    Ok(event) => yield StatsUpdate::Request(event),
                    Err(broadcast::error::RecvError::Lagged(dropped)) => {
                        tracing::warn!(dropped, "stats stream subscriber lagged; closing stream for a fresh subscription");
                        break;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                _ = snapshots.tick() => {
                    yield StatsUpdate::Kv(super::kv_stats::build_snapshot(&manager));
                }
            }
        }
    })
}

fn stats_stream_response(
    stats: EngineStats,
    manager: Arc<ModelManager>,
    shutdown: CancellationToken,
) -> Response {
    let source = stats_stream(stats, manager, shutdown);
    let stream = async_stream::stream! {
        futures::pin_mut!(source);
        while let Some(update) = futures::StreamExt::next(&mut source).await {
            let line = match update {
                StatsUpdate::Request(event) => json_line(&event),
                StatsUpdate::Kv(snapshot) => json_line(&snapshot),
            };
            yield Ok::<Bytes, Infallible>(line);
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

fn json_line(value: &impl Serialize) -> Bytes {
    let mut line = serde_json::to_vec(value).expect("stats stream DTO must serialize");
    line.push(b'\n');
    Bytes::from(line)
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde_json::Value;
    use tokio::sync::broadcast::error::TryRecvError;

    use super::*;

    fn json(value: &impl Serialize) -> Value {
        serde_json::to_value(value).unwrap()
    }

    #[tokio::test]
    async fn request_stats_are_cumulative_and_finish_once() {
        let stats = EngineStats::default();
        let mut receiver = stats.subscribe();
        {
            let mut request = RequestStats::new(stats.clone(), "dynamo-id", "dynamo-model");
            request.observe(3, 0);
            request.observe(3, 2);
            request.observe(3, 3);
            request.observe(3, 0);
        }

        let processed = json(&receiver.recv().await.unwrap());
        assert_eq!(processed["request_id"], "dynamo-id");
        assert!(processed.get("correlation_id").is_none());
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
            let mut request = RequestStats::new(stats.clone(), "dynamo-id", "dynamo-model");
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
        let request = RequestStats::new(stats.clone(), "dynamo-id", "dynamo-model");

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
        let response = stats_stream_response(
            EngineStats::default(),
            Arc::new(ModelManager::new()),
            shutdown.clone(),
        );
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/x-ndjson"
        );

        let mut body = response.into_body().into_data_stream();
        let first = body.next().await.unwrap().unwrap();
        let first: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(first["type"], "kv_stats_snapshot");
        assert_eq!(first["models"], serde_json::json!([]));
        shutdown.cancel();
        assert!(body.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn idle_stream_emits_periodic_kv_snapshots() {
        let stats = EngineStats::default();
        let response = stats_stream_response(
            stats.clone(),
            Arc::new(ModelManager::new()),
            CancellationToken::new(),
        );
        let mut body = response.into_body().into_data_stream();
        let first = body.next().await.unwrap().unwrap();

        let next = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .expect("periodic snapshot should arrive")
            .unwrap()
            .unwrap();
        let first: Value = serde_json::from_slice(&first).unwrap();
        let next: Value = serde_json::from_slice(&next).unwrap();
        assert_eq!(next["type"], "kv_stats_snapshot");
        assert_ne!(first["snapshot_id"], next["snapshot_id"]);
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
        let line = stats.chunk().await.unwrap().unwrap();
        let value: Value = serde_json::from_slice(&line).unwrap();
        assert_eq!(value["type"], "kv_stats_snapshot");

        drop(stats);
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("frontend should stop after cancellation")
            .unwrap()
            .unwrap();
    }
}
