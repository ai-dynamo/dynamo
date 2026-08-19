// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public read-only stream of canonical KV placement state.

use std::collections::HashSet;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, Method, header};
use axum::response::Response;
use axum::routing::get;
use dynamo_kv_router::protocols::RouterEvent;
use dynamo_runtime::protocols::EndpointId;
use futures::{Stream, StreamExt, stream::SelectAll};
use serde::Serialize;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::discovery::ModelManager;
use crate::kv_router::indexer::{
    PlacementFeed, PlacementStream as PlacementFeedStream, PlacementUpdate,
};

use super::{RouteDoc, service_v2};

const KV_PLACEMENTS_PATH: &str = "/v1/kv-cache/placements/stream";
const MAX_EVENTS_PER_LINE: usize = 64;
const PLACEMENT_BUFFER_LINES: usize = 4;
const SLOW_CONSUMER_TIMEOUT: Duration = Duration::from_secs(5);
static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) type PlacementUpdateStream = Pin<Box<dyn Stream<Item = PlacementStreamEvent> + Send>>;

#[derive(Clone)]
struct PlacementSource {
    model: String,
    block_size_tokens: u32,
    endpoint: EndpointId,
    feed: PlacementFeed,
}

struct PlacementSession {
    source: PlacementSource,
    stream: PlacementFeedStream,
    cursor: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct PlacementSourceInfo {
    pub(crate) model: String,
    pub(crate) block_size_tokens: u32,
    pub(crate) endpoint: EndpointId,
}

#[derive(Debug)]
pub(crate) struct PlacementEventBatch {
    pub(crate) snapshot_id: Option<u64>,
    pub(crate) source: PlacementSourceInfo,
    pub(crate) cursor: u64,
    pub(crate) batch_index: usize,
    pub(crate) batch_count: usize,
    pub(crate) events: Vec<RouterEvent>,
}

#[derive(Debug)]
pub(crate) enum PlacementStreamEvent {
    SnapshotBegin {
        snapshot_id: u64,
    },
    SnapshotEvents(PlacementEventBatch),
    SnapshotEnd {
        snapshot_id: u64,
        complete: bool,
        cursors: Vec<PlacementCursor>,
    },
    Events(PlacementEventBatch),
    SourceError {
        source: PlacementSourceInfo,
        reason: String,
    },
}

#[derive(Serialize)]
struct SnapshotBoundary<'a> {
    v: u8,
    #[serde(rename = "type")]
    event_type: &'a str,
    snapshot_id: u64,
    complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cursor: Option<&'a [PlacementCursor]>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PlacementCursor {
    pub(crate) model: String,
    pub(crate) namespace: String,
    pub(crate) component: String,
    pub(crate) endpoint: String,
    pub(crate) cursor: u64,
}

#[derive(Serialize)]
struct PlacementEvents<'a> {
    v: u8,
    #[serde(rename = "type")]
    event_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot_id: Option<u64>,
    model: &'a str,
    namespace: &'a str,
    component: &'a str,
    endpoint: &'a str,
    block_size_tokens: u32,
    cursor: u64,
    batch_index: usize,
    batch_count: usize,
    events: &'a [RouterEvent],
}

#[derive(Serialize)]
struct PlacementError<'a> {
    v: u8,
    #[serde(rename = "type")]
    event_type: &'static str,
    model: &'a str,
    namespace: &'a str,
    component: &'a str,
    endpoint: &'a str,
    reason: &'a str,
}

struct LiveBatch {
    source: PlacementSource,
    cursor: u64,
    events: Vec<RouterEvent>,
    error: Option<String>,
}

pub(super) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let docs = vec![RouteDoc::new(Method::GET, KV_PLACEMENTS_PATH)];
    let router = Router::new()
        .route(KV_PLACEMENTS_PATH, get(kv_placements_stream_handler))
        .with_state(state);
    (docs, router)
}

async fn kv_placements_stream_handler(State(state): State<Arc<service_v2::State>>) -> Response {
    kv_placements_stream_response(state.manager_clone(), state.cancel_token().clone())
}

fn kv_placements_stream_response(
    manager: Arc<ModelManager>,
    shutdown: CancellationToken,
) -> Response {
    let source = placement_stream(manager, shutdown);
    let stream = async_stream::stream! {
        futures::pin_mut!(source);
        while let Some(event) = source.next().await {
            yield Ok::<Bytes, Infallible>(placement_json_line(&event));
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

pub(crate) fn placement_stream(
    manager: Arc<ModelManager>,
    shutdown: CancellationToken,
) -> PlacementUpdateStream {
    let source = raw_placement_stream(manager, shutdown.clone());
    let (stream, _producer) = bounded_placement_stream(
        source,
        shutdown,
        PLACEMENT_BUFFER_LINES,
        SLOW_CONSUMER_TIMEOUT,
    );
    Box::pin(stream)
}

fn raw_placement_stream(
    manager: Arc<ModelManager>,
    shutdown: CancellationToken,
) -> PlacementUpdateStream {
    let source_shutdown = shutdown.clone();
    Box::pin(async_stream::stream! {
        let snapshot_id = NEXT_SNAPSHOT_ID.fetch_add(1, Ordering::Relaxed);
        yield PlacementStreamEvent::SnapshotBegin {
            snapshot_id,
        };

        let mut complete = true;
        let mut sessions = Vec::new();
        let mut snapshot_cursor = Vec::new();
        for source in placement_sources(&manager) {
            let mut stream = match source.feed.stream().await {
                Ok(stream) => stream,
                Err(error) => {
                    tracing::warn!(model = source.model, endpoint = %source.endpoint, %error, "failed to open KV placement feed");
                    complete = false;
                    yield placement_error(&source, "feed_unavailable");
                    continue;
                }
            };
            let (cursor, events) = match stream.next().await {
                Some(Ok(PlacementUpdate::Snapshot { cursor, events })) => (cursor, events),
                Some(Ok(PlacementUpdate::Events { .. })) => {
                    complete = false;
                    yield placement_error(&source, "snapshot_missing");
                    continue;
                }
                Some(Err(error)) => {
                    tracing::warn!(model = source.model, endpoint = %source.endpoint, %error, "failed to read KV placement snapshot");
                    complete = false;
                    yield placement_error(&source, "snapshot_failed");
                    continue;
                }
                None => {
                    complete = false;
                    yield placement_error(&source, "stream_closed");
                    continue;
                }
            };

            snapshot_cursor.push(PlacementCursor {
                model: source.model.clone(),
                namespace: source.endpoint.namespace.clone(),
                component: source.endpoint.component.clone(),
                endpoint: source.endpoint.name.clone(),
                cursor,
            });
            for event in placement_event_batches(
                &source,
                Some(snapshot_id),
                cursor,
                events,
            ) {
                yield PlacementStreamEvent::SnapshotEvents(event);
            }
            sessions.push(PlacementSession {
                source,
                stream,
                cursor,
            });
        }

        snapshot_cursor.sort_unstable_by(|left, right| {
            (
                left.model.as_str(),
                left.namespace.as_str(),
                left.component.as_str(),
                left.endpoint.as_str(),
            )
                .cmp(&(
                    right.model.as_str(),
                    right.namespace.as_str(),
                    right.component.as_str(),
                    right.endpoint.as_str(),
                ))
        });
        yield PlacementStreamEvent::SnapshotEnd {
            snapshot_id,
            complete,
            cursors: snapshot_cursor,
        };

        if !complete {
            return;
        }

        let expected_sources = sessions
            .iter()
            .map(|session| source_key(&session.source))
            .collect::<Vec<_>>();
        let mut live = SelectAll::new();
        for session in sessions {
            live.push(live_source_stream(session));
        }
        let mut source_check = tokio::time::interval(Duration::from_secs(1));
        source_check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        source_check.tick().await;
        loop {
            tokio::select! {
                _ = source_shutdown.cancelled() => break,
                _ = source_check.tick() => {
                    if placement_source_keys(&manager) != expected_sources {
                        tracing::debug!("KV placement source set changed; closing stream for a fresh snapshot");
                        break;
                    }
                }
                batch = async {
                    if live.is_empty() {
                        std::future::pending().await
                    } else {
                        live.next().await
                    }
                } => {
                    let Some(batch) = batch else { break };
                    if let Some(reason) = batch.error.as_deref() {
                        yield placement_error(&batch.source, reason);
                        break;
                    }
                    for event in placement_event_batches(
                        &batch.source,
                        None,
                        batch.cursor,
                        batch.events,
                    ) {
                        yield PlacementStreamEvent::Events(event);
                    }
                }
            }
        }
    })
}

fn live_source_stream(
    mut session: PlacementSession,
) -> Pin<Box<dyn Stream<Item = LiveBatch> + Send>> {
    Box::pin(async_stream::stream! {
        loop {
            match session.stream.next().await {
                Some(Ok(PlacementUpdate::Events { cursor, events })) => {
                    let expected = session.cursor.checked_add(1);
                    if expected != Some(cursor) {
                        yield live_error(session.source, "cursor_gap");
                        break;
                    }
                    session.cursor = cursor;
                    yield LiveBatch {
                        source: session.source.clone(),
                        cursor,
                        events,
                        error: None,
                    };
                }
                Some(Ok(PlacementUpdate::Snapshot { .. })) => {
                    yield live_error(session.source, "unexpected_snapshot");
                    break;
                }
                Some(Err(error)) => {
                    tracing::warn!(model = session.source.model, endpoint = %session.source.endpoint, %error, "KV placement feed failed");
                    yield live_error(session.source, "feed_error");
                    break;
                }
                None => {
                    yield live_error(session.source, "stream_closed");
                    break;
                }
            }
        }
    })
}

fn live_error(source: PlacementSource, reason: &str) -> LiveBatch {
    LiveBatch {
        source,
        cursor: 0,
        events: Vec::new(),
        error: Some(reason.to_string()),
    }
}

fn placement_sources(manager: &ModelManager) -> Vec<PlacementSource> {
    let mut seen = HashSet::new();
    let mut sources = Vec::new();
    for view in manager.committed_model_views() {
        for worker_set in view.worker_sets {
            if worker_set.card().lora.is_some() {
                continue;
            }
            if let (Some(feed), Some(endpoint)) = (
                worker_set.kv_placement_feed().cloned(),
                worker_set.topology_endpoint().map(|endpoint| endpoint.id()),
            ) && seen.insert((view.name.clone(), endpoint.clone()))
            {
                sources.push(PlacementSource {
                    model: view.name.clone(),
                    block_size_tokens: worker_set.card().kv_cache_block_size,
                    endpoint,
                    feed,
                });
            }
            if let Some((endpoint, feed)) = worker_set.prefill_placement_feed()
                && seen.insert((view.name.clone(), endpoint.clone()))
            {
                sources.push(PlacementSource {
                    model: view.name.clone(),
                    block_size_tokens: worker_set.card().kv_cache_block_size,
                    endpoint,
                    feed,
                });
            }
        }
    }
    sources.sort_unstable_by(|left, right| {
        (
            left.model.as_str(),
            left.endpoint.namespace.as_str(),
            left.endpoint.component.as_str(),
            left.endpoint.name.as_str(),
        )
            .cmp(&(
                right.model.as_str(),
                right.endpoint.namespace.as_str(),
                right.endpoint.component.as_str(),
                right.endpoint.name.as_str(),
            ))
    });
    sources
}

fn placement_source_keys(manager: &ModelManager) -> Vec<(String, EndpointId)> {
    placement_sources(manager).iter().map(source_key).collect()
}

fn source_key(source: &PlacementSource) -> (String, EndpointId) {
    (source.model.clone(), source.endpoint.clone())
}

fn placement_event_batches(
    source: &PlacementSource,
    snapshot_id: Option<u64>,
    cursor: u64,
    events: Vec<RouterEvent>,
) -> Vec<PlacementEventBatch> {
    let batch_count = events.len().div_ceil(MAX_EVENTS_PER_LINE).max(1);
    let source = placement_source_info(source);
    let mut events = events.into_iter();
    (0..batch_count)
        .map(|batch_index| PlacementEventBatch {
            snapshot_id,
            source: source.clone(),
            cursor,
            batch_index,
            batch_count,
            events: events.by_ref().take(MAX_EVENTS_PER_LINE).collect(),
        })
        .collect()
}

#[cfg(test)]
fn event_chunks(events: &[RouterEvent]) -> impl Iterator<Item = &[RouterEvent]> {
    events.chunks(MAX_EVENTS_PER_LINE)
}

fn placement_source_info(source: &PlacementSource) -> PlacementSourceInfo {
    PlacementSourceInfo {
        model: source.model.clone(),
        block_size_tokens: source.block_size_tokens,
        endpoint: source.endpoint.clone(),
    }
}

fn placement_error(source: &PlacementSource, reason: &str) -> PlacementStreamEvent {
    PlacementStreamEvent::SourceError {
        source: placement_source_info(source),
        reason: reason.to_owned(),
    }
}

fn placement_json_line(event: &PlacementStreamEvent) -> Bytes {
    match event {
        PlacementStreamEvent::SnapshotBegin { snapshot_id } => json_line(&SnapshotBoundary {
            v: 1,
            event_type: "placement_snapshot_begin",
            snapshot_id: *snapshot_id,
            complete: false,
            cursor: None,
        }),
        PlacementStreamEvent::SnapshotEvents(batch) => {
            placement_batch_json_line("placement_snapshot_events", batch)
        }
        PlacementStreamEvent::SnapshotEnd {
            snapshot_id,
            complete,
            cursors,
        } => json_line(&SnapshotBoundary {
            v: 1,
            event_type: "placement_snapshot_end",
            snapshot_id: *snapshot_id,
            complete: *complete,
            cursor: Some(cursors),
        }),
        PlacementStreamEvent::Events(batch) => placement_batch_json_line("placement_events", batch),
        PlacementStreamEvent::SourceError { source, reason } => json_line(&PlacementError {
            v: 1,
            event_type: "placement_source_error",
            model: &source.model,
            namespace: &source.endpoint.namespace,
            component: &source.endpoint.component,
            endpoint: &source.endpoint.name,
            reason,
        }),
    }
}

fn placement_batch_json_line(event_type: &str, batch: &PlacementEventBatch) -> Bytes {
    json_line(&PlacementEvents {
        v: 1,
        event_type,
        snapshot_id: batch.snapshot_id,
        model: &batch.source.model,
        namespace: &batch.source.endpoint.namespace,
        component: &batch.source.endpoint.component,
        endpoint: &batch.source.endpoint.name,
        block_size_tokens: batch.source.block_size_tokens,
        cursor: batch.cursor,
        batch_index: batch.batch_index,
        batch_count: batch.batch_count,
        events: &batch.events,
    })
}

fn json_line(value: &impl Serialize) -> Bytes {
    let mut line = serde_json::to_vec(value).expect("placement stream DTO must serialize");
    line.push(b'\n');
    Bytes::from(line)
}

struct CancelOnDropStream<T> {
    inner: ReceiverStream<T>,
    cancel: CancellationToken,
}

impl<T> Stream for CancelOnDropStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<T> Drop for CancelOnDropStream<T> {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

fn bounded_placement_stream<T: Send + 'static>(
    mut source: Pin<Box<dyn Stream<Item = T> + Send>>,
    shutdown: CancellationToken,
    capacity: usize,
    send_timeout: Duration,
) -> (CancelOnDropStream<T>, tokio::task::JoinHandle<()>) {
    let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
    let client_cancel = CancellationToken::new();
    let producer_cancel = client_cancel.clone();
    let producer = tokio::spawn(async move {
        loop {
            let item = tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = producer_cancel.cancelled() => break,
                item = source.next() => item,
            };
            let Some(item) = item else {
                break;
            };
            let sent = tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = producer_cancel.cancelled() => break,
                sent = tokio::time::timeout(send_timeout, sender.send(item)) => sent,
            };
            match sent {
                Ok(Ok(())) => {}
                Ok(Err(_)) => break,
                Err(_) => {
                    tracing::debug!("disconnecting slow KV placement stream consumer");
                    break;
                }
            }
        }
    });
    (
        CancelOnDropStream {
            inner: ReceiverStream::new(receiver),
            cancel: client_cancel,
        },
        producer,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(event_id: u64) -> RouterEvent {
        RouterEvent::new(
            7,
            dynamo_kv_router::protocols::KvCacheEvent {
                event_id,
                data: dynamo_kv_router::protocols::KvCacheEventData::Cleared,
                dp_rank: 2,
            },
        )
    }

    #[test]
    fn placement_batches_are_bounded() {
        let events = vec![event(1); MAX_EVENTS_PER_LINE + 1];
        let lengths = event_chunks(&events)
            .map(|chunk| chunk.len())
            .collect::<Vec<_>>();
        assert_eq!(lengths, [MAX_EVENTS_PER_LINE, 1]);
    }

    #[tokio::test]
    async fn empty_placement_stream_has_a_complete_snapshot_boundary() {
        let shutdown = CancellationToken::new();
        let response =
            kv_placements_stream_response(Arc::new(ModelManager::new()), shutdown.clone());
        let mut body = response.into_body().into_data_stream();
        let begin = body.next().await.unwrap().unwrap();
        let end = body.next().await.unwrap().unwrap();
        let begin: serde_json::Value = serde_json::from_slice(&begin).unwrap();
        let end: serde_json::Value = serde_json::from_slice(&end).unwrap();
        assert_eq!(begin["type"], "placement_snapshot_begin");
        assert_eq!(end["type"], "placement_snapshot_end");
        assert_eq!(end["complete"], true);
        assert_eq!(end["cursor"], serde_json::json!([]));
        shutdown.cancel();
        assert!(body.next().await.is_none());
    }

    #[tokio::test]
    async fn slow_placement_consumer_is_disconnected() {
        let source: Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>> =
            Box::pin(futures::stream::repeat(Ok(Bytes::from_static(b"line\n"))));
        let (stream, producer) = bounded_placement_stream(
            source,
            CancellationToken::new(),
            1,
            Duration::from_millis(10),
        );

        tokio::time::timeout(Duration::from_millis(100), producer)
            .await
            .expect("producer should disconnect a consumer that does not read")
            .unwrap();
        drop(stream);
    }

    #[tokio::test]
    async fn dropping_placement_body_cancels_producer() {
        let source: Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>> =
            Box::pin(futures::stream::pending());
        let (stream, producer) =
            bounded_placement_stream(source, CancellationToken::new(), 1, Duration::from_secs(1));

        drop(stream);
        tokio::time::timeout(Duration::from_millis(100), producer)
            .await
            .expect("dropping the response body should cancel its producer")
            .unwrap();
    }
}
