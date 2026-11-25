// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV event translation shared by the vLLM and SGLang sidecars: engine
//! batches in, from the engine's gRPC stream or, for older engines, its ZMQ
//! publisher; Dynamo events out, through the publisher's push input.

use tonic_v14 as tonic;

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{DynamoError, KvEventSource};
use dynamo_kv_events::zmq_wire::{KvEventBatch, RawKvEvent, ZmqEventNormalizer};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, decode_zmq_kv_batch, normalize_batch};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use futures::future::{self, BoxFuture};
use futures::stream::{BoxStream, StreamExt};
use tokio_util::sync::CancellationToken;

#[doc(hidden)]
pub use dynamo_kv_events::zmq_wire as wire;

#[cfg(test)]
mod tests;

const RESUBSCRIBE_DELAY: Duration = Duration::from_secs(1);

/// One engine batch with its sequence number, or why it could not be decoded.
pub type KvBatch = anyhow::Result<(u64, KvEventBatch)>;

/// Engine batches; the stream continues after a batch that could not be decoded.
pub type KvBatchStream = BoxStream<'static, KvBatch>;

/// What `GetKvSnapshot` returned for one rank.
pub struct KvSnapshot {
    /// The stores that recreate the blocks the engine holds, parents first.
    pub stores: Vec<RawKvEvent>,
    /// The sequence number of the first batch the snapshot does not include.
    pub next_sequence: u64,
    /// False when the engine missed KV events, so blocks may be missing or stale.
    pub is_complete: bool,
}

/// Opens one rank's `SubscribeKvEvents` stream; resolves once the engine has accepted it.
pub type Subscribe =
    Box<dyn FnMut() -> BoxFuture<'static, Result<KvBatchStream, tonic::Status>> + Send>;

/// Calls one rank's `GetKvSnapshot`.
pub type GetSnapshot =
    Box<dyn FnMut() -> BoxFuture<'static, Result<KvSnapshot, tonic::Status>> + Send>;

/// One rank's `KvEvents` calls.
pub struct KvEventCalls {
    pub subscribe: Subscribe,
    pub snapshot: GetSnapshot,
}

/// Where one rank's engine KV events come from.
pub enum KvEventFeed {
    Grpc {
        stream: KvBatchStream,
        /// The snapshot `stream` continues, when the engine served one.
        snapshot: Option<KvSnapshot>,
        calls: KvEventCalls,
    },
    /// The engine's ZMQ publisher, for engines without the gRPC stream.
    Zmq { endpoint: String, topic: String },
}

/// One local rank's KV events, before the worker builds its publisher.
pub struct SidecarKvSource {
    pub dp_rank: u32,
    pub image_token_id: Option<u32>,
    pub idempotent_stores: bool,
    pub feed: KvEventFeed,
}

impl SidecarKvSource {
    /// A push source whose publisher receives this feed's translated events until `cancel`.
    pub fn into_push(self, cancel: CancellationToken) -> KvEventSource {
        let Self {
            dp_rank,
            image_token_id,
            idempotent_stores,
            mut feed,
        } = self;
        KvEventSource::Push {
            dp_rank,
            idempotent_stores,
            on_ready: Box::new(move |publisher| {
                let mut translator = Translator::new(publisher.clone(), dp_rank, image_token_id);
                // Before the worker registers: a router's first query must find these blocks.
                if let KvEventFeed::Grpc { snapshot, .. } = &mut feed {
                    translator.load(snapshot.take(), false);
                }
                let feed = tokio::spawn(cancel.run_until_cancelled_owned(translator.run(feed)));
                tokio::spawn(async move {
                    // A feed that stops by itself stops its publisher, as lib/llm's ZMQ listener does.
                    if !matches!(feed.await, Ok(None)) {
                        let _ = publisher.stop_and_wait().await;
                    }
                });
                Ok(())
            }),
        }
    }
}

/// One feed per rank: the engine's gRPC streams, each with a snapshot of the
/// rank's cache when the engine serves one, or `zmq` for an engine without the
/// `KvEvents` service (no stream listed, or UNIMPLEMENTED). No feeds when it
/// cannot serve KV events (FAILED_PRECONDITION).
pub async fn kv_event_feeds(
    grpc: impl IntoIterator<Item = (u32, KvEventCalls)>,
    zmq: impl FnOnce() -> Result<Vec<(u32, KvEventFeed)>, DynamoError>,
) -> Result<Vec<(u32, KvEventFeed)>, DynamoError> {
    let mut feeds = Vec::new();
    for (dp_rank, mut calls) in grpc {
        match open(&mut calls, dp_rank).await {
            Ok((snapshot, stream)) => {
                let feed = KvEventFeed::Grpc {
                    stream,
                    snapshot,
                    calls,
                };
                feeds.push((dp_rank, feed));
            }
            Err(status) if status.code() == tonic::Code::Unimplemented => return zmq(),
            Err(status) if status.code() == tonic::Code::FailedPrecondition => {
                tracing::warn!(
                    dp_rank,
                    %status,
                    "The engine advertises KV events but cannot serve them; KV events are off"
                );
                return Ok(Vec::new());
            }
            Err(status) => {
                return Err(crate::error::status_to_dynamo_v14(
                    "SubscribeKvEvents",
                    status,
                ));
            }
        }
    }
    if feeds.is_empty() { zmq() } else { Ok(feeds) }
}

/// Opens the rank's stream, then takes a snapshot of its cache: an engine's
/// snapshot holds every batch it sent a subscriber before the call, so the
/// stream has every later one.
async fn open(
    calls: &mut KvEventCalls,
    dp_rank: u32,
) -> Result<(Option<KvSnapshot>, KvBatchStream), tonic::Status> {
    let batches = (calls.subscribe)().await?;
    let snapshot = match (calls.snapshot)().await {
        Ok(snapshot) => {
            if !snapshot.is_complete {
                tracing::warn!(dp_rank, "The engine's KV snapshot is incomplete");
            }
            Some(snapshot)
        }
        Err(status)
            if matches!(
                status.code(),
                tonic::Code::Unimplemented | tonic::Code::FailedPrecondition
            ) =>
        {
            tracing::info!(dp_rank, %status, "The engine serves no KV snapshot");
            None
        }
        Err(status) => {
            tracing::warn!(dp_rank, %status, "KV snapshot failed; continuing without it");
            None
        }
    };
    let batches = match &snapshot {
        Some(snapshot) => {
            let next = snapshot.next_sequence;
            // The batches the stream delivered while the snapshot was taken are in the snapshot.
            let is_in_snapshot =
                move |batch: &KvBatch| matches!(batch, Ok((sequence, _)) if *sequence < next);
            batches
                .filter(move |batch| future::ready(!is_in_snapshot(batch)))
                .boxed()
        }
        None => batches,
    };
    Ok((snapshot, batches))
}

/// A `SubscribeKvEvents` response stream as engine batches; `batch` is `None` for a response without one.
pub fn grpc_batches<T: Send + 'static>(
    stream: tonic::Streaming<T>,
    batch: fn(T) -> Option<KvBatch>,
) -> KvBatchStream {
    stream
        .scan((), move |_, response| {
            future::ready(match response {
                Ok(response) => Some(batch(response)),
                Err(status) => {
                    tracing::warn!(%status, "KV event stream failed");
                    None
                }
            })
        })
        .filter_map(future::ready)
        .boxed()
}

/// One response of a `GetKvSnapshot` stream.
pub enum KvSnapshotItem {
    Stores(Vec<RawKvEvent>),
    /// The last response; see [`KvSnapshot`].
    End {
        next_sequence: u64,
        is_complete: bool,
    },
}

/// A `GetKvSnapshot` response stream as a snapshot; `item` is `None` for a
/// response without an item. A stream that ends without its end is an error.
pub async fn grpc_snapshot<T>(
    mut responses: tonic::Streaming<T>,
    item: fn(T) -> Option<anyhow::Result<KvSnapshotItem>>,
) -> Result<KvSnapshot, tonic::Status> {
    let mut stores = Vec::new();
    while let Some(response) = responses.message().await? {
        match item(response).transpose() {
            Ok(Some(KvSnapshotItem::Stores(more))) => stores.extend(more),
            Ok(Some(KvSnapshotItem::End {
                next_sequence,
                is_complete,
            })) => {
                return Ok(KvSnapshot {
                    stores,
                    next_sequence,
                    is_complete,
                });
            }
            Ok(None) => {}
            Err(error) => return Err(tonic::Status::internal(format!("{error:#}"))),
        }
    }
    Err(tonic::Status::internal("the KV snapshot has no end"))
}

/// The engine's ZMQ publisher as engine batches, through lib/llm's ZMQ listener's decoder.
async fn zmq_batches(endpoint: &str, topic: &str) -> anyhow::Result<KvBatchStream> {
    let socket = dynamo_llm::utils::zmq::connect_sub_socket(endpoint, Some(topic)).await?;
    let endpoint = endpoint.to_string();
    Ok(socket
        .scan((), move |_, frames| {
            future::ready(match frames {
                Ok(frames) => {
                    let frames = frames.into_iter().map(|frame| frame.to_vec()).collect();
                    Some(decode_zmq_kv_batch(frames).map(|zmq| (zmq.source_cursor, zmq.batch)))
                }
                Err(error) => {
                    tracing::warn!(%endpoint, %error, "ZMQ KV event receive failed");
                    None
                }
            })
        })
        .boxed())
}

/// What lib/llm's ZMQ listener does with each batch, for one rank's feed.
struct Translator {
    publisher: Arc<KvEventPublisher>,
    normalizer: ZmqEventNormalizer,
    worker: WorkerWithDpRank,
    /// The sequence number the next batch must have, once known.
    next_sequence: Option<u64>,
    /// The engine served a snapshot for the stream being read, so it can serve another.
    has_snapshot: bool,
}

impl Translator {
    fn new(publisher: Arc<KvEventPublisher>, dp_rank: u32, image_token_id: Option<u32>) -> Self {
        Self {
            normalizer: ZmqEventNormalizer::new(publisher.kv_block_size())
                .with_image_token_id(image_token_id),
            worker: WorkerWithDpRank::new(publisher.worker_id(), dp_rank),
            next_sequence: None,
            has_snapshot: false,
            publisher,
        }
    }

    /// Returns when the feed or the publisher has stopped for good.
    async fn run(mut self, feed: KvEventFeed) {
        let dp_rank = self.worker.dp_rank;
        let (mut stream, mut calls) = match feed {
            KvEventFeed::Grpc { stream, calls, .. } => {
                tracing::info!(dp_rank, "KV events over gRPC");
                (stream, Some(calls))
            }
            KvEventFeed::Zmq { endpoint, topic } => match zmq_batches(&endpoint, &topic).await {
                Ok(stream) => {
                    tracing::info!(
                        dp_rank,
                        %endpoint,
                        "The engine serves no gRPC KV event stream; reading its ZMQ publisher"
                    );
                    (stream, None)
                }
                Err(error) => {
                    tracing::error!(dp_rank, %endpoint, %error, "ZMQ KV event subscription failed");
                    return;
                }
            },
        };
        let mut delivered = false;
        loop {
            let is_lost = match stream.next().await {
                Some(Ok((_, batch)))
                    if batch.data_parallel_rank.map(i32::cast_unsigned) != Some(dp_rank) =>
                {
                    tracing::warn!(dp_rank, batch_rank = ?batch.data_parallel_rank, "Ignoring mismatched KV source rank");
                    false
                }
                Some(Ok((sequence, batch))) => {
                    let is_next = self.next_sequence.is_none_or(|next| next == sequence);
                    self.next_sequence = sequence.checked_add(1);
                    if !is_next {
                        tracing::warn!(
                            source_cursor = sequence,
                            dp_rank,
                            "KV event sequence discontinuity"
                        );
                    }
                    if !self.publish(batch) {
                        return;
                    }
                    delivered = true;
                    // After a snapshot, a stream that skips a batch is lost, like one that ends.
                    !is_next && self.has_snapshot
                }
                Some(Err(error)) => {
                    tracing::warn!(dp_rank, %error, "Failed to decode KV event batch");
                    // Its sequence number is unknown, so the next batch is not checked against it.
                    self.next_sequence = None;
                    false
                }
                None => true,
            };
            if !is_lost {
                continue;
            }
            // Like lib/llm's ZMQ listener, a ZMQ feed is not reconnected.
            let Some(calls) = calls.as_mut() else {
                tracing::error!(dp_rank, "ZMQ KV event stream ended; stopping the relay");
                return;
            };
            tracing::warn!(dp_rank, "KV event stream lost; reopening it");
            // At once only after a delivered batch and with no snapshot to reload: reloads are spaced.
            let needs_wait = !delivered || self.has_snapshot;
            let Some(reopened) = self.reopen(calls, needs_wait).await else {
                return;
            };
            stream = reopened;
            delivered = false;
        }
    }

    /// Reopens a lost stream, at once unless `needs_wait`, and replaces what the
    /// worker holds with the engine's snapshot when it serves one. `None` once
    /// the engine serves no KV events or the publisher has stopped.
    async fn reopen(
        &mut self,
        calls: &mut KvEventCalls,
        mut needs_wait: bool,
    ) -> Option<KvBatchStream> {
        let dp_rank = self.worker.dp_rank;
        loop {
            if needs_wait {
                tokio::time::sleep(RESUBSCRIBE_DELAY).await;
            }
            needs_wait = true;
            match open(calls, dp_rank).await {
                Ok((snapshot, stream)) => return self.load(snapshot, true).then_some(stream),
                Err(status)
                    if matches!(
                        status.code(),
                        tonic::Code::Unimplemented | tonic::Code::FailedPrecondition
                    ) =>
                {
                    tracing::warn!(dp_rank, %status, "The engine stopped serving KV events");
                    return None;
                }
                Err(status) => {
                    tracing::warn!(dp_rank, %status, "KV event subscription failed; retrying");
                }
            }
        }
    }

    /// Publishes a newly opened stream's snapshot, if it has one, after a clear of
    /// the worker's blocks when `needs_clear`. False once the publisher has stopped.
    fn load(&mut self, snapshot: Option<KvSnapshot>, needs_clear: bool) -> bool {
        self.next_sequence = snapshot.as_ref().map(|snapshot| snapshot.next_sequence);
        self.has_snapshot = snapshot.is_some();
        let Some(snapshot) = snapshot else {
            return true;
        };
        let blocks = snapshot.stores.iter().map(|store| match store {
            RawKvEvent::BlockStored { block_hashes, .. } => block_hashes.len(),
            _ => 0,
        });
        tracing::info!(
            dp_rank = self.worker.dp_rank,
            blocks = blocks.sum::<usize>(),
            next_sequence = snapshot.next_sequence,
            "Loading the engine's KV snapshot"
        );
        let clear = needs_clear.then_some(RawKvEvent::AllBlocksCleared { ownership: None });
        let events = clear.into_iter().chain(snapshot.stores).collect();
        self.publish(KvEventBatch {
            ts: 0.0,
            events,
            data_parallel_rank: Some(self.worker.dp_rank.cast_signed()),
        })
    }

    /// Translate and push one batch. False once the publisher has stopped.
    fn publish(&mut self, batch: KvEventBatch) -> bool {
        let mut events = Vec::new();
        normalize_batch(
            batch,
            &mut self.normalizer,
            self.worker,
            || self.publisher.next_event_id(),
            &mut events,
        );
        let events = events
            .into_iter()
            .map(|event| (event.event, event.placement.tier, event.session_id))
            .collect();
        self.publisher
            .publish_batch_with_session_ids(events)
            .is_ok()
    }
}
