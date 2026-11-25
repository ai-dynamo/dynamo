// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV event translation shared by the vLLM and SGLang sidecars: engine
//! batches in, from the engine's gRPC stream or, for older engines, its ZMQ
//! publisher; Dynamo events out, through the publisher's push input.

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
pub use anyhow;
#[doc(hidden)]
pub use dynamo_kv_events::zmq_wire as wire;
#[doc(hidden)]
pub use tonic_v14 as tonic;

#[cfg(test)]
mod tests;

const RESUBSCRIBE_DELAY: Duration = Duration::from_secs(1);

/// One engine batch with its sequence number, or why it could not be decoded.
pub type KvBatch = anyhow::Result<(u64, KvEventBatch)>;

/// Engine batches; the stream continues after a batch that could not be decoded.
pub type KvBatchStream = BoxStream<'static, KvBatch>;

/// The first batch a snapshot does not include, and where its stream resumes.
#[derive(Clone)]
pub struct KvCursor {
    pub stream_id: String,
    pub next_sequence_number: u64,
}

/// What `GetKvSnapshot` returned for one rank.
pub struct KvSnapshot {
    /// The stores that recreate the blocks the engine holds, parents first.
    pub stores: Vec<RawKvEvent>,
    pub resume: KvCursor,
    /// False when the engine missed KV events, so blocks may be missing or stale.
    pub is_complete: bool,
}

/// Opens one rank's `SubscribeKvEvents` stream, from the cursor if any.
pub type Subscribe = Box<
    dyn FnMut(Option<KvCursor>) -> BoxFuture<'static, Result<KvBatchStream, tonic::Status>> + Send,
>;

/// Calls one rank's `GetKvSnapshot`.
pub type GetSnapshot =
    Box<dyn FnMut() -> BoxFuture<'static, Result<KvSnapshot, tonic::Status>> + Send>;

/// One rank's `KvEvents` calls.
pub struct KvEventCalls {
    pub subscribe: Subscribe,
    pub snapshot: GetSnapshot,
}

/// One rank's stream, and the snapshot it follows when the engine served one.
pub struct OpenedKvEvents {
    snapshot: Option<KvSnapshot>,
    /// The sequence number of the first batch, for a stream that resumes from the snapshot.
    next_sequence: Option<u64>,
    batches: KvBatchStream,
}

/// Where one rank's engine KV events come from.
pub enum KvEventFeed {
    Grpc {
        opened: OpenedKvEvents,
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
                if let KvEventFeed::Grpc { opened, .. } = &mut feed {
                    translator.load(opened, false);
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

/// One feed per rank: the engine's gRPC streams, each after a snapshot of the
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
            Ok(opened) => feeds.push((dp_rank, KvEventFeed::Grpc { opened, calls })),
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

/// Takes a snapshot of the rank's cache and opens the stream from its cursor,
/// or without a cursor when the engine serves no snapshot or the snapshot fails.
async fn open(calls: &mut KvEventCalls, dp_rank: u32) -> Result<OpenedKvEvents, tonic::Status> {
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
    let resume = snapshot.as_ref().map(|snapshot| snapshot.resume.clone());
    let mut next_sequence = resume.as_ref().map(|cursor| cursor.next_sequence_number);
    let batches = match (calls.subscribe)(resume).await {
        // For example an engine without a replay buffer: one live subscription, not another snapshot.
        Err(status) if next_sequence.is_some() => {
            tracing::warn!(
                dp_rank,
                %status,
                "The engine cannot replay from the KV snapshot; KV events published since it are lost"
            );
            next_sequence = None;
            (calls.subscribe)(None).await?
        }
        batches => batches?,
    };
    Ok(OpenedKvEvents {
        snapshot,
        next_sequence,
        batches,
    })
}

/// A `SubscribeKvEvents` response stream as engine batches; `batch` is `None` for its start.
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
            KvEventFeed::Grpc { opened, calls } => {
                tracing::info!(dp_rank, "KV events over gRPC");
                (opened.batches, Some(calls))
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
                    delivered |= is_next;
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
            let wait = !delivered || self.has_snapshot;
            let Some(reopened) = self.reopen(calls, wait).await else {
                return;
            };
            stream = reopened;
            delivered = false;
        }
    }

    /// Reopens a lost stream, at once unless `wait`, and replaces what the
    /// worker holds with the engine's snapshot when it serves one. `None` once
    /// the engine serves no KV events or the publisher has stopped.
    async fn reopen(&mut self, calls: &mut KvEventCalls, mut wait: bool) -> Option<KvBatchStream> {
        let dp_rank = self.worker.dp_rank;
        loop {
            if wait {
                tokio::time::sleep(RESUBSCRIBE_DELAY).await;
            }
            wait = true;
            match open(calls, dp_rank).await {
                Ok(mut opened) => return self.load(&mut opened, true).then_some(opened.batches),
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

    /// Publishes the opened stream's snapshot, if it has one, after a clear of
    /// the worker's blocks when `needs_clear`. False once the publisher has stopped.
    fn load(&mut self, opened: &mut OpenedKvEvents, needs_clear: bool) -> bool {
        self.next_sequence = opened.next_sequence;
        self.has_snapshot = opened.snapshot.is_some();
        let Some(snapshot) = opened.snapshot.take() else {
            return true;
        };
        let blocks = snapshot.stores.iter().map(|store| match store {
            RawKvEvent::BlockStored { block_hashes, .. } => block_hashes.len(),
            _ => 0,
        });
        tracing::info!(
            dp_rank = self.worker.dp_rank,
            blocks = blocks.sum::<usize>(),
            next_sequence = snapshot.resume.next_sequence_number,
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

/// Defines `calls`, one rank's calls to an engine's `KvEvents` service, and
/// `batch`, which converts one response of its stream into the batch the ZMQ
/// decoder returns for the same events, through the decoder's own helpers.
/// vLLM's `vllm` and SGLang's `sglang.kv.v1` packages define the same messages.
/// Invoke it in a module of its own.
#[macro_export]
macro_rules! kv_event_proto {
    ($pb:path) => {
        use $crate::kv_events::anyhow::{Context as _, Result, bail};
        use $crate::kv_events::wire::{
            BlockHashValue, ExtraKeyItem, KvCacheSpecKind, KvEventBatch, KvTokenIds, Locality,
            RawKvEvent, extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace,
            mark_first_raw_key_block, normalize_medium, normalize_token_ids,
        };
        use $pb as pb;

        /// Rank `dp_rank`'s `SubscribeKvEvents` and `GetKvSnapshot` over `channel`.
        pub(crate) fn calls(
            channel: $crate::kv_events::tonic::transport::Channel,
            dp_rank: u32,
        ) -> $crate::kv_events::KvEventCalls {
            let client = move || {
                pb::kv_events_client::KvEventsClient::new(channel.clone())
                    .max_decoding_message_size($crate::DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            };
            let subscriber = client.clone();
            $crate::kv_events::KvEventCalls {
                subscribe: Box::new(move |resume| {
                    let mut client = subscriber();
                    Box::pin(async move {
                        let request = pb::SubscribeKvEventsRequest {
                            data_parallel_rank: dp_rank,
                            resume: resume.map(|cursor| pb::KvEventCursor {
                                stream_id: cursor.stream_id,
                                next_sequence_number: cursor.next_sequence_number,
                            }),
                        };
                        let responses = client.subscribe_kv_events(request).await?;
                        Ok($crate::kv_events::grpc_batches(
                            responses.into_inner(),
                            batch,
                        ))
                    })
                }),
                snapshot: Box::new(move || {
                    let mut client = client();
                    Box::pin(async move {
                        use pb::get_kv_snapshot_response::Item;
                        use $crate::kv_events::tonic::Status;
                        let request = pb::GetKvSnapshotRequest {
                            data_parallel_rank: dp_rank,
                        };
                        let mut responses = client.get_kv_snapshot(request).await?.into_inner();
                        let mut stores = Vec::new();
                        while let Some(response) = responses.message().await? {
                            match response.item {
                                Some(Item::Blocks(blocks)) => {
                                    for stored in blocks.stored {
                                        let store = block_stored(stored);
                                        stores.push(store.map_err(|error| {
                                            Status::internal(format!("{error:#}"))
                                        })?);
                                    }
                                }
                                Some(Item::End(end)) => {
                                    let Some(resume) = end.resume else {
                                        return Err(Status::internal(
                                            "KV snapshot without a cursor",
                                        ));
                                    };
                                    return Ok($crate::kv_events::KvSnapshot {
                                        stores,
                                        resume: $crate::kv_events::KvCursor {
                                            stream_id: resume.stream_id,
                                            next_sequence_number: resume.next_sequence_number,
                                        },
                                        is_complete: end.complete,
                                    });
                                }
                                None => {}
                            }
                        }
                        Err(Status::internal("the KV snapshot has no end"))
                    })
                }),
            }
        }

        /// The batch of a stream response; `None` for the stream's start.
        pub(crate) fn batch(
            response: pb::SubscribeKvEventsResponse,
        ) -> Option<$crate::kv_events::KvBatch> {
            let pb::subscribe_kv_events_response::Item::Batch(batch) = response.item? else {
                return None;
            };
            let events = batch.events.into_iter().map(event).collect::<Result<_>>();
            Some(events.map(|events| {
                (
                    batch.sequence_number,
                    KvEventBatch {
                        ts: batch.ts,
                        events,
                        data_parallel_rank: batch.data_parallel_rank.map(u32::cast_signed),
                    },
                )
            }))
        }

        fn event(event: pb::KvCacheEvent) -> Result<RawKvEvent> {
            use pb::kv_cache_event::Event;
            Ok(
                match event.event.context("KV cache event without a type")? {
                    Event::BlockStored(stored) => block_stored(stored)?,
                    Event::BlockRemoved(removed) => RawKvEvent::BlockRemoved {
                        block_hashes: hashes(removed.block_hashes)?,
                        medium: normalize_medium(removed.medium),
                        group_idx: removed.group_idx,
                        kv_cache_spec_kind: None,
                        kv_cache_spec_sliding_window: None,
                        locality: removed.locality.as_deref().map(Locality::from_wire),
                        ownership: removed.ownership,
                    },
                    Event::AllBlocksCleared(_) => RawKvEvent::AllBlocksCleared { ownership: None },
                },
            )
        }

        fn block_stored(stored: pb::BlockStored) -> Result<RawKvEvent> {
            use pb::block_token_ids::Ids;
            let token_ids = match stored
                .token_ids
                .and_then(|token_ids| token_ids.ids)
                .context("BlockStored without token_ids")?
            {
                Ids::Single(list) => KvTokenIds::Single(list.token_ids),
                Ids::Pairs(list) => KvTokenIds::Bigram(
                    list.pairs
                        .into_iter()
                        .map(|pair| (pair.first, pair.second))
                        .collect(),
                ),
            };
            let (mut token_ids, is_eagle) = normalize_token_ids(token_ids);
            let block_size = stored.block_size as usize;
            // The decoder's helpers read an empty entry as they read `None`.
            let extra_keys: Vec<Option<Vec<ExtraKeyItem>>> = stored
                .extra_keys
                .into_iter()
                .map(|block| {
                    let keys = block.keys.into_iter().map(extra_key);
                    keys.collect::<Result<_>>().map(Some)
                })
                .collect::<Result<_>>()?;
            mark_first_raw_key_block(&mut token_ids, block_size, Some(&extra_keys));
            let cache_namespace = stored.cache_salt.or_else(|| {
                extra_keys_to_cache_namespace(Some(&extra_keys), stored.lora_name.as_deref())
            });
            Ok(RawKvEvent::BlockStored {
                block_hashes: hashes(stored.block_hashes)?,
                parent_block_hash: stored.parent_block_hash.map(hash).transpose()?,
                token_ids,
                block_size,
                medium: normalize_medium(stored.medium),
                lora_name: stored.lora_name,
                cache_namespace,
                block_mm_infos: extra_keys_to_block_mm_infos(Some(extra_keys)),
                is_eagle: Some(is_eagle),
                group_idx: stored.group_idx,
                kv_cache_spec_kind: stored
                    .kv_cache_spec_kind
                    .as_deref()
                    .map(KvCacheSpecKind::from_wire),
                kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
                locality: stored.locality.as_deref().map(Locality::from_wire),
                ownership: stored.ownership,
                session_id: stored.session_id,
            })
        }

        fn hashes(hashes: Vec<pb::BlockHash>) -> Result<Vec<BlockHashValue>> {
            hashes.into_iter().map(hash).collect()
        }

        fn hash(hash: pb::BlockHash) -> Result<BlockHashValue> {
            use pb::block_hash::Value;
            match hash.value {
                Some(Value::UnsignedValue(value)) => Ok(BlockHashValue::Unsigned(value)),
                Some(Value::SignedValue(value)) => Ok(BlockHashValue::Signed(value)),
                // The ZMQ decoder rejects byte hashes as well.
                Some(Value::Digest(_)) => bail!("byte block hashes are not supported"),
                None => bail!("block hash without a value"),
            }
        }

        fn extra_key(key: pb::ExtraKeyValue) -> Result<ExtraKeyItem> {
            use pb::extra_key_value::Value;
            Ok(match key.value.context("extra key without a value")? {
                Value::Text(text) => ExtraKeyItem::Hash(text),
                Value::Data(data) => ExtraKeyItem::Bytes(data.into()),
                Value::Integer(value) => ExtraKeyItem::Signed(value),
                Value::Tuple(tuple) => match <[pb::ExtraKeyValue; 2]>::try_from(tuple.items) {
                    Ok(
                        [
                            pb::ExtraKeyValue {
                                value: Some(Value::Text(hash)),
                            },
                            pb::ExtraKeyValue {
                                value: Some(Value::Integer(offset)),
                            },
                        ],
                    ) => ExtraKeyItem::HashWithSignedOffset((hash, offset)),
                    _ => bail!("unsupported extra key tuple"),
                },
            })
        }
    };
}
