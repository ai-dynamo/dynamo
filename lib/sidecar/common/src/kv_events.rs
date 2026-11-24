// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV event translation shared by the vLLM and SGLang sidecars: engine
//! batches in, from the engine's gRPC stream or, for older engines, its ZMQ
//! publisher; Dynamo events out, through the publisher's push input.

use tonic_v14 as tonic;

use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{DynamoError, KvEventSource};
use dynamo_kv_events::zmq_wire::{KvEventBatch, ZmqEventNormalizer};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, decode_zmq_kv_batch, normalize_batch};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use futures::future::{self, BoxFuture};
use futures::stream::{BoxStream, StreamExt};
use tokio_util::sync::CancellationToken;

#[doc(hidden)]
pub use anyhow;
#[doc(hidden)]
pub use dynamo_kv_events::zmq_wire as wire;

#[cfg(test)]
mod tests;

const RESUBSCRIBE_DELAY: Duration = Duration::from_secs(1);

/// One engine batch with its sequence number, or why it could not be decoded.
pub type KvBatch = anyhow::Result<(u64, KvEventBatch)>;

/// Engine batches; the stream continues after a batch that could not be decoded.
pub type KvBatchStream = BoxStream<'static, KvBatch>;

/// Opens one rank's `SubscribeKvEvents` stream, without a cursor.
pub type Subscribe =
    Box<dyn FnMut() -> BoxFuture<'static, Result<KvBatchStream, tonic::Status>> + Send>;

/// Where one rank's engine KV events come from.
pub enum KvEventFeed {
    Grpc {
        stream: KvBatchStream,
        resubscribe: Subscribe,
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
            feed,
        } = self;
        KvEventSource::Push {
            dp_rank,
            idempotent_stores,
            on_ready: Box::new(move |publisher| {
                let translator = Translator::new(publisher.clone(), dp_rank, image_token_id);
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

/// One feed per rank: the engine's gRPC streams, opened without a cursor, or
/// `zmq` for an engine without the `KvEvents` service (no stream listed, or
/// UNIMPLEMENTED). No feeds when it cannot serve KV events (FAILED_PRECONDITION).
pub async fn kv_event_feeds(
    grpc: impl IntoIterator<Item = (u32, Subscribe)>,
    zmq: impl FnOnce() -> Result<Vec<(u32, KvEventFeed)>, DynamoError>,
) -> Result<Vec<(u32, KvEventFeed)>, DynamoError> {
    let mut feeds = Vec::new();
    for (dp_rank, mut subscribe) in grpc {
        match subscribe().await {
            Ok(stream) => feeds.push((
                dp_rank,
                KvEventFeed::Grpc {
                    stream,
                    resubscribe: subscribe,
                },
            )),
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

/// Resubscribes without a cursor, at once unless `wait`; `None` once the engine serves no KV events.
async fn resubscribed(
    subscribe: &mut Subscribe,
    dp_rank: u32,
    mut wait: bool,
) -> Option<KvBatchStream> {
    loop {
        if wait {
            tokio::time::sleep(RESUBSCRIBE_DELAY).await;
        }
        wait = true;
        match subscribe().await {
            Ok(stream) => return Some(stream),
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

/// What lib/llm's ZMQ listener does with each batch, for one rank's feed.
struct Translator {
    publisher: Arc<KvEventPublisher>,
    normalizer: ZmqEventNormalizer,
    worker: WorkerWithDpRank,
    last_sequence: Option<u64>,
}

impl Translator {
    fn new(publisher: Arc<KvEventPublisher>, dp_rank: u32, image_token_id: Option<u32>) -> Self {
        Self {
            normalizer: ZmqEventNormalizer::new(publisher.kv_block_size())
                .with_image_token_id(image_token_id),
            worker: WorkerWithDpRank::new(publisher.worker_id(), dp_rank),
            last_sequence: None,
            publisher,
        }
    }

    /// Returns when the feed or the publisher has stopped for good.
    async fn run(mut self, feed: KvEventFeed) {
        let dp_rank = self.worker.dp_rank;
        let (mut stream, mut resubscribe) = match feed {
            KvEventFeed::Grpc {
                stream,
                resubscribe,
            } => {
                tracing::info!(dp_rank, "KV events over gRPC");
                (stream, Some(resubscribe))
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
            match stream.next().await {
                Some(Ok((sequence, batch))) => {
                    delivered = true;
                    if !self.publish(sequence, batch) {
                        return;
                    }
                }
                Some(Err(error)) => {
                    tracing::warn!(dp_rank, %error, "Failed to decode KV event batch");
                }
                None => {
                    // Like lib/llm's ZMQ listener, a ZMQ feed is not reconnected.
                    let Some(subscribe) = resubscribe.as_mut() else {
                        tracing::error!(dp_rank, "ZMQ KV event stream ended; stopping the relay");
                        return;
                    };
                    tracing::warn!(dp_rank, "KV event stream ended; resubscribing");
                    // A stream that ended before its first batch is not reopened at once.
                    let Some(reopened) = resubscribed(subscribe, dp_rank, !delivered).await else {
                        return;
                    };
                    stream = reopened;
                    delivered = false;
                }
            }
        }
    }

    /// Translate and push one batch. False once the publisher has stopped.
    fn publish(&mut self, sequence: u64, batch: KvEventBatch) -> bool {
        let dp_rank = self.worker.dp_rank;
        if batch.data_parallel_rank.map(i32::cast_unsigned) != Some(dp_rank) {
            tracing::warn!(dp_rank, batch_rank = ?batch.data_parallel_rank, "Ignoring mismatched KV source rank");
            return true;
        }
        if self
            .last_sequence
            .replace(sequence)
            .is_some_and(|previous| previous.checked_add(1) != Some(sequence))
        {
            tracing::warn!(
                source_cursor = sequence,
                dp_rank,
                "KV event sequence discontinuity"
            );
        }
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

/// Defines `batch`, which converts one response of an engine's `KvEvents`
/// service into the batch the ZMQ decoder returns for the same events, through
/// the decoder's own helpers. vLLM's `vllm` and SGLang's `sglang.kv.v1`
/// packages define the same messages. Invoke it in a module of its own.
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
