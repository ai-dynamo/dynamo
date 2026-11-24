// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Mocker's KV events as an engine's gRPC `KvEvents` stream.

#[doc(hidden)]
pub use dynamo_kv_events::event::KvCacheEventData;

/// Defines `KvEventStream`, a KV event sink that publishes each batch on the
/// Mocker's ZMQ sink and to the subscribers of the `KvEvents` service of proto
/// package `$pb`. vLLM's `vllm` and SGLang's `sglang.kv.v1` packages define the
/// same messages. `$hash` builds the engine's block hash value from a `u64`,
/// and `$group_idx` is the group index it reports. Invoke it in a module of its
/// own, in a crate with `anyhow`, `async_stream`, `futures`, `tokio`,
/// `tonic_v14` and `uuid`.
#[macro_export]
macro_rules! grpc_kv_events {
    ($pb:path, $hash:expr, $group_idx:expr) => {
        use std::pin::Pin;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        use ::futures::Stream;
        use ::tokio::sync::broadcast;
        use ::tonic_v14::Status;
        use $crate::common::protocols::{RawKvEvent, RawKvEventSink};
        use $crate::services::grpc_kv_events::KvCacheEventData;
        use $crate::services::zmq_events::ZmqKvEventSink;
        use $pb as pb;

        const SUBSCRIBER_BACKLOG: usize = 4096;

        pub(crate) type KvEventResponses =
            Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;

        /// Live batches only: the Mocker keeps no replay buffer.
        pub(crate) struct KvEventStream {
            zmq: ZmqKvEventSink,
            batches: broadcast::Sender<pb::KvEventBatch>,
            next_sequence: AtomicU64,
            stream_id: String,
            dp_rank: u32,
            block_size: u32,
        }

        impl KvEventStream {
            pub(crate) fn new(zmq: ZmqKvEventSink, dp_rank: u32, block_size: u32) -> Self {
                Self {
                    zmq,
                    batches: broadcast::channel(SUBSCRIBER_BACKLOG).0,
                    next_sequence: AtomicU64::new(0),
                    stream_id: ::uuid::Uuid::new_v4().to_string(),
                    dp_rank,
                    block_size,
                }
            }

            pub(crate) fn subscribe(
                &self,
                request: pb::SubscribeKvEventsRequest,
            ) -> Result<KvEventResponses, Status> {
                if request.data_parallel_rank != self.dp_rank {
                    return Err(Status::invalid_argument(format!(
                        "data_parallel_rank {} is not served",
                        request.data_parallel_rank
                    )));
                }
                if request.resume.is_some() {
                    return Err(Status::out_of_range("the Mocker keeps no KV event buffer"));
                }
                let mut batches = self.batches.subscribe();
                let start = pb::KvEventStreamStart {
                    stream_id: self.stream_id.clone(),
                    data_parallel_rank: self.dp_rank,
                };
                Ok(Box::pin(::async_stream::stream! {
                    use pb::subscribe_kv_events_response::Item;
                    let item = Some(Item::Start(start));
                    yield Ok(pb::SubscribeKvEventsResponse { item });
                    loop {
                        match batches.recv().await {
                            Ok(batch) => {
                                let item = Some(Item::Batch(batch));
                                yield Ok(pb::SubscribeKvEventsResponse { item });
                            }
                            Err(broadcast::error::RecvError::Lagged(_)) => {
                                yield Err(Status::data_loss("the subscriber fell behind"));
                                break;
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }))
            }

            /// A prefix cache reset's event, which the Mocker's ZMQ format lacks.
            pub(crate) fn publish_all_blocks_cleared(&self) {
                use pb::kv_cache_event::Event;
                let cleared = Event::AllBlocksCleared(pb::AllBlocksCleared {});
                self.send(vec![pb::KvCacheEvent {
                    event: Some(cleared),
                }]);
            }

            fn send(&self, events: Vec<pb::KvCacheEvent>) {
                if events.is_empty() {
                    return;
                }
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                let _ = self.batches.send(pb::KvEventBatch {
                    sequence_number: self.next_sequence.fetch_add(1, Ordering::Relaxed),
                    ts,
                    data_parallel_rank: Some(self.dp_rank),
                    events,
                });
            }

            /// The fields the Mocker's ZMQ sink publishes for the same event.
            fn event(&self, raw: &RawKvEvent) -> pb::KvCacheEvent {
                use pb::kv_cache_event::Event;
                let hash = |value: u64| pb::BlockHash {
                    value: Some(($hash)(value)),
                };
                let medium = raw.storage_tier.to_kv_medium().map(str::to_string);
                let event = match &raw.event.data {
                    KvCacheEventData::Stored(data) => Event::BlockStored(pb::BlockStored {
                        block_hashes: data
                            .blocks
                            .iter()
                            .map(|block| hash(block.block_hash.0))
                            .collect(),
                        parent_block_hash: data.parent_hash.map(|parent| hash(parent.0)),
                        token_ids: Some(pb::BlockTokenIds {
                            ids: Some(pb::block_token_ids::Ids::Single(pb::TokenIdList {
                                token_ids: raw
                                    .block_token_ids
                                    .iter()
                                    .flatten()
                                    .flatten()
                                    .map(|&token| u64::from(token))
                                    .collect(),
                            })),
                        }),
                        block_size: self.block_size,
                        medium,
                        group_idx: $group_idx,
                        session_id: raw.session_id.clone(),
                        ..Default::default()
                    }),
                    KvCacheEventData::Removed(data) => Event::BlockRemoved(pb::BlockRemoved {
                        block_hashes: data
                            .block_hashes
                            .iter()
                            .map(|block| hash(block.0))
                            .collect(),
                        medium,
                        group_idx: $group_idx,
                        ..Default::default()
                    }),
                    KvCacheEventData::Cleared => Event::AllBlocksCleared(pb::AllBlocksCleared {}),
                };
                pb::KvCacheEvent { event: Some(event) }
            }
        }

        impl RawKvEventSink for KvEventStream {
            fn publish(&self, event: RawKvEvent) -> ::anyhow::Result<()> {
                self.publish_batch(vec![event])
            }

            fn publish_batch(&self, events: Vec<RawKvEvent>) -> ::anyhow::Result<()> {
                self.send(events.iter().map(|event| self.event(event)).collect());
                self.zmq.publish_batch(events)
            }
        }
    };
}
