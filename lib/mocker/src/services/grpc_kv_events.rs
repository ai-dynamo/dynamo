// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Mocker's KV events as an engine's gRPC `KvEvents` stream.

#[doc(hidden)]
pub use dynamo_kv_events::event::KvCacheEventData;

/// Defines `KvEventStream`, a KV event sink that publishes each batch on the
/// Mocker's ZMQ sink and to the subscribers of the `KvEvents` service of proto
/// package `$pb`. `$rank` names the rank field of the engine's batch, and
/// `$event` converts a `RawKvEvent` and the block size into the engine's
/// `KvCacheEvent`. Invoke it in a module of its own, in a crate with `anyhow`,
/// `async_stream`, `futures`, `tokio`, `tonic_v14` and `uuid`.
#[macro_export]
macro_rules! grpc_kv_events {
    ($pb:path, $rank:ident, $event:path) => {
        use std::pin::Pin;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        use ::futures::Stream;
        use ::tokio::sync::broadcast;
        use ::tonic_v14::Status;
        use $crate::common::protocols::{RawKvEvent, RawKvEventSink};
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
                    $rank: Some(self.dp_rank),
                    events,
                });
            }
        }

        impl RawKvEventSink for KvEventStream {
            fn publish(&self, event: RawKvEvent) -> ::anyhow::Result<()> {
                self.publish_batch(vec![event])
            }

            fn publish_batch(&self, events: Vec<RawKvEvent>) -> ::anyhow::Result<()> {
                let block_size = self.block_size;
                self.send(
                    events
                        .iter()
                        .map(|event| $event(event, block_size))
                        .collect(),
                );
                self.zmq.publish_batch(events)
            }
        }
    };
}
