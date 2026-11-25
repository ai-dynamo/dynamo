// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Mocker's KV events as an engine's gRPC `KvEvents` service.

#[doc(hidden)]
pub use dynamo_kv_events::event::{KvCacheEvent, KvCacheEventData, KvCacheStoreData};

/// Defines `KvEventStream`, a KV event sink that publishes each batch on the
/// Mocker's ZMQ sink and to the subscribers of the `KvEvents` service of proto
/// package `$pb`, and serves that service's snapshot of the device blocks held.
/// `$rank` names the rank field of the engine's batch, and `$event` converts a
/// `RawKvEvent` and the block size into the engine's `KvCacheEvent`. `$start`
/// checks a subscription and gives the stream's first response, for an engine
/// that sends one before its batches; `$end` gives a snapshot's last response.
/// Both receive the stream's id. Invoke it in a module of its own, in a crate
/// with `anyhow`, `async_stream`, `futures`, `tokio`, `tonic_v14` and `uuid`.
#[macro_export]
macro_rules! grpc_kv_events {
    ($pb:path, $rank:ident, $event:path, $start:path, $end:path) => {
        use std::collections::HashMap;
        use std::pin::Pin;
        use std::sync::Mutex;
        use std::time::{SystemTime, UNIX_EPOCH};

        use ::futures::Stream;
        use ::tokio::sync::broadcast;
        use ::tonic_v14::Status;
        use $crate::common::protocols::{RawKvEvent, RawKvEventSink};
        use $crate::services::grpc_kv_events::{KvCacheEvent, KvCacheEventData, KvCacheStoreData};
        use $crate::services::zmq_events::ZmqKvEventSink;
        use $pb as pb;

        const SUBSCRIBER_BACKLOG: usize = 4096;

        pub(crate) type KvEventResponses =
            Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;
        pub(crate) type KvSnapshotResponses =
            Pin<Box<dyn Stream<Item = Result<pb::GetKvSnapshotResponse, Status>> + Send>>;

        /// The stream's live batches and the device blocks held.
        pub(crate) struct KvEventStream {
            zmq: ZmqKvEventSink,
            state: Mutex<State>,
            stream_id: String,
            dp_rank: u32,
            block_size: u32,
        }

        /// Changed under one lock, so a snapshot is one cut of the stream.
        struct State {
            batches: broadcast::Sender<pb::KvEventBatch>,
            next_sequence: u64,
            /// The device blocks held, by hash: each one's parent and the store that recreates it.
            held: HashMap<u64, (Option<u64>, RawKvEvent)>,
        }

        impl KvEventStream {
            pub(crate) fn new(zmq: ZmqKvEventSink, dp_rank: u32, block_size: u32) -> Self {
                Self {
                    zmq,
                    state: Mutex::new(State {
                        batches: broadcast::channel(SUBSCRIBER_BACKLOG).0,
                        next_sequence: 0,
                        held: HashMap::new(),
                    }),
                    stream_id: ::uuid::Uuid::new_v4().to_string(),
                    dp_rank,
                    block_size,
                }
            }

            fn check_rank(&self, dp_rank: u32) -> Result<(), Status> {
                if dp_rank == self.dp_rank {
                    return Ok(());
                }
                Err(Status::invalid_argument(format!(
                    "data_parallel_rank {dp_rank} is not served"
                )))
            }

            pub(crate) fn subscribe(
                &self,
                request: pb::SubscribeKvEventsRequest,
            ) -> Result<KvEventResponses, Status> {
                self.check_rank(request.data_parallel_rank)?;
                let start = $start(&request, &self.stream_id)?;
                let mut batches = self.state.lock().unwrap().batches.subscribe();
                Ok(Box::pin(::async_stream::stream! {
                    use pb::subscribe_kv_events_response::Item;
                    if let Some(start) = start {
                        yield Ok(start);
                    }
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

            /// The held blocks, each after its parent; a block whose parent is not held is left out.
            pub(crate) fn snapshot(
                &self,
                request: pb::GetKvSnapshotRequest,
            ) -> Result<KvSnapshotResponses, Status> {
                use pb::get_kv_snapshot_response::Item;
                use pb::kv_cache_event::Event;
                self.check_rank(request.data_parallel_rank)?;
                let state = self.state.lock().unwrap();
                let mut children: HashMap<Option<u64>, Vec<u64>> = HashMap::new();
                for (hash, (parent, _)) in &state.held {
                    children.entry(*parent).or_default().push(*hash);
                }
                // From the blocks without a parent down, one block per response.
                let mut next = children.remove(&None).unwrap_or_default();
                let mut items = Vec::new();
                while let Some(hash) = next.pop() {
                    let store = $event(&state.held[&hash].1, self.block_size);
                    if let Some(Event::BlockStored(stored)) = store.event {
                        let stored = vec![stored];
                        items.push(Item::Blocks(pb::KvSnapshotBlocks { stored }));
                    }
                    next.extend(children.remove(&Some(hash)).unwrap_or_default());
                }
                items.push(Item::End($end(&self.stream_id, state.next_sequence)));
                let responses = items
                    .into_iter()
                    .map(|item| Ok(pb::GetKvSnapshotResponse { item: Some(item) }));
                Ok(Box::pin(::futures::stream::iter(responses)))
            }

            /// A prefix cache reset's event, which the Mocker's ZMQ format lacks.
            pub(crate) fn publish_all_blocks_cleared(&self) {
                use pb::kv_cache_event::Event;
                let cleared = Event::AllBlocksCleared(pb::AllBlocksCleared {});
                let mut state = self.state.lock().unwrap();
                state.held.clear();
                self.send(
                    &mut state,
                    vec![pb::KvCacheEvent {
                        event: Some(cleared),
                    }],
                );
            }

            /// Ends every subscriber's stream, as a relay that lost its engine does.
            pub(crate) fn drop_subscribers(&self) {
                self.state.lock().unwrap().batches = broadcast::channel(SUBSCRIBER_BACKLOG).0;
            }

            fn send(&self, state: &mut State, events: Vec<pb::KvCacheEvent>) {
                if events.is_empty() {
                    return;
                }
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                let _ = state.batches.send(pb::KvEventBatch {
                    sequence_number: state.next_sequence,
                    ts,
                    $rank: Some(self.dp_rank),
                    events,
                });
                state.next_sequence += 1;
            }
        }

        impl State {
            /// Follows the device blocks an event stores or removes.
            fn track(&mut self, raw: &RawKvEvent) {
                match &raw.event.data {
                    KvCacheEventData::Stored(data) if raw.storage_tier.is_gpu() => {
                        let mut parent = data.parent_hash;
                        let tokens = raw.block_token_ids.iter().flatten();
                        for (block, tokens) in data.blocks.iter().zip(tokens) {
                            // The block's own store; a snapshot's store has no session.
                            let held = || {
                                let data = KvCacheEventData::Stored(KvCacheStoreData {
                                    parent_hash: parent,
                                    start_position: None,
                                    blocks: vec![block.clone()],
                                });
                                let store = RawKvEvent {
                                    event: KvCacheEvent { data, ..raw.event },
                                    block_token_ids: Some(vec![tokens.clone()]),
                                    storage_tier: raw.storage_tier,
                                    session_id: None,
                                };
                                (parent.map(|parent| parent.0), store)
                            };
                            self.held.entry(block.block_hash.0).or_insert_with(held);
                            parent = Some(block.block_hash);
                        }
                    }
                    KvCacheEventData::Removed(data) if raw.storage_tier.is_gpu() => {
                        for hash in &data.block_hashes {
                            self.held.remove(&hash.0);
                        }
                    }
                    KvCacheEventData::Cleared => self.held.clear(),
                    _ => {}
                }
            }
        }

        impl RawKvEventSink for KvEventStream {
            fn publish(&self, event: RawKvEvent) -> ::anyhow::Result<()> {
                self.publish_batch(vec![event])
            }

            fn publish_batch(&self, events: Vec<RawKvEvent>) -> ::anyhow::Result<()> {
                {
                    let mut state = self.state.lock().unwrap();
                    for event in &events {
                        state.track(event);
                    }
                    let block_size = self.block_size;
                    let events = events
                        .iter()
                        .map(|event| $event(event, block_size))
                        .collect();
                    self.send(&mut state, events);
                }
                self.zmq.publish_batch(events)
            }
        }
    };
}
