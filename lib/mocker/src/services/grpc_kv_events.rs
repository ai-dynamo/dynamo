// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Mocker's KV events as an engine's gRPC `KvEvents` service.

#[doc(hidden)]
pub use dynamo_kv_events::event::KvCacheEventData;

/// Defines `KvEventStream`, a KV event sink that publishes each batch on the
/// Mocker's ZMQ sink and to the subscribers of the `KvEvents` service of proto
/// package `$pb`, and serves that service's snapshot of the device blocks held.
/// vLLM's `vllm` and SGLang's `sglang.kv.v1` packages define the
/// same messages. `$hash` builds the engine's block hash value from a `u64`,
/// and `$group_idx` is the group index it reports. Invoke it in a module of its
/// own, in a crate with `anyhow`, `async_stream`, `futures`, `tokio`,
/// `tonic_v14` and `uuid`.
#[macro_export]
macro_rules! grpc_kv_events {
    ($pb:path, $hash:expr, $group_idx:expr) => {
        use std::collections::HashMap;
        use std::pin::Pin;
        use std::sync::Mutex;
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
        pub(crate) type KvSnapshotResponses =
            Pin<Box<dyn Stream<Item = Result<pb::GetKvSnapshotResponse, Status>> + Send>>;

        /// Live batches and the blocks held: the Mocker keeps no replay buffer.
        pub(crate) struct KvEventStream {
            zmq: ZmqKvEventSink,
            state: Mutex<State>,
            stream_id: String,
            dp_rank: u32,
            block_size: u32,
        }

        /// Changed under one lock, so a snapshot and its cursor are one cut of the stream.
        struct State {
            batches: broadcast::Sender<pb::KvEventBatch>,
            next_sequence: u64,
            /// The device blocks held, by hash: each one's parent and the store that recreates it.
            held: HashMap<u64, (Option<u64>, pb::BlockStored)>,
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
                let mut batches = {
                    let state = self.state.lock().unwrap();
                    // Without a buffer, only the batch still to be published can be resumed from.
                    if let Some(resume) = request.resume
                        && (resume.next_sequence_number != state.next_sequence
                            || !(resume.stream_id.is_empty() || resume.stream_id == self.stream_id))
                    {
                        return Err(Status::out_of_range("the Mocker keeps no KV event buffer"));
                    }
                    state.batches.subscribe()
                };
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

            /// The held blocks, each after its parent; a block whose parent is not held is left out.
            pub(crate) fn snapshot(
                &self,
                request: pb::GetKvSnapshotRequest,
            ) -> Result<KvSnapshotResponses, Status> {
                use pb::get_kv_snapshot_response::Item;
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
                    let stored = vec![state.held[&hash].1.clone()];
                    items.push(Item::Blocks(pb::KvSnapshotBlocks { stored }));
                    next.extend(children.remove(&Some(hash)).unwrap_or_default());
                }
                items.push(Item::End(pb::KvSnapshotEnd {
                    resume: Some(pb::KvEventCursor {
                        stream_id: self.stream_id.clone(),
                        next_sequence_number: state.next_sequence,
                    }),
                    complete: true,
                }));
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
                    data_parallel_rank: Some(self.dp_rank),
                    events,
                });
                state.next_sequence += 1;
            }

            /// Follows the device blocks an event stores or removes.
            fn track(&self, state: &mut State, raw: &RawKvEvent) {
                match &raw.event.data {
                    KvCacheEventData::Stored(data) if raw.storage_tier.is_gpu() => {
                        let mut parent = data.parent_hash.map(|parent| parent.0);
                        let tokens = raw.block_token_ids.iter().flatten();
                        for (block, tokens) in data.blocks.iter().zip(tokens) {
                            let hash = block.block_hash.0;
                            let stored = || (parent, self.stored(&[hash], parent, tokens));
                            state.held.entry(hash).or_insert_with(stored);
                            parent = Some(hash);
                        }
                    }
                    KvCacheEventData::Removed(data) if raw.storage_tier.is_gpu() => {
                        for hash in &data.block_hashes {
                            state.held.remove(&hash.0);
                        }
                    }
                    KvCacheEventData::Cleared => state.held.clear(),
                    _ => {}
                }
            }

            fn hash(value: u64) -> pb::BlockHash {
                pb::BlockHash {
                    value: Some(($hash)(value)),
                }
            }

            fn stored(
                &self,
                hashes: &[u64],
                parent: Option<u64>,
                tokens: &[u32],
            ) -> pb::BlockStored {
                pb::BlockStored {
                    block_hashes: hashes.iter().map(|&value| Self::hash(value)).collect(),
                    parent_block_hash: parent.map(Self::hash),
                    token_ids: Some(pb::BlockTokenIds {
                        ids: Some(pb::block_token_ids::Ids::Single(pb::TokenIdList {
                            token_ids: tokens.iter().map(|&token| u64::from(token)).collect(),
                        })),
                    }),
                    block_size: self.block_size,
                    group_idx: $group_idx,
                    ..Default::default()
                }
            }

            /// The fields the Mocker's ZMQ sink publishes for the same event.
            fn event(&self, raw: &RawKvEvent) -> pb::KvCacheEvent {
                use pb::kv_cache_event::Event;
                let medium = raw.storage_tier.to_kv_medium().map(str::to_string);
                let event = match &raw.event.data {
                    KvCacheEventData::Stored(data) => {
                        let hashes: Vec<_> =
                            data.blocks.iter().map(|block| block.block_hash.0).collect();
                        let tokens = raw.block_token_ids.iter().flatten().flatten();
                        let tokens: Vec<_> = tokens.copied().collect();
                        let parent = data.parent_hash.map(|parent| parent.0);
                        Event::BlockStored(pb::BlockStored {
                            medium,
                            session_id: raw.session_id.clone(),
                            ..self.stored(&hashes, parent, &tokens)
                        })
                    }
                    KvCacheEventData::Removed(data) => Event::BlockRemoved(pb::BlockRemoved {
                        block_hashes: data
                            .block_hashes
                            .iter()
                            .map(|block| Self::hash(block.0))
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
                {
                    let mut state = self.state.lock().unwrap();
                    for event in &events {
                        self.track(&mut state, event);
                    }
                    let events = events.iter().map(|event| self.event(event)).collect();
                    self.send(&mut state, events);
                }
                self.zmq.publish_batch(events)
            }
        }
    };
}
