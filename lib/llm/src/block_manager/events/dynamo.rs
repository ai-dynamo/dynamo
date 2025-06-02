// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{
    kv_router::{
        indexer::RouterEvent,
        protocols::{
            ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
            KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
        },
        KV_EVENT_SUBJECT,
    },
    tokens::BlockHash,
};
use derive_getters::{Dissolve, Getters};
use dynamo_runtime::traits::events::EventPublisher;
use dynamo_runtime::{
    component::{Component, Namespace},
    raise, Result,
};
use std::sync::Arc;
use tokio::sync::mpsc;

pub enum DynamoPublisher {
    Component(Component),
    Namespace(Namespace),
}

impl DynamoPublisher {
    pub async fn publish(&self, event: RouterEvent) -> Result<()> {
        match self {
            DynamoPublisher::Component(component) => {
                component.publish(KV_EVENT_SUBJECT, &event).await
            }
            DynamoPublisher::Namespace(namespace) => {
                namespace.publish(KV_EVENT_SUBJECT, &event).await
            }
        }
    }
}

struct EventChannel {
    tx: mpsc::UnboundedSender<Event>,
}

impl EventReleaseManager for EventChannel {
    // Generalize sequence_hash
    fn block_release(&self, sequence_hash: SequenceHash) {
        if self.tx.send(Event::RemoveSingle(sequence_hash)).is_err() {
            tracing::warn!("Failed to send remove block event");
        }
    }
}

pub struct NatsEventManager {
    event_channel: Arc<EventChannel>,
}

impl NatsEventManager {
    // todo - generalize identifier
    pub async fn new(publisher: DynamoPublisher, worker_identifier: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = NatsEventsManagerState {
            rx,
            publisher,
            worker_identifier,
        };

        tokio::spawn(progress_engine(state));

        Self {
            event_channel: Arc::new(EventChannel { tx }),
        }
    }
}

impl std::fmt::Debug for NatsEventManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NatsEventManager")
    }
}

impl EventManager for NatsEventManager {
    fn register_block(&self, token_block: &TokenBlock) -> Result<RegistrationHandle> {
        let event = Event::StoreSingle(RegisterBlockEvent {
            block_hash: LocalBlockHash(token_block.block_hash()),
            sequence_hash: ExternalSequenceBlockHash(token_block.sequence_hash()),
            parent_hash: token_block
                .parent_sequence_hash()
                .map(ExternalSequenceBlockHash),
        });
        if self.event_channel.tx.send(event).is_err() {
            tracing::warn!("Failed to send store block event");
            raise!("Failed to send store block event");
        }
        Ok(RegistrationHandle {
            sequence_hash: token_block.sequence_hash(),
            release_manager: Some(self.event_channel.clone()),
        })
    }

    fn register_blocks(&self, token_blocks: &[TokenBlock]) -> Result<Vec<RegistrationHandle>> {
        let event = Event::StoreMultiple(RegisterBlocksEvent {
            hashes: token_blocks
                .iter()
                .map(|block| {
                    (
                        LocalBlockHash(block.block_hash()),
                        ExternalSequenceBlockHash(block.sequence_hash()),
                    )
                })
                .collect(),
            parent_hash: token_blocks
                .first()
                .and_then(|block| block.parent_sequence_hash().map(ExternalSequenceBlockHash)),
        });

        let handles = token_blocks
            .iter()
            .map(|block| RegistrationHandle {
                sequence_hash: block.sequence_hash(),
                release_manager: Some(self.event_channel.clone()),
            })
            .collect();

        if self.event_channel.tx.send(event).is_err() {
            tracing::warn!("Failed to send store block event");
            raise!("Failed to send store block event");
        }

        Ok(handles)
    }
}

#[derive(Dissolve)]
struct NatsEventsManagerState {
    rx: mpsc::UnboundedReceiver<Event>,
    publisher: DynamoPublisher,
    worker_identifier: WorkerIdentifier,
}

async fn progress_engine(state: NatsEventsManagerState) {
    let (mut rx, publisher, worker_identifier) = state.dissolve();

    let mut event_id = 0;

    while let Some(event) = rx.recv().await {
        match event {
            Event::StoreSingle(event) => {
                let store_data = KvCacheStoreData {
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: event.sequence_hash,
                        tokens_hash: event.block_hash,
                    }],
                    parent_hash: event.parent_hash,
                };
                let data = KvCacheEventData::Stored(store_data);
                let event = KvCacheEvent { event_id, data };
                let event = RouterEvent::new(worker_identifier as i64, event);
                if publisher.publish(event).await.is_err() {
                    tracing::warn!("Failed to publish store event");
                }
            }
            Event::StoreMultiple(event) => {
                let store_data = KvCacheStoreData {
                    blocks: event
                        .hashes
                        .iter()
                        .map(|(local_hash, external_hash)| KvCacheStoredBlockData {
                            block_hash: *external_hash,
                            tokens_hash: *local_hash,
                        })
                        .collect(),
                    parent_hash: event.parent_hash,
                };
                let data = KvCacheEventData::Stored(store_data);
                let event = KvCacheEvent { event_id, data };
                let event = RouterEvent::new(worker_identifier as i64, event);
                if publisher.publish(event).await.is_err() {
                    tracing::warn!("Failed to publish store event");
                }
            }
            Event::RemoveSingle(sequence_hash) => {
                let remove_data = KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(sequence_hash)],
                };
                let data = KvCacheEventData::Removed(remove_data);
                let event = KvCacheEvent { event_id, data };
                let event = RouterEvent::new(worker_identifier as i64, event);
                if publisher.publish(event).await.is_err() {
                    tracing::warn!("Failed to publish remove event");
                }
            }
        }
        event_id += 1;
    }
}
