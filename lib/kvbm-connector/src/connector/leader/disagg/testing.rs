// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test doubles for conditional-disaggregation session and queue traits.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Result, anyhow};
use futures::{FutureExt, Stream, future::BoxFuture};
use kvbm_disagg_protocol::{DisaggSequenceHash, RemotePrefillRequest, SessionEndpoint, SessionId};
use kvbm_engine::disagg::{BlockSetResponse, HashSelection, UnpinAck, UnpinRequest};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::queue::RemotePrefillQueue;
use super::session::{
    PrefillSession, PrefillSessionFactory, SessionEvent, SessionEventStream, hash_to_wire,
};
use crate::{G2, SequenceHash};

pub const TEST_BLOCK_SIZE: usize = 16;

pub struct TestG2Blocks {
    #[allow(dead_code)]
    pub manager: Arc<BlockManager<G2>>,
    pub blocks: Vec<ImmutableBlock<G2>>,
}

impl TestG2Blocks {
    pub fn hashes(&self) -> Vec<SequenceHash> {
        self.blocks
            .iter()
            .map(|block| block.sequence_hash())
            .collect()
    }

    pub fn hashes_wire(&self) -> Vec<DisaggSequenceHash> {
        self.blocks
            .iter()
            .map(|block| hash_to_wire(block.sequence_hash()))
            .collect()
    }
}

pub fn test_g2_blocks(count: usize, start_token: u32) -> TestG2Blocks {
    assert!(count > 0, "test_g2_blocks count must be positive");
    let registry = TestRegistryBuilder::new().build();
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(count)
            .block_size(TEST_BLOCK_SIZE)
            .registry(registry)
            .build(),
    );
    let token_sequence = create_token_sequence(count, TEST_BLOCK_SIZE, start_token);
    let mutable = manager
        .allocate_blocks(count)
        .unwrap_or_else(|| panic!("failed to allocate {count} test G2 blocks"));
    let complete = mutable
        .into_iter()
        .zip(token_sequence.blocks().iter())
        .map(|(block, token_block)| {
            block
                .complete(token_block)
                .unwrap_or_else(|err| panic!("failed to complete test block: {err:?}"))
        })
        .collect();
    let blocks = manager.register_blocks(complete);

    TestG2Blocks { manager, blocks }
}

struct EventStream {
    receiver: mpsc::UnboundedReceiver<SessionEvent>,
}

impl Stream for EventStream {
    type Item = SessionEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

pub struct MockPrefillSession {
    id: SessionId,
    endpoint: Option<SessionEndpoint>,
    event_tx: mpsc::UnboundedSender<SessionEvent>,
    state: Mutex<MockSessionState>,
}

#[derive(Default)]
struct MockSessionState {
    ready_blocks: Vec<ImmutableBlock<G2>>,
    pending_hashes: Vec<SequenceHash>,
    event_rx: Option<mpsc::UnboundedReceiver<SessionEvent>>,
    block_set_responses: Vec<BlockSetResponse>,
    unpin_acks: Vec<UnpinAck>,
    requested_unpins: Vec<UnpinRequest>,
    closed_reason: Option<Option<String>>,
}

impl MockPrefillSession {
    pub fn new() -> Arc<Self> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        Arc::new(Self {
            id: uuid::Uuid::new_v4(),
            endpoint: Some(SessionEndpoint {
                kind: "mock".to_string(),
                payload: serde_json::json!({ "session": "mock" }),
            }),
            event_tx,
            state: Mutex::new(MockSessionState {
                event_rx: Some(event_rx),
                ..Default::default()
            }),
        })
    }

    pub fn push_event(&self, event: SessionEvent) -> Result<()> {
        self.event_tx
            .send(event)
            .map_err(|err| anyhow!("failed to push session event: {err}"))
    }

    pub fn ready_hashes(&self) -> Vec<DisaggSequenceHash> {
        self.state
            .lock()
            .ready_blocks
            .iter()
            .map(|block| hash_to_wire(block.sequence_hash()))
            .collect()
    }

    pub fn pending_hashes(&self) -> Vec<DisaggSequenceHash> {
        self.state
            .lock()
            .pending_hashes
            .iter()
            .map(|hash| hash_to_wire(*hash))
            .collect()
    }

    pub fn block_set_responses(&self) -> Vec<BlockSetResponse> {
        self.state.lock().block_set_responses.clone()
    }

    pub fn unpin_acks(&self) -> Vec<UnpinAck> {
        self.state.lock().unpin_acks.clone()
    }

    pub fn requested_unpins(&self) -> Vec<UnpinRequest> {
        self.state.lock().requested_unpins.clone()
    }

    pub fn closed_reason(&self) -> Option<Option<String>> {
        self.state.lock().closed_reason.clone()
    }
}

impl PrefillSession for MockPrefillSession {
    fn session_id(&self) -> SessionId {
        self.id
    }

    fn endpoint(&self) -> Option<SessionEndpoint> {
        self.endpoint.clone()
    }

    fn add_ready_blocks(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
        self.state.lock().ready_blocks.extend(blocks);
        Ok(())
    }

    fn add_pending_hashes(&self, hashes: Vec<SequenceHash>) -> Result<()> {
        self.state.lock().pending_hashes.extend(hashes);
        Ok(())
    }

    fn subscribe(&self) -> SessionEventStream {
        let receiver = self
            .state
            .lock()
            .event_rx
            .take()
            .expect("MockPrefillSession::subscribe called twice");
        Box::pin(EventStream { receiver })
    }

    fn respond_to_block_set_request(&self, response: BlockSetResponse) -> Result<()> {
        self.state.lock().block_set_responses.push(response);
        Ok(())
    }

    fn release_session_pins(&self, selection: &HashSelection) -> Result<Vec<DisaggSequenceHash>> {
        let mut state = self.state.lock();
        let mut released = Vec::new();
        match selection {
            HashSelection::All => {
                released = state
                    .ready_blocks
                    .iter()
                    .map(|block| hash_to_wire(block.sequence_hash()))
                    .collect();
                state.ready_blocks.clear();
            }
            HashSelection::Hashes(hashes) => {
                let selected: std::collections::HashSet<_> = hashes.iter().cloned().collect();
                state.ready_blocks.retain(|block| {
                    let hash = hash_to_wire(block.sequence_hash());
                    if selected.contains(&hash) {
                        released.push(hash);
                        false
                    } else {
                        true
                    }
                });
            }
        }
        released.sort();
        Ok(released)
    }

    fn ack_unpin(&self, ack: UnpinAck) -> Result<()> {
        self.state.lock().unpin_acks.push(ack);
        Ok(())
    }

    fn request_unpin(&self, request: UnpinRequest) -> BoxFuture<'static, Result<UnpinAck>> {
        self.state.lock().requested_unpins.push(request.clone());
        async move {
            Ok(UnpinAck {
                request_id: request.request_id,
                hashes: request.hashes,
            })
        }
        .boxed()
    }

    fn close(&self, reason: Option<String>) {
        let mut state = self.state.lock();
        state.ready_blocks.clear();
        state.closed_reason = Some(reason);
    }
}

pub struct MockPrefillSessionFactory {
    last_created: Mutex<Option<Arc<MockPrefillSession>>>,
}

impl MockPrefillSessionFactory {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            last_created: Mutex::new(None),
        })
    }

    pub fn last(&self) -> Option<Arc<MockPrefillSession>> {
        self.last_created.lock().clone()
    }
}

impl PrefillSessionFactory for MockPrefillSessionFactory {
    fn create_decode(&self, _session_id: SessionId) -> Result<Arc<dyn PrefillSession>> {
        let session = MockPrefillSession::new();
        *self.last_created.lock() = Some(session.clone());
        Ok(session)
    }
}

pub struct InMemoryRemotePrefillQueue {
    items: Mutex<Vec<RemotePrefillRequest>>,
}

impl InMemoryRemotePrefillQueue {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            items: Mutex::new(Vec::new()),
        })
    }

    pub fn snapshot(&self) -> Vec<RemotePrefillRequest> {
        self.items.lock().clone()
    }
}

impl RemotePrefillQueue for InMemoryRemotePrefillQueue {
    fn enqueue(&self, request: RemotePrefillRequest) -> BoxFuture<'static, Result<()>> {
        self.items.lock().push(request);
        async { Ok(()) }.boxed()
    }
}

pub async fn wait_until(predicate: impl Fn() -> bool) {
    for _ in 0..200 {
        if predicate() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!("condition not met within timeout");
}
