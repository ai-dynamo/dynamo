// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test doubles for conditional-disaggregation session and queue traits.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Result, anyhow};
use futures::{FutureExt, Stream, future::BoxFuture};
use kvbm_disagg_protocol::{RemotePrefillRequest, SessionEndpoint, SessionId, TransferParams};
use kvbm_engine::disagg::{
    BlockSetRequest, BlockSetResponse, HashSelection, PullAck, PullComplete, RemoteBlockSet,
    UnpinAck, UnpinRequest,
};
use tokio::sync::oneshot;

use super::transport::{
    CdBlockTransport, CdWorkerHook, InnerLeaderShim, PrefillSessionAttacher,
};
use kvbm_logical::blocks::{CompleteBlock, MutableBlock};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::testing::token_blocks::create_token_sequence;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::queue::RemotePrefillQueue;
use super::session::{PrefillSession, PrefillSessionFactory, SessionEvent, SessionEventStream};
use crate::{BlockId, G2, SequenceHash};

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
    /// Pending `request_block_sets` waiters, keyed by
    /// `request.request_id`. Resolved by
    /// `enqueue_block_set_response` when the response's
    /// `request_id` matches.
    pending_block_set_requests:
        Mutex<std::collections::HashMap<String, oneshot::Sender<BlockSetResponse>>>,
}

#[derive(Default)]
struct MockSessionState {
    ready_blocks: Vec<ImmutableBlock<G2>>,
    pending_hashes: Vec<SequenceHash>,
    event_rx: Option<mpsc::UnboundedReceiver<SessionEvent>>,
    block_set_responses: Vec<BlockSetResponse>,
    unpin_acks: Vec<UnpinAck>,
    requested_unpins: Vec<UnpinRequest>,
    pull_completes: Vec<PullComplete>,
    closed_reason: Option<Option<String>>,
    /// Prefill-side: scripted response for the next
    /// `request_block_sets` call. Pop on each call.
    block_set_responses_to_return: std::collections::VecDeque<BlockSetResponse>,
    /// Prefill-side: every `request_block_sets` call observed.
    requested_block_sets: Vec<BlockSetRequest>,
    /// Prefill-side: every `publish_output_block_sets` call.
    published_output_sets: Vec<Vec<RemoteBlockSet>>,
    /// Prefill-side: every `ack_pull_from_prefill` call.
    pull_acks: Vec<PullAck>,
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
            pending_block_set_requests: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub fn push_event(&self, event: SessionEvent) -> Result<()> {
        self.event_tx
            .send(event)
            .map_err(|err| anyhow!("failed to push session event: {err}"))
    }

    pub fn ready_hashes(&self) -> Vec<SequenceHash> {
        self.state
            .lock()
            .ready_blocks
            .iter()
            .map(|block| block.sequence_hash())
            .collect()
    }

    pub fn pending_hashes(&self) -> Vec<SequenceHash> {
        self.state.lock().pending_hashes.clone()
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

    pub fn pull_completes(&self) -> Vec<PullComplete> {
        self.state.lock().pull_completes.clone()
    }

    pub fn closed_reason(&self) -> Option<Option<String>> {
        self.state.lock().closed_reason.clone()
    }

    /// Prefill-side: deliver a scripted response.
    ///
    /// If a waiter is already in-flight (i.e.
    /// `request_block_sets` was called and is awaiting a
    /// response with the same `request_id`), resolve it
    /// directly. Otherwise stash the response so the next
    /// matching `request_block_sets` call resolves immediately.
    pub fn enqueue_block_set_response(&self, response: BlockSetResponse) {
        let request_id = response.request_id.clone();
        let pending = {
            let mut pending = self.pending_block_set_requests.lock();
            pending.remove(&request_id)
        };
        if let Some(tx) = pending {
            let _ = tx.send(response);
        } else {
            self.state
                .lock()
                .block_set_responses_to_return
                .push_back(response);
        }
    }

    pub fn requested_block_sets(&self) -> Vec<BlockSetRequest> {
        self.state.lock().requested_block_sets.clone()
    }

    pub fn published_output_sets(&self) -> Vec<Vec<RemoteBlockSet>> {
        self.state.lock().published_output_sets.clone()
    }

    pub fn pull_acks(&self) -> Vec<PullAck> {
        self.state.lock().pull_acks.clone()
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

    fn release_session_pins(&self, selection: &HashSelection) -> Result<Vec<SequenceHash>> {
        let mut state = self.state.lock();
        let mut released = Vec::new();
        match selection {
            HashSelection::All => {
                released = state
                    .ready_blocks
                    .iter()
                    .map(|block| block.sequence_hash())
                    .collect();
                state.ready_blocks.clear();
            }
            HashSelection::Hashes(hashes) => {
                let selected: std::collections::HashSet<_> = hashes.iter().cloned().collect();
                state.ready_blocks.retain(|block| {
                    let hash = block.sequence_hash();
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

    fn pull_complete_from_decode(
        &self,
        complete: PullComplete,
    ) -> BoxFuture<'static, Result<PullAck>> {
        self.state.lock().pull_completes.push(complete.clone());
        async move {
            Ok(PullAck {
                pull_id: complete.pull_id,
            })
        }
        .boxed()
    }

    fn request_block_sets(
        &self,
        request: BlockSetRequest,
    ) -> BoxFuture<'static, Result<BlockSetResponse>> {
        let request_id = request.request_id.clone();

        // Record the call (so tests can observe it).
        self.state.lock().requested_block_sets.push(request);

        // If a response was pre-enqueued for this request_id,
        // pop and return it. Otherwise install a oneshot waiter.
        let preexisting = {
            let mut state = self.state.lock();
            let queue = &mut state.block_set_responses_to_return;
            let pos = queue.iter().position(|r| r.request_id == request_id);
            pos.map(|p| queue.remove(p).expect("position valid"))
        };
        if let Some(response) = preexisting {
            return async move { Ok(response) }.boxed();
        }

        let (tx, rx) = oneshot::channel();
        self.pending_block_set_requests
            .lock()
            .insert(request_id, tx);

        async move {
            rx.await
                .map_err(|err| anyhow!("block_set oneshot dropped: {err}"))
        }
        .boxed()
    }

    fn publish_output_block_sets(
        &self,
        block_sets: Vec<RemoteBlockSet>,
    ) -> BoxFuture<'static, Result<()>> {
        self.state.lock().published_output_sets.push(block_sets);
        async move { Ok(()) }.boxed()
    }

    fn ack_pull_from_prefill(
        &self,
        ack: PullAck,
    ) -> BoxFuture<'static, Result<()>> {
        self.state.lock().pull_acks.push(ack);
        async move { Ok(()) }.boxed()
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

/// Test attacher: returns an `MockPrefillSession` per
/// `attach()` call and records the call. The returned session is
/// also kept addressable via `last()` so tests can drive it
/// (push events, queue scripted responses).
pub struct MockPrefillSessionAttacher {
    last_attached: Mutex<Option<Arc<MockPrefillSession>>>,
    attach_calls: Mutex<Vec<(SessionId, SessionEndpoint)>>,
}

impl MockPrefillSessionAttacher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            last_attached: Mutex::new(None),
            attach_calls: Mutex::new(Vec::new()),
        })
    }

    pub fn last(&self) -> Option<Arc<MockPrefillSession>> {
        self.last_attached.lock().clone()
    }

    pub fn attach_calls(&self) -> Vec<(SessionId, SessionEndpoint)> {
        self.attach_calls.lock().clone()
    }
}

impl PrefillSessionAttacher for MockPrefillSessionAttacher {
    fn attach(
        &self,
        session_id: SessionId,
        decode_endpoint: SessionEndpoint,
    ) -> BoxFuture<'static, Result<Arc<dyn PrefillSession>>> {
        let session = MockPrefillSession::new();
        *self.last_attached.lock() = Some(Arc::clone(&session));
        self.attach_calls.lock().push((session_id, decode_endpoint));
        async move { Ok(session as Arc<dyn PrefillSession>) }.boxed()
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

// ============================================================================
// MockCdBlockTransport
// ============================================================================

#[derive(Debug, Clone)]
pub struct PullRemoteCall {
    pub remote_instance: crate::InstanceId,
    pub block_sets: Vec<RemoteBlockSet>,
    pub local_dst_g2_block_ids: Vec<BlockId>,
}

#[derive(Debug, Clone)]
pub struct LocalG2ToG1Call {
    pub src_g2_block_ids: Vec<BlockId>,
    pub dst_g1_block_ids: Vec<BlockId>,
}

/// Records every transport call and lets tests resolve each one
/// independently, in order, with `Ok` or `Err`.
pub struct MockCdBlockTransport {
    pulls: Mutex<Vec<PendingPullCall>>,
    onboards: Mutex<Vec<PendingOnboardCall>>,
}

struct PendingPullCall {
    call: PullRemoteCall,
    resolver: Option<oneshot::Sender<Result<()>>>,
}

struct PendingOnboardCall {
    call: LocalG2ToG1Call,
    resolver: Option<oneshot::Sender<Result<()>>>,
}

impl MockCdBlockTransport {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            pulls: Mutex::new(Vec::new()),
            onboards: Mutex::new(Vec::new()),
        })
    }

    /// All `pull_remote` calls received, in order.
    pub fn pull_calls(&self) -> Vec<PullRemoteCall> {
        self.pulls
            .lock()
            .iter()
            .map(|p| p.call.clone())
            .collect()
    }

    /// All `local_g2_to_g1` calls received, in order.
    pub fn onboard_calls(&self) -> Vec<LocalG2ToG1Call> {
        self.onboards
            .lock()
            .iter()
            .map(|o| o.call.clone())
            .collect()
    }

    /// Resolve the Nth `pull_remote` call (0-indexed) with `result`.
    pub fn resolve_pull(&self, index: usize, result: Result<()>) {
        let mut pulls = self.pulls.lock();
        let pending = pulls
            .get_mut(index)
            .expect("pull_remote call not yet recorded");
        let resolver = pending
            .resolver
            .take()
            .expect("pull_remote call already resolved");
        let _ = resolver.send(result);
    }

    /// Resolve the Nth `local_g2_to_g1` call (0-indexed) with `result`.
    pub fn resolve_onboard(&self, index: usize, result: Result<()>) {
        let mut onboards = self.onboards.lock();
        let pending = onboards
            .get_mut(index)
            .expect("local_g2_to_g1 call not yet recorded");
        let resolver = pending
            .resolver
            .take()
            .expect("local_g2_to_g1 call already resolved");
        let _ = resolver.send(result);
    }

    /// Wait until at least `n` pull calls have been recorded.
    pub async fn wait_pull_count(&self, n: usize) {
        wait_until(|| self.pulls.lock().len() >= n).await;
    }

    /// Wait until at least `n` onboard calls have been recorded.
    pub async fn wait_onboard_count(&self, n: usize) {
        wait_until(|| self.onboards.lock().len() >= n).await;
    }
}

// ============================================================================
// MockCdWorkerHook
// ============================================================================

#[derive(Debug, Clone)]
pub struct CompleteCall {
    pub request_id: String,
}

#[derive(Debug, Clone)]
pub struct FailedCall {
    pub request_id: String,
    pub block_ids: Vec<BlockId>,
}

#[derive(Default)]
pub struct MockCdWorkerHook {
    completed: Mutex<Vec<CompleteCall>>,
    failed: Mutex<Vec<FailedCall>>,
}

impl MockCdWorkerHook {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn completed(&self) -> Vec<CompleteCall> {
        self.completed.lock().clone()
    }

    pub fn failed(&self) -> Vec<FailedCall> {
        self.failed.lock().clone()
    }

    pub fn completed_contains(&self, request_id: &str) -> bool {
        self.completed
            .lock()
            .iter()
            .any(|call| call.request_id == request_id)
    }

    pub fn failed_for(&self, request_id: &str) -> Option<FailedCall> {
        self.failed
            .lock()
            .iter()
            .find(|call| call.request_id == request_id)
            .cloned()
    }
}

// ============================================================================
// MockInnerLeaderShim
// ============================================================================
//
// A scriptable [`InnerLeaderShim`] for the decode-side E2E test. Backed by a
// real [`BlockManager<G2>`] so the lifecycle types (`MutableBlock<G2>` →
// `CompleteBlock<G2>` → `ImmutableBlock<G2>`) flow through the wrapper just
// like in production, without spinning up a `KvbmRuntime` / `nixl_agent`.

use crate::common::Request as CrateRequest;
use crate::connector::leader::FinishedStatus;
use crate::connector::leader::SlotMatchSplit;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};

/// Scripted state for a single mock slot.
pub struct MockSlot {
    pub block_size: usize,
    /// Total token blocks in the sequence (`N`).
    pub total_blocks: usize,
    /// Number of computed blocks at the start (`COMPUTED`).
    pub computed_blocks: usize,
    /// Local-match blocks (`X - COMPUTED`).
    pub local_match_blocks: usize,
    /// All sequence hashes, length == `total_blocks`.
    pub all_hashes: Vec<SequenceHash>,
    /// Real `TokenBlock`s for the full sequence, indexed 0..total_blocks.
    pub token_blocks: Vec<dynamo_tokens::TokenBlock>,
    /// Real, registered `ImmutableBlock<G2>`s for the local-match range.
    /// Drained by `take_local_match_g2_blocks`.
    pub local_match_g2: Mutex<Option<Vec<ImmutableBlock<G2>>>>,
    /// Block_ids vLLM "allocated" for the slot, set by `apply_block_assignments`.
    pub assigned_block_ids: Mutex<Option<Vec<crate::BlockId>>>,
    /// First-call inner GNMT result so the wrapper proceeds to the policy
    /// branch.
    pub gnmt_result: (Option<usize>, bool),
    /// `update_state_after_alloc` calls observed (USAA-2 path).
    pub usaa_passthrough_calls: Mutex<Vec<(Vec<crate::BlockId>, usize)>>,
    /// Scriptable CD transfer params surfaced via
    /// `slot_transfer_params`. `None` for non-CD requests.
    pub transfer_params: Option<TransferParams>,
}

pub struct MockInnerLeaderShim {
    block_size: usize,
    local_id: crate::InstanceId,
    g2_manager: Arc<BlockManager<G2>>,
    slots: Mutex<std::collections::HashMap<String, Arc<MockSlot>>>,
}

impl MockInnerLeaderShim {
    pub fn new(block_size: usize, g2_manager: Arc<BlockManager<G2>>) -> Arc<Self> {
        Arc::new(Self {
            block_size,
            local_id: uuid::Uuid::new_v4().into(),
            g2_manager,
            slots: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub fn local_id(&self) -> crate::InstanceId {
        self.local_id
    }

    pub fn install_slot(&self, request_id: impl Into<String>, slot: MockSlot) {
        self.slots.lock().insert(request_id.into(), Arc::new(slot));
    }

    pub fn slot(&self, request_id: &str) -> Option<Arc<MockSlot>> {
        self.slots.lock().get(request_id).cloned()
    }

    fn require_slot(&self, request_id: &str) -> Result<Arc<MockSlot>> {
        self.slot(request_id)
            .ok_or_else(|| anyhow!("MockInnerLeaderShim: no slot for {}", request_id))
    }
}

impl InnerLeaderShim for MockInnerLeaderShim {
    fn create_slot(&self, _request: CrateRequest) -> Result<()> {
        Ok(())
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.slots.lock().contains_key(request_id)
    }

    fn extend_slot_tokens(
        &self,
        _request_id: &str,
        _tokens: Vec<u32>,
    ) -> Result<()> {
        Ok(())
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        _num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        Ok(self.require_slot(request_id)?.gnmt_result)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<crate::BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        if let Some(slot) = self.slot(request_id) {
            slot.usaa_passthrough_calls
                .lock()
                .push((block_ids, num_external_tokens));
        }
        Ok(())
    }

    fn build_connector_meta(
        &self,
        _output: SchedulerOutput,
    ) -> Result<KvConnectorMetadata> {
        Err(anyhow!(
            "MockInnerLeaderShim::build_connector_meta not implemented"
        ))
    }

    fn update_connector_output(
        &self,
        _finished_sending: std::collections::HashSet<String>,
        _finished_recving: std::collections::HashSet<String>,
    ) -> Result<()> {
        Ok(())
    }

    fn request_finished(&self, _request_id: &str) -> FinishedStatus {
        FinishedStatus::Finished
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize> {
        let slot = self.require_slot(request_id)?;
        Ok(slot.total_blocks * slot.block_size)
    }

    fn slot_match_split(&self, request_id: &str) -> Result<SlotMatchSplit> {
        let slot = self.require_slot(request_id)?;
        Ok(SlotMatchSplit {
            block_size: slot.block_size,
            computed_blocks: slot.computed_blocks,
            local_match_blocks: slot.local_match_blocks,
            total_blocks: slot.total_blocks,
            all_sequence_hashes: slot.all_hashes.clone(),
        })
    }

    fn slot_token_ids(&self, request_id: &str) -> Result<Vec<u32>> {
        let slot = self.require_slot(request_id)?;
        let mut out = Vec::with_capacity(slot.total_blocks * slot.block_size);
        for tb in &slot.token_blocks {
            out.extend(tb.tokens().iter().copied());
        }
        Ok(out)
    }

    fn local_instance_id(&self) -> crate::InstanceId {
        self.local_id
    }

    fn apply_block_assignments(
        &self,
        request_id: &str,
        block_ids: Vec<crate::BlockId>,
    ) -> Result<()> {
        let slot = self.require_slot(request_id)?;
        *slot.assigned_block_ids.lock() = Some(block_ids);
        Ok(())
    }

    fn take_local_match_g2_blocks(
        &self,
        request_id: &str,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        let slot = self.require_slot(request_id)?;
        slot.local_match_g2
            .lock()
            .take()
            .ok_or_else(|| anyhow!("local_match_g2 already taken for {}", request_id))
    }

    fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>> {
        let slot = self.require_slot(request_id)?;
        if range.end > slot.token_blocks.len() {
            anyhow::bail!(
                "token_blocks_for_range: out of bounds {:?} (len {})",
                range,
                slot.token_blocks.len()
            );
        }
        Ok(slot.token_blocks[range].to_vec())
    }

    fn slot_transfer_params(
        &self,
        request_id: &str,
    ) -> Result<Option<TransferParams>> {
        let slot = self.require_slot(request_id)?;
        Ok(slot.transfer_params.clone())
    }

    fn allocate_g2_blocks(
        &self,
        count: usize,
    ) -> Result<Vec<MutableBlock<G2>>> {
        self.g2_manager
            .allocate_blocks(count)
            .ok_or_else(|| anyhow!("MockInnerLeaderShim: G2 alloc {} failed", count))
    }

    fn register_g2_blocks(
        &self,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        Ok(self.g2_manager.register_blocks(blocks))
    }
}

impl CdWorkerHook for MockCdWorkerHook {
    fn mark_onboarding_complete(&self, request_id: String) -> BoxFuture<'static, Result<()>> {
        self.completed.lock().push(CompleteCall { request_id });
        async { Ok(()) }.boxed()
    }

    fn mark_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>> {
        self.failed.lock().push(FailedCall {
            request_id,
            block_ids,
        });
        async { Ok(()) }.boxed()
    }
}

impl CdBlockTransport for MockCdBlockTransport {
    fn pull_remote(
        &self,
        remote_instance: crate::InstanceId,
        block_sets: Vec<RemoteBlockSet>,
        local_dst_g2_block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>> {
        let (tx, rx) = oneshot::channel();
        self.pulls.lock().push(PendingPullCall {
            call: PullRemoteCall {
                remote_instance,
                block_sets,
                local_dst_g2_block_ids,
            },
            resolver: Some(tx),
        });
        async move {
            rx.await
                .map_err(|err| anyhow!("pull_remote resolver dropped: {err}"))?
        }
        .boxed()
    }

    fn local_g2_to_g1(
        &self,
        src_g2_block_ids: Vec<BlockId>,
        dst_g1_block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>> {
        let (tx, rx) = oneshot::channel();
        self.onboards.lock().push(PendingOnboardCall {
            call: LocalG2ToG1Call {
                src_g2_block_ids,
                dst_g1_block_ids,
            },
            resolver: Some(tx),
        });
        async move {
            rx.await
                .map_err(|err| anyhow!("local_g2_to_g1 resolver dropped: {err}"))?
        }
        .boxed()
    }
}
