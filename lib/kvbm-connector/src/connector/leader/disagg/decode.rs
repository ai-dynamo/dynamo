// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side conditional-disaggregation coordinator.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use futures::StreamExt;
use kvbm_disagg_protocol::{
    BlockDescriptor, DISAGG_PROTOCOL_VERSION, DescriptorResponse, DisaggBlockRef,
    DisaggSequenceHash, HashSelection, RemotePrefillRequest, SessionId, UnpinAck,
};
use kvbm_logical::blocks::ImmutableBlock;
use parking_lot::Mutex;
use tokio::runtime::Handle;

use super::queue::RemotePrefillQueue;
use super::session::{
    PrefillSession, PrefillSessionFactory, SessionBlocks, SessionEvent, hash_to_wire,
    hashes_to_wire,
};
use super::{ConditionalDisaggPolicy, PolicyInputs, PrefillSelection};
use crate::{G2, InstanceId, SequenceHash};

const DEFAULT_ATTACH_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeginOutcome {
    Started { session_id: SessionId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemotePrefillStatus {
    AwaitingAttach,
    Attached,
    PrefillPullComplete,
    Failed,
    Released,
}

pub struct RemotePrefillState {
    pub session: Arc<dyn PrefillSession>,
    pub initiator_instance_id: InstanceId,
    pub ready_g2_blocks: Vec<ImmutableBlock<G2>>,
    pub pending_hashes: Vec<SequenceHash>,
    pub initial_ready_hashes: HashSet<DisaggSequenceHash>,
    pub session_unpinned_hashes: HashSet<DisaggSequenceHash>,
    pub token_ids: Vec<u32>,
    pub num_computed_tokens: usize,
    pub num_prefill_tokens: usize,
    pub status: RemotePrefillStatus,
    pub failure_reason: Option<String>,
}

pub struct RemotePrefillCoordinator {
    policy: Arc<dyn ConditionalDisaggPolicy>,
    session_factory: Arc<dyn PrefillSessionFactory>,
    queue: Arc<dyn RemotePrefillQueue>,
    states: DashMap<String, Arc<Mutex<RemotePrefillState>>>,
    tokio_handle: Handle,
    attach_timeout: Duration,
}

impl RemotePrefillCoordinator {
    pub fn new(
        policy: Arc<dyn ConditionalDisaggPolicy>,
        session_factory: Arc<dyn PrefillSessionFactory>,
        queue: Arc<dyn RemotePrefillQueue>,
        tokio_handle: Handle,
    ) -> Arc<Self> {
        Self::with_attach_timeout(
            policy,
            session_factory,
            queue,
            tokio_handle,
            DEFAULT_ATTACH_TIMEOUT,
        )
    }

    pub fn with_attach_timeout(
        policy: Arc<dyn ConditionalDisaggPolicy>,
        session_factory: Arc<dyn PrefillSessionFactory>,
        queue: Arc<dyn RemotePrefillQueue>,
        tokio_handle: Handle,
        attach_timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            policy,
            session_factory,
            queue,
            states: DashMap::new(),
            tokio_handle,
            attach_timeout,
        })
    }

    pub fn evaluate(&self, inputs: &PolicyInputs) -> PrefillSelection {
        self.policy.evaluate(inputs)
    }

    pub fn begin_remote_prefill(
        self: &Arc<Self>,
        request_id: &str,
        inputs: &PolicyInputs,
        initiator_instance_id: InstanceId,
        session_blocks: SessionBlocks,
        sequence_hashes: Vec<SequenceHash>,
        token_ids: Vec<u32>,
    ) -> Result<BeginOutcome> {
        if self.states.contains_key(request_id) {
            anyhow::bail!(
                "begin_remote_prefill called twice for request_id={}",
                request_id
            );
        }

        let session_id = uuid::Uuid::new_v4();
        let session = self.session_factory.create_decode(session_id)?;
        session.add_ready_blocks(session_blocks.ready_g2.clone())?;
        session.add_pending_hashes(session_blocks.pending_hashes.clone())?;
        let event_stream = session.subscribe();

        let initial_ready_hashes: HashSet<_> =
            session_blocks.ready_hashes_wire().into_iter().collect();
        let state = RemotePrefillState {
            session: session.clone(),
            initiator_instance_id,
            ready_g2_blocks: session_blocks.ready_g2,
            pending_hashes: session_blocks.pending_hashes,
            initial_ready_hashes,
            session_unpinned_hashes: HashSet::new(),
            token_ids: token_ids.clone(),
            num_computed_tokens: inputs.num_computed_tokens,
            num_prefill_tokens: inputs.num_prefill_tokens(),
            status: RemotePrefillStatus::AwaitingAttach,
            failure_reason: None,
        };

        self.states
            .insert(request_id.to_string(), Arc::new(Mutex::new(state)));

        let request = RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: request_id.to_string(),
            session_id,
            initiator_instance_id,
            decode_endpoint: session.endpoint(),
            sequence_hashes: hashes_to_wire(sequence_hashes),
            token_ids,
            num_computed_tokens: inputs.num_computed_tokens,
        };

        if let Err(err) = self.queue.enqueue(request) {
            self.states.remove(request_id);
            return Err(err);
        }

        let coord = Arc::clone(self);
        let request_id = request_id.to_string();
        self.tokio_handle.spawn(async move {
            monitor_loop(coord, request_id, event_stream).await;
        });

        Ok(BeginOutcome::Started { session_id })
    }

    pub fn state_for(&self, request_id: &str) -> Option<Arc<Mutex<RemotePrefillState>>> {
        self.states
            .get(request_id)
            .map(|entry| entry.value().clone())
    }

    pub fn status_for(&self, request_id: &str) -> Option<RemotePrefillStatus> {
        self.state_for(request_id).map(|state| state.lock().status)
    }

    pub fn active_count(&self) -> usize {
        self.states.len()
    }

    pub fn release(&self, request_id: &str) {
        if let Some((_, state)) = self.states.remove(request_id) {
            state.lock().session.close(Some("released".to_string()));
        }
    }

    fn is_awaiting_attach(&self, request_id: &str) -> bool {
        self.status_for(request_id) == Some(RemotePrefillStatus::AwaitingAttach)
    }

    fn mark_failed(&self, request_id: &str, reason: String) {
        if let Some(state) = self.state_for(request_id) {
            let session = {
                let mut state = state.lock();
                state.status = RemotePrefillStatus::Failed;
                state.failure_reason = Some(reason.clone());
                state.session.clone()
            };
            session.close(Some(reason));
        }
    }

    fn handle_attached(&self, request_id: &str) {
        if let Some(state) = self.state_for(request_id) {
            let mut state = state.lock();
            if state.status == RemotePrefillStatus::AwaitingAttach {
                state.status = RemotePrefillStatus::Attached;
            }
        }
    }

    fn handle_descriptor_request(&self, request_id: &str, selection: HashSelection) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };

        let (session, response) = {
            let state = state.lock();
            (
                state.session.clone(),
                descriptor_response(&state, &selection),
            )
        };

        if let Err(err) = session.respond_to_descriptor_request(response) {
            self.mark_failed(request_id, format!("descriptor response failed: {err}"));
        }
    }

    fn handle_unpin_requested(
        &self,
        request_id: &str,
        request: kvbm_disagg_protocol::UnpinRequest,
    ) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };

        let (session, released, complete) = {
            let mut state = state.lock();
            let session = state.session.clone();
            let released = match session.release_session_pins(&request.hashes) {
                Ok(released) => released,
                Err(err) => {
                    drop(state);
                    self.mark_failed(request_id, format!("session unpin failed: {err}"));
                    return;
                }
            };
            state
                .session_unpinned_hashes
                .extend(released.iter().cloned());
            let complete = state
                .initial_ready_hashes
                .iter()
                .all(|hash| state.session_unpinned_hashes.contains(hash));
            if complete {
                state.status = RemotePrefillStatus::PrefillPullComplete;
            }
            (session, released, complete)
        };

        let ack_hashes = if matches!(request.hashes, HashSelection::All) {
            HashSelection::All
        } else {
            HashSelection::Hashes(released)
        };

        if let Err(err) = session.ack_unpin(UnpinAck {
            request_id: request.request_id,
            hashes: ack_hashes,
        }) {
            self.mark_failed(request_id, format!("unpin ack failed: {err}"));
            return;
        }

        if complete {
            tracing::debug!(request_id, "remote prefill initial pull complete");
        }
    }

    fn handle_detached(&self, request_id: &str, reason: Option<String>) {
        let Some(state) = self.state_for(request_id) else {
            return;
        };
        let (session, status) = {
            let mut state = state.lock();
            if state.status == RemotePrefillStatus::PrefillPullComplete {
                state.status = RemotePrefillStatus::Released;
            } else {
                state.status = RemotePrefillStatus::Failed;
                state.failure_reason = Some(
                    reason
                        .clone()
                        .unwrap_or_else(|| "detached before prefill pull complete".to_string()),
                );
            }
            (state.session.clone(), state.status)
        };
        session.close(reason.or_else(|| Some(format!("detached with status {status:?}"))));
    }
}

fn descriptor_response(
    state: &RemotePrefillState,
    selection: &HashSelection,
) -> DescriptorResponse {
    let requested = requested_hashes(selection, state);
    let ready: Vec<_> = state
        .ready_g2_blocks
        .iter()
        .filter(|block| requested.contains(&hash_to_wire(block.sequence_hash())))
        .map(block_ref)
        .collect();
    let ready_hashes: HashSet<_> = ready
        .iter()
        .map(|block| block.sequence_hash.clone())
        .collect();
    let pending_hashes = requested
        .into_iter()
        .filter(|hash| !ready_hashes.contains(hash))
        .collect();

    DescriptorResponse {
        ready_blocks: ready,
        pending_hashes,
    }
}

fn requested_hashes(
    selection: &HashSelection,
    state: &RemotePrefillState,
) -> Vec<DisaggSequenceHash> {
    match selection {
        HashSelection::All => state
            .ready_g2_blocks
            .iter()
            .map(|block| hash_to_wire(block.sequence_hash()))
            .chain(hashes_to_wire(state.pending_hashes.iter().copied()))
            .collect(),
        HashSelection::Hashes(hashes) => hashes.clone(),
    }
}

fn block_ref(block: &ImmutableBlock<G2>) -> DisaggBlockRef {
    DisaggBlockRef {
        block_id: block.block_id(),
        sequence_hash: hash_to_wire(block.sequence_hash()),
        layout_handle_raw: block.block_id().to_string(),
        descriptor: Some(BlockDescriptor::new(
            block.block_id().to_le_bytes().to_vec(),
        )),
    }
}

async fn monitor_loop(
    coord: Arc<RemotePrefillCoordinator>,
    request_id: String,
    mut event_stream: super::SessionEventStream,
) {
    let attach_timer = tokio::time::sleep(coord.attach_timeout);
    tokio::pin!(attach_timer);

    loop {
        tokio::select! {
            _ = &mut attach_timer, if coord.is_awaiting_attach(&request_id) => {
                coord.mark_failed(&request_id, "prefill attach timeout".to_string());
                break;
            }
            event = event_stream.next() => {
                match event {
                    Some(SessionEvent::Attached { .. }) => {
                        coord.handle_attached(&request_id);
                    }
                    Some(SessionEvent::DescriptorRequest(request)) => {
                        coord.handle_descriptor_request(&request_id, request.hashes);
                    }
                    Some(SessionEvent::UnpinRequested(request)) => {
                        coord.handle_unpin_requested(&request_id, request);
                    }
                    Some(SessionEvent::UnpinAcked(_)) => {
                        // Future decode-output path consumes peer acks for decode-initiated unpins.
                    }
                    Some(SessionEvent::BlocksAdded { .. }) => {
                        // Prefill output path lands in a later phase.
                    }
                    Some(SessionEvent::Detached { reason }) => {
                        coord.handle_detached(&request_id, reason);
                        break;
                    }
                    Some(SessionEvent::Failed { reason }) => {
                        coord.mark_failed(&request_id, reason);
                        break;
                    }
                    None => {
                        if coord.status_for(&request_id) != Some(RemotePrefillStatus::Released) {
                            coord.mark_failed(&request_id, "session event stream closed".to_string());
                        }
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use kvbm_disagg_protocol::{DescriptorRequest, HashSelection, UnpinRequest};

    use super::*;
    use crate::connector::leader::disagg::testing::{
        InMemoryRemotePrefillQueue, MockPrefillSessionFactory, test_g2_blocks, wait_until,
    };
    use crate::connector::leader::disagg::{AlwaysRemote, SessionBlocks};

    fn policy_inputs() -> PolicyInputs {
        PolicyInputs {
            total_tokens: 64,
            num_computed_tokens: 16,
            num_connector_tokens: 32,
            transfer_params: None,
        }
    }

    fn coordinator(
        attach_timeout: Duration,
    ) -> (
        Arc<RemotePrefillCoordinator>,
        Arc<MockPrefillSessionFactory>,
        Arc<InMemoryRemotePrefillQueue>,
    ) {
        let factory = MockPrefillSessionFactory::new();
        let queue = InMemoryRemotePrefillQueue::new();
        let coord = RemotePrefillCoordinator::with_attach_timeout(
            Arc::new(AlwaysRemote),
            factory.clone(),
            queue.clone(),
            Handle::current(),
            attach_timeout,
        );
        (coord, factory, queue)
    }

    #[tokio::test]
    async fn begin_remote_prefill_registers_session_and_queue_item() {
        let (coord, factory, queue) = coordinator(Duration::from_secs(120));
        let blocks = test_g2_blocks(2, 10);
        let hashes = blocks.hashes();
        let session_blocks = SessionBlocks::new(blocks.blocks.clone(), Vec::new());
        let inputs = policy_inputs();

        let outcome = coord
            .begin_remote_prefill(
                "req-1",
                &inputs,
                uuid::Uuid::new_v4().into(),
                session_blocks,
                hashes.clone(),
                vec![1, 2, 3],
            )
            .expect("begin remote prefill");

        let BeginOutcome::Started { session_id } = outcome;
        let session = factory.last().expect("session created");
        assert_eq!(session.ready_hashes(), blocks.hashes_wire());
        assert_eq!(
            coord.status_for("req-1"),
            Some(RemotePrefillStatus::AwaitingAttach)
        );

        let queued = queue.snapshot();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].request_id, "req-1");
        assert_eq!(queued[0].session_id, session_id);
        assert_eq!(queued[0].sequence_hashes, session.ready_hashes());
        assert_eq!(queued[0].num_computed_tokens, inputs.num_computed_tokens);
    }

    #[tokio::test]
    async fn attached_event_transitions_session_status() {
        let (coord, factory, _) = coordinator(Duration::from_secs(120));
        let blocks = test_g2_blocks(1, 20);
        let hashes = blocks.hashes();

        coord
            .begin_remote_prefill(
                "req-attach",
                &policy_inputs(),
                uuid::Uuid::new_v4().into(),
                SessionBlocks::new(blocks.blocks, Vec::new()),
                hashes,
                vec![9],
            )
            .expect("begin remote prefill");

        let session = factory.last().expect("session created");
        session
            .push_event(SessionEvent::Attached {
                peer_instance_id: uuid::Uuid::new_v4().into(),
            })
            .expect("push attached");

        wait_until(|| coord.status_for("req-attach") == Some(RemotePrefillStatus::Attached)).await;
    }

    #[tokio::test]
    async fn attach_timeout_marks_session_failed_and_closes() {
        let (coord, factory, _) = coordinator(Duration::from_millis(10));
        let blocks = test_g2_blocks(1, 30);
        let hashes = blocks.hashes();

        coord
            .begin_remote_prefill(
                "req-timeout",
                &policy_inputs(),
                uuid::Uuid::new_v4().into(),
                SessionBlocks::new(blocks.blocks, Vec::new()),
                hashes,
                vec![10],
            )
            .expect("begin remote prefill");

        wait_until(|| coord.status_for("req-timeout") == Some(RemotePrefillStatus::Failed)).await;

        let session = factory.last().expect("session created");
        assert_eq!(
            session.closed_reason(),
            Some(Some("prefill attach timeout".to_string()))
        );
    }

    #[tokio::test]
    async fn descriptor_request_returns_ready_and_pending_hashes() {
        let (coord, factory, _) = coordinator(Duration::from_secs(120));
        let ready = test_g2_blocks(1, 40);
        let pending = test_g2_blocks(1, 100);
        let pending_hashes = pending.hashes();

        coord
            .begin_remote_prefill(
                "req-desc",
                &policy_inputs(),
                uuid::Uuid::new_v4().into(),
                SessionBlocks::new(ready.blocks.clone(), pending_hashes.clone()),
                ready
                    .hashes()
                    .into_iter()
                    .chain(pending_hashes.iter().copied())
                    .collect(),
                vec![11],
            )
            .expect("begin remote prefill");

        let session = factory.last().expect("session created");
        session
            .push_event(SessionEvent::Attached {
                peer_instance_id: uuid::Uuid::new_v4().into(),
            })
            .expect("push attached");
        session
            .push_event(SessionEvent::DescriptorRequest(DescriptorRequest {
                hashes: HashSelection::All,
            }))
            .expect("push descriptor request");

        wait_until(|| !session.descriptor_responses().is_empty()).await;
        let responses = session.descriptor_responses();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].ready_blocks.len(), 1);
        assert_eq!(
            responses[0].ready_blocks[0].sequence_hash,
            ready.hashes_wire()[0]
        );
        assert!(responses[0].ready_blocks[0].descriptor.is_some());
        assert_eq!(responses[0].pending_hashes, pending.hashes_wire());
    }

    #[tokio::test]
    async fn unpin_request_releases_only_session_pins_and_acks() {
        let (coord, factory, _) = coordinator(Duration::from_secs(120));
        let blocks = test_g2_blocks(2, 50);
        let hashes_wire = blocks.hashes_wire();

        coord
            .begin_remote_prefill(
                "req-unpin",
                &policy_inputs(),
                uuid::Uuid::new_v4().into(),
                SessionBlocks::new(blocks.blocks.clone(), Vec::new()),
                blocks.hashes(),
                vec![12],
            )
            .expect("begin remote prefill");

        let session = factory.last().expect("session created");
        session
            .push_event(SessionEvent::Attached {
                peer_instance_id: uuid::Uuid::new_v4().into(),
            })
            .expect("push attached");
        session
            .push_event(SessionEvent::UnpinRequested(UnpinRequest {
                request_id: "unpin-1".to_string(),
                hashes: HashSelection::Hashes(vec![hashes_wire[0].clone()]),
            }))
            .expect("push unpin request");

        wait_until(|| !session.unpin_acks().is_empty()).await;

        let acks = session.unpin_acks();
        assert_eq!(acks.len(), 1);
        assert_eq!(
            acks[0].hashes,
            HashSelection::Hashes(vec![hashes_wire[0].clone()])
        );
        assert_eq!(session.ready_hashes(), vec![hashes_wire[1].clone()]);

        let state = coord.state_for("req-unpin").expect("state");
        let state = state.lock();
        assert_eq!(state.status, RemotePrefillStatus::Attached);
        assert_eq!(state.ready_g2_blocks.len(), 2);
        assert!(state.session_unpinned_hashes.contains(&hashes_wire[0]));
    }

    #[tokio::test]
    async fn unpin_all_marks_initial_pull_complete_after_ack() {
        let (coord, factory, _) = coordinator(Duration::from_secs(120));
        let blocks = test_g2_blocks(2, 70);
        let hashes = blocks.hashes();

        coord
            .begin_remote_prefill(
                "req-unpin-all",
                &policy_inputs(),
                uuid::Uuid::new_v4().into(),
                SessionBlocks::new(blocks.blocks, Vec::new()),
                hashes,
                vec![13],
            )
            .expect("begin remote prefill");

        let session = factory.last().expect("session created");
        session
            .push_event(SessionEvent::Attached {
                peer_instance_id: uuid::Uuid::new_v4().into(),
            })
            .expect("push attached");
        session
            .push_event(SessionEvent::UnpinRequested(UnpinRequest {
                request_id: "unpin-all".to_string(),
                hashes: HashSelection::All,
            }))
            .expect("push unpin request");

        wait_until(|| {
            coord.status_for("req-unpin-all") == Some(RemotePrefillStatus::PrefillPullComplete)
        })
        .await;

        let acks = session.unpin_acks();
        assert_eq!(acks.len(), 1);
        assert_eq!(acks[0].hashes, HashSelection::All);
        assert!(session.ready_hashes().is_empty());
    }
}
