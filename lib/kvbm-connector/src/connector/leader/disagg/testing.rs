// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test doubles for conditional-disaggregation transport / worker-hook /
//! inner-leader-shim / queue traits.
//!
//! Session-side mocks (`MockSession` + `MockSessionFactory`) live in the
//! engine crate at `kvbm_engine::disagg::session::testing`.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::{FutureExt, future::BoxFuture};
use kvbm_disagg_protocol::{RemotePrefillRequest, TransferParams};
use tokio::sync::oneshot;

use super::transport::{CdBlockTransport, CdWorkerHook, InnerLeaderShim};
use kvbm_logical::blocks::{CompleteBlock, MutableBlock};
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use parking_lot::Mutex;

use super::queue::RemotePrefillQueue;
use crate::{BlockId, G2, SequenceHash};

pub const TEST_BLOCK_SIZE: usize = 16;

// ============================================================================
// InMemoryRemotePrefillQueue
// ============================================================================

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

// ============================================================================
// wait_until — sleep-poll predicate helper
// ============================================================================

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
// MockCdBlockTransport — local G2→G1 only (remote pull is now handled by
// `Session::pull` inside the engine).
// ============================================================================

#[derive(Debug, Clone)]
pub struct LocalG2ToG1Call {
    pub src_g2_block_ids: Vec<BlockId>,
    pub dst_g1_block_ids: Vec<BlockId>,
}

pub struct MockCdBlockTransport {
    onboards: Mutex<Vec<PendingOnboardCall>>,
}

struct PendingOnboardCall {
    call: LocalG2ToG1Call,
    resolver: Option<oneshot::Sender<Result<()>>>,
}

impl MockCdBlockTransport {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            onboards: Mutex::new(Vec::new()),
        })
    }

    pub fn onboard_calls(&self) -> Vec<LocalG2ToG1Call> {
        self.onboards.lock().iter().map(|o| o.call.clone()).collect()
    }

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

    pub async fn wait_onboard_count(&self, n: usize) {
        wait_until(|| self.onboards.lock().len() >= n).await;
    }
}

impl CdBlockTransport for MockCdBlockTransport {
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

// ============================================================================
// MockInnerLeaderShim — scriptable inner leader for wrapper E2E tests.
//
// Backed by a real BlockManager<G2> so the lifecycle types (MutableBlock<G2>
// → CompleteBlock<G2> → ImmutableBlock<G2>) flow through the wrapper just
// like in production, without spinning up a `KvbmRuntime` / `nixl_agent`.
// ============================================================================

use crate::common::Request as CrateRequest;
use crate::connector::leader::FinishedStatus;
use crate::connector::leader::SlotMatchSplit;
use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};

pub struct MockSlot {
    pub block_size: usize,
    pub total_blocks: usize,
    pub computed_blocks: usize,
    pub local_match_blocks: usize,
    pub all_hashes: Vec<SequenceHash>,
    pub token_blocks: Vec<dynamo_tokens::TokenBlock>,
    pub local_match_g2: Mutex<Option<Vec<ImmutableBlock<G2>>>>,
    pub assigned_block_ids: Mutex<Option<Vec<crate::BlockId>>>,
    pub gnmt_result: (Option<usize>, bool),
    pub usaa_passthrough_calls: Mutex<Vec<(Vec<crate::BlockId>, usize)>>,
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

    fn extend_slot_tokens(&self, _request_id: &str, _tokens: Vec<u32>) -> Result<()> {
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

    fn build_connector_meta(&self, _output: SchedulerOutput) -> Result<KvConnectorMetadata> {
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

    fn slot_transfer_params(&self, request_id: &str) -> Result<Option<TransferParams>> {
        let slot = self.require_slot(request_id)?;
        Ok(slot.transfer_params.clone())
    }

    fn allocate_g2_blocks(&self, count: usize) -> Result<Vec<MutableBlock<G2>>> {
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
