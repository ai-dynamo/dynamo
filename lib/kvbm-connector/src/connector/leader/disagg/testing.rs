// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test doubles for disaggregation transport / worker-hook /
//! inner-leader-shim / queue traits.
//!
//! Session-side mocks (`MockSession` + `MockSessionFactory`) live in the
//! engine crate at `kvbm_engine::p2p::session::testing`.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::{FutureExt, future::BoxFuture};
use kvbm_engine::offload::ExternalBlock;
use kvbm_protocols::disagg::{RemotePrefillRequest, TransferParams};
use tokio::sync::oneshot;

use crate::connector::leader::p2p::transport::{InnerLeaderShim, P2pBlockTransport, P2pWorkerHook};
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::blocks::{CompleteBlock, MutableBlock};
use kvbm_logical::manager::BlockManager;
use parking_lot::Mutex;

use super::queue::RemotePrefillQueue;
use crate::{BlockId, G1, G2, SequenceHash};

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
// MockP2pBlockTransport — local G2→G1 only (remote pull is now handled by
// `Session::pull` inside the engine).
// ============================================================================

#[derive(Debug, Clone)]
pub struct LocalG2ToG1Call {
    pub src_g2_block_ids: Vec<BlockId>,
    pub dst_g1_block_ids: Vec<BlockId>,
}

pub struct MockP2pBlockTransport {
    onboards: Mutex<Vec<PendingOnboardCall>>,
}

struct PendingOnboardCall {
    call: LocalG2ToG1Call,
    resolver: Option<oneshot::Sender<Result<()>>>,
}

impl MockP2pBlockTransport {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            onboards: Mutex::new(Vec::new()),
        })
    }

    pub fn onboard_calls(&self) -> Vec<LocalG2ToG1Call> {
        self.onboards
            .lock()
            .iter()
            .map(|o| o.call.clone())
            .collect()
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

impl P2pBlockTransport for MockP2pBlockTransport {
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
// MockP2pWorkerHook
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
pub struct MockP2pWorkerHook {
    completed: Mutex<Vec<CompleteCall>>,
    failed: Mutex<Vec<FailedCall>>,
}

impl MockP2pWorkerHook {
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

impl P2pWorkerHook for MockP2pWorkerHook {
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
    /// Wrapped in a mutex so tests that need to swap transfer params
    /// post-`install_slot` (e.g. the digest-mismatch verifier reject
    /// test) can do so without re-installing the whole slot. Production
    /// inner shims don't share this shape — this is a mock-only choice.
    pub transfer_params: Mutex<Option<TransferParams>>,
    /// Recorded request_ids for which a CD onboarding payload was
    /// installed via `install_cd_onboarding_payload`. Order
    /// preserved.
    pub installed_cd_payloads: Mutex<Vec<String>>,
    /// The most recently installed CD onboarding payload — held
    /// alive so its `Drop` doesn't fire until the test takes it.
    pub installed_cd_payload:
        Mutex<Option<Box<dyn crate::connector::leader::slot::CdOnboardingPayload>>>,
}

impl Default for MockSlot {
    fn default() -> Self {
        Self {
            block_size: 0,
            total_blocks: 0,
            computed_blocks: 0,
            local_match_blocks: 0,
            all_hashes: Vec::new(),
            token_blocks: Vec::new(),
            local_match_g2: Mutex::new(None),
            assigned_block_ids: Mutex::new(None),
            gnmt_result: (None, false),
            usaa_passthrough_calls: Mutex::new(Vec::new()),
            transfer_params: Mutex::new(None),
            installed_cd_payloads: Mutex::new(Vec::new()),
            installed_cd_payload: Mutex::new(None),
        }
    }
}

/// Per-call record for `promote_g1_to_g2`, recorded in
/// [`MockInnerLeaderShim::promotions`]. Tests drive completion
/// (or failure) via [`MockInnerLeaderShim::resolve_promotion`].
pub struct PendingPromotion {
    pub source_blocks: Vec<ExternalBlock<G1>>,
    /// One-shot resolver consumed by the returned future. Taken on
    /// first `resolve_promotion`; subsequent calls panic.
    resolver: Option<oneshot::Sender<Result<Vec<ImmutableBlock<G2>>>>>,
}

impl PendingPromotion {
    pub fn source_block_ids(&self) -> Vec<BlockId> {
        self.source_blocks.iter().map(|b| b.block_id).collect()
    }

    /// The sequence hashes the production offload pipeline would
    /// register the resulting G2 blocks with — read directly off
    /// each `ExternalBlock<G1>`. Tests assert on this to verify
    /// the caller built source blocks with the correct hashes
    /// (any mismatch would silently mis-register in production).
    pub fn source_hashes(&self) -> Vec<SequenceHash> {
        self.source_blocks.iter().map(|b| b.sequence_hash).collect()
    }
}

/// Per-call record for `promote_g3_to_g2`, recorded in
/// [`MockInnerLeaderShim::g3_promotions`]. Tests inspect entries
/// and resolve them manually via
/// [`MockInnerLeaderShim::resolve_g3_promotion`].
pub struct PendingG3Promotion {
    pub hashes: Vec<SequenceHash>,
    /// One-shot resolver consumed by the returned future.
    resolver: Option<oneshot::Sender<Result<Vec<ImmutableBlock<G2>>>>>,
}

pub struct MockInnerLeaderShim {
    block_size: usize,
    local_id: crate::InstanceId,
    g2_manager: Arc<BlockManager<G2>>,
    slots: Mutex<std::collections::HashMap<String, Arc<MockSlot>>>,
    /// Per-request invocation counter for `get_num_new_matched_tokens`.
    /// Used by idempotency tests to assert the wrapper short-circuits
    /// repeat gnmt calls without re-invoking inner.
    gnmt_call_counts: Mutex<std::collections::HashMap<String, usize>>,
    /// Test-only hook fired inside `apply_block_assignments`. Lets a
    /// test inject side effects (e.g., stashing `pending_failure` on
    /// the wrapper's cd_request_state) at a specific point in
    /// `commit_usaa1`'s execution — between the outer
    /// pending_failure re-check and the post-insert re-check. Used
    /// by the commit_usaa1 post-insert replay reproducer.
    apply_block_assignments_hook: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    /// G1→G2 promotion requests recorded by `promote_g1_to_g2`.
    /// Tests inspect entries and resolve them manually via
    /// `resolve_promotion(index, Ok(g2_blocks))` to drive the
    /// returned futures forward.
    promotions: Mutex<Vec<PendingPromotion>>,
    /// G3→G2 promotion requests recorded by `promote_g3_to_g2`.
    /// Tests inspect entries and resolve them via
    /// `resolve_g3_promotion(index, Ok(g2_blocks))`. Stage 2
    /// counterpart of `promotions`.
    g3_promotions: Mutex<Vec<PendingG3Promotion>>,
    /// Per-slot G3 hash universe used by `find_prefix_g3_hashes`.
    /// Empty by default — `install_g3_prefix(rid, hashes)` lets
    /// tests script which hashes the mock "has" in G3. The
    /// returned set is the prefix slice of the slot's full hash
    /// chain that matches against this universe.
    g3_universe: Mutex<std::collections::HashMap<String, Vec<SequenceHash>>>,
}

impl MockInnerLeaderShim {
    pub fn new(block_size: usize, g2_manager: Arc<BlockManager<G2>>) -> Arc<Self> {
        Arc::new(Self {
            block_size,
            local_id: uuid::Uuid::new_v4().into(),
            g2_manager,
            slots: Mutex::new(std::collections::HashMap::new()),
            gnmt_call_counts: Mutex::new(std::collections::HashMap::new()),
            apply_block_assignments_hook: Mutex::new(None),
            promotions: Mutex::new(Vec::new()),
            g3_promotions: Mutex::new(Vec::new()),
            g3_universe: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub fn g2_manager(&self) -> &Arc<BlockManager<G2>> {
        &self.g2_manager
    }

    /// Number of `promote_g1_to_g2` calls observed so far.
    pub fn promotion_count(&self) -> usize {
        self.promotions.lock().len()
    }

    /// Snapshot of recorded promotion requests as
    /// `(source_block_ids, source_sequence_hashes)`. The hashes are
    /// taken from the `ExternalBlock<G1>` entries themselves — same
    /// hashes the production offload pipeline registers the
    /// resulting G2 blocks with.
    pub fn snapshot_promotion(&self, index: usize) -> Option<(Vec<BlockId>, Vec<SequenceHash>)> {
        self.promotions
            .lock()
            .get(index)
            .map(|p| (p.source_block_ids(), p.source_hashes()))
    }

    /// Resolve a recorded promotion request. Drives the returned
    /// future forward; the caller is responsible for constructing
    /// the resolved `Vec<ImmutableBlock<G2>>` via
    /// `register_g2_blocks` against the shim's `g2_manager` (or
    /// providing an `Err` to simulate transfer failure).
    pub fn resolve_promotion(&self, index: usize, result: Result<Vec<ImmutableBlock<G2>>>) {
        let mut promotions = self.promotions.lock();
        let pending = promotions
            .get_mut(index)
            .expect("promote_g1_to_g2 call not yet recorded");
        let resolver = pending
            .resolver
            .take()
            .expect("promote_g1_to_g2 call already resolved");
        let _ = resolver.send(result);
    }

    /// Async sleep-poll wait until at least `n` promotion requests
    /// have been recorded.
    pub async fn wait_promotion_count(&self, n: usize) {
        wait_until(|| self.promotions.lock().len() >= n).await;
    }

    /// Number of `promote_g3_to_g2` calls observed so far.
    pub fn g3_promotion_count(&self) -> usize {
        self.g3_promotions.lock().len()
    }

    /// Snapshot of recorded G3 promotion requests as the hashes
    /// the caller passed in. Tests assert on these to verify the
    /// caller sourced the right prefix slice.
    pub fn snapshot_g3_promotion(&self, index: usize) -> Option<Vec<SequenceHash>> {
        self.g3_promotions
            .lock()
            .get(index)
            .map(|p| p.hashes.clone())
    }

    /// Resolve a recorded G3 promotion request. Mirrors
    /// [`Self::resolve_promotion`].
    pub fn resolve_g3_promotion(&self, index: usize, result: Result<Vec<ImmutableBlock<G2>>>) {
        let mut promotions = self.g3_promotions.lock();
        let pending = promotions
            .get_mut(index)
            .expect("promote_g3_to_g2 call not yet recorded");
        let resolver = pending
            .resolver
            .take()
            .expect("promote_g3_to_g2 call already resolved");
        let _ = resolver.send(result);
    }

    pub async fn wait_g3_promotion_count(&self, n: usize) {
        wait_until(|| self.g3_promotions.lock().len() >= n).await;
    }

    /// Script which prefix hashes the mock's G3 "has." When
    /// `find_prefix_g3_hashes(rid, n)` is called, the mock
    /// returns `slot.all_hashes[..n].to_vec()` ONLY if all those
    /// hashes appear in this universe; otherwise empty. Empty
    /// universe (the default) returns empty — falls back to the
    /// G1 promotion path in production code.
    pub fn install_g3_prefix(&self, request_id: impl Into<String>, hashes: Vec<SequenceHash>) {
        self.g3_universe.lock().insert(request_id.into(), hashes);
    }

    /// Install a one-shot side-effect hook that fires synchronously
    /// inside `apply_block_assignments` after the slot's block_ids
    /// are recorded but before the function returns. Used by tests
    /// to inject state mutations at a precise point in
    /// `commit_usaa1`'s execution.
    pub fn set_apply_block_assignments_hook(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        *self.apply_block_assignments_hook.lock() = Some(hook);
    }

    /// How many times `get_num_new_matched_tokens` was called for
    /// `request_id`. Used by idempotency tests.
    pub fn gnmt_call_count(&self, request_id: &str) -> usize {
        *self.gnmt_call_counts.lock().get(request_id).unwrap_or(&0)
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
        *self
            .gnmt_call_counts
            .lock()
            .entry(request_id.to_string())
            .or_insert(0) += 1;
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

    fn slot_lora_name(&self, _request_id: &str) -> Result<Option<String>> {
        // Mock requests have no LoRA — the CD wire's loud-fail guard
        // depends on these returning None for the assert-None contract.
        Ok(None)
    }

    fn slot_salt(&self, _request_id: &str) -> Result<Option<String>> {
        Ok(None)
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
        // Fire the test hook (if installed) AFTER the slot mutation
        // but before returning. This is the deterministic injection
        // point used by the commit_usaa1 post-insert replay test.
        let hook = self.apply_block_assignments_hook.lock().clone();
        if let Some(hook) = hook {
            hook();
        }
        Ok(())
    }

    fn take_local_match_g2_blocks(&self, request_id: &str) -> Result<Vec<ImmutableBlock<G2>>> {
        let slot = self.require_slot(request_id)?;
        slot.local_match_g2
            .lock()
            .take()
            .ok_or_else(|| anyhow!("local_match_g2 already taken for {}", request_id))
    }

    fn find_prefix_g2_blocks(
        &self,
        _request_id: &str,
        _num_prefix_blocks: usize,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        // Tests today exercise the prefix-caching-disabled code path
        // (num_computed_tokens = 0 → num_prefix_blocks = 0). When the
        // PC-enabled path is exercised, extend `MockSlot` with a
        // `prefix_match_g2: Mutex<Option<Vec<ImmutableBlock<G2>>>>`
        // field — production semantics are all-or-nothing, so the
        // fixture should store either the full prefix Vec or None
        // (representing the incomplete-backing case logged in
        // production as `prefix_g2_incomplete_skip`).
        //
        // Returning empty here matches the "incomplete backing" arm:
        // tests run cleanly until someone wires the per-slot fixture.
        Ok(Vec::new())
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
        Ok(slot.transfer_params.lock().clone())
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

    fn install_cd_onboarding_payload(
        &self,
        request_id: &str,
        cd_payload: Box<dyn crate::connector::leader::slot::CdOnboardingPayload>,
    ) -> Result<()> {
        let slot = self.require_slot(request_id)?;
        // Mock just records that a payload was installed and drops
        // it via the slot field; production transitions the slot's
        // txn_state. Tests that need to assert the install can
        // inspect `installed_cd_payloads`.
        slot.installed_cd_payloads
            .lock()
            .push(request_id.to_string());
        // Hold the payload alive on the mock slot so its `Drop`
        // doesn't fire prematurely. Tests assert against
        // `cd_payloads_dropped` to check Drop behavior.
        slot.installed_cd_payload.lock().replace(cd_payload);
        Ok(())
    }

    fn promote_g1_to_g2(
        &self,
        source_blocks: Vec<ExternalBlock<G1>>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        let (tx, rx) = oneshot::channel();
        self.promotions.lock().push(PendingPromotion {
            source_blocks,
            resolver: Some(tx),
        });
        async move {
            rx.await
                .map_err(|err| anyhow!("promote_g1_to_g2 resolver dropped: {err}"))?
        }
        .boxed()
    }

    fn find_prefix_g3_hashes(
        &self,
        request_id: &str,
        num_prefix_blocks: usize,
    ) -> Result<Vec<SequenceHash>> {
        if num_prefix_blocks == 0 {
            return Ok(Vec::new());
        }
        let slot = self.require_slot(request_id)?;
        let prefix = slot
            .all_hashes
            .get(..num_prefix_blocks)
            .ok_or_else(|| {
                anyhow!(
                    "find_prefix_g3_hashes ({}): num_prefix_blocks {} exceeds slot hashes",
                    request_id,
                    num_prefix_blocks
                )
            })?
            .to_vec();
        let universe = self.g3_universe.lock();
        let g3 = match universe.get(request_id) {
            Some(set) => set,
            None => return Ok(Vec::new()),
        };
        if prefix.iter().all(|h| g3.contains(h)) {
            Ok(prefix)
        } else {
            Ok(Vec::new())
        }
    }

    fn promote_g3_to_g2(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        let (tx, rx) = oneshot::channel();
        self.g3_promotions.lock().push(PendingG3Promotion {
            hashes,
            resolver: Some(tx),
        });
        async move {
            rx.await
                .map_err(|err| anyhow!("promote_g3_to_g2 resolver dropped: {err}"))?
        }
        .boxed()
    }
}
