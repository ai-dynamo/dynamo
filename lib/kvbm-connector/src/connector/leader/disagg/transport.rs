// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block-transport seam for the conditional-disaggregation wrapper.
//!
//! The decode-side wrapper drives two kinds of transfers during the
//! USAA-1 → `BlockSetsAdded` → completion pipeline:
//!
//! - **Remote pull** — RDMA-pull a peer's published G2 block set into our
//!   pre-allocated G2 destinations.
//! - **Local G2 → G1** — copy already-resident G2 data into vLLM's G1
//!   destination block_ids.
//!
//! These are factored behind a trait so the decode-side end-to-end test
//! can swap in a scriptable mock without spinning up real workers, RDMA,
//! or velo sessions.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::LogicalLayoutHandle;
use kvbm_disagg_protocol::{SessionEndpoint, SessionId, TransferParams};
use kvbm_engine::disagg::{DisaggSession, PrefillSession, RemoteBlockSet};
use kvbm_engine::leader::InstanceLeader;
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};
use kvbm_physical::TransferOptions;

use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{
    ConnectorLeader, FinishedStatus, Request, SlotMatchSplit,
};
use crate::{BlockId, G2, InstanceId};

/// Transport seam used by the conditional-disagg decode wrapper to drive
/// RDMA pulls and local G2→G1 transfers.
pub trait CdBlockTransport: Send + Sync {
    /// RDMA-pull peer-published G2 block sets into local G2 destinations.
    ///
    /// `local_dst_g2_block_ids` length must equal the sum of `block_sets`'
    /// `blocks.len()`. Block sets are pulled in order; within a set, the
    /// source `RemoteBlockRef` at index `i` lands in
    /// `local_dst_g2_block_ids[offset + i]`.
    fn pull_remote(
        &self,
        remote_instance: InstanceId,
        block_sets: Vec<RemoteBlockSet>,
        local_dst_g2_block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>>;

    /// Local G2 → G1 transfer for already-resident G2 blocks.
    ///
    /// `src_g2_block_ids` and `dst_g1_block_ids` must have equal length;
    /// blocks pair up in index order.
    fn local_g2_to_g1(
        &self,
        src_g2_block_ids: Vec<BlockId>,
        dst_g1_block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>>;
}

/// Production transport that bridges `CdBlockTransport` to the engine's
/// `InstanceLeader` (for remote pulls) and `parallel_worker` (for local
/// G2→G1 transfers).
pub struct EngineCdBlockTransport {
    instance_leader: Arc<InstanceLeader>,
}

impl EngineCdBlockTransport {
    pub fn new(instance_leader: Arc<InstanceLeader>) -> Arc<Self> {
        Arc::new(Self { instance_leader })
    }
}

impl CdBlockTransport for EngineCdBlockTransport {
    fn pull_remote(
        &self,
        remote_instance: InstanceId,
        block_sets: Vec<RemoteBlockSet>,
        local_dst_g2_block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>> {
        let leader = self.instance_leader.clone();
        Box::pin(async move {
            let notification = leader
                .pull_remote_block_sets(remote_instance, &block_sets, &local_dst_g2_block_ids)
                .await?;
            notification.await?;
            Ok(())
        })
    }

    fn local_g2_to_g1(
        &self,
        src_g2_block_ids: Vec<BlockId>,
        dst_g1_block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>> {
        let leader = self.instance_leader.clone();
        Box::pin(async move {
            let parallel_worker = leader
                .parallel_worker()
                .ok_or_else(|| anyhow!("no parallel worker available for CD local G2→G1"))?;
            let notification = parallel_worker.execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G1,
                Arc::from(src_g2_block_ids),
                Arc::from(dst_g1_block_ids),
                TransferOptions::default(),
            )?;
            notification.await?;
            Ok(())
        })
    }
}

/// Hook the conditional-disagg wrapper uses to notify workers when a
/// CD-bound request finishes onboarding (or fails partway). Factored out
/// of the wrapper so tests can record these calls without standing up
/// real worker clients.
pub trait CdWorkerHook: Send + Sync {
    /// All listed G1 blocks for `request_id` have been written; the
    /// scheduler can now run the request.
    fn mark_onboarding_complete(&self, request_id: String) -> BoxFuture<'static, Result<()>>;

    /// One or more G1 blocks for `request_id` could not be filled; the
    /// scheduler should not use them. `block_ids` lists the unfilled G1
    /// destinations.
    fn mark_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>>;
}

/// Production hook that delegates to a `ConnectorLeader`'s worker clients.
pub struct InnerLeaderWorkerHook {
    inner: Arc<ConnectorLeader>,
}

impl InnerLeaderWorkerHook {
    pub fn new(inner: Arc<ConnectorLeader>) -> Arc<Self> {
        Arc::new(Self { inner })
    }
}

impl CdWorkerHook for InnerLeaderWorkerHook {
    fn mark_onboarding_complete(&self, request_id: String) -> BoxFuture<'static, Result<()>> {
        self.inner.mark_workers_onboarding_complete(request_id)
    }

    fn mark_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<()>> {
        self.inner
            .mark_workers_failed_onboarding(request_id, block_ids)
    }
}

/// Trait abstraction over the connector leader the conditional-disagg
/// wrapper depends on. Production code wraps a concrete `ConnectorLeader`
/// via [`ConnectorLeaderShim`]; tests inject a mock that scripts every
/// inner call. Holding this trait object instead of `Arc<ConnectorLeader>`
/// is what lets the decode-side end-to-end test run without spinning up a
/// real `KvbmRuntime` / `nixl_agent` / GPU stack.
pub trait InnerLeaderShim: Send + Sync {
    // ---- Scheduler-facing API surface (passthrough for non-CD slots) ----

    fn create_slot(&self, request: Request) -> Result<()>;
    fn has_slot(&self, request_id: &str) -> bool;
    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()>;
    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)>;
    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()>;
    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata>;
    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()>;
    fn request_finished(&self, request_id: &str) -> FinishedStatus;

    // ---- CD-specific helpers ----

    fn block_size(&self) -> usize;
    fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize>;
    fn slot_match_split(&self, request_id: &str) -> Result<SlotMatchSplit>;
    fn slot_token_ids(&self, request_id: &str) -> Result<Vec<u32>>;
    fn local_instance_id(&self) -> InstanceId;
    fn apply_block_assignments(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
    ) -> Result<()>;
    fn take_local_match_g2_blocks(
        &self,
        request_id: &str,
    ) -> Result<Vec<ImmutableBlock<G2>>>;
    fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>>;
    /// Parsed CD transfer params on the slot, if present.
    ///
    /// Used by the prefill-side wrapper at GNMT time to detect CD-bound
    /// requests and read the decode-endpoint + sequence_hashes carried
    /// by the originating request.
    fn slot_transfer_params(
        &self,
        request_id: &str,
    ) -> Result<Option<TransferParams>>;

    // ---- G2 allocator + registration (kept narrow) ----

    /// Allocate `count` mutable G2 blocks from the leader's G2 manager.
    fn allocate_g2_blocks(&self, count: usize) -> Result<Vec<MutableBlock<G2>>>;
    /// Register a batch of G2 `CompleteBlock`s with the leader's manager,
    /// returning the corresponding `ImmutableBlock`s.
    fn register_g2_blocks(
        &self,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>>;
}

/// Prefill-side session-attach seam.
///
/// The prefill wrapper attaches to a decode session using the
/// `decode_endpoint` carried on the request's transfer_params.
/// Production wraps `DisaggSession::attach_prefill` over real velo.
/// Tests inject [`super::testing::MockPrefillSessionAttacher`].
pub trait PrefillSessionAttacher: Send + Sync {
    fn attach(
        &self,
        session_id: SessionId,
        decode_endpoint: SessionEndpoint,
    ) -> BoxFuture<'static, Result<Arc<dyn PrefillSession>>>;
}

/// Production attacher backed by a real velo handle.
pub struct VeloPrefillSessionAttacher {
    velo: Arc<velo::Velo>,
}

impl VeloPrefillSessionAttacher {
    pub fn new(velo: Arc<velo::Velo>) -> Arc<Self> {
        Arc::new(Self { velo })
    }
}

impl PrefillSessionAttacher for VeloPrefillSessionAttacher {
    fn attach(
        &self,
        session_id: SessionId,
        decode_endpoint: SessionEndpoint,
    ) -> BoxFuture<'static, Result<Arc<dyn PrefillSession>>> {
        let velo = Arc::clone(&self.velo);
        Box::pin(async move {
            let session = DisaggSession::attach_prefill(velo, session_id, &decode_endpoint).await?;
            Ok(session as Arc<dyn PrefillSession>)
        })
    }
}

/// Production [`InnerLeaderShim`] that wraps a concrete `ConnectorLeader`.
pub struct ConnectorLeaderShim {
    inner: Arc<ConnectorLeader>,
}

impl ConnectorLeaderShim {
    pub fn new(inner: Arc<ConnectorLeader>) -> Arc<Self> {
        Arc::new(Self { inner })
    }

    pub fn inner(&self) -> &Arc<ConnectorLeader> {
        &self.inner
    }
}

impl InnerLeaderShim for ConnectorLeaderShim {
    fn create_slot(&self, request: Request) -> Result<()> {
        self.inner.create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.inner.has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.inner.extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        self.inner
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        ConnectorLeader::update_state_after_alloc(
            &self.inner,
            request_id,
            block_ids,
            num_external_tokens,
        )
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        self.inner.build_connector_meta(output)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.inner
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        self.inner.request_finished(request_id)
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize> {
        self.inner.get_slot_total_tokens(request_id)
    }

    fn slot_match_split(&self, request_id: &str) -> Result<SlotMatchSplit> {
        self.inner.slot_match_split(request_id)
    }

    fn slot_token_ids(&self, request_id: &str) -> Result<Vec<u32>> {
        self.inner.slot_token_ids(request_id)
    }

    fn local_instance_id(&self) -> InstanceId {
        self.inner.local_instance_id()
    }

    fn apply_block_assignments(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
    ) -> Result<()> {
        self.inner.apply_block_assignments(request_id, block_ids)
    }

    fn take_local_match_g2_blocks(
        &self,
        request_id: &str,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        self.inner.take_local_match_g2_blocks(request_id)
    }

    fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>> {
        self.inner.token_blocks_for_range(request_id, range)
    }

    fn slot_transfer_params(
        &self,
        request_id: &str,
    ) -> Result<Option<TransferParams>> {
        self.inner.slot_transfer_params(request_id)
    }

    fn allocate_g2_blocks(&self, count: usize) -> Result<Vec<MutableBlock<G2>>> {
        let leader = self
            .inner
            .instance_leader()
            .ok_or_else(|| anyhow!("InstanceLeader not initialized for allocate_g2_blocks"))?;
        leader
            .g2_manager()
            .allocate_blocks(count)
            .ok_or_else(|| anyhow!("G2 allocator: failed to allocate {} blocks", count))
    }

    fn register_g2_blocks(
        &self,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        let leader = self
            .inner
            .instance_leader()
            .ok_or_else(|| anyhow!("InstanceLeader not initialized for register_g2_blocks"))?;
        Ok(leader.g2_manager().register_blocks(blocks))
    }
}
