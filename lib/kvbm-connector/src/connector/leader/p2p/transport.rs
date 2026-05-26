// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block-transport seam for the disaggregation wrapper.
//!
//! The wrapper drives only one kind of transfer through this seam now:
//! **Local G2 → G1** — copy already-resident G2 data into vLLM's G1
//! destination block_ids. (The remote pull path was subsumed by
//! `Session::pull` in the symmetric session refactor.)
//!
//! This is factored behind a trait so the wrapper E2E tests can swap
//! in a scriptable mock without spinning up real workers.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::FutureExt;
use futures::future::BoxFuture;
use kvbm_common::LogicalLayoutHandle;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::offload::ExternalBlock;
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};
use kvbm_physical::TransferOptions;
use kvbm_protocols::disagg::TransferParams;

use crate::connector::leader::scheduler::{KvConnectorMetadata, SchedulerOutput};
use crate::connector::leader::{ConnectorLeader, FinishedStatus, Request, SlotMatchSplit};
use crate::{BlockId, G1, G2, G3, InstanceId, SequenceHash};

/// Transport seam used by the disagg wrapper to drive
/// local G2→G1 transfers (decode's local-match kick at USAA-1, and
/// the per-chunk onboard after pulling remote outputs).
pub trait P2pBlockTransport: Send + Sync {
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

/// Production transport that bridges `P2pBlockTransport` to the engine's
/// `InstanceLeader::parallel_worker` for local G2→G1 transfers.
pub struct EngineP2pBlockTransport {
    instance_leader: Arc<InstanceLeader>,
}

impl EngineP2pBlockTransport {
    pub fn new(instance_leader: Arc<InstanceLeader>) -> Arc<Self> {
        Arc::new(Self { instance_leader })
    }
}

impl P2pBlockTransport for EngineP2pBlockTransport {
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

/// Hook the disagg wrapper uses to notify workers when a
/// CD-bound request finishes onboarding (or fails partway). Factored out
/// of the wrapper so tests can record these calls without standing up
/// real worker clients.
pub trait P2pWorkerHook: Send + Sync {
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

impl P2pWorkerHook for InnerLeaderWorkerHook {
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

/// Trait abstraction over the connector leader the disagg
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
    /// Borrow the slot's LoRA adapter name for CD wire propagation.
    fn slot_lora_name(&self, request_id: &str) -> Result<Option<String>>;
    /// Borrow the slot's raw salt string for CD wire propagation.
    fn slot_salt(&self, request_id: &str) -> Result<Option<String>>;
    fn local_instance_id(&self) -> InstanceId;
    fn apply_block_assignments(&self, request_id: &str, block_ids: Vec<BlockId>) -> Result<()>;
    fn take_local_match_g2_blocks(&self, request_id: &str) -> Result<Vec<ImmutableBlock<G2>>>;
    /// Search G2 for the prefix range `sequence_hashes[0 .. num_prefix_blocks]`
    /// — the blocks vLLM reports as already in G1 but which the inner search
    /// at `process_match` did not cover (it starts at
    /// `num_computed_tokens / block_size`).
    ///
    /// CD sessions ideally want every block in `[0, total)` reachable by the
    /// prefill peer via decode's G2/G3. With vLLM prefix-caching disabled
    /// `num_computed_tokens` is always 0 and this method is a no-op (returns
    /// empty); with PC enabled the prefix range gets searched here.
    ///
    /// # Returns — all-or-nothing
    ///
    /// Either the full `num_prefix_blocks` G2 blocks (every prefix block
    /// was G2-resident) or an empty Vec (any G2 miss → drop the partial
    /// hits). Partial publication would tell prefill "decode has prefix
    /// `[0..M)` but not `[M..P)`" while vLLM's G1 actually holds the
    /// full prefix — an inconsistent advertisement.
    ///
    /// The empty arm is the Stage 1 promotion trigger: the caller
    /// (`decode_leader.rs::commit_gnmt_remote`) plans a G1→G2
    /// promotion for the full prefix window and fires the actual
    /// transfer at USAA via `Self::promote_g1_to_g2`.
    ///
    /// # Errors
    ///
    /// `Err` only on infrastructure failures (slot missing, leader not
    /// initialized). G2-miss cases return `Ok(Vec::new())` per the
    /// all-or-nothing contract above. The production impl uses
    /// `g2_manager.match_blocks` directly (no `find_matches_with_options`
    /// / no G3 fallback / no AsyncSession allocation) — G3-tier prefix
    /// blocks are not pursued at GNMT time and don't change this contract.
    ///
    /// Reachable from `decode_gnmt` (the vLLM `get_num_new_matched_tokens`
    /// PyO3 entrypoint), so failures must be `Err` (request-scoped abort) and
    /// never `panic!` (would poison the worker across all in-flight requests).
    fn find_prefix_g2_blocks(
        &self,
        request_id: &str,
        num_prefix_blocks: usize,
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
    fn slot_transfer_params(&self, request_id: &str) -> Result<Option<TransferParams>>;

    // ---- G2 allocator + registration (kept narrow) ----

    /// Allocate `count` mutable G2 blocks from the leader's G2 manager.
    fn allocate_g2_blocks(&self, count: usize) -> Result<Vec<MutableBlock<G2>>>;
    /// Register a batch of G2 `CompleteBlock`s with the leader's manager,
    /// returning the corresponding `ImmutableBlock`s.
    fn register_g2_blocks(&self, blocks: Vec<CompleteBlock<G2>>)
    -> Result<Vec<ImmutableBlock<G2>>>;

    /// Install or attach a [`crate::connector::leader::slot::CdOnboardingPayload`]
    /// on the slot's transaction state, transitioning `Inactive →
    /// Onboarding(cd_payload=Some)` (cold-cache CD) or attaching to
    /// an existing `Onboarding/PreparingToOnboard` state (CD with
    /// local match).
    ///
    /// Idempotent: returns `Ok(())` even if the slot already carries
    /// a payload (the wrapper's `cd_request_state` DashMap entry
    /// already guards against duplicate installs).
    fn install_cd_onboarding_payload(
        &self,
        request_id: &str,
        cd_payload: Box<dyn crate::connector::leader::slot::CdOnboardingPayload>,
    ) -> Result<()>;

    /// Promote a set of G1-resident blocks into G2 and return the
    /// registered `ImmutableBlock<G2>`s once the offload pipeline has
    /// completed the transfer + register step.
    ///
    /// The returned future encapsulates the whole alloc + transfer +
    /// register flow. The caller is expected to spawn it in a task
    /// whose lifetime is independent of the originating request — if
    /// the request is torn down mid-promotion the resulting G2
    /// blocks remain in cache and benefit future requests.
    ///
    /// The sequence hashes carried by `source_blocks` are what the
    /// offload pipeline registers each resulting G2 block with; the
    /// future re-queries the G2 manager by those same hashes after
    /// the transfer completes and errors if any hash is missing
    /// from the registered set (partial promotion). Passing a
    /// `source_blocks` entry with a mismatched
    /// `(block_id, sequence_hash)` pair therefore produces either a
    /// hash-collision register failure or an "expected hash absent
    /// after transfer" error from this future — both surface as
    /// `Err` and are caller-recoverable.
    ///
    /// Used by Stage 1 conditional-disagg decode-side G1→G2 prefix
    /// promotion at USAA. The CD `ConditionalDecodeG2Observer`
    /// installed at init (for the prefill chunked-output path) also
    /// observes the same G1→G2 register events for these blocks but
    /// silently ignores them — its per-request `pending` map has no
    /// residual hash set for decode-role requests.
    fn promote_g1_to_g2(
        &self,
        source_blocks: Vec<ExternalBlock<G1>>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>>;

    /// Search G3 (NVMe/SSD tier) for the prefix range
    /// `sequence_hashes[0..num_prefix_blocks]`. Stage 2 analog of
    /// [`Self::find_prefix_g2_blocks`].
    ///
    /// # Returns — all-or-nothing
    ///
    /// Either the full `num_prefix_blocks` sequence hashes (every
    /// prefix block was G3-resident) or an empty Vec (any G3 miss
    /// → drop the partial knowledge; the caller falls back to G1
    /// promotion). Partial G3 advertisement would conflict with
    /// Invariant A in the same way partial G2 advertisement
    /// would.
    ///
    /// Returns hashes (not pinned blocks). Between this call and
    /// the USAA-time `promote_g3_to_g2`, the matched G3 blocks
    /// could be evicted under pressure; that's an accepted small
    /// eviction window and surfaces as `promote_g3_to_g2` Err
    /// (which routes through the standard CD failure path).
    ///
    /// # Errors
    ///
    /// `Err` only on infrastructure failures (slot missing, leader
    /// not initialized). G3-miss returns `Ok(Vec::new())`. When the
    /// G3 manager is not configured (no NVMe tier), returns
    /// `Ok(Vec::new())` — caller falls back to G1 path.
    fn find_prefix_g3_hashes(
        &self,
        request_id: &str,
        num_prefix_blocks: usize,
    ) -> Result<Vec<SequenceHash>>;

    /// Stage 2 G3→G2 prefix promotion. Mirrors
    /// [`Self::promote_g1_to_g2`] but the source tier is G3.
    ///
    /// The hashes are re-matched against the G3 manager inside the
    /// returned future; if any hash is no longer G3-resident
    /// (evicted between GNMT and USAA), the future errors. On
    /// match, the future stages G3→G2 via
    /// `kvbm_engine::leader::staging::stage_g3_to_g2` and returns
    /// the registered G2 blocks.
    ///
    /// The caller (the spawn at `commit_usaa1`) treats Err the
    /// same as `promote_g1_to_g2` Err — calls `session.close` so
    /// the prefill peer routes through its cleanup_failed_request
    /// path and vLLM is notified via `mark_failed_onboarding`.
    fn promote_g3_to_g2(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>>;
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

    fn slot_lora_name(&self, request_id: &str) -> Result<Option<String>> {
        self.inner.slot_lora_name(request_id)
    }

    fn slot_salt(&self, request_id: &str) -> Result<Option<String>> {
        self.inner.slot_salt(request_id)
    }

    fn local_instance_id(&self) -> InstanceId {
        self.inner.local_instance_id()
    }

    fn apply_block_assignments(&self, request_id: &str, block_ids: Vec<BlockId>) -> Result<()> {
        self.inner.apply_block_assignments(request_id, block_ids)
    }

    fn take_local_match_g2_blocks(&self, request_id: &str) -> Result<Vec<ImmutableBlock<G2>>> {
        self.inner.take_local_match_g2_blocks(request_id)
    }

    fn find_prefix_g2_blocks(
        &self,
        request_id: &str,
        num_prefix_blocks: usize,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        self.inner
            .find_prefix_g2_blocks(request_id, num_prefix_blocks)
    }

    fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>> {
        self.inner.token_blocks_for_range(request_id, range)
    }

    fn slot_transfer_params(&self, request_id: &str) -> Result<Option<TransferParams>> {
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

    fn install_cd_onboarding_payload(
        &self,
        request_id: &str,
        cd_payload: Box<dyn crate::connector::leader::slot::CdOnboardingPayload>,
    ) -> Result<()> {
        let shared_slot = self
            .inner
            .get_slot(request_id)
            .map_err(|e| anyhow!("install_cd_onboarding_payload: {}", e))?;
        let mut slot = shared_slot.lock();
        slot.txn_install_or_attach_cd_payload(cd_payload)
            .map_err(|e| anyhow!("install_cd_onboarding_payload({}): {}", request_id, e))
    }

    fn promote_g1_to_g2(
        &self,
        source_blocks: Vec<ExternalBlock<G1>>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        // The offload pipeline registers each destination G2 block
        // with the `sequence_hash` carried in its `ExternalBlock`.
        // Capture those hashes here so we can re-query the G2
        // manager after the transfer completes — the offload
        // pipeline's register-observer hands blocks to G2 but does
        // not return them to the caller of `enqueue_g1_to_g2`.
        let expected_hashes: Vec<SequenceHash> =
            source_blocks.iter().map(|b| b.sequence_hash).collect();
        let inner = Arc::clone(&self.inner);
        async move {
            let engine = inner
                .offload_engine()
                .ok_or_else(|| anyhow!("offload engine not initialized for promote_g1_to_g2"))?;
            let mut handle = engine.enqueue_g1_to_g2(source_blocks)?;
            // Drives the transfer to a terminal TransferStatus. The
            // pipeline's register step puts blocks into the G2
            // manager before this resolves.
            let _result = handle.wait().await?;
            let leader = inner.instance_leader().ok_or_else(|| {
                anyhow!("InstanceLeader not initialized for promote_g1_to_g2 re-query")
            })?;
            let g2_blocks = leader.g2_manager().match_blocks(&expected_hashes);
            if g2_blocks.len() != expected_hashes.len() {
                anyhow::bail!(
                    "promote_g1_to_g2: registered {}/{} expected hashes after transfer",
                    g2_blocks.len(),
                    expected_hashes.len(),
                );
            }
            Ok(g2_blocks)
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
        // Pull the canonical prefix-window hashes off the slot's
        // PLH chain. Same source of truth as
        // `find_prefix_g2_blocks` uses for its G2 query.
        let prefix_hashes: Vec<SequenceHash> = self
            .inner
            .slot_match_split(request_id)?
            .all_sequence_hashes
            .get(..num_prefix_blocks)
            .ok_or_else(|| {
                anyhow!(
                    "find_prefix_g3_hashes ({}): num_prefix_blocks {} exceeds slot's hashes",
                    request_id,
                    num_prefix_blocks
                )
            })?
            .to_vec();

        let leader = self.inner.instance_leader().ok_or_else(|| {
            anyhow!("InstanceLeader not set; find_prefix_g3_hashes called before initialized")
        })?;

        let Some(g3_manager) = leader.g3_manager() else {
            // No NVMe tier configured — caller falls through to
            // the G1 promotion path.
            return Ok(Vec::new());
        };

        // Pin briefly to verify presence, then drop. The hashes
        // are what we return; the pins are not preserved (Stage 2
        // accepts the GNMT→USAA eviction window — see trait doc).
        let matched = g3_manager.match_blocks(&prefix_hashes);
        if matched.len() != num_prefix_blocks {
            tracing::debug!(
                request_id,
                num_prefix_blocks,
                hits = matched.len(),
                "find_prefix_g3_hashes: incomplete G3 backing; falling back to G1 promotion"
            );
            crate::audit!(
                "prefix_g3_incomplete_skip",
                role = "decode",
                request_id,
                num_prefix_blocks,
                hits = matched.len()
            );
            return Ok(Vec::new());
        }
        Ok(prefix_hashes)
    }

    fn promote_g3_to_g2(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        let inner = Arc::clone(&self.inner);
        async move {
            let leader = inner
                .instance_leader()
                .ok_or_else(|| anyhow!("InstanceLeader not initialized for promote_g3_to_g2"))?;
            let g3_manager = leader
                .g3_manager()
                .ok_or_else(|| {
                    anyhow!("promote_g3_to_g2: G3 manager not configured (no NVMe tier)")
                })?
                .clone();
            let matched: Vec<ImmutableBlock<G3>> = g3_manager.match_blocks(&hashes);
            if matched.len() != hashes.len() {
                anyhow::bail!(
                    "promote_g3_to_g2: matched {}/{} G3 blocks at USAA \
                     (evicted between GNMT and USAA?)",
                    matched.len(),
                    hashes.len(),
                );
            }
            let parallel_worker = leader
                .parallel_worker()
                .ok_or_else(|| anyhow!("promote_g3_to_g2: no parallel worker configured"))?;
            let holder = kvbm_engine::leader::BlockHolder::new(matched);
            let result = kvbm_engine::leader::stage_g3_to_g2(
                &holder,
                leader.g2_manager(),
                &*parallel_worker,
            )
            .await?;
            if result.new_g2_blocks.len() != hashes.len() {
                anyhow::bail!(
                    "promote_g3_to_g2: stage_g3_to_g2 produced {}/{} G2 blocks",
                    result.new_g2_blocks.len(),
                    hashes.len(),
                );
            }
            Ok(result.new_g2_blocks)
        }
        .boxed()
    }
}
