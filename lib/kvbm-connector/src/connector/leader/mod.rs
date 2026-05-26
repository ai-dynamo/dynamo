// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod audit;
pub mod disagg;
mod request;
mod slot;

use super::worker::ConnectorWorkerClient;
use crate::{BlockId, G2, InstanceId, KvbmRuntime};
use kvbm_config::OnboardMode;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::leader::{EventSource, FindMatchesOptions, Leader, StagingMode};
use kvbm_engine::offload::OffloadEngine;
use kvbm_engine::worker::SerializedLayout;
use kvbm_engine::worker::VeloWorkerClient;
use kvbm_hub::{ConditionalDisaggClient, HubClient};
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_observability::CacheStatsTracker;

use anyhow::{Context, Result, anyhow, bail};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use scheduler::{DefaultOracle, ForwardPassSample, Oracle};
use slot::{MatchCheckOutcome, RequestSlot, TransactionState};

pub use request::Request;
pub use scheduler::{CachedRequestData, NewRequestData, SchedulerOutput};
pub use slot::{CdOnboardingPayload, FinishedStatus};
// SlotMatchSplit is defined further down in this file; re-exported here so
// downstream code (e.g., the disagg `InnerLeaderShim` trait)
// can name it.

pub trait ConnectorLeaderInterface: Send + Sync {}

/// Parameters for the in-process kv-router consolidator.
///
/// Supplied at construction time via [`ConnectorLeader::new`] and forwarded to
/// [`InstanceLeader::with_consolidator`] inside `initialize_async`.  All three
/// fields must be non-empty strings; `vllm_zmq_endpoint` is optional and when
/// `None` the consolidator runs in KVBM-only mode (no ZMQ ingress).
pub struct ConsolidatorEndpoints {
    /// ZMQ endpoint vLLM publishes G1 events on (e.g. `"tcp://127.0.0.1:5557"`).
    /// `None` disables ZMQ ingress.
    pub vllm_zmq_endpoint: Option<String>,
    /// ZMQ endpoint the consolidator binds for egress (e.g. `"tcp://0.0.0.0:57001"`).
    pub egress_endpoint: String,
    /// Origin tag for ZMQ-ingress events (`Vllm`, `Trtllm`, or `Kvbm`).
    pub engine_source: EventSource,
}

pub struct ConnectorLeader {
    pub(crate) runtime: Arc<KvbmRuntime>,
    block_size: usize,
    init: Arc<Mutex<WorkerClients>>,
    workers: OnceLock<Arc<WorkerClients>>,
    instance_leader: OnceLock<InstanceLeader>,
    slots: DashMap<String, Arc<Mutex<RequestSlot>>>,
    #[allow(dead_code)] // Will be used for scheduling decisions
    oracle: Arc<dyn Oracle>,
    /// Optional consolidator endpoints.  When `Some`, `initialize_async` will
    /// create an `EventsManager`, wire it to the `BlockRegistry`, and spawn an
    /// in-process consolidator via `InstanceLeader::with_consolidator`.
    consolidator_endpoints: Option<ConsolidatorEndpoints>,
    /// Offload engine for G1→G2→G3 transfers (initialized in initialize_async)
    offload_engine: OnceLock<OffloadEngine>,
    /// Accumulated G2 blocks for intra-pass onboarding.
    ///
    /// These blocks are collected from each request's find session during
    /// `prepare_intra_pass_onboarding` and held until the forward pass completes.
    /// A cleanup task (spawned in `process_scheduler_output`) waits on the
    /// forward pass completion event and then drops these blocks.
    pending_intra_pass_g2_blocks: Mutex<Vec<ImmutableBlock<G2>>>,

    forward_pass_samples: Mutex<Option<ForwardPassSample>>,
    cache_stats: CacheStatsTracker,
    /// Conditional-disagg hub client. Populated in `initialize_async` when
    /// `config.disagg` is present. Holds the hub registration alive for the
    /// life of this leader.
    disagg_client: OnceLock<Arc<ConditionalDisaggClient>>,
    /// KV-index hub publisher, kept alive for the life of this leader.
    /// Populated in `initialize_async` when `leader.hub` enables the KV-indexer
    /// feature. Dropping it aborts the publish task, so it must be held here.
    indexer_publisher: OnceLock<kvbm_logical::events::KvbmCacheEventsPublisher>,
    /// KV-index hub client, holding the lightweight `Feature::Indexer`
    /// registration alive (RAII `DELETE` on drop → hub reclaims the index
    /// entries). Populated only on the kv-index-only path; the disagg path
    /// folds Indexer into its own registration via `disagg_client`.
    indexer_hub_client: OnceLock<Arc<HubClient>>,
    /// Standalone-P2P hub client, holding the `Feature::P2P` registration alive
    /// (RAII `DELETE` on drop). Populated only on the p2p-without-CD path; the
    /// CD path holds its registration via `disagg_client`.
    p2p_hub_client: OnceLock<Arc<HubClient>>,
    /// CD-role dispatcher (`Arc<ConditionalDisaggLeader>`) installed by
    /// `initialize_async` when `config.disagg` is present. Bindings route
    /// `ConnectorLeaderApi` methods through this when set so role-specific
    /// wrappers (decode/prefill) can intercept GNMT/USAA/etc. without
    /// changing the binding-side handle type.
    cd_api: OnceLock<Arc<dyn disagg::ConnectorLeaderApi>>,
}

#[derive(Default, Clone)]
pub(crate) struct WorkerClients {
    worker_instance_ids: Vec<InstanceId>,
    worker_connector_clients: Vec<ConnectorWorkerClient>,
    worker_transfer_clients: Vec<VeloWorkerClient>,
    worker_metadata: Vec<SerializedLayout>,
}

// Connector leader implementation extensions
mod init;

// KV-index hub publisher (ZMQ PUB egress) wired from the hub aggregate.
pub(crate) mod hub_indexer;

// Hub handshake: GET /v1/config, resolve the effective feature set + runtime
// summary the connector registers with.
pub(crate) mod hub_handshake;

// P2P feature foundation (transport seam, hub peer resolver, P2P wiring).
// ConditionalDisagg builds on top of this. `pub` so integration tests can
// reach the transport/peer_resolver types (parity with the old `disagg` path).
pub mod p2p;

/// Implementation of the request_finished function.
mod finish;

/// Calls to coordinator workers.
mod clients;

// Implementation of search tools for the get_num_new_matched_tokens function.
mod search;

// Hub-indexer-backed `RemoteBlockDiscovery` impl injected into the engine leader.
mod remote_search;

// Implementation of onboarding tools for the update_state_after_alloc function.
mod onboard;

// Implementation of offloading engine which will be triggered by the build_connector_metadata function.
mod offload;

// Implementation of the [`scheduler::SchedulerOutput`] struct.
pub mod scheduler;

impl ConnectorLeader {
    pub fn new(runtime: Arc<KvbmRuntime>, block_size: usize) -> Self {
        Self::new_with_consolidator(runtime, block_size, None)
    }

    pub fn new_with_consolidator(
        runtime: Arc<KvbmRuntime>,
        block_size: usize,
        consolidator_endpoints: Option<ConsolidatorEndpoints>,
    ) -> Self {
        // Pull onboard mode from runtime config
        let onboard_mode = runtime.config().onboard.mode;
        tracing::info!(
            ?onboard_mode,
            "ConnectorLeader initialized with onboard mode"
        );

        Self {
            cache_stats: CacheStatsTracker::new(
                runtime.config().metrics.cache_stats_max_requests,
                Duration::from_secs(runtime.config().metrics.cache_stats_log_interval_secs),
                Some(runtime.messenger().instance_id().to_string()),
            ),
            runtime,
            block_size,
            consolidator_endpoints,
            init: Arc::new(Mutex::new(WorkerClients::default())),
            workers: OnceLock::new(),
            instance_leader: OnceLock::new(),
            slots: DashMap::new(),
            oracle: Arc::new(DefaultOracle::default()),
            offload_engine: OnceLock::new(),
            pending_intra_pass_g2_blocks: Mutex::new(Vec::new()),
            forward_pass_samples: Mutex::new(None),
            disagg_client: OnceLock::new(),
            indexer_publisher: OnceLock::new(),
            indexer_hub_client: OnceLock::new(),
            p2p_hub_client: OnceLock::new(),
            cd_api: OnceLock::new(),
        }
    }

    /// Store the kv-index-only hub client to keep its registration guard alive
    /// for the life of the leader. Called once from `initialize_async`.
    pub(crate) fn set_indexer_hub_client(&self, client: Arc<HubClient>) -> Result<()> {
        self.indexer_hub_client
            .set(client)
            .map_err(|_| anyhow!("indexer_hub_client already set"))
    }

    /// Store the standalone-P2P hub client to keep its registration guard alive
    /// for the life of the leader. Called once from `initialize_async`.
    pub(crate) fn set_p2p_hub_client(&self, client: Arc<HubClient>) -> Result<()> {
        self.p2p_hub_client
            .set(client)
            .map_err(|_| anyhow!("p2p_hub_client already set"))
    }

    /// Store the KV-index hub publisher to keep its background task alive.
    /// Called once from `initialize_async`.
    pub(crate) fn set_indexer_publisher(
        &self,
        publisher: kvbm_logical::events::KvbmCacheEventsPublisher,
    ) -> Result<()> {
        self.indexer_publisher
            .set(publisher)
            .map_err(|_| anyhow!("indexer_publisher already set"))
    }

    /// Conditional-disagg dispatcher, populated in `initialize_async` when
    /// `config.disagg` is present. Bindings consult this to route the
    /// `ConnectorLeaderApi` method set through the CD wrapper.
    pub fn cd_api(&self) -> Option<&Arc<dyn disagg::ConnectorLeaderApi>> {
        self.cd_api.get()
    }

    /// Set the CD dispatcher. Called once from `initialize_async`.
    pub(crate) fn set_cd_api(&self, api: Arc<dyn disagg::ConnectorLeaderApi>) -> Result<()> {
        self.cd_api
            .set(api)
            .map_err(|_| anyhow!("cd_api already set"))
    }

    /// Conditional-disagg hub client, if this leader was configured with
    /// `disagg` and `initialize_async` has completed.
    pub fn disagg_client(&self) -> Option<&Arc<ConditionalDisaggClient>> {
        self.disagg_client.get()
    }

    /// Whichever [`HubClient`] this leader registered with, regardless of which
    /// feature path ran. The three registration paths stash it differently
    /// (kv-index-only, standalone-P2P, or folded into the CD client), so this
    /// reaches into each in turn. `None` before any hub registration completes.
    fn registered_hub_client(&self) -> Option<Arc<HubClient>> {
        if let Some(h) = self.indexer_hub_client.get() {
            return Some(Arc::clone(h));
        }
        if let Some(h) = self.p2p_hub_client.get() {
            return Some(Arc::clone(h));
        }
        if let Some(c) = self.disagg_client.get() {
            return Some(Arc::clone(c.hub()));
        }
        None
    }

    /// Get the current onboard mode.
    pub fn onboard_mode(&self) -> OnboardMode {
        self.runtime.config().onboard.mode
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Access the InstanceLeader (available after initialize_workers()).
    pub(crate) fn instance_leader(&self) -> Option<&InstanceLeader> {
        self.instance_leader.get()
    }

    /// Access the worker clients, if `initialize_async` has completed.
    ///
    /// Used by the disagg wrapper to drive
    /// `mark_onboarding_complete` after a remote-prefill request finishes its
    /// G2 → G1 onboarding. Currently dead in this crate; the call site
    /// lands with the `on_block_sets_added` output-pull wiring.
    #[allow(dead_code)]
    pub(crate) fn worker_clients(&self) -> Option<&Arc<WorkerClients>> {
        self.workers.get()
    }

    /// Tokio runtime handle for spawning async work. See `worker_clients`
    /// — the call site lands with the CD output-pull integration.
    #[allow(dead_code)]
    pub(crate) fn tokio_handle(&self) -> tokio::runtime::Handle {
        self.runtime.tokio().clone()
    }

    /// Set the InstanceLeader (called by test infrastructure after worker initialization).
    ///
    /// This is typically called by ConnectorInstance after workers are initialized
    /// and we have access to their DirectWorker instances.
    pub(crate) fn set_instance_leader(&self, leader: InstanceLeader) -> Result<()> {
        self.instance_leader
            .set(leader)
            .map_err(|_| anyhow!("InstanceLeader already set"))
    }

    /// Get the offload engine.
    ///
    /// Returns `None` if `initialize_async()` has not been called.
    pub fn offload_engine(&self) -> Option<&OffloadEngine> {
        self.offload_engine.get()
    }

    /// Check if a slot exists for the given request ID.
    pub fn has_slot(&self, request_id: &str) -> bool {
        self.slots.contains_key(request_id)
    }

    /// Get a slot for the given request ID.
    pub fn get_slot(&self, request_id: &str) -> Result<Arc<Mutex<RequestSlot>>> {
        self.slots
            .get(request_id)
            .map(|slot| slot.clone())
            .ok_or_else(|| anyhow!("Slot not found for request ID: {}", request_id))
    }

    /// Create a new slot for the given request ID, tokens and salt hash.
    pub fn create_slot(&self, request: Request) -> Result<()> {
        let request_id = request.request_id.clone();
        if self.has_slot(&request_id) {
            bail!("Slot already exists for request ID: {}", request_id);
        }
        let slot = RequestSlot::new(request, self.block_size)?;
        self.slots.insert(request_id, Arc::new(Mutex::new(slot)));
        Ok(())
    }

    fn record_cache_metrics(
        &self,
        breakdown: &kvbm_engine::leader::MatchBreakdown,
        blocks_queried: usize,
    ) {
        self.cache_stats.record(
            breakdown.host_blocks,
            breakdown.disk_blocks,
            breakdown.object_blocks,
            blocks_queried,
        );
        let (host_rate, disk_rate, object_rate) = self.cache_stats.rates();
        self.runtime
            .observability()
            .compat_metrics()
            .set_cache_hit_rates(host_rate, disk_rate, object_rate);
        self.runtime
            .observability()
            .transfer_metrics()
            .record_cache_hits("host", breakdown.host_blocks as u64);
        self.runtime
            .observability()
            .transfer_metrics()
            .record_cache_hits("disk", breakdown.disk_blocks as u64);
        self.runtime
            .observability()
            .transfer_metrics()
            .record_cache_hits("object", breakdown.object_blocks as u64);
        self.cache_stats.maybe_log();
    }

    /// Get the total number of tokens in a slot's sequence.
    ///
    /// This is used to compare with the vLLM Request's token count to detect
    /// when new tokens have been generated during decoding.
    pub fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize> {
        let slot = self.get_slot(request_id)?;
        Ok(slot.lock().total_tokens())
    }

    /// Apply G1 block_id assignments to a slot's token sequence without
    /// triggering the inner USAA's G2→G1 transfer.
    ///
    /// Conditional-disagg uses this on USAA-1 to align the vLLM-allocated
    /// G1 block_ids with the slot's sequence hashes; the wrapper then
    /// drives transfers for the local-match and remote-prefill slices
    /// separately via `P2pBlockTransport` (step 2 onward).
    #[allow(dead_code)]
    pub(crate) fn apply_block_assignments(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
    ) -> Result<()> {
        let slot = self.get_slot(request_id)?;
        let mut slot = slot.lock();
        slot.set_match_requires_reset(true);
        slot.apply_new_blocks(block_ids);
        Ok(())
    }

    /// Read the local-match split for a slot mid-onboard.
    ///
    /// Non-destructive — does not consume G2 blocks. The wrapper uses this
    /// to decide how many of the vLLM-allocated G1 blocks point at
    /// already-resident local data versus remote-prefill output.
    ///
    /// Slots whose inner GNMT yielded zero local matches don't have an
    /// onboarding state installed — for those the split is degenerate
    /// (`local_match_blocks=0`, `computed_blocks=0`) and the wrapper can
    /// still proceed with a fully-remote prefill.
    #[allow(dead_code)]
    pub(crate) fn slot_match_split(&self, request_id: &str) -> Result<SlotMatchSplit> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        let block_size = slot.block_size();
        let total_blocks = slot.sequence.blocks().len();
        let all_sequence_hashes = slot.all_sequence_hashes();

        let onboarding = slot.onboarding_state().map(|o| {
            // The CD wrapper requires `local == take_local_match_g2_blocks().len()`,
            // which itself routes through `collect_g2_blocks_from_shards`. Both
            // must apply matched-span (first-hole) semantics with head-mask +
            // tail-truncate; using `aggregate_breakdown()` (raw per-tier sums)
            // would over-count when a later shard partially matches or when
            // num_computed_tokens advanced after the shard was issued.
            //
            // Empty shards = CD-only state (cold-cache): zero local match —
            // `matched_span` debug_asserts non-empty, so short-circuit here.
            let local = if o.shards.is_empty() {
                0
            } else {
                let (effective_start, final_end) = o.matched_span(block_size);
                final_end - effective_start
            };
            (local, o.num_computed_tokens)
        });

        Ok(compute_slot_match_split(
            block_size,
            total_blocks,
            all_sequence_hashes,
            onboarding,
        ))
    }

    /// Take ownership of the local-match G2 blocks staged by the slot's
    /// find session.
    ///
    /// Destructive: after this returns the slot's onboarding state has no
    /// G2 blocks left to give. The wrapper hands these to
    /// [`P2pBlockTransport::local_g2_to_g1`] for the matched portion of the
    /// G1 destination slice.
    ///
    /// Returns an empty vec when the slot has no onboarding state — that
    /// case corresponds to a 0-local-match request (fully remote prefill);
    /// there's no G2 to take. Mirrors `slot_match_split`'s degenerate case.
    #[allow(dead_code)]
    pub(crate) fn take_local_match_g2_blocks(
        &self,
        request_id: &str,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        let slot = self.get_slot(request_id)?;
        let mut slot = slot.lock();
        let block_size = slot.block_size();
        let Some(onboarding) = slot.onboarding_state_mut() else {
            return Ok(Vec::new());
        };
        // CD-only OnboardingState has no shards — there's nothing to take.
        // Mirrors `slot_match_split`'s `local_match_blocks = 0` for empty shards;
        // the CD wrapper asserts the two agree.
        if onboarding.shards.is_empty() {
            return Ok(Vec::new());
        }
        // Delegate to the canonical helper so this matches the intra-pass /
        // execute_onboarding paths: head-mask leading_skip (skipped when
        // num_computed_tokens advances after a shard was issued), tail-truncate
        // to `final_end - effective_start`. Without this the wrapper's
        // `local_g2.len() == split.local_match_blocks` invariant fails on any
        // shard with a partial match or any race against vLLM's eviction.
        crate::connector::leader::onboard::collect_g2_blocks_from_shards(onboarding, block_size)
            .with_context(|| format!("slot {} G2 collection failed", request_id))
    }

    /// Search the connector's G2 cache for `sequence_hashes[0 .. num_prefix_blocks]`
    /// — the prefix range vLLM reports as already in G1 but which the inner
    /// search at `process_match` did NOT cover (it starts at
    /// `num_computed_tokens / block_size`).
    ///
    /// CD sessions ideally want every block in `[0, total)` to be reachable
    /// by the prefill peer via decode's G2/G3. The inner search only covers
    /// `[num_computed_tokens / bs, total / bs)`; this method backfills the
    /// prefix from the connector-tracked G2.
    ///
    /// # Implementation: direct `g2_manager.match_blocks`, no session
    ///
    /// Uses `InstanceLeader::g2_manager().match_blocks(prefix_hashes)`
    /// rather than `find_matches_with_options`. The higher-level API:
    ///
    /// * Falls back to G3 on G2 miss → wraps the result in `AsyncSession`
    ///   for G3→G2 staging. We don't want to stage G3 prefix at GNMT —
    ///   that's synchronous-PyO3-blocking work and the wrong tier
    ///   promotion for "prefix vLLM has in G1." If G3 hits exist they're
    ///   dropped here (G3 prefix backfill is future work alongside the
    ///   G1→G2 path).
    /// * Allocates `SessionState` keyed by a new `session_id` and stores
    ///   matched_g2_blocks + matched_g3_blocks under it. Without an
    ///   explicit `release_session` those blocks leak until session
    ///   timeout — and on `release_session` we'd also drop the valid G2
    ///   hits stored alongside the G3 hits in the same `SessionState`.
    ///
    /// `g2_manager.match_blocks` returns the G2 hits directly with no
    /// session bookkeeping and no G3 fallback — perfect fit for "what's
    /// already in G2 and pinable by RAII right now."
    ///
    /// # Return value — all-or-nothing
    ///
    /// * **Full hit** (`g2_manager.match_blocks(prefix_hashes).len() ==
    ///   num_prefix_blocks`) — every prefix block was in G2; returns the
    ///   full Vec and the wrapper publishes them all on the session.
    /// * **Any miss** (returned len `< num_prefix_blocks`, including 0) —
    ///   drops the partial G2 hits (RAII-release returns them to G2's
    ///   inactive pool) and returns an empty Vec. The wrapper handles
    ///   the empty arm via the Stage 1 promotion plan (see
    ///   `decode_leader.rs::commit_gnmt_remote`): the planned-promoted
    ///   hashes are committed up front on the session and the actual
    ///   G1→G2 transfer fires at USAA when vLLM hands over the G1
    ///   `block_ids`.
    ///
    /// Why all-or-nothing rather than partial-publish: G2's first-hole
    /// policy returns the leading-contiguous G2-resident prefix. If
    /// blocks `[0..M)` are in G2 but `[M..P)` aren't, publishing `[0..M)`
    /// to the session tells prefill "decode has prefix blocks `[0..M)`
    /// available, not the rest." But on the decode side vLLM holds G1
    /// for the FULL prefix `[0..P)` — the "missing" `[M..P)` is a hole
    /// only in G2, not in the conversation state. Advertising the partial
    /// set creates an inconsistent view: prefill would believe positions
    /// `[M..P)` are unavailable when in reality decode can serve them via
    /// the promotion path.
    ///
    /// # Errors
    ///
    /// Returns `Err` only for true infrastructure failures (slot missing,
    /// `InstanceLeader` not initialized). All G2-miss cases — full miss
    /// and incomplete-prefix miss — return `Ok(Vec::new())` per the
    /// all-or-nothing contract above.
    ///
    /// Reachable from `decode_gnmt` (the vLLM `get_num_new_matched_tokens`
    /// PyO3 entrypoint), so failures are `Err` (request-scoped abort) and
    /// never `panic!` (would poison the worker across all in-flight
    /// requests).
    #[allow(dead_code)]
    pub(crate) fn find_prefix_g2_blocks(
        &self,
        request_id: &str,
        num_prefix_blocks: usize,
    ) -> Result<Vec<ImmutableBlock<G2>>> {
        if num_prefix_blocks == 0 {
            return Ok(Vec::new());
        }

        let prefix_hashes: Vec<crate::SequenceHash> = {
            let slot = self.get_slot(request_id)?;
            let slot = slot.lock();
            let all = slot.all_sequence_hashes();
            if num_prefix_blocks > all.len() {
                bail!(
                    "find_prefix_g2_blocks ({}): num_prefix_blocks {} exceeds slot's total \
                     sequence_hashes {}",
                    request_id,
                    num_prefix_blocks,
                    all.len(),
                );
            }
            all[..num_prefix_blocks].to_vec()
        };

        let leader = self
            .instance_leader
            .get()
            .ok_or_else(|| anyhow!("InstanceLeader not set; called before initialized"))?;

        // Direct G2 lookup. No session, no G3 fallback, no staging side
        // effects. The returned ImmutableBlock<G2> entries are
        // RAII-pinned for the caller's lifetime (and after `make_available`,
        // for the session's lifetime).
        let g2 = leader.g2_manager().match_blocks(&prefix_hashes);

        if g2.len() != num_prefix_blocks {
            // Any miss → return empty (all-or-nothing). Publishing
            // the partial leading-contiguous G2 hits would tell
            // prefill "decode has [0..M) of the prefix" while
            // vLLM's G1 actually holds the full [0..P) — an
            // inconsistent advertisement that makes the holes look
            // like real cache misses to prefill.
            //
            // Drop the partial G2 hits (RAII returns them to the
            // inactive pool intact, available for future requests).
            // The caller (`commit_gnmt_remote`) plans a Stage 1
            // G1→G2 promotion for the full prefix window in the
            // empty arm — the actual transfer fires at USAA when
            // vLLM hands over the G1 `block_ids`.
            let hits = g2.len();
            let misses = num_prefix_blocks - hits;
            drop(g2);
            tracing::debug!(
                request_id,
                num_prefix_blocks,
                hits,
                misses,
                "find_prefix_g2_blocks: incomplete G2 backing; deferring to Stage 1 promotion"
            );
            crate::audit!(
                "prefix_g2_incomplete_skip",
                role = "decode",
                request_id,
                num_prefix_blocks,
                hits,
                misses
            );
            return Ok(Vec::new());
        }

        Ok(g2)
    }

    /// Clone a contiguous slice of the slot's `TokenBlock`s.
    ///
    /// Used by the wrapper to feed `MutableBlock<G2>::complete(token_block)`
    /// once an RDMA pull lands and we need to register the new G2 blocks.
    #[allow(dead_code)]
    pub(crate) fn token_blocks_for_range(
        &self,
        request_id: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<dynamo_tokens::TokenBlock>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        let blocks = slot.sequence.blocks();
        if range.end > blocks.len() {
            bail!(
                "token_blocks_for_range out of bounds: range {:?}, len {}",
                range,
                blocks.len()
            );
        }
        Ok(blocks[range].to_vec())
    }

    /// Drive `mark_onboarding_complete` across all worker clients for
    /// CD-bound requests once the wrapper has finished the G2→G1 transfer.
    #[allow(dead_code)]
    pub(crate) fn mark_workers_onboarding_complete(
        &self,
        request_id: String,
    ) -> futures::future::BoxFuture<'static, Result<()>> {
        use futures::FutureExt;
        let workers = self.workers.get().cloned();
        async move {
            let workers = workers.ok_or_else(|| {
                anyhow!("WorkerClients not initialized for mark_workers_onboarding_complete")
            })?;
            workers.mark_onboarding_complete(request_id).await
        }
        .boxed()
    }

    /// Read the full token sequence for a slot.
    ///
    /// Used by the disagg wrapper to populate
    /// `RemotePrefillRequest.token_ids` so the prefill peer knows which
    /// tokens to compute.
    #[allow(dead_code)]
    pub(crate) fn slot_token_ids(&self, request_id: &str) -> Result<Vec<u32>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        let total = slot.sequence.total_tokens();
        let tokens = slot.sequence.tokens_at(0..total);
        Ok(Vec::<u32>::from(tokens))
    }

    /// Local velo instance_id — used as the `initiator_instance_id` on
    /// CD remote-prefill requests so the prefill peer can attach back.
    #[allow(dead_code)]
    pub(crate) fn local_instance_id(&self) -> InstanceId {
        self.runtime.messenger().instance_id()
    }

    /// Borrow the slot's LoRA adapter name. Used by the CD wire to
    /// forward the canonical `kv_hashing::Request` shape to the prefill
    /// side so it can recompute identical hashes locally.
    #[allow(dead_code)]
    pub(crate) fn slot_lora_name(&self, request_id: &str) -> Result<Option<String>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        Ok(slot.request.lora_name.clone())
    }

    /// Borrow the slot's raw salt string (the input to `salt_hash`
    /// computation). Forwarded onto the CD wire alongside `lora_name`.
    #[allow(dead_code)]
    pub(crate) fn slot_salt(&self, request_id: &str) -> Result<Option<String>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        Ok(slot.request.salt.clone())
    }

    /// Read the slot's parsed CD transfer params, if any. Returns `None`
    /// when the slot has no metadata or no `kv_transfer_params`.
    ///
    /// Used by the prefill-side disagg wrapper to detect
    /// CD-bound requests at GNMT time.
    #[allow(dead_code)]
    pub(crate) fn slot_transfer_params(
        &self,
        request_id: &str,
    ) -> Result<Option<kvbm_protocols::disagg::TransferParams>> {
        let slot = self.get_slot(request_id)?;
        let slot = slot.lock();
        slot.transfer_params()
            .map_err(|err| anyhow!("decode kv_transfer_params for {}: {}", request_id, err))
    }

    /// Drive `mark_failed_onboarding` across all worker clients for the
    /// listed G1 block_ids when a CD pipeline cannot complete.
    #[allow(dead_code)]
    pub(crate) fn mark_workers_failed_onboarding(
        &self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> futures::future::BoxFuture<'static, Result<()>> {
        use futures::FutureExt;
        let workers = self.workers.get().cloned();
        async move {
            let workers = workers.ok_or_else(|| {
                anyhow!("WorkerClients not initialized for mark_workers_failed_onboarding")
            })?;
            workers.mark_failed_onboarding(request_id, block_ids).await
        }
        .boxed()
    }
}

/// Snapshot of a slot's match-side state taken mid-onboard, used by the
/// disagg wrapper to split USAA's G1 block_ids into
/// `[computed | local_match | remote_prefill]` ranges.
///
/// Methods/fields are `#[allow(dead_code)]` until `decode_usaa` rewrite
/// (step 6) wires them in.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SlotMatchSplit {
    pub block_size: usize,
    pub computed_blocks: usize,
    pub local_match_blocks: usize,
    pub total_blocks: usize,
    pub all_sequence_hashes: Vec<crate::SequenceHash>,
}

/// Pure split-computation. Extracted for unit-testing the
/// no-onboarding-state degenerate case (cold-cache prefill request)
/// without standing up a full `ConnectorLeader` fixture.
///
/// `onboarding` carries `(local_match_blocks, num_computed_tokens)`
/// when the slot has an onboarding state attached, `None` otherwise.
fn compute_slot_match_split(
    block_size: usize,
    total_blocks: usize,
    all_sequence_hashes: Vec<crate::SequenceHash>,
    onboarding: Option<(usize, usize)>,
) -> SlotMatchSplit {
    let (local_match_blocks, computed_blocks) = match onboarding {
        Some((local, num_computed)) => (local, num_computed / block_size),
        None => (0, 0),
    };
    SlotMatchSplit {
        block_size,
        computed_blocks,
        local_match_blocks,
        total_blocks,
        all_sequence_hashes,
    }
}

#[allow(dead_code)]
impl SlotMatchSplit {
    pub fn remote_blocks(&self) -> usize {
        self.total_blocks
            .saturating_sub(self.computed_blocks)
            .saturating_sub(self.local_match_blocks)
    }

    pub fn local_match_range(&self) -> std::ops::Range<usize> {
        self.computed_blocks..self.computed_blocks + self.local_match_blocks
    }

    pub fn remote_range(&self) -> std::ops::Range<usize> {
        let start = self.computed_blocks + self.local_match_blocks;
        start..self.total_blocks
    }

    pub fn expected_remote_hashes(&self) -> Vec<crate::SequenceHash> {
        self.all_sequence_hashes[self.remote_range()].to_vec()
    }
}

impl ConnectorLeader {
    /// Extend a slot's token sequence with new tokens.
    ///
    /// This is called during decoding when new tokens have been generated
    /// and need to be synchronized to the slot.
    pub fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        let slot = self.get_slot(request_id)?;
        let mut slot = slot.lock();
        slot.extend_tokens(tokens)
    }

    /// Get the number of new tokens that can be loaded from external KV cache.
    ///
    /// This implements the vLLM KVConnector interface for `get_num_new_matched_tokens`:
    /// - Returns `(None, false)` while the find operation is still in progress
    /// - Returns `(Some(0), false)` if no external blocks are found
    /// - Returns `(Some(n), true)` if n tokens can be loaded asynchronously (inter-pass mode)
    /// - Returns `(Some(n), false)` if n tokens will be loaded synchronously (intra-pass mode)
    ///
    /// The first call for a request starts the find operation. Subsequent calls check
    /// the status of the operation and return results when complete.
    ///
    /// The second boolean in the return tuple indicates whether async loading is in progress:
    /// - `true` for inter-pass mode (async out-of-band via Velo messages)
    /// - `false` for intra-pass mode (sync layer-wise during forward pass)
    #[tracing::instrument(level = "info", skip(self), ret)]
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        let shared_slot = self
            .slots
            .get(request_id)
            .map(|slot| slot.clone())
            .expect("slot should exist");

        let mut slot = shared_slot.lock();

        // Bounded escape for a hung async find session. Without this a stuck
        // PreparingToOnboard slot would survive arbitrarily many GNMT calls
        // and only get cleaned up when the request was explicitly finished.
        if slot.force_error_if_preparing_timed_out(slot::PREPARING_TO_ONBOARD_DEFAULT_TIMEOUT) {
            crate::audit!(
                "preparing_to_onboard_timeout",
                request_id,
                num_computed_tokens,
                timeout_secs = slot::PREPARING_TO_ONBOARD_DEFAULT_TIMEOUT.as_secs(),
            );
            // Drain the find sessions held in the now-Error state and fall
            // back to local prefill. `recover_from_match_error` takes the
            // Error state, releases per-shard sessions on the InstanceLeader,
            // and returns the slot to Inactive.
            self.recover_from_match_error(&mut slot);
            return Ok((None, false));
        }

        if slot.match_requires_reset() {
            // Check for inflight offloads - potential race with block freeing
            // vLLM preemption frees G1 blocks immediately, but we may still have
            // offload transfers reading from those blocks.
            // if slot.has_inflight_offloads() {
            //     tracing::error!(
            //         request_id,
            //         "Preemption detected while offloads inflight - deferring reset until transfers complete"
            //     );
            //     // Return (None, false) to signal "not ready" - vLLM will retry next cycle
            //     return Ok((None, false));
            // }

            // Safe to reset - no inflight offloads
            tracing::debug!(request_id, "Resetting slot state after preemption");
            slot.reset_for_preemption();
            // Fall through to normal matching flow
        }

        // Determine the match outcome
        let outcome = self.process_match(&mut slot, num_computed_tokens);
        let match_breakdown = slot
            .onboarding_state()
            .map(|state| state.aggregate_breakdown())
            .unwrap_or_default();
        let blocks_queried = slot.total_query_blocks();

        // Single point for state transition
        match slot.finalize_match_check(outcome) {
            Ok((count, uses_async)) => {
                if let Some(count) = count {
                    self.record_cache_metrics(&match_breakdown, blocks_queried);

                    if count > 0 && slot.mark_matched_tokens_reported() {
                        self.runtime
                            .observability()
                            .compat_metrics()
                            .matched_tokens
                            .inc_by(count as u64);
                    }
                }

                // For intra-pass mode, we always return false for the async flag
                // since loading happens synchronously during the forward pass.
                // For inter-pass mode, we preserve the async flag from finalize_match_check.
                let actual_async = match self.onboard_mode() {
                    OnboardMode::Intra => false,
                    OnboardMode::Inter => uses_async,
                };
                // Request-scoped trace marker. `count == None` is the stall:
                // a find (incl. an in-flight remote search) is still running, so
                // vLLM re-polls. `Some(n)` resolves it to an external match.
                match count {
                    None => crate::audit!("gnmt_pending", request_id, num_computed_tokens),
                    Some(n) => {
                        crate::audit!(
                            "gnmt_matched",
                            request_id,
                            num_computed_tokens,
                            matched_tokens = n,
                            async_load = actual_async
                        )
                    }
                }
                Ok((count, actual_async))
            }
            Err(e) => {
                self.recover_from_match_error(&mut slot);
                if cfg!(debug_assertions) {
                    // If we are in debug mode, we want to fail the request so we can find and diagnose errors.
                    // Often times, errors will result in a misalignment in the understanding of the frameworks policy
                    // and how it calls the connnector api. Notably, these policies can change subtly across versions.
                    Err(e)
                } else {
                    // If we are in release mode, we want to ensure the request can still be processed normally,
                    // albeit without the benefits of getting an external kv cache match.
                    Ok((None, false))
                }
            }
        }
    }

    /// If this is called with `num_external_tokens` > 0, it will be called with all the blocks upto the block(s)
    /// that need to be onboarded, this included any matched G1 blocks.
    ///
    /// In this case, we compute the start block to onboard by scanning back from the end of the block lists by
    /// the `num_external_tokens/block_size`.
    ///
    /// If this is called with `num_external_tokens` == 0, we will be given the remainder of the blocks destined
    /// for prefill.
    ///
    /// The behavior depends on the configured onboard mode:
    /// - **Inter-pass mode**: Spawns an async task to transfer blocks from G2 to G1 via Velo messages.
    /// - **Intra-pass mode**: Stores G2/G1 block pairs for later aggregation in `process_scheduler_output`,
    ///   which will pass them to workers via `KvConnectorMetadata.intra_pass_load`.
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id))]
    pub fn update_state_after_alloc(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let block_size = self.block_size;
        self.get_slot(request_id)?
            .lock()
            .set_match_requires_reset(true);

        if num_external_tokens == 0 {
            return Ok(());
        }

        if !num_external_tokens.is_multiple_of(block_size) {
            bail!(
                "num_external_tokens {} is not a multiple of block size {}",
                num_external_tokens,
                block_size
            );
        }

        let result = match self.onboard_mode() {
            OnboardMode::Inter => {
                // Async out-of-band onboarding via Velo messages
                self.start_onboarding(request_id, block_ids, num_external_tokens)
            }
            OnboardMode::Intra => {
                // Sync layer-wise onboarding - store G2/G1 pairs for later
                self.prepare_intra_pass_onboarding(request_id, block_ids, num_external_tokens)
            }
        };

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                tracing::error!("Failed to start onboarding: {}", e);
                todo!("clean up session and free resources")
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = output.iteration))]
    pub fn build_connector_meta(
        &self,
        output: scheduler::SchedulerOutput,
    ) -> Result<scheduler::KvConnectorMetadata> {
        self.process_scheduler_output(output)
    }

    // ========================================================================
    // Eviction Query Methods (Scheduler Integration)
    // ========================================================================
    //
    // These methods extend the vLLM KVConnector API to support intelligent
    // eviction decisions. The scheduler calls these during preemption to:
    //
    // 1. `can_evict()` - Check if a request can be safely evicted (no inflight offloads)
    // 2. `get_eviction_score()` - Get G2 availability for ranking eviction candidates
    // 3. `get_block_boundary_info()` - Get alignment info for block-boundary eviction
    //
    // These methods are designed to be called from the scheduler's `try_preempt()`
    // method, typically in a loop over candidate requests.

    /// Check if a request can be safely evicted.
    ///
    /// A request **cannot** be evicted if it has inflight G1→G2 offload transfers
    /// in progress. Evicting would free G1 blocks that are being read by RDMA,
    /// causing data corruption or undefined behavior.
    ///
    /// # Returns
    ///
    /// - `true` if the request can be safely evicted (no inflight offloads)
    /// - `false` if the request has inflight offloads and must not be evicted
    ///
    /// # Onboarding Protection
    ///
    /// Requests actively loading KV data from G2 (onboarding) are automatically
    /// protected because they are in the `Waiting` queue (status `WAITING_FOR_REMOTE_KVS`),
    /// not the `Running` queue. Only running requests are candidates for eviction,
    /// so onboarding requests are never considered.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's try_preempt():
    /// for candidate in &running_requests {
    ///     if let Some(connector) = &self.connector {
    ///         if !connector.can_evict(&candidate.request_id()) {
    ///             continue; // Skip - has inflight offloads
    ///         }
    ///     }
    ///     // Candidate is safe to evict
    /// }
    /// ```
    pub fn can_evict(&self, request_id: &str) -> bool {
        // If no slot exists, the request has no connector state - safe to evict
        let Some(slot_ref) = self.slots.get(request_id) else {
            return true;
        };

        let slot = slot_ref.lock();

        // Check for inflight offloads
        !slot.has_inflight_offloads()
    }

    /// Get eviction score for a request based on G2 block coverage.
    ///
    /// Requests with more blocks already offloaded to G2 are preferred for eviction
    /// because:
    ///
    /// - They can be resumed with minimal prefill (onboarding from G2 is fast)
    /// - The work invested in offloading is preserved
    /// - Memory is freed without losing computation
    ///
    /// # Returns
    ///
    /// An [`EvictionScore`] containing:
    /// - `g2_block_count`: Number of blocks available in G2 for this request
    /// - `total_block_count`: Total number of blocks assigned to this request
    /// - `coverage_ratio`: Fraction of blocks in G2 (g2_block_count / total_block_count)
    ///
    /// Higher `coverage_ratio` = better eviction candidate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's victim selection:
    /// let scores: Vec<_> = candidates.iter()
    ///     .filter_map(|c| {
    ///         connector.get_eviction_score(&c.request_id())
    ///             .ok()
    ///             .map(|s| (c, s))
    ///     })
    ///     .collect();
    ///
    /// // Select candidate with highest G2 coverage
    /// let best_victim = scores.iter()
    ///     .max_by(|a, b| a.1.coverage_ratio.partial_cmp(&b.1.coverage_ratio).unwrap());
    /// ```
    pub fn get_eviction_score(&self, request_id: &str) -> Result<EvictionScore> {
        let slot_ref = self.get_slot(request_id)?;
        let slot = slot_ref.lock();

        // TODO: Implement actual G2 block counting by querying InstanceLeader
        // For now, return a stub that indicates no G2 coverage
        //
        // The real implementation would:
        // 1. Get the block IDs assigned to this request from the slot
        // 2. Query the InstanceLeader to see which blocks exist in G2
        // 3. Return the count and ratio

        let total_block_count = slot.assigned_block_count();

        // Stub: No G2 coverage information available yet
        Ok(EvictionScore {
            g2_block_count: 0,
            total_block_count,
            coverage_ratio: 0.0,
        })
    }

    /// Get block boundary alignment information for a request.
    ///
    /// For efficient eviction, we want to evict at block boundaries:
    ///
    /// - Continuing generation until a block is full costs zero extra resources
    ///   (the block is already allocated and partially filled)
    /// - Evicting at a boundary preserves complete blocks that can be resumed
    /// - On resume, we prefill just the known next token for the new block
    ///
    /// # Returns
    ///
    /// A [`BlockBoundaryInfo`] containing:
    /// - `is_at_boundary`: True if the request is at a block boundary (last block is full)
    /// - `tokens_until_boundary`: Tokens remaining until the next block boundary
    /// - `current_block_fill`: Tokens in the current (partial) block
    ///
    /// Prefer evicting requests where `is_at_boundary == true` or `tokens_until_boundary`
    /// is small (close to completing a block).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's victim selection:
    /// for candidate in &candidates {
    ///     let boundary_info = connector.get_block_boundary_info(&candidate.request_id())?;
    ///     if boundary_info.is_at_boundary {
    ///         // Ideal candidate - evict immediately
    ///         return Some(candidate);
    ///     }
    /// }
    /// ```
    pub fn get_block_boundary_info(&self, request_id: &str) -> Result<BlockBoundaryInfo> {
        let slot_ref = self.get_slot(request_id)?;
        let slot = slot_ref.lock();

        let total_tokens = slot.total_tokens();
        let block_size = self.block_size;

        // Calculate position within current block
        let current_block_fill = total_tokens % block_size;
        let is_at_boundary = current_block_fill == 0 && total_tokens > 0;
        let tokens_until_boundary = if is_at_boundary {
            0
        } else {
            block_size - current_block_fill
        };

        Ok(BlockBoundaryInfo {
            is_at_boundary,
            tokens_until_boundary,
            current_block_fill,
        })
    }

    // =========================================================================
    // Projection System Integration - Priority Offload
    // =========================================================================

    /// Request priority offload for blocks planned for eviction.
    ///
    /// When the projection system identifies requests that need to be evicted,
    /// this method requests priority G2 offload for their blocks. This ensures
    /// that blocks are safely in G2 before eviction, enabling faster resume.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The request whose blocks should be offloaded
    /// * `block_ids` - Specific blocks to prioritize (if empty, all blocks)
    ///
    /// # Returns
    ///
    /// The number of blocks that were queued for priority offload.
    ///
    /// # Note
    ///
    /// This is currently a stub. The real implementation will:
    /// 1. Add blocks to a priority queue in the offload engine
    /// 2. Bump their priority above normal background offload
    /// 3. Track completion via the existing inflight offload mechanism
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's planned eviction processing:
    /// if let Some(connector) = &self.connector {
    ///     let block_ids = request.block_ids();
    ///     let queued = connector.request_priority_offload(
    ///         request.request_id(),
    ///         &block_ids,
    ///     )?;
    ///     tracing::debug!(
    ///         request_id = %request.request_id(),
    ///         blocks_queued = queued,
    ///         "Requested priority offload for planned eviction"
    ///     );
    /// }
    /// ```
    pub fn request_priority_offload(
        &self,
        request_id: &str,
        block_ids: &[crate::BlockId],
    ) -> Result<usize> {
        let _slot_ref = self.get_slot(request_id)?;

        // TODO: Implement priority queue in offload engine
        // For now, return 0 to indicate no blocks were queued
        //
        // The real implementation would:
        // 1. Get the OffloadEngine reference from the slot
        // 2. Add blocks to the priority offload queue
        // 3. Track the request in the slot's inflight offloads
        //
        // This requires changes to:
        // - OffloadEngine to support priority queue
        // - RequestSlot to track priority offload state

        tracing::debug!(
            request_id = request_id,
            block_count = block_ids.len(),
            "Priority offload requested (stub - not implemented)"
        );

        Ok(0)
    }

    /// Get per-block G2 status for a request.
    ///
    /// Returns a map of block IDs to their G2 presence status. This is used
    /// by the projection system to determine which blocks are safe to release
    /// from paused requests without losing computed state.
    ///
    /// # Returns
    ///
    /// A HashMap where:
    /// - Key: Block ID
    /// - Value: `true` if the block exists in G2, `false` otherwise
    ///
    /// # Note
    ///
    /// This is currently a stub. The real implementation will query the
    /// InstanceLeader's block registry for G2 presence information.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scheduler's progressive block release:
    /// if let Some(connector) = &self.connector {
    ///     let g2_status = connector.get_block_g2_status(request_id)?;
    ///     let releasable: Vec<_> = request.block_ids()
    ///         .into_iter()
    ///         .filter(|id| g2_status.get(id).copied().unwrap_or(false))
    ///         .collect();
    ///     // These blocks are in G2 and can be safely released
    /// }
    /// ```
    pub fn get_block_g2_status(
        &self,
        request_id: &str,
    ) -> Result<std::collections::HashMap<crate::BlockId, bool>> {
        // Verify request exists
        let _slot_ref = self.get_slot(request_id)?;

        // TODO: Query InstanceLeader for block-level G2 presence
        // For now, return empty map indicating no G2 status available
        //
        // The real implementation would:
        // 1. Get block IDs from the slot (requires adding a method)
        // 2. Query InstanceLeader's G2 block registry
        // 3. Return presence status for each block
        //
        // Note: The slot currently only tracks block count, not individual IDs.
        // The real implementation will need to either:
        // - Add block ID tracking to RequestSlot, or
        // - Query the scheduler's block state for block IDs

        Ok(std::collections::HashMap::new())
    }
}

/// Eviction score for a request based on G2 block coverage.
///
/// Used by the scheduler to rank eviction candidates. Requests with higher
/// G2 coverage are preferred for eviction because they can resume faster.
#[derive(Debug, Clone, Copy)]
pub struct EvictionScore {
    /// Number of blocks for this request that exist in G2 (host memory).
    pub g2_block_count: usize,

    /// Total number of blocks assigned to this request.
    pub total_block_count: usize,

    /// Coverage ratio: g2_block_count / total_block_count.
    ///
    /// - 1.0 = All blocks in G2, request can resume with zero prefill
    /// - 0.0 = No blocks in G2, request must fully recompute
    pub coverage_ratio: f32,
}

/// Block boundary alignment information for a request.
///
/// Used by the scheduler to prefer evicting at block boundaries, which
/// is more efficient for resume operations.
#[derive(Debug, Clone, Copy)]
pub struct BlockBoundaryInfo {
    /// True if the request is exactly at a block boundary (last block is full).
    pub is_at_boundary: bool,

    /// Number of tokens until the next block boundary.
    ///
    /// - 0 if `is_at_boundary` is true
    /// - `block_size - current_block_fill` otherwise
    pub tokens_until_boundary: usize,

    /// Number of tokens in the current (partial) block.
    ///
    /// - 0 if `is_at_boundary` is true
    /// - 1..block_size otherwise
    pub current_block_fill: usize,
}

impl Deref for ConnectorLeader {
    type Target = dyn Leader;

    fn deref(&self) -> &Self::Target {
        self.instance_leader.get().expect("InstanceLeader not set")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SequenceHash;

    /// Regression for the B.2 disagg-bringup smoke failure:
    /// `commit_gnmt_remote → slot_match_split` panicked with
    /// "slot has no active onboarding state" on cold-cache requests
    /// because the inner GNMT path doesn't install onboarding state
    /// when there's nothing to match locally. The wrapper must still
    /// be able to compute a (degenerate) split and proceed with a
    /// fully-remote prefill.
    #[test]
    fn slot_match_split_handles_no_onboarding_state() {
        let block_size = 16;
        let total_blocks = 8;
        let hashes: Vec<SequenceHash> = (0..total_blocks)
            .map(|i| SequenceHash::new(i as u64, None, i as u64))
            .collect();

        // No onboarding state — the cold-cache path that crashed.
        let split = compute_slot_match_split(block_size, total_blocks, hashes.clone(), None);

        assert_eq!(split.local_match_blocks, 0);
        assert_eq!(split.computed_blocks, 0);
        assert_eq!(split.total_blocks, total_blocks);
        assert_eq!(split.block_size, block_size);
        assert_eq!(split.all_sequence_hashes, hashes);
        // The wrapper relies on these range helpers for slicing the
        // prefill request — confirm they're sensible in the
        // degenerate case.
        assert_eq!(split.local_match_range(), 0..0);
        assert_eq!(split.remote_range(), 0..total_blocks);
        assert_eq!(split.remote_blocks(), total_blocks);
    }

    /// Onboarding-state present: split reflects local + computed.
    #[test]
    fn slot_match_split_with_onboarding_state() {
        let block_size = 16;
        let total_blocks = 10;
        let hashes: Vec<SequenceHash> = (0..total_blocks)
            .map(|i| SequenceHash::new(i as u64, None, i as u64))
            .collect();

        // 2 blocks already computed (32 tokens), 3 blocks matched locally.
        let split =
            compute_slot_match_split(block_size, total_blocks, hashes.clone(), Some((3, 32)));

        assert_eq!(split.computed_blocks, 2);
        assert_eq!(split.local_match_blocks, 3);
        assert_eq!(split.local_match_range(), 2..5);
        assert_eq!(split.remote_range(), 5..total_blocks);
        assert_eq!(split.remote_blocks(), 5);
    }
}
