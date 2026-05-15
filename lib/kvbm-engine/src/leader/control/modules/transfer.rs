// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../../../docs/control-transfer.md")]
//!
//! ## Module implementation
//!
//! The handlers in this file are thin shims: deserialize the request,
//! call an `InstanceLeader` method, wrap the result in a [`ControlReply`],
//! and return. The substantive logic lives below as free functions
//! invoked by both the `InstanceLeader` methods and the legacy
//! `search_prefix` / `search_scatter` back-compat shims. Putting the
//! work on `InstanceLeader` keeps the public surface discoverable for
//! in-process callers and avoids forcing every caller through the velo
//! wire.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use velo::{Handler, Messenger};

use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};

use crate::G3;
use crate::leader::session::BlockHolder;
use crate::leader::session::staging::stage_g3_to_g2;
use kvbm_protocols::control::modules::transfer::{
    CLOSE_SESSION_HANDLER, CloseTransferSessionRequest, CloseTransferSessionResponse, FindMode,
    MatchBreakdown, OPEN_SESSION_HANDLER, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PULL_FROM_SESSION_HANDLER, PullFromSessionRequest, PullFromSessionResponse,
    SEARCH_PREFIX_HANDLER, SEARCH_SCATTER_HANDLER, SearchMode, SearchRequest, SearchResponse,
    TierSelection, TransferSessionCapability,
};
use kvbm_protocols::control::{ControlError, ControlReply, ModuleId};

use crate::disagg::session::{AvailabilityDelta, CommitDelta, Session};
use crate::leader::InstanceLeader;
use crate::leader::control::ControlModule;
use crate::{G2, SequenceHash};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// The `transfer` control module — always enabled.
///
/// Carries only an `Arc<InstanceLeader>`; everything the handlers need
/// (g2_manager, g3_manager, session_factory_cell, session_manager,
/// runtime, messenger.instance_id) is reached through it.
pub struct TransferModule {
    leader: Arc<InstanceLeader>,
}

impl TransferModule {
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self { leader }
    }
}

impl ControlModule for TransferModule {
    fn id(&self) -> ModuleId {
        ModuleId::Transfer
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        register_open_session(messenger, &self.leader)?;
        register_pull_from_session(messenger, &self.leader)?;
        register_close_session(messenger, &self.leader)?;
        // Legacy back-compat handlers, retained as shims over open_session.
        register_search_prefix(messenger, &self.leader)?;
        register_search_scatter(messenger, &self.leader)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

fn register_open_session(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(OPEN_SESSION_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: OpenTransferSessionRequest = ctx.input;
            let reply: ControlReply<OpenTransferSessionResponse> =
                leader.open_transfer_session(req).await.into();
            Ok::<ControlReply<OpenTransferSessionResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({OPEN_SESSION_HANDLER}): {e}"))?;
    Ok(())
}

fn register_close_session(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(CLOSE_SESSION_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: CloseTransferSessionRequest = ctx.input;
            let reply: ControlReply<CloseTransferSessionResponse> =
                leader.close_transfer_session(req).await.into();
            Ok::<ControlReply<CloseTransferSessionResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({CLOSE_SESSION_HANDLER}): {e}"))?;
    Ok(())
}

fn register_pull_from_session(
    messenger: &Arc<Messenger>,
    leader: &Arc<InstanceLeader>,
) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(PULL_FROM_SESSION_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: PullFromSessionRequest = ctx.input;
            let reply: ControlReply<PullFromSessionResponse> =
                leader.pull_from_session(req).await.into();
            Ok::<ControlReply<PullFromSessionResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({PULL_FROM_SESSION_HANDLER}): {e}"))?;
    Ok(())
}

fn register_search_prefix(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    register_search_shim(messenger, leader, SEARCH_PREFIX_HANDLER, SearchMode::Prefix)
}

fn register_search_scatter(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    register_search_shim(
        messenger,
        leader,
        SEARCH_SCATTER_HANDLER,
        SearchMode::Scatter,
    )
}

/// Adapter: legacy `SearchRequest`/`SearchResponse` over the new
/// `open_session` path with `find_mode = Sync` and `tiers = default`.
fn register_search_shim(
    messenger: &Arc<Messenger>,
    leader: &Arc<InstanceLeader>,
    handler_name: &'static str,
    mode: SearchMode,
) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(handler_name, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: SearchRequest = ctx.input;
            let open_req = OpenTransferSessionRequest {
                sequence_hashes: req.sequence_hashes,
                search_mode: mode,
                find_mode: FindMode::Sync,
                tiers: TierSelection::default(),
                watchdog_ms: None,
            };
            let reply: ControlReply<SearchResponse> = leader
                .open_transfer_session(open_req)
                .await
                .map(|resp| match resp {
                    OpenTransferSessionResponse::NoBlocksFound => SearchResponse::NoBlocksFound,
                    OpenTransferSessionResponse::Sync { capability, .. } => {
                        SearchResponse::Session {
                            session_id: capability.session_id,
                        }
                    }
                    // Sync request must produce Sync or NoBlocksFound; treat
                    // anything else as internal error.
                    OpenTransferSessionResponse::Async { .. } => SearchResponse::Session {
                        session_id: uuid::Uuid::nil(),
                    },
                })
                .into();
            Ok::<ControlReply<SearchResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({handler_name}): {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// open_transfer_session — substantive logic
// ---------------------------------------------------------------------------

/// Result of the populator's `find_phase`. Held only briefly: in Sync
/// mode we read `g2_committed`, `g3_committed`, and `breakdown` into the
/// response; in either mode the `*_blocks` are consumed by `stage_phase`.
struct FindOutcome {
    g2_committed: Vec<SequenceHash>,
    g2_blocks: Vec<ImmutableBlock<G2>>,
    g3_committed: Vec<SequenceHash>,
    g3_blocks: Vec<ImmutableBlock<G3>>,
    breakdown: MatchBreakdown,
}

impl FindOutcome {
    fn committed(&self) -> Vec<SequenceHash> {
        let mut out = Vec::with_capacity(self.g2_committed.len() + self.g3_committed.len());
        out.extend(&self.g2_committed);
        out.extend(&self.g3_committed);
        out
    }
}

/// Scan local tiers per `search_mode` / `tiers`. Synchronous body — both
/// G2 and G3 scans are in-memory hashmap lookups. `async fn` for
/// forward-compat with G4 (object-store) scans in v1.1.
///
/// In v1, **G3 is only consulted in `SearchMode::Scatter`**. The Prefix
/// mode preserves the existing semantic (contiguous G2 prefix); extending
/// the prefix walk into G3 requires careful gap handling that doesn't
/// pay for itself yet.
async fn find_phase(
    leader: &Arc<InstanceLeader>,
    hashes: &[SequenceHash],
    search_mode: SearchMode,
    tiers: TierSelection,
) -> FindOutcome {
    match search_mode {
        SearchMode::Prefix => {
            // Contiguous prefix in G2 only. G3/G4 not searched.
            let g2_blocks = leader.g2_manager().match_blocks(hashes);
            let g2_committed: Vec<SequenceHash> =
                g2_blocks.iter().map(|b| b.sequence_hash()).collect();
            let breakdown = MatchBreakdown {
                host_blocks: g2_blocks.len(),
                disk_blocks: 0,
                object_blocks: 0,
            };
            FindOutcome {
                g2_committed,
                g2_blocks,
                g3_committed: Vec::new(),
                g3_blocks: Vec::new(),
                breakdown,
            }
        }
        SearchMode::Scatter => {
            let g2_map = leader.g2_manager().scan_matches(hashes, /* touch */ false);
            let g2_committed: Vec<SequenceHash> = g2_map.keys().copied().collect();
            let g2_blocks: Vec<ImmutableBlock<G2>> = g2_map.into_values().collect();

            let mut g3_committed: Vec<SequenceHash> = Vec::new();
            let mut g3_blocks: Vec<ImmutableBlock<G3>> = Vec::new();

            if tiers.g3
                && let Some(g3_manager) = leader.g3_manager()
            {
                let g2_set: std::collections::HashSet<SequenceHash> =
                    g2_committed.iter().copied().collect();
                let remaining: Vec<SequenceHash> = hashes
                    .iter()
                    .filter(|h| !g2_set.contains(h))
                    .copied()
                    .collect();
                if !remaining.is_empty() {
                    let g3_map = g3_manager.scan_matches(&remaining, false);
                    for (h, b) in g3_map {
                        g3_committed.push(h);
                        g3_blocks.push(b);
                    }
                }
            }

            let breakdown = MatchBreakdown {
                host_blocks: g2_blocks.len(),
                disk_blocks: g3_blocks.len(),
                object_blocks: 0,
            };
            FindOutcome {
                g2_committed,
                g2_blocks,
                g3_committed,
                g3_blocks,
                breakdown,
            }
        }
    }
}

/// Drive the disagg session's commit / make_available / finish_*
/// based on `FindOutcome`. G2 blocks are made available immediately;
/// G3 hashes are committed up front, then staged G3→G2 in the
/// background before make_available.
///
/// Errors propagate as `ControlError::Internal`; on error the caller
/// is expected to call `session.close(...)` to surface
/// `LifecycleEvent::Failed` to the puller.
async fn stage_phase(
    leader: Arc<InstanceLeader>,
    session: Arc<dyn Session>,
    find: FindOutcome,
) -> Result<(), ControlError> {
    let FindOutcome {
        g2_committed,
        g2_blocks,
        g3_committed,
        g3_blocks,
        ..
    } = find;

    if !g2_committed.is_empty() {
        session
            .commit(g2_committed)
            .map_err(|e| ControlError::Internal(format!("commit g2: {e:#}")))?;
        session
            .make_available(g2_blocks)
            .map_err(|e| ControlError::Internal(format!("make_available g2: {e:#}")))?;
    }

    if !g3_committed.is_empty() {
        // Commit G3 hashes up front so the puller sees the full
        // committed set via `commits()` before staging completes.
        session
            .commit(g3_committed)
            .map_err(|e| ControlError::Internal(format!("commit g3: {e:#}")))?;

        let parallel_worker = leader.parallel_worker().ok_or_else(|| {
            ControlError::Internal(
                "G3 staging requires a parallel_worker; leader was built without workers".into(),
            )
        })?;

        let holder = BlockHolder::<G3>::new(g3_blocks);
        let staged = stage_g3_to_g2(&holder, leader.g2_manager(), &*parallel_worker)
            .await
            .map_err(|e| ControlError::Internal(format!("stage_g3_to_g2: {e:#}")))?;

        session
            .make_available(staged.new_g2_blocks)
            .map_err(|e| ControlError::Internal(format!("make_available staged: {e:#}")))?;
    }

    session
        .finish_commits()
        .map_err(|e| ControlError::Internal(format!("finish_commits: {e:#}")))?;
    session
        .finish_availability()
        .map_err(|e| ControlError::Internal(format!("finish_availability: {e:#}")))?;
    Ok(())
}

/// Engine-side implementation behind [`InstanceLeader::open_transfer_session`].
///
/// G2 + G3 in v1; G4 in v1.1.
pub(crate) async fn open_transfer_session(
    leader: &Arc<InstanceLeader>,
    req: OpenTransferSessionRequest,
) -> Result<OpenTransferSessionResponse, ControlError> {
    let find = find_phase(leader, &req.sequence_hashes, req.search_mode, req.tiers).await;
    let committed = find.committed();
    let breakdown = find.breakdown;

    // Pre-flight: if find_phase produced G3 matches we cannot stage, fail
    // *before* opening a session. `stage_g3_to_g2` requires a
    // `parallel_worker`; without it the background populator would close
    // the session moments after `open_transfer_session` returned, leaving
    // the caller with a capability that points to a teardown-in-progress
    // session — a "usable-looking session that cannot serve blocks". The
    // honest answer is to reject the open and tell the operator what to
    // fix.
    if !find.g3_blocks.is_empty() && leader.parallel_worker().is_none() {
        return Err(ControlError::Internal(
            "g3_requires_parallel_worker: leader has no parallel_worker but G3 matches \
             were found; configure a parallel_worker on this leader or set \
             tiers.g3=false in the request"
                .into(),
        ));
    }

    // Sync mode + zero matches across all selected tiers: short-circuit,
    // do not open a session.
    if matches!(req.find_mode, FindMode::Sync) && committed.is_empty() {
        crate::engine_audit!(
            "transfer_session_no_matches",
            requested = req.sequence_hashes.len(),
            search_mode = ?req.search_mode
        );
        return Ok(OpenTransferSessionResponse::NoBlocksFound);
    }

    let factory = leader
        .session_factory_cell()
        .get()
        .ok_or(ControlError::NotInitialized)?
        .clone();

    let session_id = uuid::Uuid::new_v4();
    let session = factory
        .open(session_id)
        .map_err(|e| ControlError::Internal(format!("open session: {e:#}")))?;
    let endpoint = session
        .endpoint()
        .ok_or_else(|| ControlError::Internal("opened session has no endpoint".into()))?;
    let instance_id = leader.messenger().instance_id();

    let capability = TransferSessionCapability {
        session_id,
        instance_id,
        endpoint,
    };

    // Park the session before kicking off the populator. Per-session
    // watchdog override is v1.1 (SessionManager currently has a single
    // fixed watchdog at construction time); accept the field on the
    // request now so the wire is stable.
    let _ = req.watchdog_ms;
    leader.session_manager().register(Arc::clone(&session));

    crate::engine_audit!(
        "transfer_session_opened",
        %session_id,
        find_mode = ?req.find_mode,
        search_mode = ?req.search_mode,
        committed = committed.len(),
        g2_hits = breakdown.host_blocks,
        g3_hits = breakdown.disk_blocks
    );

    // Always spawn stage_phase in the background. Sync mode awaits the
    // (synchronous-body) find_phase before returning so its response
    // can include `committed` + `breakdown`; stage_phase runs after
    // regardless.
    let runtime = leader.runtime();
    let leader_for_task = Arc::clone(leader);
    let session_for_task = Arc::clone(&session);
    runtime.spawn(async move {
        match stage_phase(leader_for_task, Arc::clone(&session_for_task), find).await {
            Ok(()) => {
                crate::engine_audit!(
                    "transfer_populator_complete",
                    %session_id
                );
            }
            Err(err) => {
                tracing::error!(error = %err, %session_id, "transfer populator failed");
                crate::engine_audit!(
                    "transfer_populator_failed",
                    %session_id,
                    error = %err
                );
                session_for_task.close(Some(format!("populator: {err}")));
            }
        }
    });

    match req.find_mode {
        FindMode::Sync => Ok(OpenTransferSessionResponse::Sync {
            capability,
            committed,
            breakdown,
        }),
        FindMode::Async => Ok(OpenTransferSessionResponse::Async { capability }),
    }
}

// ---------------------------------------------------------------------------
// close_transfer_session — substantive logic
// ---------------------------------------------------------------------------

/// Engine-side implementation behind [`InstanceLeader::close_transfer_session`].
///
/// Idempotent: a missing session returns `Ok(was_present: false)`.
pub(crate) async fn close_transfer_session(
    leader: &Arc<InstanceLeader>,
    req: CloseTransferSessionRequest,
) -> Result<CloseTransferSessionResponse, ControlError> {
    let removed = leader.session_manager().remove(&req.session_id);
    let was_present = removed.is_some();
    let session_id = req.session_id;
    let reason_for_audit = req.reason.clone();
    if let Some(session) = removed {
        session.close(req.reason);
    }
    crate::engine_audit!(
        "transfer_session_closed",
        %session_id,
        was_present,
        reason = ?reason_for_audit
    );
    Ok(CloseTransferSessionResponse { was_present })
}

// ---------------------------------------------------------------------------
// pull_from_session — substantive logic
// ---------------------------------------------------------------------------

/// Engine-side implementation behind [`InstanceLeader::pull_from_session`].
///
/// Attach to the holder's session, drain `commits()` to learn what's
/// committed, then drain `availability()` and pull each batch into
/// freshly-allocated G2 mutables on this side. Stage + register each
/// pulled mutable so the blocks land in the local registry.
pub(crate) async fn pull_from_session(
    leader: &Arc<InstanceLeader>,
    req: PullFromSessionRequest,
) -> Result<PullFromSessionResponse, ControlError> {
    let endpoint = req.endpoint.ok_or_else(|| {
        ControlError::Internal(
            "endpoint_required: pull_from_session requires an explicit endpoint in v1 \
             (hub-registry resolution is v1.1)"
                .into(),
        )
    })?;

    let factory = leader
        .session_factory_cell()
        .get()
        .ok_or(ControlError::NotInitialized)?
        .clone();

    crate::engine_audit!(
        "transfer_pull_started",
        session_id = %req.session_id,
        source = %req.source_instance_id,
        selector_present = req.selector.is_some()
    );

    let session = factory
        .attach(req.session_id, req.source_instance_id, endpoint)
        .await
        .map_err(|e| ControlError::Internal(format!("attach: {e:#}")))?;

    // Drain commits to build the committed set. Replay-on-first-subscribe
    // means anything that arrived before this subscribe is buffered and
    // delivered as a single Added batch.
    let mut commit_stream = session.commits();
    let mut committed: HashSet<SequenceHash> = HashSet::new();
    while let Some(delta) = commit_stream.next().await {
        match delta {
            CommitDelta::Added(hashes) => committed.extend(hashes),
            CommitDelta::Closed => break,
        }
    }
    drop(commit_stream);

    // Resolve target set against selector.
    let target_hashes: Vec<SequenceHash> = match req.selector {
        None => committed.iter().copied().collect(),
        Some(selector) => {
            let missing: Vec<SequenceHash> = selector
                .iter()
                .copied()
                .filter(|h| !committed.contains(h))
                .collect();
            if !missing.is_empty() {
                // Best-effort close so the holder's session_manager
                // can evict promptly; do not propagate this error.
                session.close(Some("selector references uncommitted hashes".into()));
                return Err(ControlError::Internal(format!(
                    "hashes_not_committed: {} hash(es) in selector are not committed",
                    missing.len()
                )));
            }
            selector
                .into_iter()
                .filter(|h| committed.contains(h))
                .collect()
        }
    };

    if target_hashes.is_empty() {
        session.finalize(None);
        return Ok(PullFromSessionResponse::default());
    }

    let target_set: HashSet<SequenceHash> = target_hashes.iter().copied().collect();

    // Drain availability, pulling each chunk.
    let block_size = leader.g2_manager().block_size();
    let mut pulled_order: Vec<SequenceHash> = Vec::with_capacity(target_set.len());
    let mut pulled_set: HashSet<SequenceHash> = HashSet::new();

    let mut avail_stream = session.availability();
    'drain: while let Some(delta) = avail_stream.next().await {
        match delta {
            AvailabilityDelta::Available(blocks) => {
                // Filter to what we actually want and haven't pulled yet.
                let chunk_hashes: Vec<SequenceHash> = blocks
                    .into_iter()
                    .filter_map(|b| {
                        if target_set.contains(&b.hash) && !pulled_set.contains(&b.hash) {
                            Some(b.hash)
                        } else {
                            None
                        }
                    })
                    .collect();
                if chunk_hashes.is_empty() {
                    continue;
                }

                let chunk_len = chunk_hashes.len();
                let dst = leader
                    .g2_manager()
                    .allocate_blocks(chunk_len)
                    .ok_or_else(|| {
                        ControlError::Internal(format!(
                            "pull: failed to allocate {chunk_len} G2 mutable blocks"
                        ))
                    })?;

                let filled = session
                    .pull(chunk_hashes.clone(), dst)
                    .await
                    .map_err(|e| ControlError::Internal(format!("session.pull: {e:#}")))?;

                if filled.len() != chunk_len {
                    return Err(ControlError::Internal(format!(
                        "pull: session.pull returned {} blocks, expected {}",
                        filled.len(),
                        chunk_len
                    )));
                }

                // Stage + register the filled mutables so the blocks
                // join the local G2 registry as ImmutableBlocks.
                let mut completes: Vec<CompleteBlock<G2>> = Vec::with_capacity(chunk_len);
                for (mutable, hash) in filled.into_iter().zip(chunk_hashes.iter()) {
                    let complete = mutable.stage(*hash, block_size).map_err(|e| {
                        ControlError::Internal(format!("stage pulled block: {e:#}"))
                    })?;
                    completes.push(complete);
                }
                let _registered = leader.g2_manager().register_blocks(completes);

                pulled_order.extend(&chunk_hashes);
                pulled_set.extend(&chunk_hashes);

                if pulled_set.len() == target_set.len() {
                    break 'drain;
                }
            }
            AvailabilityDelta::Drained => break 'drain,
        }
    }
    drop(avail_stream);

    // If availability drained before we got everything we wanted,
    // surface that as an error — the holder's commits promised more
    // than it could make available.
    if pulled_set.len() < target_set.len() {
        session.finalize(None);
        return Err(ControlError::Internal(format!(
            "pull: availability drained with {} of {} target hashes pulled",
            pulled_set.len(),
            target_set.len()
        )));
    }

    // Cooperative shutdown. The holder side will see Finished + (if it
    // also finalizes) trigger the wire-level finalize; otherwise the
    // SessionManager's watchdog evicts. Either way the puller-side
    // arc drops when this function returns.
    session.finalize(None);

    crate::engine_audit!(
        "transfer_pull_completed",
        session_id = %req.session_id,
        pulled = pulled_order.len()
    );

    Ok(PullFromSessionResponse {
        pulled: pulled_order,
        breakdown: MatchBreakdown {
            host_blocks: pulled_set.len(),
            disk_blocks: 0,
            object_blocks: 0,
        },
    })
}
