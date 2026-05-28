// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-based transport for the standalone KV cache indexer (DEP-976).
//!
//! This module replaces the HTTP + ZMQ transport stack with velo:
//!
//! - **Event ingestion** (`subscriber`): workers fire-and-forget
//!   [`IndexerEventBatch`] messages via velo active messaging — no per-worker
//!   ZMQ socket, no acknowledgement overhead.
//! - **Prefix-match queries** (`query_engine`): gateways call a velo unary RPC
//!   and receive a [`crate::indexer::IndexerQueryResponse`] back.
//! - **Peer discovery** (`discovery`): the indexer publishes its `PeerInfo` to a
//!   filesystem path; workers/routers pick it up automatically.
//!
//! The existing HTTP server path (`indexer-runtime` feature) is untouched.
//!
//! ## Ordering
//!
//! Velo's `DispatchMode::Spawn` delivers concurrent handler tasks that may
//! enqueue out of wire order.  Each [`IndexerEventBatch`] and
//! [`IndexerUnregisterRequest`] carries a `sender_id` + monotonic `seq`; the
//! per-shard consumer reorders per sender via a [`SenderState`] buffer.
//! `sender_id = 0` bypasses reordering (legacy / query-engine use).
//! The per-shard channel is bounded at [`SHARD_CHANNEL_CAPACITY`]; overflow
//! is logged at ERROR level and dropped.

pub mod discovery;
pub mod query_engine;
pub mod subscriber;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::protocols::{LocalBlockHash, RouterEvent};
use crate::standalone_indexer::indexer::{Indexer, create_indexer};

// ── Handler endpoint names ────────────────────────────────────────────────

/// Velo active-messaging endpoint for KV cache event batches (write path).
pub const INDEXER_EVENT_HANDLER: &str = "kv_router.indexer.event";

/// Velo active-messaging endpoint for worker registration (optional; lazy
/// shard creation also happens on the first event batch).
pub const INDEXER_REGISTER_HANDLER: &str = "kv_router.indexer.register";

/// Velo active-messaging endpoint for worker unregistration.
///
/// Workers should send an [`IndexerUnregisterRequest`] on clean shutdown so
/// that stale `WorkerWithDpRank` entries are pruned from the shard before any
/// replacement worker registers.  The request is enqueued on the same ordered
/// channel as event batches, so it is applied after all in-flight events from
/// the departing worker have been processed.
pub const INDEXER_UNREGISTER_HANDLER: &str = "kv_router.indexer.unregister";

/// Velo unary-RPC endpoint for prefix-match queries (read path).
pub const INDEXER_QUERY_HANDLER: &str = "kv_router.indexer.query";

// ── Wire types ────────────────────────────────────────────────────────────

/// A batch of KV cache events sent from a worker to the indexer.
///
/// Workers may bundle multiple events into one message to amortise per-call
/// overhead; the handler applies them sequentially.
///
/// ## Ordering
///
/// `sender_id` is the velo `instance_id` of the sending worker.  Set to `0`
/// to bypass per-sender reordering (batch applied immediately on arrival).
/// `seq` is a monotonically increasing per-sender counter starting at `0`;
/// ignored when `sender_id == 0`.  Legacy senders that omit both fields
/// default to `sender_id = 0` via `#[serde(default)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerEventBatch {
    pub model_name: String,
    pub tenant_id: String,
    /// Token-block size used by the worker; must match the indexer shard.
    pub block_size: u32,
    pub events: Vec<RouterEvent>,
    /// Velo instance ID of the sender; `0` means no ordering guarantee.
    #[serde(default)]
    pub sender_id: u64,
    /// Per-sender monotonic sequence number; ignored when `sender_id == 0`.
    #[serde(default)]
    pub seq: u64,
}

/// Optional registration message sent by a worker before its first event batch.
///
/// Sending this message pre-creates the indexer shard for `(model_name,
/// tenant_id)` so that it is ready before the first event arrives.
/// Registration is **not** required — shards are also created lazily on the
/// first event batch.
///
/// **Note:** `threads` in this request is ignored server-side; shard thread
/// allocation is controlled by the [`RuntimeRegistry`] default set at startup
/// via [`RuntimeRegistry::new`].  The field is retained for wire compatibility.
///
/// ## Recovery after unhealthy mark
///
/// Registration does **not** clear an existing unhealthy-sender mark.  Two
/// safe recovery paths exist:
///
/// 1. **Send [`IndexerUnregisterRequest`]**: evicts stale indexed blocks and
///    replaces the `u64::MAX` watermark with the Unregister's `seq = N`.
///    Subsequent event batches with `seq > N` are accepted; the consumer
///    seeds `next_expected` at `N + 1` so gaps within the fresh epoch are
///    detected correctly rather than silently skipped.
/// 2. **Fresh `sender_id`**: restarting the velo `Messenger` produces a new
///    `instance_id`; the new `sender_id` has no watermark entry and is
///    accepted immediately without requiring any prior Unregister.
///
/// Clearing the watermark in the registration handler is not safe because the
/// consumer may not yet have processed the reset, and delayed events from the
/// previous epoch could be applied against a fresh `SenderState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerRegisterRequest {
    pub worker_id: u64,
    pub dp_rank: u32,
    pub model_name: String,
    pub tenant_id: String,
    pub block_size: u32,
    /// Ignored server-side; shard threads are controlled by [`RuntimeRegistry`].
    pub threads: usize,
}

/// Unregistration message sent by a worker on clean shutdown.
///
/// Causes the server to call `remove_worker_dp_rank` on the shard, evicting
/// the worker's blocks from the index.  The message is enqueued on the same
/// per-sender sequenced channel as event batches; when `sender_id != 0` the
/// consumer's reorder buffer ensures it is applied only after all preceding
/// batches with lower `seq` values.
///
/// `seq` must be one greater than the last [`IndexerEventBatch::seq`] sent by
/// this worker for the same `(model_name, tenant_id)` shard — i.e. the
/// Unregister is the terminal event in the sender's sequence.
///
/// **Limitation:** crashes without a clean shutdown leave stale entries until
/// the shard is otherwise evicted.  A heartbeat/lease mechanism would be
/// needed for full crash recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerUnregisterRequest {
    pub worker_id: u64,
    pub dp_rank: u32,
    pub model_name: String,
    pub tenant_id: String,
    /// Velo WorkerId of the sending worker; must match the `sender_id` used in
    /// [`IndexerEventBatch`] messages.  Set to `0` for legacy senders (applied
    /// immediately, no ordering guarantee).
    #[serde(default)]
    pub sender_id: u64,
    /// Per-sender monotonic sequence number; must be one past the last event
    /// batch `seq` from this worker.  Ignored when `sender_id == 0`.
    #[serde(default)]
    pub seq: u64,
    /// Token-block size of the shard.  When non-zero the handler will create
    /// the shard if it does not exist yet, ensuring Unregister is not silently
    /// dropped when it races ahead of the first event batch.  Set to `0` for
    /// legacy senders that do not know the block size; the handler falls back
    /// to the existing shard lookup in that case.
    #[serde(default)]
    pub block_size: u32,
}

/// Request body for a prefix-match query sent to the indexer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeQueryRequest {
    pub model_name: String,
    pub tenant_id: String,
    /// Sequence of block hashes to look up.
    pub block_hashes: Vec<LocalBlockHash>,
    /// When `true`, only the device-tier (GPU) overlap is returned via
    /// [`Indexer::find_matches`], skipping the lower-tier walk entirely.
    pub device_only: bool,
}

// ── RuntimeRegistry ───────────────────────────────────────────────────────

/// Capacity of the per-shard [`ShardMessage`] channel.
///
/// When the channel is full the AM handler logs an error and drops the message.
/// This is the bounded failure policy: visible (ERROR-level log) and bounded
/// (no unbounded memory growth).  A worker whose event rate consistently
/// saturates the indexer channel should be co-located or the indexer scaled.
pub const SHARD_CHANNEL_CAPACITY: usize = 4096;

/// Maximum out-of-order batches buffered per sender before gap recovery.
///
/// When exceeded the sender is marked unhealthy and its indexed blocks are
/// evicted.  Bounds memory when a sequence predecessor is permanently lost.
const MAX_PENDING_PER_SENDER: usize = 256;

/// Maximum time a sequence gap may persist before gap recovery triggers.
///
/// Complements `MAX_PENDING_PER_SENDER`: a small gap (e.g. one lost batch
/// before an Unregister) may never fill the buffer.  After this timeout the
/// missing predecessor is declared permanently lost and the sender is evicted.
const GAP_STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// How often the consumer sweeps open gaps for timeout expiry.
///
/// The inline check only fires on new arrivals from the same sender; this
/// sweep catches senders that go silent after opening a gap (e.g. a crash).
const GAP_SWEEP_PERIOD: std::time::Duration = std::time::Duration::from_secs(10);

/// Capacity of the per-shard control channel for out-of-band eviction requests.
///
/// Separate from the event channel so eviction is never dropped when the event
/// channel is full.  `mark_sender_unhealthy` spawns a task calling
/// `send().await` (not `try_send`), so a momentarily full control channel
/// merely blocks the task rather than dropping the request.  At most one task
/// per sender per unhealthy epoch is spawned (deduplication on `u64::MAX`).
const CONTROL_CHANNEL_CAPACITY: usize = 128;

/// Unified action type for the per-sender reorder buffer.
///
/// Both event batches and lifecycle messages (Unregister) are sequenced through
/// the same `BTreeMap` so the consumer applies them in the correct order.
enum SequencedAction {
    Events(Vec<RouterEvent>),
    Unregister { worker_id: u64, dp_rank: u32 },
}

/// Per-sender reorder state maintained by the per-shard background consumer.
///
/// Both `Events` and `Unregister` messages from the same sender share a single
/// monotonic sequence namespace.  An `Unregister` must carry a `seq` one past
/// the last event batch's `seq` for that sender so the consumer applies it only
/// after all preceding events have been processed.
///
/// The entry is removed from `sender_states` when the `Unregister` action is
/// applied, discarding any pending events buffered after it.
///
/// `worker_ids` tracks every `(worker_id, dp_rank)` pair seen in event batches
/// from this sender.  When the sender is marked unhealthy (gap recovery or
/// channel overflow), the consumer evicts these entries from the indexer so that
/// stale blocks are not returned by prefix-match queries.
#[derive(Default)]
struct SenderState {
    next_expected: u64,
    pending: BTreeMap<u64, SequencedAction>,
    /// All `(worker_id, dp_rank)` pairs ever seen in Events from this sender.
    worker_ids: HashSet<(u64, u32)>,
    /// When the current gap started — i.e. when the first out-of-order action
    /// was buffered with the predecessor still missing.  `None` when `pending`
    /// is empty (no open gap).  Compared against [`GAP_STALL_TIMEOUT`] on every
    /// subsequent out-of-order arrival to detect permanently-lost predecessors
    /// that would never fill the buffer to [`MAX_PENDING_PER_SENDER`].
    gap_started: Option<std::time::Instant>,
}

/// Messages routed through the per-shard ordered channel.
///
/// All writes and lifecycle events are sequenced through a single bounded
/// `mpsc` channel.  The channel is bounded to prevent unbounded memory growth
/// when the consumer is slower than the producer; overflow is logged at ERROR
/// level and the message is dropped.
pub enum ShardMessage {
    /// A batch of KV cache events to apply sequentially.
    ///
    /// `sender_id == 0` means "apply immediately, no ordering"; any other value
    /// routes through the per-sender reorder buffer.
    Events {
        sender_id: u64,
        seq: u64,
        events: Vec<RouterEvent>,
    },
    /// Remove a specific worker's blocks from the shard index.
    ///
    /// `sender_id` must match the value used in preceding `Events` messages so
    /// the reorder buffer applies this after all lower-`seq` batches.
    /// `worker_id` is the KV router worker id passed to `remove_worker_dp_rank`.
    Unregister {
        sender_id: u64,
        seq: u64,
        worker_id: u64,
        dp_rank: u32,
    },
}

/// Sender half of the per-shard [`ShardMessage`] channel.
///
/// Bounded at [`SHARD_CHANNEL_CAPACITY`].  Use `try_send`; log at ERROR level
/// and return an error from the AM handler closure when the channel is full so
/// the overflow is visible without silently corrupting indexer state.
pub type EventSender = mpsc::Sender<ShardMessage>;

/// Sender half of the per-shard control channel.
///
/// Carries `sender_id` values that the consumer should immediately evict.
/// Separate from the event channel (bounded at [`CONTROL_CHANNEL_CAPACITY`])
/// so that eviction requests are never dropped when the event channel is full.
type ControlSender = mpsc::Sender<u64>;

/// Lightweight shard registry for the velo-based indexer.
///
/// Unlike [`crate::standalone_indexer::registry::WorkerRegistry`] this type
/// does not manage ZMQ listeners — events arrive over velo active messaging
/// instead.  It maps `(model_name, tenant_id)` pairs to [`Indexer`] shards
/// and creates new shards on demand.
///
/// ## Ordering
///
/// Each shard has a single background consumer that drains the bounded
/// [`EventSender`] channel sequentially.  Non-zero-`sender_id` batches are
/// reordered per sender via [`SenderState`]; `sender_id == 0` is applied
/// immediately.  `ShardMessage::Unregister` shares the same sequence space.
pub struct RuntimeRegistry {
    /// Per-shard state: `(indexer, block_size, event_tx, control_tx, unhealthy_senders)`.
    ///
    /// `unhealthy_senders` maps `sender_id → watermark_seq` for senders that
    /// experienced a sequencing failure:
    ///
    /// - `watermark_seq == u64::MAX` — permanently unhealthy: the bounded
    ///   channel overflowed or the reorder buffer exceeded
    ///   [`MAX_PENDING_PER_SENDER`].  **All** subsequent event batches from this
    ///   sender are discarded until the sender sends an `Unregister` message.
    /// - `watermark_seq == N` — post-Unregister tombstone: the sender's
    ///   `Unregister` was processed with `seq = N`.  Any delayed in-flight
    ///   event batches with `seq <= N` (from the pre-Unregister stream) are
    ///   discarded.  Batches with `seq > N` are treated as a fresh epoch.
    ///
    /// `Unregister` messages are always passed through to `apply_action`
    /// regardless of the watermark; `apply_action` then replaces any existing
    /// entry with the Unregister's `seq` as the new post-Unregister tombstone.
    ///
    /// `control_tx` is a dedicated bounded channel ([`CONTROL_CHANNEL_CAPACITY`])
    /// for out-of-band eviction requests.  It is separate from `event_tx` so
    /// that eviction requests from [`Self::mark_sender_unhealthy`] are never
    /// dropped when the event channel is full.
    indexers: DashMap<
        (String, String),
        (Arc<Indexer>, u32, EventSender, ControlSender, Arc<DashMap<u64, u64>>),
    >,
    num_threads: usize,
}

impl RuntimeRegistry {
    /// Create a new registry.  `num_threads` is the default thread-pool size
    /// used when creating new indexer shards; 0 produces a single-threaded
    /// shard.
    pub fn new(num_threads: usize) -> Self {
        Self {
            indexers: DashMap::new(),
            num_threads,
        }
    }

    /// Return the [`Indexer`] and block size for a shard, or `None` if no
    /// shard has been created for `(model_name, tenant_id)` yet.
    pub fn get(&self, model_name: &str, tenant_id: &str) -> Option<(Arc<Indexer>, u32)> {
        self.indexers
            .get(&(model_name.to_owned(), tenant_id.to_owned()))
            .map(|e| {
                let (indexer, block_size, _, _, _) = e.value().clone();
                (indexer, block_size)
            })
    }

    /// Return the [`EventSender`] for `(model_name, tenant_id)`, or `None` if
    /// the shard has not been created yet.
    ///
    /// Used by the unregister handler (legacy path without block_size) to
    /// route lifecycle messages through the same ordered channel as event batches.
    pub fn get_sender(&self, model_name: &str, tenant_id: &str) -> Option<EventSender> {
        self.indexers
            .get(&(model_name.to_owned(), tenant_id.to_owned()))
            .map(|e| e.value().2.clone())
    }

    /// Mark `sender_id` as permanently unhealthy (`watermark = u64::MAX`) and
    /// request proactive block eviction via the dedicated control channel.
    ///
    /// `overflow_seq` is the `seq` of the triggering batch.  When a finite
    /// tombstone `N` already exists for this sender and `overflow_seq <= N`,
    /// the overflow would have been discarded at the tombstone gate anyway —
    /// the tombstone is left intact rather than clobbered with `u64::MAX`.
    ///
    /// The eviction task is spawned only on the first transition to `u64::MAX`
    /// (deduplication).  Only a [`ShardMessage::Unregister`] can clear the
    /// `u64::MAX` mark.  If the shard does not exist yet this is a no-op.
    ///
    /// Recovery: send [`IndexerUnregisterRequest`] (sets a fresh-epoch tombstone)
    /// or use a new `sender_id` (no watermark entry, accepted immediately).
    pub fn mark_sender_unhealthy(
        &self,
        model_name: &str,
        tenant_id: &str,
        sender_id: u64,
        overflow_seq: u64,
    ) {
        if sender_id == 0 {
            return; // Legacy senders without ordering; nothing to mark.
        }
        if let Some(entry) = self.indexers.get(&(model_name.to_owned(), tenant_id.to_owned())) {
            let (_, _, _, ctrl_tx, unhealthy) = entry.value();

            // If a post-Unregister tombstone (finite watermark N) already covers
            // this overflow — i.e. overflow_seq <= N — the consumer would have
            // discarded the batch at the tombstone gate anyway.  Do not escalate
            // to u64::MAX: that would clobber the tombstone and prevent the
            // fresh-epoch path (seq > N) from accepting replacement batches.
            if let Some(wm_ref) = unhealthy.get(&sender_id) {
                let wm = *wm_ref;
                drop(wm_ref);
                if wm != u64::MAX && overflow_seq <= wm {
                    return;
                }
            }

            // DashMap::insert returns Some(prev) when the key already existed.
            // Skip spawning a new eviction task when the sender is already
            // marked MAX — prevents duplicate eviction tasks from rapid-fire
            // overflow events flooding the control channel.
            let prev = unhealthy.insert(sender_id, u64::MAX);
            let newly_unhealthy = prev.map_or(true, |v| v != u64::MAX);
            if newly_unhealthy {
                let ctrl_tx = ctrl_tx.clone();
                // spawn() + send().await delivers the eviction request even
                // when the control channel is momentarily full.  The channel
                // is small (CONTROL_CHANNEL_CAPACITY) and receives at most
                // one task per sender per unhealthy epoch, so in practice
                // send().await returns immediately.
                tokio::spawn(async move {
                    if ctrl_tx.send(sender_id).await.is_err() {
                        tracing::warn!(
                            sender_id,
                            "Control channel closed before eviction delivery (shard exited)"
                        );
                    }
                });
            }
        }
    }

    /// Return the ordered [`EventSender`] for `(model_name, tenant_id)`,
    /// creating a new shard and its background consumer task if needed.
    ///
    /// The AM event handler must use this to enqueue batches rather than
    /// calling [`Indexer`] methods directly, so that ordering is preserved
    /// across concurrent handler invocations.
    ///
    /// Returns an error if a shard already exists with a different block size.
    pub fn get_or_create_sender(
        &self,
        model_name: &str,
        tenant_id: &str,
        block_size: u32,
        threads: usize,
    ) -> Result<EventSender> {
        let (_, _, tx, _, _) =
            self.get_or_create_inner(model_name, tenant_id, block_size, threads)?;
        Ok(tx)
    }

    /// Return the [`Indexer`] for `(model_name, tenant_id)`, creating a new
    /// shard if one does not exist yet.
    ///
    /// Prefer [`get_or_create_sender`][Self::get_or_create_sender] on the
    /// write path; use this method for shard pre-creation (registration
    /// handler) and read-path access (query engine).
    ///
    /// `threads` overrides the registry default when creating a new shard;
    /// pass `0` to use the registry's `num_threads`.
    ///
    /// Returns an error if a shard already exists with a different block size.
    pub fn get_or_create(
        &self,
        model_name: &str,
        tenant_id: &str,
        block_size: u32,
        threads: usize,
    ) -> Result<Arc<Indexer>> {
        let (indexer, _, _, _, _) =
            self.get_or_create_inner(model_name, tenant_id, block_size, threads)?;
        Ok(indexer)
    }

    fn get_or_create_inner(
        &self,
        model_name: &str,
        tenant_id: &str,
        block_size: u32,
        threads: usize,
    ) -> Result<(Arc<Indexer>, u32, EventSender, ControlSender, Arc<DashMap<u64, u64>>)> {
        let key = (model_name.to_owned(), tenant_id.to_owned());
        let threads = if threads == 0 { self.num_threads } else { threads };

        let mn = model_name.to_owned();
        let ti = tenant_id.to_owned();

        let entry = self.indexers.entry(key).or_insert_with(|| {
            let indexer = Arc::new(create_indexer(block_size, threads));
            let (tx, mut rx) = mpsc::channel::<ShardMessage>(SHARD_CHANNEL_CAPACITY);
            // Dedicated control channel: carries sender_ids to evict.
            // Separate from the event channel so eviction is never lost when
            // the event channel is full (precisely when eviction is most needed).
            let (ctrl_tx, mut ctrl_rx) = mpsc::channel::<u64>(CONTROL_CHANNEL_CAPACITY);
            // Watermark map: sender_id → u64::MAX (permanently unhealthy) or N
            // (post-Unregister tombstone).  Written by AM handler and consumer.
            let unhealthy_senders: Arc<DashMap<u64, u64>> = Arc::new(DashMap::new());

            // Single background consumer; tokio::select! (biased) multiplexes:
            //  1. ctrl_rx — proactive eviction requests (highest priority)
            //  2. gap_sweep — periodic sweep for stalled gaps
            //  3. rx — sequenced event and lifecycle messages
            let indexer_task = indexer.clone();
            let mn_task = mn.clone();
            let ti_task = ti.clone();
            let unhealthy_ext = unhealthy_senders.clone();
            tokio::spawn(async move {
                let mut sender_states: HashMap<u64, SenderState> = HashMap::new();
                let mut gap_sweep = tokio::time::interval(GAP_SWEEP_PERIOD);
                gap_sweep.tick().await; // discard the immediate first tick

                loop {
                    tokio::select! {
                        biased; // drain control and check sweep before events

                        // ── Control: proactive eviction ───────────────────
                        Some(evict_id) = ctrl_rx.recv() => {
                            if let Some(state) = sender_states.remove(&evict_id) {
                                if !state.worker_ids.is_empty() {
                                    tracing::info!(
                                        sender_id = evict_id,
                                        count = state.worker_ids.len(),
                                        model_name = %mn_task,
                                        tenant_id  = %ti_task,
                                        "Proactively evicting indexed blocks for unhealthy sender",
                                    );
                                    for (wid, dp) in state.worker_ids {
                                        indexer_task.remove_worker_dp_rank(wid, dp).await;
                                    }
                                }
                            }
                        }

                        // ── Sweep: detect stalled gaps ────────────────────
                        _ = gap_sweep.tick() => {
                            // Collect sender_ids whose gap has timed out.
                            let stalled: Vec<u64> = sender_states
                                .iter()
                                .filter_map(|(&id, state)| {
                                    let timed_out = state
                                        .gap_started
                                        .map_or(false, |t| t.elapsed() >= GAP_STALL_TIMEOUT);
                                    if timed_out { Some(id) } else { None }
                                })
                                .collect();

                            for sender_id in stalled {
                                if let Some(mut state) = sender_states.remove(&sender_id) {
                                    tracing::error!(
                                        sender_id,
                                        buffered      = state.pending.len(),
                                        model_name    = %mn_task,
                                        tenant_id     = %ti_task,
                                        "Gap sweep: stalled gap timed out. Evicting sender's \
                                         indexed blocks and marking unhealthy — worker must \
                                         send IndexerUnregisterRequest or use a fresh \
                                         sender_id to recover."
                                    );
                                    for (wid, dp) in std::mem::take(&mut state.worker_ids) {
                                        indexer_task.remove_worker_dp_rank(wid, dp).await;
                                    }
                                    unhealthy_ext.insert(sender_id, u64::MAX);
                                }
                            }
                        }

                        // ── Events: sequenced event / lifecycle messages ───
                        msg = rx.recv() => {
                            let Some(msg) = msg else { break; };

                            // Decode message into a (sender_id, seq, action) triple.
                            let (sender_id, seq, action) = match msg {
                                ShardMessage::Events { sender_id, seq, events } => {
                                    (sender_id, seq, SequencedAction::Events(events))
                                }
                                ShardMessage::Unregister { sender_id, seq, worker_id, dp_rank } => {
                                    (
                                        sender_id,
                                        seq,
                                        SequencedAction::Unregister { worker_id, dp_rank },
                                    )
                                }
                            };

                            if sender_id == 0 {
                                // No ordering requested; apply immediately.
                                apply_action(
                                    sender_id,
                                    seq,
                                    action,
                                    &indexer_task,
                                    &mut sender_states,
                                    &unhealthy_ext,
                                    &mn_task,
                                    &ti_task,
                                )
                                .await;
                                continue;
                            }

                            // Watermark gate: u64::MAX = permanently unhealthy (discard
                            // events, pass Unregister for cleanup); N = post-Unregister
                            // tombstone (discard seq <= N, fresh epoch if seq > N).
                            if let Some(wm_ref) = unhealthy_ext.get(&sender_id) {
                                let watermark = *wm_ref;
                                drop(wm_ref); // release DashMap shard lock

                                if matches!(action, SequencedAction::Unregister { .. }) {
                                    // Apply cleanup directly — the sender is in an
                                    // undefined state; apply_action sets a tombstone.
                                    apply_action(
                                        sender_id,
                                        seq,
                                        action,
                                        &indexer_task,
                                        &mut sender_states,
                                        &unhealthy_ext,
                                        &mn_task,
                                        &ti_task,
                                    )
                                    .await;
                                    continue;
                                }

                                if seq <= watermark {
                                    // Lazy eviction on first discardable event (catches
                                    // channel-overflow case where eager eviction could not run).
                                    if let Some(state) = sender_states.remove(&sender_id) {
                                        if !state.worker_ids.is_empty() {
                                            tracing::info!(
                                                sender_id,
                                                count = state.worker_ids.len(),
                                                model_name = %mn_task,
                                                tenant_id = %ti_task,
                                                "Evicting stale indexed blocks \
                                                 for unhealthy sender",
                                            );
                                            for (wid, dp) in state.worker_ids {
                                                indexer_task
                                                    .remove_worker_dp_rank(wid, dp)
                                                    .await;
                                            }
                                        }
                                    }
                                    tracing::debug!(
                                        sender_id,
                                        seq,
                                        watermark,
                                        "Discarding event batch from unhealthy/tombstoned sender"
                                    );
                                    continue;
                                }

                                // seq > watermark: fresh epoch — same sender_id reused
                                // after its Unregister.  Seed next_expected at
                                // watermark + 1, not the arriving seq, so gap detection
                                // works correctly within the new epoch.
                                unhealthy_ext.remove(&sender_id);
                                sender_states.remove(&sender_id);
                                sender_states.insert(
                                    sender_id,
                                    SenderState {
                                        next_expected: watermark + 1,
                                        ..Default::default()
                                    },
                                );
                            }

                            let state = sender_states.entry(sender_id).or_default();

                            if seq < state.next_expected {
                                // Stale duplicate (e.g. retry after gap recovery) — discard.
                                tracing::debug!(
                                    sender_id,
                                    seq,
                                    next_expected = state.next_expected,
                                    "Dropping stale duplicate action"
                                );
                                continue;
                            }

                            if seq > state.next_expected {
                                // Out-of-order arrival.

                                // Record when this gap opened (first out-of-order action).
                                if state.pending.is_empty() {
                                    state.gap_started = Some(std::time::Instant::now());
                                }

                                // Gap recovery: buffer full OR stall timeout (handles
                                // small gaps that would never fill the buffer).
                                let gap_stalled = state
                                    .gap_started
                                    .map_or(false, |t| t.elapsed() > GAP_STALL_TIMEOUT);

                                if state.pending.len() >= MAX_PENDING_PER_SENDER || gap_stalled {
                                    let reason = if gap_stalled {
                                        "stalled gap (predecessor permanently lost)"
                                    } else {
                                        "reorder buffer full"
                                    };
                                    tracing::error!(
                                        sender_id,
                                        seq,
                                        next_expected = state.next_expected,
                                        buffered = state.pending.len(),
                                        gap_stalled,
                                        model_name = %mn_task,
                                        tenant_id = %ti_task,
                                        "Gap recovery triggered ({reason}). Eagerly evicting \
                                         sender's indexed blocks and marking unhealthy — \
                                         worker must send IndexerUnregisterRequest to clear \
                                         the unhealthy mark before events are accepted again."
                                    );
                                    // Eagerly evict all blocks indexed by this sender so
                                    // they are not returned by queries while the sender
                                    // is in an undefined state.
                                    let worker_ids = std::mem::take(&mut state.worker_ids);
                                    state.pending.clear();
                                    drop(state); // release borrow before mutation
                                    sender_states.remove(&sender_id);
                                    for (wid, dp) in worker_ids {
                                        indexer_task.remove_worker_dp_rank(wid, dp).await;
                                    }
                                    unhealthy_ext.insert(sender_id, u64::MAX);
                                } else {
                                    tracing::debug!(
                                        sender_id,
                                        seq,
                                        next_expected = state.next_expected,
                                        buffered = state.pending.len(),
                                        "Buffering out-of-order action"
                                    );
                                    state.pending.insert(seq, action);
                                }
                                continue;
                            }

                            // seq == next_expected: collect contiguous actions into a
                            // Vec before calling apply_action (releases borrow first).
                            let mut to_apply: Vec<(u64, SequencedAction)> = vec![(seq, action)];
                            let is_terminal =
                                matches!(&to_apply[0].1, SequencedAction::Unregister { .. });
                            if !is_terminal {
                                state.next_expected += 1;
                                loop {
                                    match state.pending.remove(&state.next_expected) {
                                        None => break,
                                        Some(next_action) => {
                                            let terminal = matches!(
                                                next_action,
                                                SequencedAction::Unregister { .. }
                                            );
                                            to_apply.push((state.next_expected, next_action));
                                            state.next_expected += 1;
                                            if terminal {
                                                break;
                                            }
                                        }
                                    }
                                }
                                // Gap closed when pending drains to empty.
                                if state.pending.is_empty() {
                                    state.gap_started = None;
                                }
                            }
                            drop(state); // release borrow before apply_action

                            for (act_seq, act) in to_apply {
                                let terminal = matches!(act, SequencedAction::Unregister { .. });
                                apply_action(
                                    sender_id,
                                    act_seq,
                                    act,
                                    &indexer_task,
                                    &mut sender_states,
                                    &unhealthy_ext,
                                    &mn_task,
                                    &ti_task,
                                )
                                .await;
                                if terminal {
                                    break;
                                }
                            }
                        } // close msg = rx.recv() arm
                    } // close tokio::select!
                } // close loop
                tracing::debug!(
                    model_name = %mn_task,
                    tenant_id  = %ti_task,
                    "Velo-runtime shard consumer exiting"
                );
            });

            tracing::info!(
                model_name = %mn,
                tenant_id  = %ti,
                block_size,
                threads,
                "Creating velo-runtime indexer shard"
            );
            (indexer, block_size, tx, ctrl_tx, unhealthy_senders)
        });

        let (indexer, existing_block_size, tx, ctrl_tx, unhealthy) = entry.value().clone();
        if existing_block_size != block_size {
            anyhow::bail!(
                "block-size mismatch for shard ({mn}, {ti}): \
                 existing={existing_block_size} requested={block_size}"
            );
        }
        Ok((indexer, existing_block_size, tx, ctrl_tx, unhealthy))
    }
}

// ── Consumer helper ───────────────────────────────────────────────────────

/// Apply one [`SequencedAction`] to the indexer.
///
/// `Events`: records each `(worker_id, dp_rank)` in `sender_states` for
/// later eager eviction if the sender is marked unhealthy.
///
/// `Unregister`: removes `sender_states[sender_id]` and sets a post-Unregister
/// tombstone (`sender_id → seq`) so delayed in-flight batches are discarded.
/// `sender_id == 0` skips watermark tracking in both branches.
async fn apply_action(
    sender_id: u64,
    seq: u64,
    action: SequencedAction,
    indexer: &Indexer,
    sender_states: &mut HashMap<u64, SenderState>,
    unhealthy_senders: &DashMap<u64, u64>,
    model_name: &str,
    tenant_id: &str,
) {
    match action {
        SequencedAction::Events(events) => {
            // Track (worker_id, dp_rank) for later eviction if unhealthy.
            if sender_id != 0 {
                if let Some(state) = sender_states.get_mut(&sender_id) {
                    for event in &events {
                        state.worker_ids.insert((event.worker_id, event.event.dp_rank));
                    }
                }
            }
            for event in events {
                indexer.apply_event_routed(event).await;
            }
        }
        SequencedAction::Unregister { worker_id, dp_rank } => {
            sender_states.remove(&sender_id);
            if sender_id != 0 {
                // Post-Unregister tombstone: discard delayed pre-Unregister
                // batches (seq <= N) without reinstating stale blocks.
                unhealthy_senders.insert(sender_id, seq);
            }
            indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
            tracing::info!(
                worker_id, dp_rank, seq, %model_name, %tenant_id,
                "Worker removed from velo-runtime shard; post-Unregister watermark set"
            );
        }
    }
}
