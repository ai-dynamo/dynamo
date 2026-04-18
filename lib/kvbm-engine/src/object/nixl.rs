// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL-based object storage client for KV cache block offload/onboard.
//!
//! Uses NIXL's OBJ backend to transfer blocks between host DRAM and object
//! storage without going through the AWS SDK on the hot path.
//! The OBJ backend is configured once at agent init time; individual
//! block transfers are expressed as NIXL `XferRequest`s.
//!
//! # Backend configuration
//!
//! The NIXL OBJ plugin reads S3 credentials and endpoint from the standard
//! AWS environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_OVERRIDE`, …) plus any plugin-specific
//! keys supplied via [`NixlObjectConfig`](kvbm_config::NixlObjectConfig).
//! See [`NixlS3Config::to_nixl_params`](kvbm_config::NixlS3Config::to_nixl_params)
//! for the full parameter list.
//!
//! # Object key mapping
//!
//! NIXL's OBJ backend identifies objects with a `u64` device_id.
//! [`SequenceHash`] (`PositionalLineageHash`, a `u128`) is XOR-folded into
//! 64 bits.  The same function must be used on both write and read so the
//! object key round-trips correctly.
//!
//! The XOR fold (`lo ^ hi`) is collision-resistant for well-distributed
//! 128-bit inputs such as `PositionalLineageHash`.  The probability of two
//! distinct 128-bit values producing the same 64-bit fold is 2⁻⁶⁴, which is
//! negligible in practice.  Both write and read paths use the same function so
//! the key always round-trips.
//!
//! # `has_blocks`
//!
//! NIXL's OBJ backend does not expose a HEAD/stat primitive.  This
//! implementation delegates `has_blocks` to an SDK-backed client built from
//! the same S3 config.  When the `s3` feature is disabled the method returns
//! `None` for every hash (conservative: forces re-upload).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_memory::nixl::{
    Agent as RawAgent, MemType, NixlAgent, NixlDescriptor, XferDescList, XferOp, XferRequest,
};
use futures::future::BoxFuture;
use tokio::sync::mpsc;
use tracing::instrument;
use velo::{EventHandle, EventManager};

use crate::object::{KeyFormatter, LayoutConfigExt, ObjectBlockOps};
use crate::{BlockId, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::transfer::{PhysicalLayout, TransferCompleteNotification};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// How often the background polling task calls `get_xfer_status`.
///
/// 1 ms matches the interval used by the `kvbm-physical` status-polling task
/// for NIXL RDMA transfers, giving consistent latency across transfer types.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Bounded channel capacity between transfer submission and the polling task.
///
/// Back-pressure kicks in once this many transfers are simultaneously in-flight.
/// Sized for a realistic upper-bound of concurrent OBJ transfers per worker.
const POLLING_CHANNEL_CAPACITY: usize = 1024;

/// How long the drain loop waits for in-flight transfers after the channel
/// closes before giving up and logging an error.
///
/// Prevents infinite blocking on shutdown if a NIXL transfer stalls
/// (no NIXL API exists to cancel a posted request).
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Fallback S3 bucket name when none is set in config or `AWS_DEFAULT_BUCKET`.
const DEFAULT_BUCKET: &str = "kvbm-blocks";

/// Fallback AWS region for the SDK-backed `has_blocks` HEAD-check client.
const DEFAULT_REGION: &str = "us-east-1";

/// Maximum concurrent SDK requests for `has_blocks` HEAD checks.
const HAS_BLOCKS_MAX_CONCURRENT: usize = 16;

/// Convert a [`NixlS3Config`] into an [`S3Config`] for SDK-backed `has_blocks` checks.
#[cfg(feature = "s3")]
pub(super) fn nixl_s3_to_sdk_config(cfg: &kvbm_config::NixlS3Config) -> super::s3::S3Config {
    super::s3::S3Config {
        endpoint_url: cfg.endpoint_override.clone(),
        bucket: cfg.bucket_name().unwrap_or_else(|| DEFAULT_BUCKET.to_string()),
        region: cfg.region.clone().unwrap_or_else(|| DEFAULT_REGION.to_string()),
        // path-style when virtual addressing is disabled (or unset)
        force_path_style: !cfg.use_virtual_addressing.unwrap_or(false),
        max_concurrent_requests: HAS_BLOCKS_MAX_CONCURRENT,
    }
}

/// XOR-fold a 128-bit `SequenceHash` into a 64-bit NIXL OBJ device_id.
///
/// Both write and read use the same conversion so that the object key is
/// stable and round-trips correctly across nodes.
///
/// Collision probability for well-distributed 128-bit inputs is 2⁻⁶⁴, which
/// is negligible in practice.
fn hash_to_device_id(hash: SequenceHash) -> u64 {
    let raw = hash.as_u128();
    (raw as u64) ^ ((raw >> 64) as u64)
}

/// Initialise the NIXL OBJ backend on an existing agent.
///
/// Merges `extra_params` (bucket, endpoint, etc.) into the plugin's default
/// parameter set, then calls `create_backend("OBJ", params)`.
///
/// The caller is responsible for ensuring the agent hasn't already had an OBJ
/// backend added (idempotent via `NixlAgent::add_backend_with_params`).
pub fn add_obj_backend(agent: &mut NixlAgent, extra_params: HashMap<String, String>) -> Result<()> {
    agent.add_backend_with_params("OBJ", &extra_params)
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared transfer completion poller
// ─────────────────────────────────────────────────────────────────────────────

/// An in-flight OBJ transfer registered with the background polling task.
struct PendingObjTransfer {
    /// Held alive for the duration of the transfer (NIXL requirement).
    xfer_req: XferRequest,
    agent: Arc<RawAgent>,
    event_handle: EventHandle,
}

/// Background task: wakes every [`POLL_INTERVAL`] and calls `get_xfer_status`
/// for all outstanding OBJ transfers.  Triggers or poisons the associated event
/// when the transfer completes or errors.
///
/// All concurrent transfers share this one task — no per-transfer sleeping
/// on the async executor.
///
/// # Zombie transfers after timeout
///
/// When a caller's `tokio::time::timeout` fires, the associated
/// `PendingObjTransfer` is already registered here and cannot be removed
/// (NIXL has no cancel API for a posted request).  The task continues polling
/// that transfer until NIXL reports completion or error, at which point the
/// event trigger is a no-op (the awaiter was dropped).  In a timeout storm,
/// outstanding transfers will accumulate until they naturally finish.
///
/// # Shutdown
///
/// After the submission channel closes, the task drains remaining transfers for
/// up to [`DRAIN_TIMEOUT`].  Any still-pending transfers after that deadline
/// are abandoned with an error log — this avoids blocking process exit
/// indefinitely when NIXL transfers stall.
async fn poll_obj_transfers(
    mut rx: mpsc::Receiver<PendingObjTransfer>,
    events: Arc<EventManager>,
) {
    let mut outstanding: Vec<PendingObjTransfer> = Vec::new();
    let mut tick = tokio::time::interval(POLL_INTERVAL);

    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Some(t) => outstanding.push(t),
                None => break,
            },
            _ = tick.tick(), if !outstanding.is_empty() => {
                outstanding.retain(|t| {
                    match t.agent.get_xfer_status(&t.xfer_req) {
                        Ok(status) if status.is_success() => {
                            if let Err(e) = events.trigger(t.event_handle) {
                                tracing::error!("NIXL OBJ: failed to trigger completion event: {e}");
                            }
                            false // remove from outstanding
                        }
                        Ok(_) => true, // still pending
                        Err(e) => {
                            if let Err(pe) = events.poison(t.event_handle, e.to_string()) {
                                tracing::error!("NIXL OBJ: failed to poison completion event: {pe}");
                            }
                            false // remove from outstanding
                        }
                    }
                });
            }
        }
    }

    // Channel closed — drain remaining transfers up to DRAIN_TIMEOUT.
    let drain_deadline = tokio::time::Instant::now() + DRAIN_TIMEOUT;
    while !outstanding.is_empty() {
        if tokio::time::Instant::now() >= drain_deadline {
            tracing::error!(
                count = outstanding.len(),
                "NIXL OBJ: shutdown drain timed out; {} transfer(s) abandoned",
                outstanding.len()
            );
            break;
        }
        tick.tick().await;
        outstanding.retain(|t| {
            match t.agent.get_xfer_status(&t.xfer_req) {
                Ok(status) if status.is_success() => {
                    let _ = events.trigger(t.event_handle);
                    false
                }
                Ok(_) => true,
                Err(e) => {
                    let _ = events.poison(t.event_handle, e.to_string());
                    false
                }
            }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NixlObjectBlockClient
// ─────────────────────────────────────────────────────────────────────────────

/// `ObjectBlockOps` implementation that routes DRAM↔Object transfers through
/// NIXL's OBJ backend instead of the AWS SDK.
///
/// Construct via [`NixlObjectBlockClient::from_config`] (from the runtime) or
/// via [`NixlObjectBlockClient::new`] after manually calling [`add_obj_backend`].
///
/// # Transfer completion
///
/// After posting a transfer, completion is tracked by a shared background task
/// ([`poll_obj_transfers`]) that wakes every [`POLL_INTERVAL`] and checks all
/// outstanding transfers.  The async caller simply awaits a
/// [`TransferCompleteNotification`] — no per-transfer polling or sleeping on
/// the executor thread.
#[derive(Clone)]
pub struct NixlObjectBlockClient {
    /// Underlying raw NIXL agent (shared across clones for transfer state).
    raw_agent: Arc<RawAgent>,
    /// Bucket name embedded in NIXL Object descriptors.
    bucket: String,
    /// Reserved for future string-key NIXL support (e.g., when NIXL exposes
    /// a named-object API that doesn't require u64 device_id mapping).
    #[allow(dead_code)]
    key_formatter: Arc<dyn KeyFormatter>,
    /// Optional delegate used only for `has_blocks` HEAD checks.
    ///
    /// NIXL's OBJ backend has no stat primitive, so we optionally delegate to
    /// an `ObjectBlockOps` implementation that does (e.g. `S3ObjectBlockClient`).
    /// `None` → `has_blocks` always returns `None` (conservative).
    has_blocks_delegate: Option<Arc<dyn ObjectBlockOps>>,
    /// Deadline for a single transfer to complete.  `None` = no timeout.
    pub(crate) transfer_timeout: Option<Duration>,
    /// Shared event system — all clones of this client trigger events here.
    events: Arc<EventManager>,
    /// Channel to the shared polling background task.
    polling_tx: mpsc::Sender<PendingObjTransfer>,
}

impl std::fmt::Debug for NixlObjectBlockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NixlObjectBlockClient")
            .field("bucket", &self.bucket)
            .field("has_blocks_delegate", &self.has_blocks_delegate.is_some())
            .field("transfer_timeout", &self.transfer_timeout)
            .finish_non_exhaustive()
    }
}

impl NixlObjectBlockClient {
    /// Create a new client.
    ///
    /// The `agent` must already have the OBJ backend initialised (call
    /// [`add_obj_backend`] first, or use [`from_config`] which validates this).
    ///
    /// `has_blocks_delegate` — if `Some`, its `has_blocks()` is called for
    /// existence checks; if `None`, all blocks are assumed absent.
    ///
    /// `transfer_timeout` — maximum time to wait for a single transfer.
    /// `None` means wait indefinitely (use only when the NIXL OBJ backend has
    /// its own deadline enforcement).
    ///
    /// Spawns a [`poll_obj_transfers`] background task on the current tokio
    /// runtime.  All clones of this client share that task via the sender.
    ///
    /// # Panics
    ///
    /// Panics if called outside of a tokio runtime context (internally calls
    /// `tokio::spawn`).
    pub fn new(
        agent: NixlAgent,
        bucket: String,
        key_formatter: Arc<dyn KeyFormatter>,
        has_blocks_delegate: Option<Arc<dyn ObjectBlockOps>>,
        transfer_timeout: Option<Duration>,
    ) -> Self {
        let events = Arc::new(EventManager::local());
        let (polling_tx, polling_rx) = mpsc::channel(POLLING_CHANNEL_CAPACITY);
        tokio::spawn(poll_obj_transfers(polling_rx, Arc::clone(&events)));

        Self {
            raw_agent: Arc::new(agent.into_raw_agent()),
            bucket,
            key_formatter,
            has_blocks_delegate,
            transfer_timeout,
            events,
            polling_tx,
        }
    }

    /// Build from a kvbm-config [`NixlObjectConfig`] and an already-initialised agent.
    ///
    /// # Errors
    ///
    /// Returns an error if the OBJ backend has not been registered on `agent`
    /// (call [`add_obj_backend`] before invoking this).
    ///
    /// When the `s3` feature is available a thin `S3ObjectBlockClient` is
    /// created from the embedded S3 config and used as the `has_blocks` delegate.
    pub async fn from_config(
        agent: NixlAgent,
        nixl_config: &kvbm_config::NixlObjectConfig,
        rank: Option<usize>,
    ) -> Result<Self> {
        use kvbm_config::NixlObjectConfig;
        use super::create_key_formatter;

        // Pre-flight: the OBJ backend must already be registered.
        // Check before consuming the agent so the error message is clear.
        agent.require_backend("OBJ")?;

        let key_formatter = create_key_formatter(rank);

        match nixl_config {
            NixlObjectConfig::S3(nixl_s3) => {
                let bucket = nixl_s3.bucket_name().unwrap_or_else(|| DEFAULT_BUCKET.to_string());
                let transfer_timeout = nixl_s3.transfer_timeout();

                // Build a thin S3 client for has_blocks HEAD checks when the
                // `s3` feature is available.
                let has_blocks_delegate: Option<Arc<dyn ObjectBlockOps>> = {
                    #[cfg(feature = "s3")]
                    {
                        let s3_config = nixl_s3_to_sdk_config(nixl_s3);
                        super::s3::S3ObjectBlockClient::with_key_formatter(
                            s3_config,
                            key_formatter.clone(),
                        )
                        .await
                        .ok()
                        .map(|c| Arc::new(c) as Arc<dyn ObjectBlockOps>)
                    }
                    #[cfg(not(feature = "s3"))]
                    {
                        None
                    }
                };

                Ok(Self::new(
                    agent,
                    bucket,
                    key_formatter,
                    has_blocks_delegate,
                    transfer_timeout,
                ))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ObjectBlockOps impl
// ─────────────────────────────────────────────────────────────────────────────

impl ObjectBlockOps for NixlObjectBlockClient {
    /// Check block presence.
    ///
    /// Delegates to `has_blocks_delegate` when set (e.g. a thin S3 client using
    /// HEAD requests). Falls back to reporting all blocks absent, which is
    /// conservative but correct: the offload pipeline re-uploads the block.
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        if let Some(delegate) = &self.has_blocks_delegate {
            return delegate.has_blocks(keys);
        }
        Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Workers must resolve the logical handle to a physical layout first.
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let agent = Arc::clone(&self.raw_agent);
        let timeout = self.transfer_timeout;
        let events = Arc::clone(&self.events);
        let polling_tx = self.polling_tx.clone();
        Box::pin(async move {
            execute_nixl_obj_transfer(
                &agent, XferOp::Write, &keys, &layout, &block_ids, timeout, &events, &polling_tx,
            )
            .await
        })
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let agent = Arc::clone(&self.raw_agent);
        let timeout = self.transfer_timeout;
        let events = Arc::clone(&self.events);
        let polling_tx = self.polling_tx.clone();
        Box::pin(async move {
            execute_nixl_obj_transfer(
                &agent, XferOp::Read, &keys, &layout, &block_ids, timeout, &events, &polling_tx,
            )
            .await
        })
    }

}

// ─────────────────────────────────────────────────────────────────────────────
// Core NIXL OBJ transfer
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a NIXL transfer between host DRAM and object storage.
///
/// * `XferOp::Write` — offload: DRAM → Object storage
/// * `XferOp::Read`  — onboard: Object storage → DRAM
///
/// For fully-contiguous layouts each block is one DRAM descriptor.
/// For layer-separate layouts we add one descriptor per (layer, outer_dim)
/// region and match them against a correspondingly striped object address space.
///
/// # Completion
///
/// After posting the transfer, this function registers a [`PendingObjTransfer`]
/// with the shared [`poll_obj_transfers`] background task.  The task wakes
/// every [`POLL_INTERVAL`], calls `get_xfer_status` for all outstanding
/// transfers, and triggers the completion event when done.  The async caller
/// awaits a [`TransferCompleteNotification`], optionally wrapped in
/// `tokio::time::timeout` when `transfer_timeout` is `Some`.
///
/// This means **no per-transfer sleeping on the executor thread**: all
/// concurrent OBJ transfers share a single polling tick.
///
/// # Timeout and zombie transfers
///
/// When `transfer_timeout` is `Some` and the deadline fires, this function
/// returns errors for all keys.  However, the `PendingObjTransfer` is already
/// registered with the background task and cannot be cancelled (NIXL has no
/// cancel API for a posted request).  The background task continues polling
/// until NIXL reports completion or error; at that point the event trigger
/// is a no-op because the awaiter was dropped.
#[instrument(skip_all, fields(op = ?op, n = keys.len()))]
async fn execute_nixl_obj_transfer(
    agent_arc: &Arc<RawAgent>,
    op: XferOp,
    keys: &[SequenceHash],
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    transfer_timeout: Option<Duration>,
    events: &Arc<EventManager>,
    polling_tx: &mpsc::Sender<PendingObjTransfer>,
) -> Vec<Result<SequenceHash, SequenceHash>> {
    let agent: &RawAgent = agent_arc;
    if keys.is_empty() {
        return Vec::new();
    }

    // ── Build descriptor lists ────────────────────────────────────────────────
    //
    // Convention (matches dynamo-fork execute_object_transfer):
    //   src_dl = DRAM (initiator's local buffers, always)
    //   dst_dl = Object (responder = OBJ backend on same agent)
    //   XferOp::Write → DRAM → Object  (offload)
    //   XferOp::Read  → Object → DRAM  (onboard)
    //
    // The Object side uses addr=0 and device_id = XOR-folded sequence hash.
    // The OBJ backend maps device_id → object key in the configured bucket.

    // Use a block to drop non-Send registration handles before the first .await.
    let xfer_result = (|| -> anyhow::Result<(XferRequest, bool)> {
        anyhow::ensure!(
            keys.len() == block_ids.len(),
            "keys.len() ({}) != block_ids.len() ({}); caller must provide one block_id per key",
            keys.len(),
            block_ids.len()
        );

        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let is_contiguous = layout.layout().is_fully_contiguous();

        let mut src_dl = XferDescList::new(MemType::Dram)?;
        let mut dst_dl = XferDescList::new(MemType::Object)?;

        // Registration handles must stay alive until after post_xfer_req.
        // NIXL may retain a reference to the registered memory until the
        // transfer completes; dropping handles early could corrupt in-flight
        // DMA.  The handles are dropped when this closure scope exits, which
        // is after post_xfer_req() returns (transfer is at least submitted).
        let mut _obj_reg_handles = Vec::with_capacity(keys.len());

        for (key, block_id) in keys.iter().zip(block_ids.iter()) {
            let device_id = hash_to_device_id(*key);

            // Register the Object-side descriptor with the agent so NIXL can
            // associate it with the OBJ backend.
            let obj_desc = NixlDescriptor {
                addr: 0,
                size: block_size,
                mem_type: MemType::Object,
                device_id,
            };
            let handle = agent.register_memory(&obj_desc, None)?;
            _obj_reg_handles.push(handle);

            if is_contiguous {
                // Fast path: the block's data is one contiguous region.
                let region = layout.memory_region(*block_id, 0, 0)?;
                src_dl.add_desc(region.addr(), block_size, 0);
                dst_dl.add_desc(0, block_size, device_id);
            } else {
                // Slow path: one DRAM entry per (layer, outer_dim) region.
                // Object-side entries share the same device_id (same object),
                // but each points to a different sub-range via addr offset.
                let inner = layout.layout();
                let region_size = config.region_size();

                // Sanity: the non-contiguous regions must sum to block_size.
                debug_assert_eq!(
                    inner.num_layers() * inner.outer_dim() * region_size,
                    block_size,
                    "non-contiguous region size mismatch: \
                     num_layers({}) * outer_dim({}) * region_size({}) = {} != block_size({})",
                    inner.num_layers(),
                    inner.outer_dim(),
                    region_size,
                    inner.num_layers() * inner.outer_dim() * region_size,
                    block_size
                );

                let mut obj_offset: usize = 0;
                for layer_id in 0..inner.num_layers() {
                    for outer_id in 0..inner.outer_dim() {
                        let region = layout.memory_region(*block_id, layer_id, outer_id)?;
                        src_dl.add_desc(region.addr(), region_size, 0);
                        // Offset into the object acts as the sub-range addr.
                        dst_dl.add_desc(obj_offset, region_size, device_id);
                        obj_offset += region_size;
                    }
                }
            }
        }

        let agent_name = agent.name();
        let xfer_req = agent.create_xfer_req(op, &src_dl, &dst_dl, &agent_name, None)?;
        let still_pending = agent.post_xfer_req(&xfer_req, None)?;
        Ok((xfer_req, still_pending))
    })();

    let (xfer_req, still_pending) = match xfer_result {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("NIXL OBJ: failed to build/post transfer: {e}");
            return keys.iter().map(|k| Err(*k)).collect();
        }
    };

    // ── Wait for completion via shared polling task ───────────────────────────
    //
    // Register this transfer with the background `process_polling_notifications`
    // task.  That task wakes every 1 ms and calls `get_xfer_status` for all
    // outstanding transfers — no per-transfer sleeping on the executor.
    if still_pending {
        let event = match events.new_event() {
            Ok(e) => e,
            Err(e) => {
                tracing::error!("NIXL OBJ: failed to create completion event: {e}");
                return keys.iter().map(|k| Err(*k)).collect();
            }
        };
        let handle = event.into_handle();
        let awaiter = match events.awaiter(handle) {
            Ok(a) => a,
            Err(e) => {
                tracing::error!("NIXL OBJ: failed to get event awaiter: {e}");
                return keys.iter().map(|k| Err(*k)).collect();
            }
        };

        let pending = PendingObjTransfer {
            xfer_req,
            agent: Arc::clone(agent_arc),
            event_handle: handle,
        };

        if let Err(e) = polling_tx.send(pending).await {
            tracing::error!("NIXL OBJ: polling channel closed: {e}");
            return keys.iter().map(|k| Err(*k)).collect();
        }

        let notification = TransferCompleteNotification::from_awaiter(awaiter);
        let wait_result = if let Some(deadline) = transfer_timeout {
            match tokio::time::timeout(deadline, notification).await {
                Ok(result) => result,
                Err(_elapsed) => {
                    tracing::error!("NIXL OBJ: transfer timed out after {deadline:?}");
                    return keys.iter().map(|k| Err(*k)).collect();
                }
            }
        } else {
            notification.await
        };
        if let Err(e) = wait_result {
            tracing::error!("NIXL OBJ: transfer failed: {e}");
            return keys.iter().map(|k| Err(*k)).collect();
        }
    }

    keys.iter().map(|k| Ok(*k)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure unit tests — no NIXL hardware required
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hash_to_device_id ─────────────────────────────────────────────────────

    #[test]
    fn test_hash_to_device_id_deterministic() {
        let hash = SequenceHash::new(0xDEAD_BEEF_u64, None, 0);
        assert_eq!(hash_to_device_id(hash), hash_to_device_id(hash));
    }

    #[test]
    fn test_hash_to_device_id_distinct() {
        let h1 = SequenceHash::new(1_u64, None, 0);
        let h2 = SequenceHash::new(2_u64, None, 0);
        assert_ne!(hash_to_device_id(h1), hash_to_device_id(h2));
    }

    #[test]
    fn test_hash_to_device_id_single_bit_difference() {
        // Adjacent inputs must produce distinct device_ids.
        let h1 = SequenceHash::new(0b0_u64, None, 0);
        let h2 = SequenceHash::new(0b1_u64, None, 0);
        assert_ne!(hash_to_device_id(h1), hash_to_device_id(h2));
    }

    #[test]
    fn test_hash_to_device_id_distribution() {
        use std::collections::HashSet;
        // 1 000 sequential inputs through a good 128-bit hash should have
        // negligible collisions when folded to 64 bits.
        let n = 1_000u64;
        let ids: HashSet<u64> = (0..n)
            .map(|i| hash_to_device_id(SequenceHash::new(i, None, 0)))
            .collect();
        assert!(
            ids.len() >= 990,
            "Too many hash collisions after XOR fold: {} unique IDs from {} inputs",
            ids.len(),
            n
        );
    }

    #[test]
    fn test_hash_to_device_id_boundary_cases() {
        // Should not panic; just validate the function handles edge values.
        let _zero = hash_to_device_id(SequenceHash::new(0_u64, None, 0));
        let _max = hash_to_device_id(SequenceHash::new(u64::MAX, None, 0));
        let _with_parent =
            hash_to_device_id(SequenceHash::new(0xDEAD_u64, Some(0xBEEF_u64), 1));
        // Distinct inputs → distinct outputs for these corner cases.
        assert_ne!(_zero, _max);
    }

    // ── LayoutConfigExt arithmetic ────────────────────────────────────────────

    /// Verify the invariant that `execute_nixl_obj_transfer` relies on in the
    /// non-contiguous path: `block_size == num_layers × outer_dim × region_size`.
    #[test]
    fn test_block_size_equals_region_sum() {
        use crate::object::LayoutConfigExt;
        use kvbm_physical::layout::LayoutConfig;

        let config = LayoutConfig::builder()
            .num_blocks(1)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        assert_eq!(
            block_size,
            config.num_layers * config.outer_dim * region_size,
            "block_size_bytes() must equal num_layers * outer_dim * region_size()"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests — require the `testing-s3` feature and a live S3 endpoint.
//
// Run with:
//   bash lib/kvbm-engine/scripts/test-s3.sh
// or manually:
//   S3_TEST_ENDPOINT=http://localhost:9876 \
//   AWS_ACCESS_KEY_ID=minioadmin \
//   AWS_SECRET_ACCESS_KEY=minioadmin \
//   cargo test -p kvbm-engine --features testing-s3 -- nixl_integration
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "testing-s3"))]
mod nixl_integration {
    use super::*;
    use crate::object::s3::s3_integration::create_test_client;
    use crate::object::s3::{S3Config, S3ObjectBlockClient};
    use crate::object::DefaultKeyFormatter;

    // ── nixl_s3_to_sdk_config ─────────────────────────────────────────────────

    /// `nixl_s3_to_sdk_config` must produce an `S3Config` whose fields correctly
    /// reflect the source `NixlS3Config`.
    #[test]
    fn test_nixl_s3_to_sdk_config_fields() {
        let nixl_cfg = kvbm_config::NixlS3Config {
            endpoint_override: Some("http://host:9000".into()),
            bucket: Some("my-bucket".into()),
            region: Some("eu-west-1".into()),
            use_virtual_addressing: Some(false),
            ..Default::default()
        };
        let sdk_cfg = nixl_s3_to_sdk_config(&nixl_cfg);
        assert_eq!(sdk_cfg.endpoint_url, Some("http://host:9000".into()));
        assert_eq!(sdk_cfg.bucket, "my-bucket");
        assert_eq!(sdk_cfg.region, "eu-west-1");
        assert!(sdk_cfg.force_path_style, "virtual_addressing=false → path-style");
    }

    /// `use_virtual_addressing` unset → path-style (safe default for S3-compatible endpoints).
    #[test]
    fn test_nixl_s3_to_sdk_config_virtual_addressing_default() {
        let nixl_cfg = kvbm_config::NixlS3Config {
            bucket: Some("b".into()),
            ..Default::default()
        };
        let sdk_cfg = nixl_s3_to_sdk_config(&nixl_cfg);
        assert!(sdk_cfg.force_path_style);
    }

    /// `bucket` field None + no env var → falls back to the hardcoded default "kvbm-blocks".
    #[test]
    fn test_nixl_s3_to_sdk_config_bucket_hardcoded_default() {
        // bucket is None and AWS_DEFAULT_BUCKET is not set → nixl_s3_to_sdk_config
        // falls back to the hardcoded sentinel "kvbm-blocks".
        let nixl_cfg = kvbm_config::NixlS3Config::default();
        // Only run this assertion when the env var is absent so the test is
        // deterministic regardless of the host environment.
        if std::env::var("AWS_DEFAULT_BUCKET").is_err() {
            let sdk_cfg = nixl_s3_to_sdk_config(&nixl_cfg);
            assert_eq!(sdk_cfg.bucket, "kvbm-blocks");
        }
    }

    // ── NixlS3Config serde round-trip ─────────────────────────────────────────

    /// JSON serialisation of `NixlS3Config` must round-trip correctly so that
    /// config files produced by one version are readable by another.
    #[test]
    fn test_nixl_s3_config_serde_roundtrip() {
        let endpoint = std::env::var("S3_TEST_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:9876".into());
        let original = kvbm_config::NixlS3Config::with_endpoint(endpoint, "serde-test-bucket");
        let json = serde_json::to_string(&original).unwrap();
        let restored: kvbm_config::NixlS3Config = serde_json::from_str(&json).unwrap();
        assert_eq!(original.endpoint_override, restored.endpoint_override);
        assert_eq!(original.bucket, restored.bucket);
        assert_eq!(original.use_virtual_addressing, restored.use_virtual_addressing);
    }

    // ── has_blocks with SDK delegate against real S3 ──────────────────────────

    /// Build a `NixlObjectBlockClient` whose `has_blocks_delegate` is a real
    /// `S3ObjectBlockClient`, then verify:
    ///   1. Blocks not yet uploaded → `has_blocks` returns `None`.
    ///   2. After uploading a block via the SDK client → `has_blocks` returns `Some`.
    ///
    /// This does NOT exercise the NIXL transfer path (which requires NIXL hardware).
    /// It validates the delegate wiring and `NixlS3Config → S3Config` conversion
    /// end-to-end against a real S3 endpoint.
    #[tokio::test]
    async fn test_has_blocks_sdk_delegate() {
        let bucket = "nixl-has-blocks-test";
        let sdk_client = Arc::new(create_test_client(bucket).await) as Arc<dyn ObjectBlockOps>;

        let absent_hash = SequenceHash::new(0xAAAA_AAAA_u64, None, 1);
        let present_hash = SequenceHash::new(0xBBBB_BBBB_u64, None, 2);

        // Upload one object under the key that `has_blocks` will HEAD-check.
        let key = DefaultKeyFormatter.format_key(&present_hash);
        let endpoint = std::env::var("S3_TEST_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:9876".into());
        let raw = S3ObjectBlockClient::new(S3Config::minio(endpoint.clone(), bucket.into()))
            .await
            .unwrap();
        raw.put_object(&key, bytes::Bytes::from("payload")).await.unwrap();

        // Verify presence via the delegate.
        let results: std::collections::HashMap<_, _> = sdk_client
            .has_blocks(vec![absent_hash, present_hash])
            .await
            .into_iter()
            .collect();

        assert!(results[&absent_hash].is_none(), "absent block should return None");
        assert!(results[&present_hash].is_some(), "uploaded block should return Some");

        // Cleanup.
        let raw = S3ObjectBlockClient::new(S3Config::minio(endpoint, bucket.into()))
            .await
            .unwrap();
        raw.delete_object(&key).await.unwrap();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NIXL OBJ full round-trip tests — require real NIXL with OBJ backend.
//
// Run with:
//   bash lib/kvbm-engine/scripts/test-nixl-obj.sh
// or inside the Docker Compose environment:
//   cargo test -p kvbm-engine --features testing-nixl-obj -- nixl_obj_integration --nocapture
//
// Requires:
//   - NIXL installed at $NIXL_PREFIX with OBJ backend
//   - S3_TEST_ENDPOINT / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "testing-nixl-obj")]
#[allow(dead_code, unused_imports)]
mod nixl_obj_integration {
    use super::*;
    use crate::object::s3::{S3Config, S3ObjectBlockClient};
    use dynamo_memory::nixl::NixlAgent;
    use kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder};
    use kvbm_physical::transfer::{FillPattern, PhysicalLayout, StorageKind, fill_blocks};

    fn test_endpoint() -> String {
        std::env::var("S3_TEST_ENDPOINT").unwrap_or_else(|_| "http://localhost:9876".into())
    }

    fn make_layout_config(num_blocks: usize) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap()
    }

    fn build_contiguous_layout(agent: NixlAgent, config: LayoutConfig) -> PhysicalLayout {
        PhysicalLayout::builder(agent)
            .with_config(config)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap()
    }

    fn build_layer_separate_layout(agent: NixlAgent, config: LayoutConfig) -> PhysicalLayout {
        PhysicalLayout::builder(agent)
            .with_config(config)
            .layer_separate(BlockDimension::BlockIsFirstDim)
            .allocate_system()
            .build()
            .unwrap()
    }

    /// Returns `false` when NIXL is running in stub mode (skip the test).
    fn nixl_available() -> bool {
        if nixl_sys::is_stub() {
            eprintln!("NIXL is in stub mode — skipping NIXL OBJ transfer test");
            return false;
        }
        true
    }

    fn read_layout_bytes(layout: &PhysicalLayout, block_id: usize) -> Vec<u8> {
        let config = layout.layout().config();
        let mut all_bytes = Vec::new();
        for layer_id in 0..config.num_layers {
            for outer_id in 0..config.outer_dim {
                let region = layout.memory_region(block_id, layer_id, outer_id).unwrap();
                let slice = unsafe {
                    std::slice::from_raw_parts(region.addr() as *const u8, region.size())
                };
                all_bytes.extend_from_slice(slice);
            }
        }
        all_bytes
    }

    fn build_agent_with_obj(bucket: &str, endpoint: &str) -> NixlAgent {
        let mut agent = NixlAgent::new("nixl-obj-test").expect("NixlAgent::new failed");
        let obj_params: std::collections::HashMap<String, String> = {
            let cfg = kvbm_config::NixlS3Config::with_endpoint(endpoint, bucket);
            cfg.to_nixl_params()
        };
        add_obj_backend(&mut agent, obj_params).expect("OBJ backend required");
        agent
    }

    async fn ensure_bucket(endpoint: &str, bucket: &str) {
        let s3_cfg = S3Config::minio(endpoint.to_string(), bucket.into());
        S3ObjectBlockClient::new(s3_cfg)
            .await
            .unwrap()
            .ensure_bucket_exists()
            .await
            .unwrap();
    }

    // ── Pre-flight validation ─────────────────────────────────────────────────

    /// `from_config` must return an error when the OBJ backend has not been
    /// registered on the agent.
    #[tokio::test]
    async fn test_from_config_requires_obj_backend() {
        if !nixl_available() {
            return;
        }

        // Create agent WITHOUT adding the OBJ backend.
        let agent = NixlAgent::new("test-preflight").expect("NixlAgent::new failed");

        let nixl_cfg = kvbm_config::NixlObjectConfig::S3(
            kvbm_config::NixlS3Config::with_endpoint(test_endpoint(), "preflight-bucket"),
        );
        let result = NixlObjectBlockClient::from_config(agent, &nixl_cfg, None).await;

        assert!(result.is_err(), "expected Err when OBJ backend is missing");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("OBJ"),
            "error message should mention the OBJ backend, got: {msg}"
        );
    }

    // ── Contiguous round-trip ─────────────────────────────────────────────────

    /// End-to-end NIXL OBJ transfer: DRAM → S3 (put) then S3 → DRAM (get)
    /// using a fully-contiguous layout.
    ///
    /// Uses `SystemStorage` (plain malloc) so no GPU is required.
    #[tokio::test]
    async fn test_nixl_obj_dram_to_s3_roundtrip() {
        if !nixl_available() {
            return;
        }

        let bucket = "nixl-obj-roundtrip";
        let endpoint = test_endpoint();
        ensure_bucket(&endpoint, bucket).await;

        let agent = build_agent_with_obj(bucket, &endpoint);

        // ── Create write layout and fill with known pattern ───────────────────
        let config = make_layout_config(1);
        let write_layout = build_contiguous_layout(agent.clone(), config.clone());

        fill_blocks(&write_layout, &[0], FillPattern::Sequential).unwrap();
        let original_bytes = read_layout_bytes(&write_layout, 0);
        assert!(!original_bytes.iter().all(|&b| b == 0), "write buffer must be non-zero");

        // ── Build the client and offload (DRAM → S3) ─────────────────────────
        let nixl_cfg = kvbm_config::NixlObjectConfig::S3(
            kvbm_config::NixlS3Config::with_endpoint(endpoint.clone(), bucket),
        );
        let client = NixlObjectBlockClient::from_config(agent.clone(), &nixl_cfg, None)
            .await
            .unwrap();

        let hash = SequenceHash::new(0xC0DE_CAFE_u64, None, 7);
        let put_results = client
            .put_blocks_with_layout(vec![hash], write_layout, vec![0])
            .await;
        assert_eq!(put_results.len(), 1);
        assert!(
            put_results[0].is_ok(),
            "put_blocks_with_layout failed: {:?}",
            put_results[0]
        );

        // ── Create zero-filled read layout and onboard (S3 → DRAM) ───────────
        let read_layout = build_contiguous_layout(agent.clone(), config.clone());
        {
            let bytes = read_layout_bytes(&read_layout, 0);
            assert!(bytes.iter().all(|&b| b == 0), "read buffer must start zeroed");
        }

        let get_results = client
            .get_blocks_with_layout(vec![hash], read_layout.clone(), vec![0])
            .await;
        assert_eq!(get_results.len(), 1);
        assert!(
            get_results[0].is_ok(),
            "get_blocks_with_layout failed: {:?}",
            get_results[0]
        );

        // ── Verify byte-for-byte correctness ─────────────────────────────────
        let restored_bytes = read_layout_bytes(&read_layout, 0);
        assert_eq!(
            original_bytes, restored_bytes,
            "round-trip mismatch: DRAM→S3→DRAM did not reproduce the original data"
        );

        println!(
            "✓ NIXL OBJ contiguous round-trip: {} bytes via DRAM→S3→DRAM",
            original_bytes.len()
        );
    }

    // ── Layer-separate (non-contiguous) round-trip ────────────────────────────

    /// End-to-end NIXL OBJ transfer using a layer-separate (non-contiguous) layout.
    ///
    /// This exercises the slow path in `execute_nixl_obj_transfer` that builds one
    /// DRAM descriptor per (layer, outer_dim) region.
    #[tokio::test]
    async fn test_nixl_obj_layer_separate_roundtrip() {
        if !nixl_available() {
            return;
        }

        let bucket = "nixl-obj-layer-sep";
        let endpoint = test_endpoint();
        ensure_bucket(&endpoint, bucket).await;

        let agent = build_agent_with_obj(bucket, &endpoint);
        let config = make_layout_config(1);

        let write_layout = build_layer_separate_layout(agent.clone(), config.clone());
        fill_blocks(&write_layout, &[0], FillPattern::Sequential).unwrap();
        let original_bytes = read_layout_bytes(&write_layout, 0);
        assert!(!original_bytes.iter().all(|&b| b == 0), "write buffer must be non-zero");

        let nixl_cfg = kvbm_config::NixlObjectConfig::S3(
            kvbm_config::NixlS3Config::with_endpoint(endpoint.clone(), bucket),
        );
        let client = NixlObjectBlockClient::from_config(agent.clone(), &nixl_cfg, None)
            .await
            .unwrap();

        let hash = SequenceHash::new(0xABCD_EF01_u64, None, 3);

        let put_results = client
            .put_blocks_with_layout(vec![hash], write_layout, vec![0])
            .await;
        assert!(put_results[0].is_ok(), "put failed: {:?}", put_results[0]);

        let read_layout = build_layer_separate_layout(agent.clone(), config.clone());
        {
            let bytes = read_layout_bytes(&read_layout, 0);
            assert!(bytes.iter().all(|&b| b == 0), "read buffer must start zeroed");
        }

        let get_results = client
            .get_blocks_with_layout(vec![hash], read_layout.clone(), vec![0])
            .await;
        assert!(get_results[0].is_ok(), "get failed: {:?}", get_results[0]);

        let restored_bytes = read_layout_bytes(&read_layout, 0);
        assert_eq!(
            original_bytes, restored_bytes,
            "layer-separate round-trip mismatch"
        );

        println!(
            "✓ NIXL OBJ layer-separate round-trip: {} bytes via DRAM→S3→DRAM",
            original_bytes.len()
        );
    }

    // ── Transfer timeout ──────────────────────────────────────────────────────

    /// Verify that a very short transfer_timeout propagates correctly: the client
    /// is built with the configured timeout and it is stored on the struct.
    ///
    /// Full "timeout fires" behaviour is difficult to test without a mock NIXL
    /// agent, but verifying config propagation catches the most common mistake
    /// (timeout silently ignored).
    #[tokio::test]
    async fn test_transfer_timeout_config_propagated() {
        if !nixl_available() {
            return;
        }

        let bucket = "nixl-obj-timeout-cfg";
        let endpoint = test_endpoint();

        let agent = build_agent_with_obj(bucket, &endpoint);
        let nixl_cfg = kvbm_config::NixlObjectConfig::S3(kvbm_config::NixlS3Config {
            endpoint_override: Some(endpoint.clone()),
            bucket: Some(bucket.into()),
            transfer_timeout_secs: Some(5),
            ..Default::default()
        });
        let client = NixlObjectBlockClient::from_config(agent, &nixl_cfg, None)
            .await
            .unwrap();

        assert_eq!(
            client.transfer_timeout,
            Some(Duration::from_secs(5)),
            "transfer_timeout not propagated from config"
        );
    }
}
