// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-based RPC implementation for distributed worker communication.
//!
//! # RPC Pattern Guidelines
//!
//! This module uses only two Velo RPC patterns:
//!
//! 1. **`am_send` (fire-and-forget)**: Use when no response is needed.
//!    - Client sends message and returns immediately
//!    - Handler processes asynchronously, no response sent back
//!    - Use `Handler::am_handler` or `am_handler_async`
//!
//! 2. **`unary` (request-response)**: Use when waiting for completion.
//!    - Client sends request and awaits response
//!    - Handler returns `Ok(Some(Bytes))` or `Ok(None)` which is sent back
//!    - Use `Handler::unary_handler` or `unary_handler_async`
//!
//! # Why Not `am_sync`?
//!
//! We avoid `am_sync` due to observed issues where it does not reliably
//! receive completion signals when paired with `am_handler_async`. While
//! `am_sync` should theoretically behave like `unary` (both await completion),
//! in practice pairing `am_sync` client with `am_handler_async` handler caused
//! indefinite blocking during RDMA transfer tests.
//!
//! The root cause appears to be a mismatch in how responses are routed:
//! - `am_handler_async` returns `Result<()>` - the return value is NOT sent back
//! - `unary_handler_async` returns `Result<Option<Bytes>>` - the return value IS sent back
//!
//! Until the `am_sync` completion path is validated, prefer the simpler and
//! more predictable patterns: `am_send` for fire-and-forget, `unary` for
//! request-response.

mod client;
mod service;

pub use client::VeloWorkerClient;
pub use service::{VeloWorkerService, VeloWorkerServiceBuilder};

/// Canonical handler names for the Velo RPCs this worker exposes.
///
/// Centralising these as constants keeps the client (`client.rs`) and
/// service (`service.rs`) in lock-step — a rename or addition lands in
/// exactly one place and the compiler enforces both sides see the same
/// string. Adding a new RPC: add a const here, register a handler in
/// `service.rs`, and use the const at the call site in `client.rs`.
pub(crate) mod handler_names {
    pub const LOCAL_TRANSFER: &str = "kvbm.worker.local_transfer";
    pub const REMOTE_ONBOARD: &str = "kvbm.worker.remote_onboard";
    pub const REMOTE_OFFLOAD: &str = "kvbm.worker.remote_offload";
    pub const IMPORT_METADATA: &str = "kvbm.worker.import_metadata";
    pub const EXPORT_METADATA: &str = "kvbm.worker.export_metadata";
    pub const CONNECT_REMOTE: &str = "kvbm.worker.connect_remote";
    pub const REMOTE_ONBOARD_FOR_INSTANCE: &str = "kvbm.worker.remote_onboard_for_instance";
    pub const REMOTE_ONBOARD_FOR_INSTANCE_RANK: &str =
        "kvbm.worker.remote_onboard_for_instance_rank";
    /// AB-3: multi-shard cross-parallelism pull. Carries a
    /// `crate::leader::dispatch::WorkerPullPlan` whose `shards` list
    /// drives one or more sliced reads from rank-aware remote handles.
    pub const REMOTE_PULL_PLAN: &str = "kvbm.worker.remote_pull_plan";
    pub const OBJECT_HAS_BLOCKS: &str = "kvbm.worker.object_has_blocks";
    pub const OBJECT_PUT_BLOCKS: &str = "kvbm.worker.object_put_blocks";
    pub const OBJECT_GET_BLOCKS: &str = "kvbm.worker.object_get_blocks";
}

use super::DirectWorker;
use super::*;
use kvbm_common::KvbmTransferRoute;
use kvbm_physical::layout::LayoutConfig;
use kvbm_physical::transfer::TransferOptions;

use ::velo::Messenger;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

// Serializable transfer options for remote operations
#[derive(Serialize, Deserialize, Clone)]
struct SerializableTransferOptions {
    layer_range: Option<std::ops::Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer_handle: Option<LayoutHandle>,
    bounce_buffer_block_ids: Option<Vec<BlockId>>,
    metric_route: Option<KvbmTransferRoute>,
}

impl From<SerializableTransferOptions> for TransferOptions {
    fn from(opts: SerializableTransferOptions) -> Self {
        TransferOptions {
            layer_range: opts.layer_range,
            nixl_write_notification: opts.nixl_write_notification,
            // bounce_buffer requires TransportManager to resolve handle to layout
            bounce_buffer: None,
            cuda_stream: None,
            // KV layout overrides are not serialized; they must be set locally
            src_kv_layout: None,
            dst_kv_layout: None,
            metric_route: opts.metric_route,
            // use_planner is a per-side optimization toggle; the wire form
            // intentionally does not propagate it. Each receiver picks its
            // own planner policy. c6 made the receiver-side rule load-
            // bearing: `executor::execute_transfer` auto-promotes
            // use_planner = true when `requires_transform(src, dst)` is
            // true, so the `false` hardcode here is safe for the
            // cross-leader case (Universal↔Universal under c3 semantics
            // → requires_transform = false → auto-promote never fires).
            use_planner: false,
        }
    }
}

impl SerializableTransferOptions {
    /// Extract bounce buffer handle and block IDs if present
    fn bounce_buffer_parts(&self) -> Option<(LayoutHandle, Vec<BlockId>)> {
        match (&self.bounce_buffer_handle, &self.bounce_buffer_block_ids) {
            (Some(handle), Some(block_ids)) => Some((*handle, block_ids.clone())),
            _ => None,
        }
    }
}

impl From<TransferOptions> for SerializableTransferOptions {
    fn from(opts: TransferOptions) -> Self {
        // Extract bounce buffer parts if present using into_parts()
        let (bounce_buffer_handle, bounce_buffer_block_ids) = opts
            .bounce_buffer
            .map(|bb| {
                let (handle, block_ids) = bb.into_parts();
                (Some(handle), Some(block_ids))
            })
            .unwrap_or((None, None));

        Self {
            layer_range: opts.layer_range,
            nixl_write_notification: opts.nixl_write_notification,
            bounce_buffer_handle,
            bounce_buffer_block_ids,
            metric_route: opts.metric_route,
        }
    }
}

// Message types for remote worker operations
#[derive(Serialize, Deserialize)]
struct LocalTransferMessage {
    src: LogicalLayoutHandle,
    dst: LogicalLayoutHandle,
    src_block_ids: Vec<BlockId>,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

#[derive(Serialize, Deserialize)]
struct RemoteOnboardMessage {
    src: RemoteDescriptor,
    dst: LogicalLayoutHandle,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

#[derive(Serialize, Deserialize)]
struct RemoteOffloadMessage {
    src: LogicalLayoutHandle,
    dst: RemoteDescriptor,
    src_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// Message for connect_remote RPC - stores remote instance metadata in local worker
#[derive(Serialize, Deserialize)]
struct ConnectRemoteMessage {
    instance_id: InstanceId,
    /// Metadata serialized as raw bytes (SerializedLayout uses bincode internally)
    metadata: Vec<Vec<u8>>,
}

/// Message for execute_remote_onboard_for_instance RPC - pulls from remote using instance ID
#[derive(Serialize, Deserialize)]
struct ExecuteRemoteOnboardForInstanceMessage {
    instance_id: InstanceId,
    remote_logical_type: LogicalLayoutHandle,
    src_block_ids: Vec<BlockId>,
    dst: LogicalLayoutHandle,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// Rank-aware variant for AB-1c. Targets a specific remote rank under
/// `instance_id` so the worker can resolve via its
/// `remote_handles_rank` map and serve asymmetric-TP pulls.
#[derive(Serialize, Deserialize)]
struct ExecuteRemoteOnboardForInstanceRankMessage {
    instance_id: InstanceId,
    remote_rank: usize,
    remote_logical_type: LogicalLayoutHandle,
    src_block_ids: Vec<BlockId>,
    dst: LogicalLayoutHandle,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// Multi-shard pull message for AB-3. The plan was produced by the
/// peer leader's [`crate::leader::dispatch::plan_pull`] and carries
/// everything one local worker needs to execute its share of an
/// asymmetric-TP transfer: a target [`InstanceId`], source/destination
/// [`LogicalLayoutHandle`]s, paired `(src_block_id, dst_block_id)`
/// vectors that apply to every shard, the per-remote-rank shard list
/// with coordinate-space slices on both sides, and a
/// [`crate::leader::dispatch::WirePullOptions`].
///
/// On the wire we wrap the plan in a one-field struct so future
/// additions (e.g. retry hints) can land without re-encoding existing
/// fields — serde_json named fields handle the forward-compat
/// gracefully, but the wrapper keeps the door open for sibling fields
/// without forcing a fresh top-level type.
#[derive(Serialize, Deserialize)]
struct RemotePullPlanMessage {
    plan: crate::leader::dispatch::WorkerPullPlan,
}

// ============================================================================
// Object Storage Message Types
// ============================================================================

/// Message for object_has_blocks RPC - check if blocks exist in object storage
#[derive(Serialize, Deserialize)]
struct ObjectHasBlocksMessage {
    keys: Vec<SequenceHash>,
}

/// Response for object_has_blocks RPC
#[derive(Serialize, Deserialize)]
struct ObjectHasBlocksResponse {
    results: Vec<(SequenceHash, Option<usize>)>,
}

/// Message for object_put_blocks RPC - upload blocks to object storage
#[derive(Serialize, Deserialize)]
struct ObjectPutBlocksMessage {
    keys: Vec<SequenceHash>,
    layout: LogicalLayoutHandle,
    block_ids: Vec<BlockId>,
}

/// Message for object_get_blocks RPC - download blocks from object storage
#[derive(Serialize, Deserialize)]
struct ObjectGetBlocksMessage {
    keys: Vec<SequenceHash>,
    layout: LogicalLayoutHandle,
    block_ids: Vec<BlockId>,
}

/// Response for object put/get operations
#[derive(Serialize, Deserialize)]
struct ObjectPutGetBlocksResponse {
    /// Ok(key) for success, Err(key) for failure - serialized as (bool, key)
    results: Vec<(bool, SequenceHash)>,
}

impl ObjectPutGetBlocksResponse {
    fn from_results(results: Vec<Result<SequenceHash, SequenceHash>>) -> Self {
        Self {
            results: results
                .into_iter()
                .map(|r| match r {
                    Ok(k) => (true, k),
                    Err(k) => (false, k),
                })
                .collect(),
        }
    }

    fn into_results(self) -> Vec<Result<SequenceHash, SequenceHash>> {
        self.results
            .into_iter()
            .map(|(ok, k)| if ok { Ok(k) } else { Err(k) })
            .collect()
    }
}
