// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nova-based RPC implementation for distributed worker communication.
//!
//! # RPC Pattern Guidelines
//!
//! This module uses only two Nova RPC patterns:
//!
//! 1. **`am_send` (fire-and-forget)**: Use when no response is needed.
//!    - Client sends message and returns immediately
//!    - Handler processes asynchronously, no response sent back
//!    - Use `NovaHandler::am_handler` or `am_handler_async`
//!
//! 2. **`unary` (request-response)**: Use when waiting for completion.
//!    - Client sends request and awaits response
//!    - Handler returns `Ok(Some(Bytes))` or `Ok(None)` which is sent back
//!    - Use `NovaHandler::unary_handler` or `unary_handler_async`
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

pub use client::NovaWorkerClient;
pub use service::{NovaWorkerService, NovaWorkerServiceBuilder};

use super::direct::DirectWorker;
use super::*;
use crate::physical::layout::LayoutConfig;
use crate::physical::transfer::TransferOptions;

use bytes::Bytes;
use dynamo_nova::Nova;
use serde::{Deserialize, Serialize};

// Serializable transfer options for remote operations
#[derive(Serialize, Deserialize, Clone)]
struct SerializableTransferOptions {
    layer_range: Option<std::ops::Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer_handle: Option<LayoutHandle>,
    bounce_buffer_block_ids: Option<Vec<BlockId>>,
}

impl From<SerializableTransferOptions> for TransferOptions {
    fn from(opts: SerializableTransferOptions) -> Self {
        TransferOptions {
            layer_range: opts.layer_range,
            nixl_write_notification: opts.nixl_write_notification,
            // bounce_buffer requires TransportManager to resolve handle to layout
            bounce_buffer: None,
            cuda_stream: None,
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
