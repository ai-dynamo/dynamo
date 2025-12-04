// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
