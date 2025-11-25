// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod service;

pub use client::NovaWorkerClient;
pub use service::NovaWorkerService;

use super::direct::DirectWorker;
use super::*;
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
