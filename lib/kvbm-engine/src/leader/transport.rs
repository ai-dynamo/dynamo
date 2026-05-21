// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-to-leader metadata exchange transport.
//!
//! Slim client for the `kvbm.leader.export_metadata` unary RPC. The leader
//! requests worker transfer metadata (`Vec<SerializedLayout>`) from a remote
//! leader so it can import the remote's layout handles before an RDMA pull.
//! Used by [`crate::leader::InstanceLeader::ensure_remote_metadata`] on the
//! live p2p RDMA-pull path.

use ::velo::Messenger;
use anyhow::Result;
use bytes::Bytes;

use std::sync::Arc;

use crate::InstanceId;
use kvbm_physical::manager::SerializedLayout;

/// Velo-based client for requesting remote leader metadata.
pub struct MetadataTransport {
    messenger: Arc<Messenger>,
}

impl MetadataTransport {
    pub fn new(messenger: Arc<Messenger>) -> Self {
        Self { messenger }
    }

    /// Request worker metadata from a remote leader for RDMA transfers.
    ///
    /// Makes a unary RPC call to the remote leader's `kvbm.leader.export_metadata`
    /// handler and returns the `Vec<SerializedLayout>` from all remote workers.
    pub async fn request_metadata(&self, target: InstanceId) -> Result<Vec<SerializedLayout>> {
        tracing::debug!(target = %target, "Requesting metadata from instance");

        let response: Bytes = self
            .messenger
            .unary("kvbm.leader.export_metadata")?
            .instance(target)
            .send()
            .await?;

        let metadata: Vec<SerializedLayout> = serde_json::from_slice(&response)?;

        tracing::debug!(
            count = metadata.len(),
            target = %target,
            "Received metadata entries"
        );

        Ok(metadata)
    }
}
