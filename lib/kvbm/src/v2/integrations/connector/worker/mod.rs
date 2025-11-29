// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dynamo_nova::Nova;
use std::sync::Arc;

use crate::v2::distributed::worker::NovaWorkerService;

pub trait ConnectorWorker: Send + Sync {
    fn register_kv_caches(&self) -> Result<()>;
    fn shutdown(&self) -> Result<()>;

    /// Allows the vLLM worker to export metadata which will be all-gathered to the leader
    /// and passed to the scheduler-side connector/leader as a rank-ordered list of metdata
    /// objects. This will
    fn handshake_metadata(&self) -> Result<Bytes>;
}

// todo: the first iteration of the worker will consist of:
// - creating a nova instance with a default config: tcp only, no discovery
// -
pub struct ConnectorWorkerImpl {
    nova: Arc<Nova>,
    inner: Option<NovaWorkerService>,
}

impl ConnectorWorker for ConnectorWorkerImpl {
    fn handshake_metadata(&self) -> Result<Bytes> {
        Ok(self.nova.peer_info().worker_address().to_bytes())
    }

    fn register_kv_caches(&self) -> Result<()> {
        // todo:
        // - inspect the kv_cache tensors
        // - build the layout
        // - extract the cuda device
        // - construct the transfer manager
        // - construct a direct worker
        // - attach the layout handle for the g1 caches to teh direct worker
        // - construct the nova worker service

        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        todo!("shutdown the connector worker");
    }
}
