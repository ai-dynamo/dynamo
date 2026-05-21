// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ::velo::{Handler, Messenger};
use anyhow::Result;
use bytes::Bytes;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use kvbm_physical::manager::SerializedLayout;

/// Type alias for async export metadata callback.
/// Returns a boxed future that resolves to `Vec<SerializedLayout>`.
pub type ExportMetadataCallback = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<Vec<SerializedLayout>>> + Send>> + Send + Sync,
>;

/// Velo leader service for the leader-to-leader metadata-export RPC.
///
/// Registers the `kvbm.leader.export_metadata` handler, which returns worker
/// layout metadata (`Vec<SerializedLayout>`) so a remote leader can import our
/// layout handles before an RDMA pull.
pub struct VeloLeaderService {
    messenger: Arc<Messenger>,

    // RDMA metadata export
    /// Callback to export worker metadata for RDMA transfers.
    export_metadata: Option<ExportMetadataCallback>,
}

impl VeloLeaderService {
    pub fn new(messenger: Arc<Messenger>) -> Self {
        Self {
            messenger,
            export_metadata: None,
        }
    }

    /// Set the callback for exporting worker metadata (RDMA).
    ///
    /// This callback is invoked when a remote leader requests metadata
    /// to enable RDMA transfers. The callback should return `Vec<SerializedLayout>`
    /// containing metadata from all workers.
    pub fn with_export_metadata(mut self, callback: ExportMetadataCallback) -> Self {
        self.export_metadata = Some(callback);
        self
    }

    /// Register all Velo handlers for leader-to-leader communication.
    pub fn register_handlers(self) -> Result<()> {
        // Register export_metadata handler if callback is configured
        if self.export_metadata.is_some() {
            self.register_export_metadata_handler()?;
        }

        Ok(())
    }

    /// Register the "kvbm.leader.export_metadata" handler.
    ///
    /// This handler returns `Vec<SerializedLayout>` containing metadata from all workers.
    /// Used by remote leaders to enable RDMA transfers.
    fn register_export_metadata_handler(&self) -> Result<()> {
        let export_metadata = self
            .export_metadata
            .clone()
            .expect("export_metadata callback required for handler registration");

        let handler = Handler::unary_handler_async("kvbm.leader.export_metadata", move |_ctx| {
            let export_metadata = export_metadata.clone();

            async move {
                tracing::debug!("Received export_metadata request");

                // Call the async callback to get metadata from all workers
                let metadata_vec = export_metadata().await?;

                // Serialize the Vec<SerializedLayout> for transport
                let serialized = serde_json::to_vec(&metadata_vec)?;

                tracing::debug!(
                    count = metadata_vec.len(),
                    "Returning worker metadata entries"
                );

                Ok(Some(Bytes::from(serialized)))
            }
        })
        .build();

        self.messenger.register_handler(handler)?;

        Ok(())
    }
}
