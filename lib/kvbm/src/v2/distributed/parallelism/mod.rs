// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod spmd;

use std::sync::Arc;

use super::object::ObjectBlockOps;
use super::worker::{
    ImportMetadataResponse, SerializedLayout, SerializedLayoutResponse, Worker, WorkerTransfers, *,
};
use anyhow::Result;

pub use spmd::ReplicatedWorker;

pub trait ParallelWorker: WorkerTransfers + ObjectBlockOps + Send + Sync {
    /// Export the local metadata for a set of workers.
    ///
    /// Layouts will be returned in rank order.
    ///
    /// # Returns
    /// A [`crate::physical::manager::SerializedLayout`] containing the local metadata
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>>;

    /// Import the remote metadata for this worker.
    ///
    /// Handles will be returned in rank order.
    ///
    /// # Arguments
    /// * `metadata` - A [`crate::physical::manager::SerializedLayout`] containing the remote metadata
    ///
    /// # Returns
    /// A vector of [`crate::physical::manager::LayoutHandle`] for the imported remote layouts
    fn import_metadata(
        &self,
        metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>>;

    /// Get the number of workers.
    fn worker_count(&self) -> usize;

    /// Get access to the underlying workers for metadata/handle queries.
    ///
    /// This is useful for operations that need to query individual workers
    /// (e.g., collecting layout handles) without executing transfers.
    fn workers(&self) -> &[Arc<dyn Worker>];
}
