// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod spmd;

use super::worker::{
    ImportMetadataResponse, SerializedLayout, SerializedLayoutResponse, WorkerTransfers, *,
};
use anyhow::Result;

pub use spmd::ReplicatedWorker;

pub trait ParallelWorker: WorkerTransfers + Send + Sync {
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
}
