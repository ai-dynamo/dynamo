// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed Workers
//!
//! This module provides the interface for how the leader will drive multiple workers.

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod spmd;

use std::sync::Arc;

use super::{
    ImportMetadataResponse, SerializedLayout, SerializedLayoutResponse, Worker, WorkerTransfers, *,
};
use crate::object::ObjectBlockOps;
use anyhow::Result;
use kvbm_physical::manager::ParallelismDescriptor;

pub use spmd::SpmdParallelWorkers;

/// A cohort of parallel workers.
///
/// This trait is used to drive one or more parallel workers.
pub trait ParallelWorkers: WorkerTransfers + ObjectBlockOps + Send + Sync {
    /// Export the local metadata for a set of workers.
    ///
    /// Layouts will be returned in rank order.
    ///
    /// # Returns
    /// A [`kvbm_physical::manager::SerializedLayout`] containing the local metadata
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>>;

    /// Import the remote metadata for this worker.
    ///
    /// Handles will be returned in rank order.
    ///
    /// # Arguments
    /// * `metadata` - A [`kvbm_physical::manager::SerializedLayout`] containing the remote metadata
    ///
    /// # Returns
    /// A vector of [`kvbm_physical::manager::LayoutHandle`] for the imported remote layouts
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

    /// AB-4: return the full per-rank [`ParallelismDescriptor`] set for a
    /// peer instance, if one was cached during `connect_remote`'s Strict
    /// import path.
    ///
    /// The leader-level [`crate::leader::InstanceLeader::rdma_pull`]
    /// consults this to feed [`crate::leader::dispatch::plan_pull`]
    /// without duplicating descriptor cache state. Returns `None` when
    /// the peer was imported via the Legacy path (unstamped) or hasn't
    /// been imported at all.
    ///
    /// Default impl returns `None` so Worker types that don't track
    /// cross-parallelism state don't have to override.
    fn remote_descriptors_for(
        &self,
        instance_id: InstanceId,
    ) -> Option<Vec<ParallelismDescriptor>> {
        let _ = instance_id;
        None
    }
}
