// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use anyhow::Result;
use dynamo_nova::events::{LocalEvent, LocalEventSystem};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A parallel [`Worker`]. Actions on this [`Worker`] are executed in
/// parallel on all workers.
pub struct ReplicatedWorker {
    workers: Vec<Arc<dyn Worker>>,
    events: Arc<LocalEventSystem>,
    runtime: tokio::runtime::Handle,

    /// Remote handle mappings: (InstanceId, worker_idx, LogicalLayoutHandle) -> remote LayoutHandle.
    /// Populated by `connect_remote` for later use by `execute_remote_onboard_for_instance`.
    remote_handles: RwLock<HashMap<(InstanceId, usize, LogicalLayoutHandle), LayoutHandle>>,
}

impl ReplicatedWorker {
    /// Create a new ReplicatedWorker.
    ///
    /// # Arguments
    /// * `workers` - The underlying workers (one per rank)
    /// * `events` - The event system for aggregating completion notifications
    /// * `runtime` - The tokio runtime handle for spawning aggregation tasks
    pub fn new(
        workers: Vec<Arc<dyn Worker>>,
        events: Arc<LocalEventSystem>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            workers,
            events,
            runtime,
            remote_handles: RwLock::new(HashMap::new()),
        }
    }

    /// Get the number of workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl WorkerTransfers for ReplicatedWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let notifications = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_local_transfer(
                    src,
                    dst,
                    src_block_ids.clone(),
                    dst_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let notifications = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_remote_onboard(
                    src.clone(),
                    dst,
                    dst_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let notifications = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_remote_offload(
                    src,
                    dst.clone(),
                    src_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // Validate metadata count matches worker count
        if metadata.len() != self.workers.len() {
            anyhow::bail!(
                "Metadata count ({}) doesn't match worker count ({})",
                metadata.len(),
                self.workers.len()
            );
        }

        // Collect handles to store and responses to await
        let mut new_handles = Vec::new();
        let mut import_responses = Vec::new();

        for (worker_idx, (worker, meta)) in
            self.workers.iter().zip(metadata.into_iter()).enumerate()
        {
            // Unpack to extract logical type info
            let unpacked = meta.unpack()?;

            // Collect handle mappings
            for descriptor in &unpacked.layouts {
                new_handles.push((
                    (instance_id, worker_idx, descriptor.logical_type),
                    descriptor.handle,
                ));
            }

            // Repack for the underlying worker's import_metadata
            let repacked = SerializedLayout::pack(
                unpacked.worker_address,
                unpacked.nixl_metadata,
                unpacked.layouts,
            )?;

            // Call underlying worker's import_metadata
            import_responses.push(worker.import_metadata(repacked)?);
        }

        // Store all handle mappings
        {
            let mut handles = self.remote_handles.write().unwrap();
            for (key, value) in new_handles {
                handles.insert(key, value);
            }
        }

        // If all responses are ready (synchronous), return immediately
        if import_responses.iter().all(|r| !r.could_yield()) {
            return Ok(ConnectRemoteResponse::ready());
        }

        // Create an event to aggregate all import completions
        let event = self.events.new_event()?;
        let awaiter = self.events.awaiter(event.handle())?;

        // Spawn task to await all import responses and signal completion
        self.runtime
            .spawn(await_import_responses(import_responses, event));

        Ok(ConnectRemoteResponse::from_awaiter(awaiter))
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        let handles = self.remote_handles.read().unwrap();
        handles.keys().any(|(id, _, _)| *id == instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let handles = self.remote_handles.read().unwrap();
        let mut notifications = Vec::with_capacity(self.workers.len());

        // SPMD: Execute SAME transfer on EVERY worker, each with its own remote handle
        for (worker_idx, worker) in self.workers.iter().enumerate() {
            let remote_handle = handles
                .get(&(instance_id, worker_idx, remote_logical_type))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No remote {:?} handle for instance {} worker {}",
                        remote_logical_type,
                        instance_id,
                        worker_idx
                    )
                })?;

            let descriptor = RemoteDescriptor::Layout {
                handle: *remote_handle,
                block_ids: src_block_ids.clone(),
            };

            notifications.push(worker.execute_remote_onboard(
                descriptor,
                dst,
                dst_block_ids.clone(),
                options.clone(),
            )?);
        }

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }
}

/// Helper to await all import metadata responses and signal completion via an event.
/// Helper to await all import metadata responses and signal completion via an event.
async fn await_import_responses(responses: Vec<ImportMetadataResponse>, event: LocalEvent) {
    let results: Vec<Result<Vec<LayoutHandle>>> =
        futures::future::join_all(responses.into_iter().map(|r| r.into_future())).await;

    // Check for any failures
    let errors: Vec<_> = results.into_iter().filter_map(|r| r.err()).collect();

    if errors.is_empty() {
        let _ = event.trigger();
    } else {
        let error_msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        let _ = event.poison(error_msg);
    }
}

impl ParallelWorker for ReplicatedWorker {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        let metadata = self
            .workers
            .iter()
            .map(|worker| worker.export_metadata())
            .collect::<Result<Vec<_>>>()?;

        Ok(metadata)
    }

    fn import_metadata(
        &self,
        metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        // validate the size of the metadata is the same as the number of workers
        if metadata.len() != self.workers.len() {
            return Err(anyhow::anyhow!(
                "Metadata size does not match number of workers"
            ));
        }

        let results = self
            .workers
            .iter()
            .zip(metadata.iter())
            .map(|(worker, metadata)| worker.import_metadata(metadata.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(results)
    }
}
