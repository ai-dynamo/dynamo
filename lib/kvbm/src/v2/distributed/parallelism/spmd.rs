// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use anyhow::Result;
use dynamo_nova::events::{LocalEvent, LocalEventSystem};

use std::sync::Arc;

/// A parallel [`Worker`]. Actions on this [`Worker`] are executed in
/// parallel on all workers.
pub struct ReplicatedWorker {
    workers: Vec<Arc<dyn Worker>>,
    events: Arc<LocalEventSystem>,
    runtime: tokio::runtime::Handle,
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
        let event = self.events.new_event()?;
        let awaiter = self.events.awaiter(event.handle())?;

        let results = self
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

        self.runtime.spawn(await_all_results(results, event));

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.events.new_event()?;
        let awaiter = self.events.awaiter(event.handle())?;

        let results = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_remote_onboard(
                    src.clone(),
                    dst.clone(),
                    dst_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.runtime.spawn(await_all_results(results, event));

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.events.new_event()?;
        let awaiter = self.events.awaiter(event.handle())?;

        let results = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_remote_offload(
                    src.clone(),
                    dst.clone(),
                    src_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.runtime.spawn(await_all_results(results, event));

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
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

async fn await_all_results(
    notifications: Vec<TransferCompleteNotification>,
    local_event: LocalEvent,
) -> Result<()> {
    match futures::future::try_join_all(
        notifications
            .into_iter()
            .map(TransferCompleteNotification::into_future),
    )
    .await
    {
        Ok(_) => local_event.trigger(),
        Err(err) => local_event.poison(err.to_string()),
    }
}
