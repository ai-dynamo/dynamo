// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub struct NovaWorkerClient {
    nova: Arc<Nova>,
    remote: InstanceId,
}

impl WorkerTransfers for NovaWorkerClient {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // Create a single local event for this operation
        let event = self.nova.events().new_event()?;
        let handle = event.handle();
        let awaiter = self.nova.events().awaiter(handle)?;

        // Convert to serializable options
        // TODO: Extract bounce buffer handle if present in options.bounce_buffer
        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
        };

        // Create the message
        let message = LocalTransferMessage {
            src,
            dst,
            src_block_ids: src_block_ids.to_vec(),
            dst_block_ids: dst_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Spawn a task for the remote instance
        let nova = self.nova.clone();
        let bytes = bytes.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .am_sync("kvbm.worker.local_transfer")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
        };

        let message = RemoteOnboardMessage {
            src,
            dst,
            dst_block_ids: dst_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let nova = self.nova.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .am_sync("kvbm.worker.remote_onboard")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.nova.events().new_event()?;
        let awaiter = self.nova.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
        };

        let message = RemoteOffloadMessage {
            src,
            dst,
            src_block_ids: src_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let nova = self.nova.clone();
        let remote_instance = self.remote;

        self.nova.tracker().spawn_on(
            async move {
                let result = nova
                    .am_sync("kvbm.worker.remote_offload")?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.nova.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

impl Worker for NovaWorkerClient {
    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        let instance = self.remote;

        let awaiter = self
            .nova
            .typed_unary::<SerializedLayout>("kvbm.worker.export_metadata")?
            .instance(instance)
            .send();

        Ok(SerializedLayoutResponse::from_awaiter(awaiter))
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        let instance = self.remote;

        let awaiter = self
            .nova
            .typed_unary::<Vec<LayoutHandle>>("kvbm.worker.import_metadata")?
            .payload(metadata)?
            .instance(instance)
            .send();

        Ok(ImportMetadataResponse::from_awaiter(awaiter))
    }
}
