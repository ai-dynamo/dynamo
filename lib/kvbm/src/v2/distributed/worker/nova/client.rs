// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub struct NovaWorkerClient {
    nova: Arc<Nova>,
    remote_instances: Vec<InstanceId>,
    tracker: TaskTracker,
}

impl Worker for NovaWorkerClient {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst_block_ids: Vec<BlockId>,
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
            src_block_ids,
            dst_block_ids,
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Spawn a task for each remote instance
        let mut tasks = Vec::new();
        for instance in &self.remote_instances {
            let nova = self.nova.clone();
            let bytes = bytes.clone();
            let instance = *instance;

            let task = tokio::spawn(async move {
                nova.am_sync("kvbm.worker.local_transfer")?
                    .raw_payload(bytes)
                    .instance(instance)
                    .send()
                    .await
            });

            tasks.push(task);
        }

        // Spawn a background task to await all and complete the event
        NovaWorkerClient::spawn_completion_task(
            self.nova.clone(),
            event,
            handle,
            tasks,
            "local_transfer",
            "One or more remote transfers failed",
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Vec<BlockId>,
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
        let message = RemoteOnboardMessage {
            src,
            dst,
            dst_block_ids,
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Spawn a task for each remote instance
        let mut tasks = Vec::new();
        for instance in &self.remote_instances {
            let nova = self.nova.clone();
            let bytes = bytes.clone();
            let instance = *instance;

            let task = tokio::spawn(async move {
                nova.am_sync("kvbm.worker.remote_onboard")?
                    .raw_payload(bytes)
                    .instance(instance)
                    .send()
                    .await
            });

            tasks.push(task);
        }

        // Spawn a background task to await all and complete the event
        NovaWorkerClient::spawn_completion_task(
            self.nova.clone(),
            event,
            handle,
            tasks,
            "remote_onboard",
            "One or more remote onboards failed",
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Vec<BlockId>,
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
        let message = RemoteOffloadMessage {
            src,
            dst,
            src_block_ids,
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Spawn a task for each remote instance
        let mut tasks = Vec::new();
        for instance in &self.remote_instances {
            let nova = self.nova.clone();
            let bytes = bytes.clone();
            let instance = *instance;

            let task = tokio::spawn(async move {
                nova.am_sync("kvbm.worker.remote_offload")?
                    .raw_payload(bytes)
                    .instance(instance)
                    .send()
                    .await
            });

            tasks.push(task);
        }

        // Spawn a background task to await all and complete the event
        NovaWorkerClient::spawn_completion_task(
            self.nova.clone(),
            event,
            handle,
            tasks,
            "remote_offload",
            "One or more remote offloads failed",
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

impl NovaWorkerClient {
    async fn complete_event_from_tasks(
        nova: Arc<Nova>,
        event: LocalEvent,
        handle: EventHandle,
        tasks: Vec<tokio::task::JoinHandle<Result<()>>>,
        operation: &'static str,
        poison_reason: &'static str,
    ) {
        let results = futures::future::join_all(tasks).await;

        let any_failed = results.into_iter().any(|res| match res {
            Ok(Ok(())) => false,
            Ok(Err(e)) => {
                tracing::error!("Remote {operation} failed: {e}");
                true
            }
            Err(e) => {
                tracing::error!("Task panicked: {e}");
                true
            }
        });

        if any_failed {
            if let Err(e) = nova.events().poison(handle, poison_reason).await {
                tracing::error!("Failed to poison event: {e}");
            }
        } else if let Err(e) = event.trigger() {
            tracing::error!("Failed to trigger event: {e}");
        }
    }

    fn spawn_completion_task(
        nova: Arc<Nova>,
        event: LocalEvent,
        handle: EventHandle,
        tasks: Vec<tokio::task::JoinHandle<Result<()>>>,
        operation: &'static str,
        poison_reason: &'static str,
    ) {
        tokio::spawn(Self::complete_event_from_tasks(
            nova,
            event,
            handle,
            tasks,
            operation,
            poison_reason,
        ));
    }
}
