// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::physical::manager::SerializedLayout;

use super::{
    Arc, ConnectRemoteMessage, DirectWorker, ExecuteRemoteOnboardForInstanceMessage,
    LocalTransferMessage, ObjectGetBlocksMessage, ObjectHasBlocksMessage, ObjectHasBlocksResponse,
    ObjectPutBlocksMessage, ObjectPutGetBlocksResponse, RemoteOffloadMessage, RemoteOnboardMessage,
    Result, TransferOptions, WorkerTransfers,
};
use crate::v2::distributed::object::ObjectBlockOps;

use bytes::Bytes;
use derive_builder::Builder;

use dynamo_nova::Nova;
use dynamo_nova::am::NovaHandler;

/// Builder for NovaWorkerService - provides flexibility in construction.
///
/// Use this builder when you need to:
/// - Pass a pre-built DirectWorker (when caller manages layout registration)
/// - Pass a pre-built TransferManager (service creates DirectWorker)
/// - Have more control over worker configuration
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct NovaWorkerService {
    nova: Arc<Nova>,
    worker: Arc<DirectWorker>,
}

impl NovaWorkerService {
    pub fn new(nova: Arc<Nova>, worker: Arc<DirectWorker>) -> Result<Self> {
        let service = Self { nova, worker };
        service.register_handlers()?;
        Ok(service)
    }

    /// Access the underlying DirectWorker.
    ///
    /// This is useful for:
    /// - Registering additional layouts after service creation
    /// - Exporting metadata for handshake
    /// - Accessing the TransferManager
    pub fn worker(&self) -> &Arc<DirectWorker> {
        &self.worker
    }

    /// Register all worker handlers with Nova
    fn register_handlers(&self) -> Result<()> {
        self.register_local_transfer_handler()?;
        self.register_remote_onboard_handler()?;
        self.register_remote_offload_handler()?;
        self.register_import_metadata_handler()?;
        self.register_export_metadata_handler()?;
        self.register_connect_remote_handler()?;
        self.register_execute_remote_onboard_for_instance_handler()?;
        // Object storage handlers
        self.register_object_has_blocks_handler()?;
        self.register_object_put_blocks_handler()?;
        self.register_object_get_blocks_handler()?;
        Ok(())
    }

    fn register_local_transfer_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        // Use unary_handler_async for explicit response (client waits for transfer completion)
        let handler = NovaHandler::unary_handler_async("kvbm.worker.local_transfer", move |ctx| {
            let worker = worker.clone();

            async move {
                // Deserialize the message
                let message: LocalTransferMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = worker.execute_local_transfer(
                    message.src,
                    message.dst,
                    Arc::from(message.src_block_ids),
                    Arc::from(message.dst_block_ids),
                    options,
                )?;

                // Await the transfer completion
                notification.await?;

                // Return empty response to signal success
                Ok(Some(Bytes::new()))
            }
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    fn register_remote_onboard_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        // Use unary_handler_async for explicit response (works with unary client)
        let handler = NovaHandler::unary_handler_async("kvbm.worker.remote_onboard", move |ctx| {
            let worker = worker.clone();

            async move {
                let message: RemoteOnboardMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = worker.execute_remote_onboard(
                    message.src,
                    message.dst,
                    Arc::from(message.dst_block_ids),
                    options,
                )?;

                notification.await?;

                Ok(Some(Bytes::new()))
            }
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    fn register_remote_offload_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        // Use unary_handler_async for explicit response (works with unary client)
        let handler = NovaHandler::unary_handler_async("kvbm.worker.remote_offload", move |ctx| {
            let worker = worker.clone();

            async move {
                let message: RemoteOffloadMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = worker.execute_remote_offload(
                    message.src,
                    Arc::from(message.src_block_ids),
                    message.dst,
                    options,
                )?;

                notification.await?;

                Ok(Some(Bytes::new()))
            }
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    fn register_import_metadata_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::unary_handler("kvbm.worker.import_metadata", move |ctx| {
            let metadata = SerializedLayout::from_bytes(ctx.payload.clone());
            let handles = worker.import_metadata(metadata)?;
            Ok(Some(Bytes::from(serde_json::to_vec(&handles)?)))
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    fn register_export_metadata_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::unary_handler("kvbm.worker.export_metadata", move |_ctx| {
            let response = worker.export_metadata()?;
            Ok(Some(response.as_bytes().clone()))
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for connect_remote - stores remote instance metadata in local worker
    fn register_connect_remote_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::unary_handler("kvbm.worker.connect_remote", move |ctx| {
            let message: ConnectRemoteMessage = serde_json::from_slice(&ctx.payload)?;

            // Deserialize metadata (SerializedLayout stored as raw bytes)
            let metadata: Vec<SerializedLayout> = message
                .metadata
                .into_iter()
                .map(|bytes| SerializedLayout::from_bytes(Bytes::from(bytes)))
                .collect();

            // Call DirectWorker.connect_remote()
            worker.connect_remote(message.instance_id, metadata)?;

            // Return empty response to signal success
            Ok(Some(Bytes::new()))
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for execute_remote_onboard_for_instance - pulls from remote using instance ID
    fn register_execute_remote_onboard_for_instance_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::unary_handler_async(
            "kvbm.worker.remote_onboard_for_instance",
            move |ctx| {
                let worker = worker.clone();
                async move {
                    let message: ExecuteRemoteOnboardForInstanceMessage =
                        serde_json::from_slice(&ctx.payload)?;

                    // Convert options and resolve bounce buffer if present
                    let bounce_buffer_parts = message.options.bounce_buffer_parts();
                    let mut options: TransferOptions = message.options.into();
                    if let Some((handle, block_ids)) = bounce_buffer_parts {
                        options.bounce_buffer =
                            Some(worker.create_bounce_buffer(handle, block_ids)?);
                    }

                    let notification = worker.execute_remote_onboard_for_instance(
                        message.instance_id,
                        message.remote_logical_type,
                        message.src_block_ids,
                        message.dst,
                        Arc::from(message.dst_block_ids),
                        options,
                    )?;

                    notification.await?;
                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    // ========================================================================
    // Object Storage Handlers
    // ========================================================================

    /// Register handler for object_has_blocks - check if blocks exist in object storage
    fn register_object_has_blocks_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler =
            NovaHandler::unary_handler_async("kvbm.worker.object_has_blocks", move |ctx| {
                let worker = worker.clone();

                async move {
                    let message: ObjectHasBlocksMessage = serde_json::from_slice(&ctx.payload)?;

                    // Call DirectWorker's ObjectBlockOps implementation
                    let results = worker.has_blocks(message.keys).await;

                    let response = ObjectHasBlocksResponse { results };
                    Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
                }
            })
            .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for object_put_blocks - upload blocks to object storage
    fn register_object_put_blocks_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler =
            NovaHandler::unary_handler_async("kvbm.worker.object_put_blocks", move |ctx| {
                let worker = worker.clone();

                async move {
                    let message: ObjectPutBlocksMessage = serde_json::from_slice(&ctx.payload)?;

                    // Resolve logical handle to physical layout
                    let layout = worker.resolve_layout(message.layout)?;

                    // Call DirectWorker's ObjectBlockOps implementation
                    let results = worker
                        .put_blocks(message.keys, layout, message.block_ids)
                        .await;

                    let response = ObjectPutGetBlocksResponse::from_results(results);
                    Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
                }
            })
            .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for object_get_blocks - download blocks from object storage
    fn register_object_get_blocks_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler =
            NovaHandler::unary_handler_async("kvbm.worker.object_get_blocks", move |ctx| {
                let worker = worker.clone();

                async move {
                    let message: ObjectGetBlocksMessage = serde_json::from_slice(&ctx.payload)?;

                    // Resolve logical handle to physical layout
                    let layout = worker.resolve_layout(message.layout)?;

                    // Call DirectWorker's ObjectBlockOps implementation
                    let results = worker
                        .get_blocks(message.keys, layout, message.block_ids)
                        .await;

                    let response = ObjectPutGetBlocksResponse::from_results(results);
                    Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
                }
            })
            .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }
}
