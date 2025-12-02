// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::physical::manager::SerializedLayout;

use super::{
    Arc, DirectWorker, LocalTransferMessage, RemoteOffloadMessage, RemoteOnboardMessage, Result,
    TransferOptions, WorkerTransfers,
};

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
        Ok(())
    }

    fn register_local_transfer_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::am_handler_async("kvbm.worker.local_transfer", move |ctx| {
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

                Ok(())
            }
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    fn register_remote_onboard_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::am_handler_async("kvbm.worker.remote_onboard", move |ctx| {
            let worker = worker.clone();

            async move {
                // Deserialize the message
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

                // Await the transfer completion
                notification.await?;

                Ok(())
            }
        })
        .build();

        self.nova.register_handler(handler)?;
        Ok(())
    }

    fn register_remote_offload_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = NovaHandler::am_handler_async("kvbm.worker.remote_offload", move |ctx| {
            let worker = worker.clone();

            async move {
                // Deserialize the message
                let message: RemoteOffloadMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = worker.execute_remote_offload(
                    message.src,
                    message.dst,
                    Arc::from(message.src_block_ids),
                    options,
                )?;

                // Await the transfer completion
                notification.await?;

                Ok(())
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

    // /// Handler: "kvbm.worker.get_layout_config"
    // ///
    // /// Returns the current G1 layout config for validation by the leader.
    // /// The leader gathers this from all workers and validates they match.
    // fn register_get_layout_config_handler(&self) -> Result<()> {
    //     let worker = self.worker.clone();

    //     let handler = NovaHandler::unary_handler("kvbm.worker.get_layout_config", move |_ctx| {
    //         let config = worker.export_layout_config()?;
    //         Ok(Some(Bytes::from(serde_json::to_vec(&config)?)))
    //     })
    //     .build();

    //     self.nova.register_handler(handler)?;
    //     Ok(())
    // }

    // /// Handler: "kvbm.worker.configure_layouts"
    // ///
    // /// Creates G2/G3 layouts based on leader-provided configuration.
    // /// Called during Phase 3 coordination after the leader validates layout configs.
    // fn register_configure_layouts_handler(&self) -> Result<()> {
    //     let worker = self.worker.clone();

    //     let handler = NovaHandler::typed_unary_async("kvbm.worker.configure_layouts", move |ctx| {
    //         let worker = worker.clone();

    //         async move {
    //             let config: LeaderLayoutConfig = ctx.input;
    //             let response = worker.configure_additional_layouts(config)?;
    //             Ok(response)
    //         }
    //     })
    //     .build();

    //     self.nova.register_handler(handler)?;
    //     Ok(())
    // }
}
