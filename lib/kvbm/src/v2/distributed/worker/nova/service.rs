// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::physical::{manager::SerializedLayout, transfer::context::TransferConfigBuilder};

use super::{
    Arc, DirectWorker, LocalTransferMessage, RemoteOffloadMessage, RemoteOnboardMessage, Result,
    TransferOptions, Worker,
};

use bytes::Bytes;
use dynamo_nova::Nova;
use dynamo_nova::am::NovaHandler;

pub struct NovaWorkerService {
    nova: Arc<Nova>,
    worker: Arc<DirectWorker>,
}

impl NovaWorkerService {
    // todo: this is where we need to expand the options with a builder/figment to pass configuration parameters to the
    // TransferConfigBuilder produced from TransferManager::builder().
    /// Create a new NovaWorkerService and register all handlers
    pub fn new(nova: Arc<Nova>, transfer_builder: TransferConfigBuilder) -> Result<Self> {
        let transport_manager = transfer_builder
            .event_system(nova.events().local().clone())
            .build()?;
        let worker = Arc::new(DirectWorker::new(transport_manager));
        let service = Self { nova, worker };

        service.register_handlers()?;
        Ok(service)
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
                    message.src_block_ids,
                    message.dst_block_ids,
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
                    message.dst_block_ids,
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
                    message.src_block_ids,
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
}
