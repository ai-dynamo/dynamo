// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Arc, DirectWorker, LocalTransferMessage, RemoteOffloadMessage, RemoteOnboardMessage, Result,
    TransferOptions, Worker,
};

use dynamo_nova::Nova;
use dynamo_nova::am::NovaHandler;

pub struct NovaWorkerService {
    direct_worker: Arc<DirectWorker>,
    nova: Arc<Nova>,
}

impl NovaWorkerService {
    /// Create a new NovaWorkerService and register all handlers
    pub fn new(direct_worker: Arc<DirectWorker>, nova: Arc<Nova>) -> Result<Self> {
        let service = Self {
            direct_worker,
            nova,
        };

        service.register_handlers()?;
        Ok(service)
    }

    /// Register all worker handlers with Nova
    fn register_handlers(&self) -> Result<()> {
        self.register_local_transfer_handler()?;
        self.register_remote_onboard_handler()?;
        self.register_remote_offload_handler()?;
        Ok(())
    }

    fn register_local_transfer_handler(&self) -> Result<()> {
        let direct_worker = self.direct_worker.clone();

        let handler = NovaHandler::am_handler_async("kvbm.worker.local_transfer", move |ctx| {
            let direct_worker = direct_worker.clone();

            async move {
                // Deserialize the message
                let message: LocalTransferMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(direct_worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = direct_worker.execute_local_transfer(
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
        let direct_worker = self.direct_worker.clone();

        let handler = NovaHandler::am_handler_async("kvbm.worker.remote_onboard", move |ctx| {
            let direct_worker = direct_worker.clone();

            async move {
                // Deserialize the message
                let message: RemoteOnboardMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(direct_worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = direct_worker.execute_remote_onboard(
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
        let direct_worker = self.direct_worker.clone();

        let handler = NovaHandler::am_handler_async("kvbm.worker.remote_offload", move |ctx| {
            let direct_worker = direct_worker.clone();

            async move {
                // Deserialize the message
                let message: RemoteOffloadMessage = serde_json::from_slice(&ctx.payload)?;

                // Convert options and resolve bounce buffer if present
                let bounce_buffer_parts = message.options.bounce_buffer_parts();
                let mut options: TransferOptions = message.options.into();
                if let Some((handle, block_ids)) = bounce_buffer_parts {
                    options.bounce_buffer = Some(direct_worker.create_bounce_buffer(handle, block_ids)?);
                }

                let notification = direct_worker.execute_remote_offload(
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
}
