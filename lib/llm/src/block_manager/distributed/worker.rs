// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use nixl_sys::Agent as NixlAgent;
use tokio_util::sync::CancellationToken;
use tracing::info;
use validator::Validate;

use crate::block_manager::{
    block::{self, *},
    layout::*,
    state::*,
    storage::{self, *},
    DeviceStorage, KvBlockManagerConfig, NixlOptions, WorkerID,
};

use super::active_message::{
    ActiveMessageFactory, ActiveMessageHandlerFactory, ActiveMessageReceiver, ActiveMessageSender,
    SharedResponseReceiver,
};

pub struct KvBlockManagerWorker {
    worker_id: WorkerID,
    cancellation_token: CancellationToken,

    nixl_agent: NixlAgent,
    nixl_backends: HashMap<String, Arc<nixl_sys::Backend>>,

    disk_layout: Option<Arc<dyn BlockLayout<StorageType = DiskStorage>>>,
    host_layout: Option<Arc<dyn BlockLayout<StorageType = PinnedStorage>>>,
    device_layout: Arc<dyn BlockLayout<StorageType = DeviceStorage>>,

    // Active message handling
    am_receiver: Option<ActiveMessageReceiver>,
    am_sender: Option<ActiveMessageSender>,
}

impl KvBlockManagerWorker {
    pub fn new(config: KvBlockManagerConfig) -> Result<Self> {
        config
            .runtime
            .validate()
            .context("Validating runtime config")?;

        config.model.validate().context("Validating model config")?;

        let worker_id = config.runtime.worker_id;
        let cancellation_token = config.runtime.cancellation_token;

        // Create a map of NIXL backends
        let mut nixl_backends: HashMap<String, Arc<nixl_sys::Backend>> = HashMap::new();

        // Create a NIXL agent if NIXL is enabled and instantiate requested backends
        // TODO: Build a map of NIXL backends to block pools/sets
        let nixl_agent = match config.runtime.nixl {
            NixlOptions::Enabled => {
                tracing::debug!("Creating NIXL agent");
                let agent = NixlAgent::new(&worker_id.to_string())?;

                tracing::debug!("Creating NIXL backends");

                if let Ok((_, ucx_params)) = agent.get_plugin_params("UCX") {
                    let backend = agent.create_backend("UCX", &ucx_params)?;
                    nixl_backends.insert("UCX".to_string(), Arc::new(backend));
                } else {
                    tracing::warn!("No UCX plugin found; will not create UCX backend");
                }

                if config.disk_layout.is_some() {
                    if let Ok((_, gds_params)) = agent.get_plugin_params("GDS") {
                        let backend = agent.create_backend("GDS", &gds_params)?;
                        nixl_backends.insert("GDS".to_string(), Arc::new(backend));
                    } else {
                        tracing::warn!("No GDS plugin found; will not create GDS backend");
                    }
                }

                Some(agent)
            }
            NixlOptions::EnabledWithAgent(agent) => Some(agent),
            NixlOptions::Disabled => None,
        };

        if nixl_agent.is_none() {
            anyhow::bail!("KvBlockManagerWorkers must be configured with NIXL enabled");
        }

        // Initialize model-specific layout config. The layout_builder is incomplete at this point.
        // We will clone this builder and apply the storage-specific configs to each clone in the
        // following steps.
        let model = &config.model;
        let mut layout_builder = LayoutConfig::builder();

        layout_builder
            .num_layers(model.num_layers)
            .outer_dim(model.outer_dim)
            .page_size(model.page_size)
            .inner_dim(model.inner_dim)
            .dtype(model.dtype);

        let mut next_block_set_idx = 0;
        let mut local_block_set = block::nixl::NixlBlockSet::new(worker_id);

        let disk_layout = if let Some(config) = config.disk_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing disk pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            Some(layout)
        } else {
            tracing::debug!("No disk layout provided; will not allocate disk blocks.");
            None
        };

        // Create the host block pool if a host layout is provided
        let host_layout = if let Some(config) = config.host_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing host pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            Some(layout)
        } else {
            tracing::debug!("No host layout provided; will not allocate host blocks.");
            None
        };

        // Create the device block pool if a device layout is provided
        let device_layout = if let Some(config) = config.device_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing device pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            Some(layout)
        } else {
            tracing::debug!("No device layout provided; will not allocate device blocks.");
            None
        };

        // Finalize the local block set by adding NIXL metadata
        if let Some(nixl_agent) = nixl_agent.as_ref() {
            tracing::debug!("Finalize NixlBlockSet: adding NIXL metadata.");
            local_block_set.set_nixl_metadata(nixl_agent.get_local_md()?);
        }

        Ok(Self {
            worker_id,
            cancellation_token,
            nixl_agent: nixl_agent.unwrap(),
            nixl_backends,

            disk_layout: disk_layout
                .map(|l| l as Arc<dyn BlockLayout<StorageType = disk::DiskStorage>>),
            host_layout: host_layout
                .map(|l| l as Arc<dyn BlockLayout<StorageType = storage::cuda::PinnedStorage>>),
            device_layout: device_layout.expect("device layout is required")
                as Arc<dyn BlockLayout<StorageType = storage::cuda::DeviceStorage>>,
            am_receiver: None,
            am_sender: None,
        })
    }

    /// Initialize the active message system with the specified concurrency
    pub fn init_active_message_system(&mut self, concurrency: usize) -> Result<()> {
        let (receiver, sender) =
            ActiveMessageFactory::create(concurrency, self.cancellation_token.clone())?;

        self.am_receiver = Some(receiver);
        self.am_sender = Some(sender);
        Ok(())
    }

    /// Register handlers with the active message receiver
    pub fn register_handlers(
        &mut self,
        handlers: HashMap<String, ActiveMessageHandlerFactory>,
    ) -> Result<()> {
        if let Some(ref mut receiver) = self.am_receiver {
            receiver.register_handlers(handlers)?;
            Ok(())
        } else {
            anyhow::bail!(
                "Active message system not initialized. Call init_active_message_system first."
            );
        }
    }

    /// Register a single handler with the active message receiver
    pub fn register_handler(
        &mut self,
        message_type: String,
        handler: ActiveMessageHandlerFactory,
    ) -> Result<()> {
        if let Some(ref mut receiver) = self.am_receiver {
            receiver.register_handler(message_type, handler)?;
            Ok(())
        } else {
            anyhow::bail!("Active message system not initialized");
        }
    }

    /// Start the active message receiver
    pub fn start_active_message_receiver(&mut self) -> Result<()> {
        if let Some(ref mut receiver) = self.am_receiver {
            receiver.start()?;
            info!(
                "Active message receiver started for worker {}",
                self.worker_id
            );
            Ok(())
        } else {
            anyhow::bail!("Active message system not initialized");
        }
    }

    /// Stop the active message receiver
    pub async fn stop_active_message_receiver(&mut self) -> Result<()> {
        if let Some(ref mut receiver) = self.am_receiver {
            receiver.stop().await?;
            info!(
                "Active message receiver stopped for worker {}",
                self.worker_id
            );
            Ok(())
        } else {
            Ok(()) // Already stopped or never started
        }
    }

    /// Get a sender for active messages
    pub fn get_message_sender(&self) -> Result<ActiveMessageSender> {
        if let Some(ref sender) = self.am_sender {
            Ok(sender.clone())
        } else {
            anyhow::bail!("Active message system not initialized")
        }
    }

    /// Get a receiver for response notifications
    pub fn get_response_receiver(&self) -> Result<SharedResponseReceiver> {
        if let Some(ref receiver) = self.am_receiver {
            Ok(receiver.get_response_receiver())
        } else {
            anyhow::bail!("Active message system not initialized")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("worker_test.rs");
}
