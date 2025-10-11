// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl Resources {
    /// Create a new [`Resources`] instance
    pub fn new(config: KvBlockManagerConfig) -> Result<Self> {
        config
            .runtime
            .validate()
            .context("Validating runtime config")?;

        config.model.validate().context("Validating model config")?;

        let worker_id = config.runtime.worker_id;
        let cancellation_token = config.runtime.cancellation_token.clone();

        let global_registry = GlobalRegistry::default();

        let event_manager = config
            .event_manager
            .clone()
            .unwrap_or_else(|| NullEventManager::new());

        // Create a NIXL agent if NIXL is enabled and instantiate requested backends
        // TODO: Build a map of NIXL backends to block pools/sets

        let mut nixl_backends: HashMap<String, Arc<nixl_sys::Backend>> = HashMap::new();

        let nixl_agent = Arc::new(match &config.runtime.nixl {
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
                    if let Ok((_, gds_mt_params)) = agent.get_plugin_params("GDS_MT") {
                        let backend = agent.create_backend("GDS_MT", &gds_mt_params)?;
                        nixl_backends.insert("GDS_MT".to_string(), Arc::new(backend));
                    } else {
                        tracing::warn!("No GDS_MT plugin found; will not create GDS_MT backend");
                    }
                }

                Some(agent)
            }
            NixlOptions::EnabledWithAgent(agent) => Some(agent.clone()),
            NixlOptions::Disabled => None,
        });

        let async_rt_handle = match &config.runtime.async_runtime {
            Some(rt) => rt.handle().clone(),
            None => match Handle::try_current() {
                Ok(handle) => handle,
                Err(e) => anyhow::bail!(e),
            },
        };

        Ok(Self {
            worker_id,
            cancellation_token,
            async_rt_handle,
            nixl_agent,
            nixl_backends,
            global_registry,
            event_manager,
            config,
        })
    }

    /// Create a new [`LayoutConfigBuilder`] with the model configuration
    pub fn layout_builder(&self) -> LayoutConfigBuilder {
        let mut layout_builder = LayoutConfig::builder();

        let model = &self.config.model;

        layout_builder
            .num_layers(model.num_layers)
            .outer_dim(model.outer_dim)
            .page_size(model.page_size)
            .inner_dim(model.inner_dim)
            .dtype_width_bytes(model.dtype_width_bytes);

        layout_builder
    }
}
