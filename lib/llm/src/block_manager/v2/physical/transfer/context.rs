// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer context.

use std::sync::Arc;

use crate::block_manager::v2::kernels::OperationalCopyBackend;
use crate::block_manager::v2::device::{DeviceBackend, DeviceContext, DeviceStream, DeviceEvent};
use anyhow::Result;
use derive_builder::Builder;
use nixl_sys::XferRequest;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::nixl_agent::{NixlAgent, NixlBackendConfig};

use crate::block_manager::v2::physical::manager::TransportManager;

// Notifications module is declared in ../mod.rs
// Re-export for convenience
use super::TransferCapabilities;
pub use super::notifications;
pub use super::notifications::TransferCompleteNotification;

#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"), public)]
#[allow(dead_code)] // Fields are used in build() but derive macros confuse dead code analysis
pub(crate) struct TransferConfig {
    worker_id: u64,

    /// Optional custom name for the NIXL agent. If not provided, defaults to "worker-{worker_id}"
    #[builder(default = "None", setter(strip_option))]
    nixl_agent_name: Option<String>,

    /// Backend configuration for NIXL backends to enable
    #[builder(default = "NixlBackendConfig::new()")]
    nixl_backend_config: NixlBackendConfig,

    /// Device backend (CUDA, HPU, XPU) - auto-detected if not specified
    #[builder(default = "DeviceBackend::auto_detect().unwrap_or(DeviceBackend::Cuda)")]
    device_backend: DeviceBackend,

    #[builder(default = "0")]
    device_id: u32,

    #[builder(default = "get_tokio_runtime()")]
    tokio_runtime: TokioRuntime,

    #[builder(default = "TransferCapabilities::default()")]
    capabilities: TransferCapabilities,

    #[builder(default = "OperationalCopyBackend::Auto")]
    operational_backend: OperationalCopyBackend,
}

impl TransferConfigBuilder {
    /// Directly provide a pre-configured wrapped NIXL agent (mainly for testing).
    ///
    /// This bypasses the agent creation and backend initialization logic,
    /// using the provided agent directly. Useful for tests that need full
    /// control over agent configuration.
    pub fn nixl_agent(self, agent: NixlAgent) -> TransferConfigBuilderWithAgent {
        TransferConfigBuilderWithAgent {
            builder: self,
            agent,
        }
    }

    /// Add a NIXL backend to enable (uses default plugin parameters).
    pub fn nixl_backend(mut self, backend: impl Into<String>) -> Self {
        let config = self
            .nixl_backend_config
            .get_or_insert_with(NixlBackendConfig::new);
        *config = config.clone().with_backend(backend);
        self
    }

    /// Load NIXL backend configuration from environment variables.
    ///
    /// This merges environment-based configuration with any backends already
    /// configured via the builder.
    pub fn with_env_backends(mut self) -> Result<Self> {
        let env_config = NixlBackendConfig::from_env()?;
        let config = self
            .nixl_backend_config
            .get_or_insert_with(NixlBackendConfig::new);
        *config = config.clone().merge(env_config);
        Ok(self)
    }

    pub fn build(self) -> Result<TransportManager> {
        let mut config = self.build_internal()?;

        // Merge environment backends if not explicitly configured
        if config.nixl_backend_config.backends().is_empty() {
            config.nixl_backend_config = NixlBackendConfig::from_env()?;
        }

        // Derive agent name from worker_id if not provided
        let agent_name = config
            .nixl_agent_name
            .unwrap_or_else(|| format!("worker-{}", config.worker_id));

        // Create wrapped NIXL agent with configured backends
        let backend_names: Vec<&str> = config
            .nixl_backend_config
            .backends()
            .iter()
            .map(|s| s.as_str())
            .collect();

        let nixl_agent = if backend_names.is_empty() {
            // No backends configured - create agent without backends
            NixlAgent::new_with_backends(&agent_name, &[])?
        } else {
            // Create agent with requested backends
            NixlAgent::new_with_backends(&agent_name, &backend_names)?
        };

        let device_context = Arc::new(DeviceContext::new(config.device_backend, config.device_id)?);
        let context = TransferContext::new(
            config.worker_id,
            nixl_agent,
            device_context,
            config.tokio_runtime,
            config.capabilities,
            config.operational_backend,
        )?;
        Ok(TransportManager::from_context(context))
    }
}

/// Builder that already has a pre-configured NIXL agent.
///
/// This is generally used for testing when you want to pass in an agent directly
/// rather than having it created by the builder.
pub struct TransferConfigBuilderWithAgent {
    builder: TransferConfigBuilder,
    agent: NixlAgent,
}

impl TransferConfigBuilderWithAgent {
    /// Build the TransportManager using the pre-configured agent.
    pub fn build(self) -> Result<TransportManager> {
        let config = self.builder.build_internal()?;
        let device_context = Arc::new(DeviceContext::new(config.device_backend, config.device_id)?);
        let context = TransferContext::new(
            config.worker_id,
            self.agent,
            device_context,
            config.tokio_runtime,
            config.capabilities,
            config.operational_backend,
        )?;
        Ok(TransportManager::from_context(context))
    }

    // Proxy methods to allow configuring other builder fields
    pub fn worker_id(mut self, worker_id: u64) -> Self {
        self.builder = self.builder.worker_id(worker_id);
        self
    }

    pub fn device_backend(mut self, device_backend: DeviceBackend) -> Self {
        self.builder = self.builder.device_backend(device_backend);
        self
    }

    pub fn device_id(mut self, device_id: u32) -> Self {
        self.builder = self.builder.device_id(device_id);
        self
    }
}

fn get_tokio_runtime() -> TokioRuntime {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => TokioRuntime::Handle(handle),
        Err(_) => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .max_blocking_threads(4)
                .worker_threads(2)
                .build()
                .expect("failed to build tokio runtime");

            TokioRuntime::Shared(Arc::new(rt))
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum TokioRuntime {
    Handle(tokio::runtime::Handle),
    Shared(Arc<tokio::runtime::Runtime>),
}

impl TokioRuntime {
    pub fn handle(&self) -> &tokio::runtime::Handle {
        match self {
            TokioRuntime::Handle(handle) => handle,
            TokioRuntime::Shared(runtime) => runtime.handle(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransferContext {
    worker_id: u64,
    nixl_agent: NixlAgent,
    #[allow(dead_code)]
    device_context: Arc<DeviceContext>,
    d2h_stream: Arc<DeviceStream>,
    h2d_stream: Arc<DeviceStream>,
    #[allow(dead_code)]
    tokio_runtime: TokioRuntime,
    capabilities: TransferCapabilities,
    operational_backend: OperationalCopyBackend,
    // Channels for background notification handlers
    tx_nixl_status:
        mpsc::Sender<notifications::RegisterPollingNotification<notifications::NixlStatusChecker>>,
    tx_device_event:
        mpsc::Sender<notifications::RegisterPollingNotification<notifications::DeviceEventChecker>>,
    #[allow(dead_code)]
    tx_nixl_events: mpsc::Sender<notifications::RegisterNixlNotification>,
}

impl TransferContext {
    pub fn builder() -> TransferConfigBuilder {
        TransferConfigBuilder::default()
    }

    pub(crate) fn new(
        worker_id: u64,
        nixl_agent: NixlAgent,
        device_context: Arc<DeviceContext>,
        tokio_runtime: TokioRuntime,
        capabilities: TransferCapabilities,
        operational_backend: OperationalCopyBackend,
    ) -> Result<Self> {
        // Disable automatic event tracking (CUDA-specific, no-op for HPU/XPU)
        unsafe { device_context.ops.disable_event_tracking()? };

        // Create channels for background notification handlers
        let (tx_nixl_status, rx_nixl_status) = mpsc::channel(64);
        let (tx_device_event, rx_device_event) = mpsc::channel(64);
        let (tx_nixl_events, rx_nixl_events) = mpsc::channel(64);

        // Spawn background handlers
        let handle = tokio_runtime.handle();

        // Spawn NIXL status polling handler
        handle.spawn(notifications::process_polling_notifications(rx_nixl_status));

        // Spawn device event polling handler (supports CUDA/HPU/XPU)
        handle.spawn(notifications::process_polling_notifications(rx_device_event));

        // Spawn NIXL notification events handler
        handle.spawn(notifications::process_nixl_notification_events(
            nixl_agent.raw_agent().clone(),
            rx_nixl_events,
        ));

        Ok(Self {
            worker_id,
            nixl_agent,
            device_context: device_context.clone(),
            d2h_stream: Arc::new(device_context.create_stream()?),
            h2d_stream: Arc::new(device_context.create_stream()?),
            tokio_runtime,
            capabilities,
            operational_backend,
            tx_nixl_status,
            tx_device_event,
            tx_nixl_events,
        })
    }

    pub(crate) fn nixl_agent(&self) -> &NixlAgent {
        &self.nixl_agent
    }

    #[allow(dead_code)]
    pub(crate) fn device_context(&self) -> &Arc<DeviceContext> {
        &self.device_context
    }

    pub(crate) fn d2h_stream(&self) -> &Arc<DeviceStream> {
        &self.d2h_stream
    }

    pub(crate) fn h2d_stream(&self) -> &Arc<DeviceStream> {
        &self.h2d_stream
    }

    #[allow(dead_code)]
    pub(crate) fn tokio(&self) -> &tokio::runtime::Handle {
        self.tokio_runtime.handle()
    }

    pub(crate) fn capabilities(&self) -> &TransferCapabilities {
        &self.capabilities
    }

    pub(crate) fn operational_backend(&self) -> OperationalCopyBackend {
        self.operational_backend
    }

    /// Register a NIXL transfer request for status polling completion.
    ///
    /// This method enqueues the transfer request to be polled for completion
    /// using `agent.get_xfer_status()`. Returns a notification object that
    /// can be awaited for completion.
    pub(crate) fn register_nixl_status(
        &self,
        xfer_req: XferRequest,
    ) -> TransferCompleteNotification {
        let (done_tx, done_rx) = oneshot::channel();

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::NixlStatusChecker::new(
                self.nixl_agent.raw_agent().clone(),
                xfer_req,
            ),
            done: done_tx,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = self.tx_nixl_status.try_send(notification);

        TransferCompleteNotification { status: done_rx }
    }

    /// Register a device event for polling completion (supports CUDA, HPU, XPU).
    ///
    /// This method enqueues the device event to be polled for completion.
    /// Returns a notification object that can be awaited for completion.
    pub(crate) fn register_device_event(&self, event: DeviceEvent) -> TransferCompleteNotification {
        let (done_tx, done_rx) = oneshot::channel();

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::DeviceEventChecker::new(event),
            done: done_tx,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = self.tx_device_event.try_send(notification);

        TransferCompleteNotification { status: done_rx }
    }

    /// Register a NIXL transfer request for notification-based completion.
    ///
    /// This method enqueues the transfer request to be completed via NIXL
    /// notification events. Returns a notification object that can be awaited
    /// for completion.
    #[allow(dead_code)]
    pub(crate) fn register_nixl_event(
        &self,
        xfer_req: XferRequest,
    ) -> TransferCompleteNotification {
        let (done_tx, done_rx) = oneshot::channel();

        let notification = notifications::RegisterNixlNotification {
            uuid: Uuid::new_v4(),
            xfer_req,
            done: done_tx,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = self.tx_nixl_events.try_send(notification);

        TransferCompleteNotification { status: done_rx }
    }

    /// Get the worker ID for this context.
    pub(crate) fn worker_id(&self) -> u64 {
        self.worker_id
    }
}
