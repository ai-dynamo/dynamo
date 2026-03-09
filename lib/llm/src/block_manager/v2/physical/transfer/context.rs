// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer context.
//!
//! Supports dual-backend operation: CUDA (NVIDIA GPUs) and Level Zero (Intel XPU).
//! Either or both backends may be active depending on the hardware available.

use std::sync::Arc;

use crate::block_manager::v2::kernels::OperationalCopyBackend;
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
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

    #[builder(default = "Some(0)")]
    cuda_device_id: Option<usize>,

    /// Level Zero device ordinal for XPU transfers.
    /// Set to Some(id) to enable the Level Zero transfer backend.
    #[builder(default = "None", setter(strip_option))]
    #[cfg(feature = "level-zero")]
    xpu_device_id: Option<u32>,

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

    /// Disable the CUDA backend entirely.
    pub fn no_cuda(mut self) -> Self {
        self.cuda_device_id = Some(None);
        self
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

        // Create CUDA context if a CUDA device was requested
        let cuda_context = match config.cuda_device_id {
            Some(device_id) => Some(CudaContext::new(device_id)?),
            None => None,
        };

        let context = TransferContext::new(
            config.worker_id,
            nixl_agent,
            cuda_context,
            config.tokio_runtime,
            config.capabilities,
            config.operational_backend,
            #[cfg(feature = "level-zero")]
            config.xpu_device_id,
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

        // Create CUDA context if a CUDA device was requested
        let cuda_context = match config.cuda_device_id {
            Some(device_id) => Some(CudaContext::new(device_id)?),
            None => None,
        };

        let context = TransferContext::new(
            config.worker_id,
            self.agent,
            cuda_context,
            config.tokio_runtime,
            config.capabilities,
            config.operational_backend,
            #[cfg(feature = "level-zero")]
            config.xpu_device_id,
        )?;
        Ok(TransportManager::from_context(context))
    }

    // Proxy methods to allow configuring other builder fields
    pub fn worker_id(mut self, worker_id: u64) -> Self {
        self.builder = self.builder.worker_id(worker_id);
        self
    }

    pub fn cuda_device_id(mut self, cuda_device_id: usize) -> Self {
        self.builder = self.builder.cuda_device_id(Some(cuda_device_id));
        self
    }

    /// Disable the CUDA backend entirely.
    pub fn no_cuda(mut self) -> Self {
        self.builder = self.builder.no_cuda();
        self
    }

    /// Set the Level Zero device ordinal for XPU transfers.
    #[cfg(feature = "level-zero")]
    pub fn xpu_device_id(mut self, xpu_device_id: u32) -> Self {
        self.builder = self.builder.xpu_device_id(xpu_device_id);
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

    // -- CUDA backend (optional) --
    #[allow(dead_code)]
    cuda_context: Option<Arc<CudaContext>>,
    d2h_stream: Option<Arc<CudaStream>>,
    h2d_stream: Option<Arc<CudaStream>>,

    // -- Level Zero backend (optional) --
    #[cfg(feature = "level-zero")]
    #[allow(dead_code)]
    xpu_device_id: Option<u32>,
    #[cfg(feature = "level-zero")]
    tx_ze_event:
        Option<mpsc::Sender<notifications::RegisterPollingNotification<notifications::ZeEventChecker>>>,

    // -- Shared infrastructure --
    #[allow(dead_code)]
    tokio_runtime: TokioRuntime,
    capabilities: TransferCapabilities,
    operational_backend: OperationalCopyBackend,
    // Channels for background notification handlers
    tx_nixl_status:
        mpsc::Sender<notifications::RegisterPollingNotification<notifications::NixlStatusChecker>>,
    tx_cuda_event:
        Option<mpsc::Sender<notifications::RegisterPollingNotification<notifications::CudaEventChecker>>>,
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
        cuda_context: Option<Arc<CudaContext>>,
        tokio_runtime: TokioRuntime,
        capabilities: TransferCapabilities,
        operational_backend: OperationalCopyBackend,
        #[cfg(feature = "level-zero")] xpu_device_id: Option<u32>,
    ) -> Result<Self> {
        // Spawn background handlers on the tokio runtime
        let handle = tokio_runtime.handle();

        // -- NIXL channels (always created) --
        let (tx_nixl_status, rx_nixl_status) = mpsc::channel(64);
        let (tx_nixl_events, rx_nixl_events) = mpsc::channel(64);

        handle.spawn(notifications::process_polling_notifications(rx_nixl_status));

        handle.spawn(notifications::process_nixl_notification_events(
            nixl_agent.raw_agent().clone(),
            rx_nixl_events,
        ));

        // -- CUDA backend setup (optional) --
        let (tx_cuda_event, d2h_stream, h2d_stream) = if let Some(ref ctx) = cuda_context {
            unsafe { ctx.disable_event_tracking() };

            // CUDA event polling channel
            let (tx, rx) = mpsc::channel(64);
            handle.spawn(notifications::process_polling_notifications(rx));

            let d2h = ctx.new_stream()?;
            let h2d = ctx.new_stream()?;

            (Some(tx), Some(d2h), Some(h2d))
        } else {
            (None, None, None)
        };

        // -- Level Zero backend setup (optional) --
        #[cfg(feature = "level-zero")]
        let tx_ze_event = if xpu_device_id.is_some() {
            let (tx, rx) = mpsc::channel(64);
            handle.spawn(notifications::process_polling_notifications(rx));
            Some(tx)
        } else {
            None
        };

        Ok(Self {
            worker_id,
            nixl_agent,
            cuda_context,
            d2h_stream,
            h2d_stream,
            #[cfg(feature = "level-zero")]
            xpu_device_id,
            #[cfg(feature = "level-zero")]
            tx_ze_event,
            tokio_runtime,
            capabilities,
            operational_backend,
            tx_nixl_status,
            tx_cuda_event,
            tx_nixl_events,
        })
    }

    // ---- Shared accessors ----

    pub(crate) fn nixl_agent(&self) -> &NixlAgent {
        &self.nixl_agent
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

    /// Get the worker ID for this context.
    pub(crate) fn worker_id(&self) -> u64 {
        self.worker_id
    }

    // ---- CUDA accessors ----

    #[allow(dead_code)]
    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        self.cuda_context
            .as_ref()
            .expect("cuda_context() called but CUDA backend is not initialised")
    }

    pub(crate) fn d2h_stream(&self) -> &Arc<CudaStream> {
        self.d2h_stream
            .as_ref()
            .expect("d2h_stream() called but CUDA backend is not initialised")
    }

    pub(crate) fn h2d_stream(&self) -> &Arc<CudaStream> {
        self.h2d_stream
            .as_ref()
            .expect("h2d_stream() called but CUDA backend is not initialised")
    }

    // ---- NIXL notification registration ----

    /// Register a NIXL transfer request for status polling completion.
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

    // ---- CUDA event notification ----

    /// Register a CUDA event for polling completion.
    pub(crate) fn register_cuda_event(&self, event: CudaEvent) -> TransferCompleteNotification {
        let tx = self
            .tx_cuda_event
            .as_ref()
            .expect("register_cuda_event() called but CUDA backend is not initialised");

        let (done_tx, done_rx) = oneshot::channel();

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::CudaEventChecker::new(event),
            done: done_tx,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = tx.try_send(notification);

        TransferCompleteNotification { status: done_rx }
    }

    // ---- Level Zero event notification ----

    /// Register a Level Zero event for polling completion.
    #[cfg(feature = "level-zero")]
    pub(crate) fn register_ze_event(
        &self,
        event: syclrc::ZeEvent,
    ) -> TransferCompleteNotification {
        let tx = self
            .tx_ze_event
            .as_ref()
            .expect("register_ze_event() called but Level Zero backend is not initialised");

        let (done_tx, done_rx) = oneshot::channel();

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::ZeEventChecker::new(event),
            done: done_tx,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = tx.try_send(notification);

        TransferCompleteNotification { status: done_rx }
    }

    // ---- NIXL notification events ----

    /// Register a NIXL transfer request for notification-based completion.
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
}
