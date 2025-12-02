// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer context.

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use derive_builder::Builder;
use kvbm_kernels::OperationalCopyBackend;
use tokio::sync::mpsc;
use uuid::Uuid;

use dynamo_memory::nixl::{NixlAgent, NixlBackendConfig, XferRequest};
use dynamo_nova::events::LocalEventSystem;

use crate::v2::physical::manager::TransferManager;

// Notifications module is declared in ../mod.rs
// Re-export for convenience
use super::TransferCapabilities;
use notifications::RegisterPollingNotification;

pub use super::notifications;
pub use super::notifications::TransferCompleteNotification;

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"), public)]
#[allow(dead_code)] // Fields are used in build() but derive macros confuse dead code analysis
pub struct TransferConfig {
    #[builder(default = "LocalEventSystem::new_local_only()")]
    event_system: Arc<LocalEventSystem>,

    /// Optional custom name for the NIXL agent. If not provided, defaults to "worker-{worker_id}"
    #[builder(default = "None", setter(strip_option))]
    nixl_agent_name: Option<String>,

    /// Backend configuration for NIXL backends to enable
    #[builder(default = "NixlBackendConfig::default()")]
    nixl_backend_config: NixlBackendConfig,

    #[builder(default = "0")]
    cuda_device_id: usize,

    #[builder(default = "get_tokio_runtime()")]
    tokio_runtime: TokioRuntime,

    #[builder(default = "TransferCapabilities::default()")]
    capabilities: TransferCapabilities,

    #[builder(default = "OperationalCopyBackend::Auto")]
    operational_backend: OperationalCopyBackend,
}

impl TransferConfigBuilder {
    /// Initialize builder with components from KvbmRuntime.
    ///
    /// This extracts the event_system and tokio runtime handle from the runtime,
    /// ensuring consistency with Nova's event system. Use this when the runtime
    /// has already been constructed and you want components to share the same
    /// event notification infrastructure.
    ///
    /// # Example
    /// ```ignore
    /// let runtime = KvbmRuntime::from_env_leader().await?;
    /// let transfer_mgr = TransferConfigBuilder::default()
    ///     .from_runtime(&runtime)
    ///     .cuda_device_id(0)
    ///     .build()?;
    /// ```
    pub fn from_runtime(self, runtime: &crate::v2::KvbmRuntime) -> Self {
        self.event_system(runtime.event_system())
            .tokio_runtime(TokioRuntime::Handle(runtime.handle()))
    }

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
            .get_or_insert_with(NixlBackendConfig::default);
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
            .get_or_insert_with(NixlBackendConfig::default);
        *config = config.clone().merge(env_config);
        Ok(self)
    }

    pub fn build(self) -> Result<TransferManager> {
        let mut config = self.build_internal()?;

        let worker_id = config.event_system.worker_id();

        // Merge environment backends if not explicitly configured
        if config.nixl_backend_config.backends().is_empty() {
            config.nixl_backend_config = NixlBackendConfig::from_env()?;
        }

        // Derive agent name from worker_id if not provided
        let agent_name = config
            .nixl_agent_name
            .unwrap_or_else(|| format!("worker-{}", worker_id));

        let nixl_agent =
            NixlAgent::from_nixl_backend_config(&agent_name, config.nixl_backend_config)?;

        let cuda_context = CudaContext::new(config.cuda_device_id)?;
        let context = TransferContext::new(
            nixl_agent,
            config.event_system,
            cuda_context,
            config.tokio_runtime,
            config.capabilities,
            config.operational_backend,
        )?;
        Ok(TransferManager::from_context(context))
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
    /// Build the TransferManager using the pre-configured agent.
    pub fn build(self) -> Result<TransferManager> {
        let config = self.builder.build_internal()?;
        let cuda_context = CudaContext::new(config.cuda_device_id)?;
        let context = TransferContext::new(
            self.agent,
            config.event_system,
            cuda_context,
            config.tokio_runtime,
            config.capabilities,
            config.operational_backend,
        )?;
        Ok(TransferManager::from_context(context))
    }

    pub fn cuda_device_id(mut self, cuda_device_id: usize) -> Self {
        self.builder = self.builder.cuda_device_id(cuda_device_id);
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

#[derive(Clone)]
pub(crate) struct TransferContext {
    worker_id: u64,
    nixl_agent: NixlAgent,
    #[allow(dead_code)]
    cuda_context: Arc<CudaContext>,
    d2h_stream: Arc<CudaStream>,
    h2d_stream: Arc<CudaStream>,
    #[allow(dead_code)]
    tokio_runtime: TokioRuntime,
    capabilities: TransferCapabilities,
    operational_backend: OperationalCopyBackend,
    event_system: Arc<LocalEventSystem>,
    // Channels for background notification handlers
    tx_nixl_status: mpsc::Sender<RegisterPollingNotification<notifications::NixlStatusChecker>>,
    tx_cuda_event: mpsc::Sender<RegisterPollingNotification<notifications::CudaEventChecker>>,
    #[allow(dead_code)]
    tx_nixl_events: mpsc::Sender<notifications::RegisterNixlNotification>,
}

impl TransferContext {
    pub fn builder() -> TransferConfigBuilder {
        TransferConfigBuilder::default()
    }

    pub(crate) fn new(
        nixl_agent: NixlAgent,
        event_system: Arc<LocalEventSystem>,
        cuda_context: Arc<CudaContext>,
        tokio_runtime: TokioRuntime,
        capabilities: TransferCapabilities,
        operational_backend: OperationalCopyBackend,
    ) -> Result<Self> {
        unsafe { cuda_context.disable_event_tracking() };

        // Create channels for background notification handlers
        let (tx_nixl_status, rx_nixl_status) = mpsc::channel(64);
        let (tx_cuda_event, rx_cuda_event) = mpsc::channel(64);
        let (tx_nixl_events, rx_nixl_events) = mpsc::channel(64);

        // Spawn background handlers
        let handle = tokio_runtime.handle();

        // Spawn NIXL status polling handler
        handle.spawn(notifications::process_polling_notifications(
            rx_nixl_status,
            event_system.clone(),
        ));

        // Spawn CUDA event polling handler
        handle.spawn(notifications::process_polling_notifications(
            rx_cuda_event,
            event_system.clone(),
        ));

        // Spawn NIXL notification events handler
        handle.spawn(notifications::process_nixl_notification_events(
            nixl_agent.raw_agent().clone(),
            rx_nixl_events,
            event_system.clone(),
        ));

        Ok(Self {
            worker_id: event_system.worker_id(),
            nixl_agent,
            cuda_context: cuda_context.clone(),
            d2h_stream: cuda_context.new_stream()?,
            h2d_stream: cuda_context.new_stream()?,
            tokio_runtime,
            capabilities,
            operational_backend,
            event_system,
            tx_nixl_status,
            tx_cuda_event,
            tx_nixl_events,
        })
    }

    pub(crate) fn nixl_agent(&self) -> &NixlAgent {
        &self.nixl_agent
    }

    #[allow(dead_code)]
    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    pub(crate) fn d2h_stream(&self) -> &Arc<CudaStream> {
        &self.d2h_stream
    }

    pub(crate) fn h2d_stream(&self) -> &Arc<CudaStream> {
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

    pub(crate) fn event_system(&self) -> &Arc<LocalEventSystem> {
        &self.event_system
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
        let event = self
            .event_system
            .new_event()
            .expect("Failed to allocate event");
        let handle = event.handle();
        let awaiter = self
            .event_system
            .awaiter(handle)
            .expect("Failed to get awaiter");

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::NixlStatusChecker::new(
                self.nixl_agent.raw_agent().clone(),
                xfer_req,
            ),
            event_handle: handle,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = self.tx_nixl_status.try_send(notification);

        TransferCompleteNotification::from_awaiter(awaiter)
    }

    /// Register a CUDA event for polling completion.
    ///
    /// This method enqueues the CUDA event to be polled for completion.
    /// Returns a notification object that can be awaited for completion.
    pub(crate) fn register_cuda_event(&self, event: CudaEvent) -> TransferCompleteNotification {
        let new_event = self
            .event_system
            .new_event()
            .expect("Failed to allocate event");
        let handle = new_event.handle();
        let awaiter = self
            .event_system
            .awaiter(handle)
            .expect("Failed to get awaiter");

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::CudaEventChecker::new(event),
            event_handle: handle,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = self.tx_cuda_event.try_send(notification);

        TransferCompleteNotification::from_awaiter(awaiter)
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
        let event = self
            .event_system
            .new_event()
            .expect("Failed to allocate event");
        let handle = event.handle();
        let awaiter = self
            .event_system
            .awaiter(handle)
            .expect("Failed to get awaiter");

        let notification = notifications::RegisterNixlNotification {
            uuid: Uuid::new_v4(),
            xfer_req,
            event_handle: handle,
        };

        // Send to background handler (ignore error if receiver dropped)
        let _ = self.tx_nixl_events.try_send(notification);

        TransferCompleteNotification::from_awaiter(awaiter)
    }

    /// Get the worker ID for this context.
    pub(crate) fn worker_id(&self) -> u64 {
        self.worker_id
    }
}
