// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer context.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};


use anyhow::Result;
use derive_builder::Builder;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::device::{DeviceBackend, DeviceContext, DeviceStream, DeviceEvent, DeviceMemPool, EngineHint};
use dynamo_memory::nixl::{NixlAgent, NixlBackendConfig, XferRequest};
use velo::EventManager;

use crate::manager::TransferManager;

// Notifications module is declared in ../mod.rs
// Re-export for convenience
use super::TransferCapabilities;
use notifications::RegisterPollingNotification;

pub(crate) use super::notifications;
pub use super::notifications::TransferCompleteNotification;

#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"), public)]
#[allow(dead_code)] // Fields are used in build() but derive macros confuse dead code analysis
pub struct TransferConfig {
    #[builder(default = "Arc::new(EventManager::local())")]
    event_system: Arc<EventManager>,

    /// Optional custom name for the NIXL agent. If not provided, defaults to "worker-{worker_id}"
    #[builder(default = "None", setter(strip_option))]
    nixl_agent_name: Option<String>,

    /// Backend configuration for NIXL backends to enable
    #[builder(default = "NixlBackendConfig::default()")]
    nixl_backend_config: NixlBackendConfig,

    /// Device ID for the compute backend (default: 0)
    #[builder(default = "0")]
    device_id: usize,

    /// Device backend type (default: Cuda)
    #[builder(default = "DeviceBackend::Cuda")]
    device_backend: DeviceBackend,

    #[builder(default = "get_tokio_runtime()")]
    tokio_runtime: TokioRuntime,

    #[builder(default = "TransferCapabilities::default()")]
    capabilities: TransferCapabilities,

    /// Size in bytes to pre-allocate for the device memory pool (default: 64 MiB)
    #[builder(default = "64 * 1024 * 1024")]
    pool_reserve_size: usize,

    /// Release threshold for the device memory pool (default: Some(64 MiB))
    /// Memory above this threshold is returned to the system when freed.
    /// If None, no release threshold is set.
    #[builder(default = "Some(64 * 1024 * 1024)")]
    pool_release_threshold: Option<u64>,

    /// Number of device streams per direction for round-robin concurrency (default: 4)
    #[builder(default = "4")]
    num_streams: usize,
}

impl TransferConfigBuilder {
    /// Initialize builder with event system and tokio handle.
    ///
    /// This sets the event_system and tokio runtime handle, ensuring consistency
    /// with Nova's event system. Use this when the runtime has already been
    /// constructed and you want components to share the same event notification
    /// infrastructure.
    pub fn from_event_system_and_handle(
        self,
        event_system: Arc<EventManager>,
        handle: tokio::runtime::Handle,
    ) -> Self {
        self.event_system(event_system)
            .tokio_runtime(TokioRuntime::Handle(handle))
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

        let worker_id = config.event_system.system_id();

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

        let context = TransferContext::new(
            nixl_agent,
            config.event_system,
            config.device_backend,
            config.device_id as u32,
            config.tokio_runtime,
            config.capabilities,
            config.pool_reserve_size,
            config.pool_release_threshold,
            config.num_streams,
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
        let context = TransferContext::new(
            self.agent,
            config.event_system,
            config.device_backend,
            config.device_id as u32,
            config.tokio_runtime,
            config.capabilities,
            config.pool_reserve_size,
            config.pool_release_threshold,
            config.num_streams,
        )?;
        Ok(TransferManager::from_context(context))
    }

    pub fn device_id(mut self, device_id: usize) -> Self {
        self.builder = self.builder.device_id(device_id);
        self
    }

    pub fn device_backend(mut self, backend: DeviceBackend) -> Self {
        self.builder = self.builder.device_backend(backend);
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
#[doc(hidden)]
pub enum TokioRuntime {
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
#[doc(hidden)]
pub struct TransferContext {
    worker_id: u64,
    nixl_agent: NixlAgent,
    #[allow(dead_code)]
    tokio_runtime: TokioRuntime,
    capabilities: TransferCapabilities,
    event_system: Arc<EventManager>,
    // Device abstraction context and streams (multi-backend)
    #[allow(dead_code)]
    device_context: Arc<DeviceContext>,
    /// Copy-engine stream pools for whole-block DMA transfers (batch_copy).
    copy_h2d_streams: Vec<Arc<DeviceStream>>,
    copy_d2h_streams: Vec<Arc<DeviceStream>>,
    current_copy_h2d_stream: Arc<AtomicUsize>,
    current_copy_d2h_stream: Arc<AtomicUsize>,
    /// Compute-engine stream pools for per-chunk kernel transfers (vectorized_copy).
    compute_h2d_streams: Vec<Arc<DeviceStream>>,
    compute_d2h_streams: Vec<Arc<DeviceStream>>,
    current_compute_h2d_stream: Arc<AtomicUsize>,
    current_compute_d2h_stream: Arc<AtomicUsize>,
    // Device memory pool for kernel allocations (multi-backend)
    #[allow(dead_code)]
    device_pool: Arc<DeviceMemPool>,
    // Channels for background notification handlers
    tx_nixl_status: mpsc::Sender<RegisterPollingNotification<notifications::NixlStatusChecker>>,
    tx_device_event: mpsc::Sender<RegisterPollingNotification<notifications::DeviceEventChecker>>,
    #[allow(dead_code)]
    tx_nixl_events: mpsc::Sender<notifications::RegisterNixlNotification>,
}

impl TransferContext {
    pub fn builder() -> TransferConfigBuilder {
        TransferConfigBuilder::default()
    }

    pub(crate) fn new(
        nixl_agent: NixlAgent,
        event_system: Arc<EventManager>,
        device_backend: DeviceBackend,
        device_id: u32,
        tokio_runtime: TokioRuntime,
        capabilities: TransferCapabilities,
        pool_reserve_size: usize,
        pool_release_threshold: Option<u64>,
        num_streams: usize,
    ) -> Result<Self> {
        // Create device context for the specified backend
        let device_ctx = DeviceContext::new(device_backend, device_id)?;

        // Disable event tracking (no-op on non-CUDA backends)
        unsafe { device_ctx.ops.disable_event_tracking()? };

        // Create device memory pool
        let device_pool = Arc::new(
            device_ctx.create_memory_pool(pool_reserve_size, pool_release_threshold)?
        );

        // Create device stream pools (num_streams per engine×direction, round-robin)
        let num_streams = num_streams.max(1); // Ensure at least 1 stream

        // Copy-engine pools: whole-block batch_copy → BCS on ZE, regular on CUDA
        let copy_h2d_streams: Vec<Arc<DeviceStream>> = (0..num_streams)
            .map(|_| device_ctx.create_stream(EngineHint::Copy).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let copy_d2h_streams: Vec<Arc<DeviceStream>> = (0..num_streams)
            .map(|_| device_ctx.create_stream(EngineHint::Copy).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;

        // Compute-engine pools: fc_lw vectorized_copy → CCS on ZE, regular on CUDA
        let compute_h2d_streams: Vec<Arc<DeviceStream>> = (0..num_streams)
            .map(|_| device_ctx.create_stream(EngineHint::Compute).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let compute_d2h_streams: Vec<Arc<DeviceStream>> = (0..num_streams)
            .map(|_| device_ctx.create_stream(EngineHint::Compute).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;

        // Create channels for background notification handlers
        let (tx_nixl_status, rx_nixl_status) = mpsc::channel(64);
        let (tx_device_event, rx_device_event) = mpsc::channel(64);
        let (tx_nixl_events, rx_nixl_events) = mpsc::channel(64);

        // Spawn background handlers
        let handle = tokio_runtime.handle();

        // Spawn NIXL status polling handler
        handle.spawn(notifications::process_polling_notifications(
            rx_nixl_status,
            event_system.clone(),
        ));

        // Spawn device event polling handler (multi-backend)
        handle.spawn(notifications::process_polling_notifications(
            rx_device_event,
            event_system.clone(),
        ));

        // Spawn NIXL notification events handler
        handle.spawn(notifications::process_nixl_notification_events(
            nixl_agent.raw_agent().clone(),
            rx_nixl_events,
            event_system.clone(),
        ));

        Ok(Self {
            worker_id: event_system.system_id(),
            nixl_agent,
            tokio_runtime,
            capabilities,
            event_system,
            device_context: Arc::new(device_ctx),
            copy_h2d_streams,
            copy_d2h_streams,
            current_copy_h2d_stream: Arc::new(AtomicUsize::new(0)),
            current_copy_d2h_stream: Arc::new(AtomicUsize::new(0)),
            compute_h2d_streams,
            compute_d2h_streams,
            current_compute_h2d_stream: Arc::new(AtomicUsize::new(0)),
            current_compute_d2h_stream: Arc::new(AtomicUsize::new(0)),
            device_pool,
            tx_nixl_status,
            tx_device_event,
            tx_nixl_events,
        })
    }

    pub(crate) fn nixl_agent(&self) -> &NixlAgent {
        &self.nixl_agent
    }

    /// Get the device context (multi-backend).
    #[allow(dead_code)]
    pub(crate) fn device_context(&self) -> &Arc<DeviceContext> {
        &self.device_context
    }



    #[allow(dead_code)]
    #[doc(hidden)]
    pub fn tokio(&self) -> &tokio::runtime::Handle {
        self.tokio_runtime.handle()
    }

    pub(crate) fn capabilities(&self) -> &TransferCapabilities {
        &self.capabilities
    }

    #[doc(hidden)]
    pub fn event_system(&self) -> &Arc<EventManager> {
        &self.event_system
    }

    /// Get the device memory pool for kernel allocations (multi-backend).
    #[allow(dead_code)]
    pub(crate) fn device_pool(&self) -> &Arc<DeviceMemPool> {
        &self.device_pool
    }

    /// Get next copy-engine H2D stream (whole-block DMA).
    pub(crate) fn next_copy_h2d_stream(&self) -> Arc<DeviceStream> {
        let idx = self.current_copy_h2d_stream.fetch_add(1, Ordering::Relaxed);
        self.copy_h2d_streams[idx % self.copy_h2d_streams.len()].clone()
    }

    /// Get next copy-engine D2H stream (whole-block DMA).
    pub(crate) fn next_copy_d2h_stream(&self) -> Arc<DeviceStream> {
        let idx = self.current_copy_d2h_stream.fetch_add(1, Ordering::Relaxed);
        self.copy_d2h_streams[idx % self.copy_d2h_streams.len()].clone()
    }

    /// Get next compute-engine H2D stream (fc_lw vectorized_copy).
    pub(crate) fn next_compute_h2d_stream(&self) -> Arc<DeviceStream> {
        let idx = self.current_compute_h2d_stream.fetch_add(1, Ordering::Relaxed);
        self.compute_h2d_streams[idx % self.compute_h2d_streams.len()].clone()
    }

    /// Get next compute-engine D2H stream (fc_lw vectorized_copy).
    pub(crate) fn next_compute_d2h_stream(&self) -> Arc<DeviceStream> {
        let idx = self.current_compute_d2h_stream.fetch_add(1, Ordering::Relaxed);
        self.compute_d2h_streams[idx % self.compute_d2h_streams.len()].clone()
    }

    /// Acquire an H2D stream (public API for external callers, defaults to copy engine).
    #[doc(hidden)]
    pub fn acquire_h2d_stream(&self) -> Arc<DeviceStream> {
        self.next_copy_h2d_stream()
    }

    /// Acquire a D2H stream (public API for external callers, defaults to copy engine).
    #[doc(hidden)]
    pub fn acquire_d2h_stream(&self) -> Arc<DeviceStream> {
        self.next_copy_d2h_stream()
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
        let handle = event.into_handle();
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

        // Send to background handler — log error if channel is full or closed
        if let Err(e) = self.tx_nixl_status.try_send(notification) {
            tracing::error!(
                "Failed to enqueue NIXL status notification: channel full or closed: {}",
                e
            );
        }

        TransferCompleteNotification::from_awaiter(awaiter)
    }

    /// Register a device event for polling completion (multi-backend).
    ///
    /// Works with any DeviceEvent (CUDA, XPU).
    pub(crate) fn register_device_event(&self, event: DeviceEvent) -> TransferCompleteNotification {
        let new_event = self
            .event_system
            .new_event()
            .expect("Failed to allocate event");
        let handle = new_event.into_handle();
        let awaiter = self
            .event_system
            .awaiter(handle)
            .expect("Failed to get awaiter");

        let notification = notifications::RegisterPollingNotification {
            uuid: Uuid::new_v4(),
            checker: notifications::DeviceEventChecker::new(event),
            event_handle: handle,
        };

        if let Err(e) = self.tx_device_event.try_send(notification) {
            tracing::error!(
                "Failed to enqueue device event notification: channel full or closed: {}",
                e
            );
        }

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
        let handle = event.into_handle();
        let awaiter = self
            .event_system
            .awaiter(handle)
            .expect("Failed to get awaiter");

        let notification = notifications::RegisterNixlNotification {
            uuid: Uuid::new_v4(),
            xfer_req,
            event_handle: handle,
        };

        // Send to background handler — log error if channel is full or closed
        if let Err(e) = self.tx_nixl_events.try_send(notification) {
            tracing::error!(
                "Failed to enqueue NIXL event notification: channel full or closed: {}",
                e
            );
        }

        TransferCompleteNotification::from_awaiter(awaiter)
    }


    /// Get the worker ID for this context.
    pub(crate) fn worker_id(&self) -> u64 {
        self.worker_id
    }
}
