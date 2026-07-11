// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer context.
//!
//! Holds the resources shared by transfer executors: NIXL agent, device
//! context/streams, memory pool, and the planner-driven graph/benchmark/
//! prepared-plan caches.
//!
//! ## Stream pools
//!
//! One multi-backend `Arc<DeviceStream>` pool per direction
//! (`h2d_streams`, `d2h_streams`). Round-robin across the pool is shared
//! by whole-block `batch_copy` and per-chunk `vectorized_copy` — neither
//! backend binds queues to distinct engine classes today.
//!
//! C-FFI launch sites (CUDA graph capture/replay, transform kernel
//! FFI on both backends, benchmark direct DMA) downcast at the
//! boundary via [`crate::device::DeviceStream::cuda_stream_arc`] /
//! [`crate::device::DeviceStream::sycl_stream`] when they need a
//! raw `cudaStream_t` / `sycl::queue*` to launch a C kernel. All
//! other planner / executor / benchmark code paths use the
//! device-agnostic `DeviceStream` API directly.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};


use anyhow::Result;
use derive_builder::Builder;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::device::{DeviceBackend, DeviceContext, DeviceStream, DeviceEvent, DeviceMemPool};
use dynamo_memory::nixl::{NixlAgent, NixlBackendConfig, XferRequest};
use kvbm_observability::SharedKvbmObservability;
use velo::EventManager;

use crate::manager::TransferManager;
use crate::transfer::benchmark::{BenchmarkCache, BenchmarkCandidate, BenchmarkKey, BenchmarkOutcome};
use crate::transfer::graph_cache::GraphCache;
use crate::transfer::prepared::PreparedPlanCache;

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

    /// Device ID for the compute backend (default: 0).
    ///
    /// Named `device_id` rather than the legacy `cuda_device_id` because
    /// the same field selects an XPU ordinal under the SYCL backend.
    #[builder(default = "0")]
    device_id: usize,

    /// Device backend type (default: Cuda).
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

    /// Shared observability registry and metrics.
    #[builder(default, setter(strip_option))]
    observability: Option<SharedKvbmObservability>,

    /// Enable compact prepared-plan caching for manager-driven transfers.
    #[builder(default = "true")]
    prepared_plan_cache_enabled: bool,

    /// Maximum entries in the remote prepared-plan LRU.
    #[builder(default = "1024")]
    prepared_plan_remote_capacity: usize,
}

impl TransferConfigBuilder {
    /// Initialize builder with event system and tokio handle.
    ///
    /// This sets the event_system and tokio runtime handle, ensuring consistency
    /// with Velo's event system. Use this when the runtime has already been
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

    /// Legacy alias for `device_id`. Some external callers (notably the
    /// kvbm-connector worker) still use the old name; we accept it as a
    /// thin shim so call sites compile unchanged across the rebase.
    pub fn cuda_device_id(self, device_id: usize) -> Self {
        self.device_id(device_id)
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
            .take()
            .unwrap_or_else(|| format!("worker-{}", worker_id));

        let nixl_agent =
            NixlAgent::from_nixl_backend_config(&agent_name, config.nixl_backend_config.clone())?;

        let context = TransferContext::new(nixl_agent, config)?;
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
        let context = TransferContext::new(self.agent, config)?;
        Ok(TransferManager::from_context(context))
    }

    pub fn device_id(mut self, device_id: usize) -> Self {
        self.builder = self.builder.device_id(device_id);
        self
    }

    /// Legacy alias for `device_id`.
    pub fn cuda_device_id(mut self, device_id: usize) -> Self {
        self.builder = self.builder.device_id(device_id);
        self
    }

    pub fn device_backend(mut self, backend: DeviceBackend) -> Self {
        self.builder = self.builder.device_backend(backend);
        self
    }

    /// Override the prepared-plan cache enabled flag after the agent
    /// has been wired. Forwarded to the underlying builder.
    pub fn prepared_plan_cache_enabled(mut self, on: bool) -> Self {
        self.builder = self.builder.prepared_plan_cache_enabled(on);
        self
    }

    /// Override the remote-LRU capacity after the agent has been wired.
    pub fn prepared_plan_remote_capacity(mut self, capacity: usize) -> Self {
        self.builder = self.builder.prepared_plan_remote_capacity(capacity);
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
    /// Stream pools, round-robin per direction. Shared by whole-block
    /// `batch_copy` and per-chunk `vectorized_copy` — neither backend
    /// binds queues to distinct engines (CUDA: unified stream, SYCL:
    /// engine ordinal is chosen by the runtime, not by the queue).
    h2d_streams: Vec<Arc<DeviceStream>>,
    d2h_streams: Vec<Arc<DeviceStream>>,
    current_h2d_stream: Arc<AtomicUsize>,
    current_d2h_stream: Arc<AtomicUsize>,
    // Device memory pool for kernel allocations (multi-backend)
    #[allow(dead_code)]
    device_pool: Arc<DeviceMemPool>,

    // Channels for background notification handlers
    tx_nixl_status: mpsc::Sender<RegisterPollingNotification<notifications::NixlStatusChecker>>,
    tx_device_event: mpsc::Sender<RegisterPollingNotification<notifications::DeviceEventChecker>>,
    #[allow(dead_code)]
    tx_nixl_events: mpsc::Sender<notifications::RegisterNixlNotification>,
    observability: Option<SharedKvbmObservability>,
    /// PR-7.4.1: CUDA graph exec handle cache for replay-based transfers.
    ///
    /// Shared across clones of `TransferContext` so all executor calls on
    /// the same context share the same cache. Drops with the last
    /// `TransferContext` clone; each `ManagedExecHandle` entry calls
    /// `cuGraphExecDestroy` on drop.
    graph_cache: Arc<GraphCache>,

    /// PR-7.5: Benchmark outcome cache for the scorer.
    ///
    /// Shared across clones of `TransferContext`. Populated by
    /// `benchmark_pair` at startup when
    /// `TransferCapabilities::startup_benchmark` is enabled; consulted by
    /// `score_candidate` via `SelectionContext::benchmark_outcome`.
    /// Drops with the last `TransferContext` clone.
    benchmark_cache: Arc<BenchmarkCache>,

    /// Compact prepared-plan cache for handle-keyed transfer templates.
    prepared_plan_cache: Arc<PreparedPlanCache>,
}

impl TransferContext {
    pub fn builder() -> TransferConfigBuilder {
        TransferConfigBuilder::default()
    }

    pub(crate) fn new(
        nixl_agent: NixlAgent,
        config: TransferConfig,
    ) -> Result<Self> {
        let TransferConfig {
            event_system,
            tokio_runtime,
            capabilities,
            device_id,
            device_backend,
            pool_reserve_size,
            pool_release_threshold,
            num_streams,
            observability,
            prepared_plan_cache_enabled,
            prepared_plan_remote_capacity,
            // Fields already consumed by the builder path before this fn runs:
            nixl_agent_name: _,
            nixl_backend_config: _,
        } = config;

        // Create device context for the specified backend
        let device_ctx = DeviceContext::new(device_backend, device_id as u32)?;

        // Disable event tracking (no-op on non-CUDA backends)
        unsafe { device_ctx.disable_event_tracking()? };

        // Create device memory pool
        let device_pool = Arc::new(
            device_ctx.create_memory_pool(pool_reserve_size, pool_release_threshold)?
        );

        // Create device stream pools (num_streams per direction, round-robin).
        // One pool per direction matches the upstream CUDA design; both
        // whole-block `batch_copy` and kernel `vectorized_copy` share the
        // same pool, because neither backend binds queues to separate
        // engine classes today.
        let num_streams = num_streams.max(1); // Ensure at least 1 stream

        let h2d_streams: Vec<Arc<DeviceStream>> = (0..num_streams)
            .map(|_| device_ctx.create_stream().map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        let d2h_streams: Vec<Arc<DeviceStream>> = (0..num_streams)
            .map(|_| device_ctx.create_stream().map(Arc::new))
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
            h2d_streams,
            d2h_streams,
            current_h2d_stream: Arc::new(AtomicUsize::new(0)),
            current_d2h_stream: Arc::new(AtomicUsize::new(0)),
            device_pool,
            tx_nixl_status,
            tx_device_event,
            tx_nixl_events,
            observability,
            graph_cache: Arc::new(GraphCache::new()),
            benchmark_cache: Arc::new(BenchmarkCache::new()),
            prepared_plan_cache: Arc::new(PreparedPlanCache::new(
                prepared_plan_cache_enabled,
                prepared_plan_remote_capacity,
            )),
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

    #[doc(hidden)]
    pub fn observability(&self) -> Option<&SharedKvbmObservability> {
        self.observability.as_ref()
    }

    /// Get the device memory pool for kernel allocations (multi-backend).
    #[allow(dead_code)]
    pub(crate) fn device_pool(&self) -> &Arc<DeviceMemPool> {
        &self.device_pool
    }

    /// Get next H2D stream (round-robin across the H2D pool).
    pub(crate) fn next_h2d_stream(&self) -> Arc<DeviceStream> {
        let idx = self.current_h2d_stream.fetch_add(1, Ordering::Relaxed);
        self.h2d_streams[idx % self.h2d_streams.len()].clone()
    }

    /// Get next D2H stream (round-robin across the D2H pool).
    pub(crate) fn next_d2h_stream(&self) -> Arc<DeviceStream> {
        let idx = self.current_d2h_stream.fetch_add(1, Ordering::Relaxed);
        self.d2h_streams[idx % self.d2h_streams.len()].clone()
    }

    /// Acquire an H2D stream (public API for external callers).
    #[doc(hidden)]
    pub fn acquire_h2d_stream(&self) -> Arc<DeviceStream> {
        self.next_h2d_stream()
    }

    /// Acquire a D2H stream (public API for external callers).
    #[doc(hidden)]
    pub fn acquire_d2h_stream(&self) -> Arc<DeviceStream> {
        self.next_d2h_stream()
    }

    /// PR-7.4.1: Get the CUDA graph exec handle cache.
    ///
    /// Shared across all clones of this context. Used by
    /// `dispatch_cuda_graph_replay_planner` to look up or insert
    /// instantiated exec handles keyed by transfer shape.
    pub(crate) fn graph_cache(&self) -> &Arc<GraphCache> {
        &self.graph_cache
    }

    /// PR-7.5: Get the benchmark outcome cache.
    ///
    /// Used by callers of `score_candidate` (in `executor::planner`) to
    /// populate `SelectionContext::benchmark_outcome` before selection.
    /// Shared across all clones of this context.
    pub(crate) fn benchmark_cache(&self) -> &Arc<BenchmarkCache> {
        &self.benchmark_cache
    }

    /// Get the compact prepared-plan cache.
    pub(crate) fn prepared_plan_cache(&self) -> &Arc<PreparedPlanCache> {
        &self.prepared_plan_cache
    }

    /// Eagerly build and cache a prepared transfer plan for one direction
    /// of a `(src, dst)` handle pair using the strategy that
    /// [`crate::transfer::strategy::select_strategy`] would pick.
    ///
    /// No-op when the cache is disabled, when the layout pair is
    /// same-shape direct (no plan needed — the planner projects
    /// `AnnotatedLayout`s inline), or when the strategy is two-hop.
    /// Sliced transfers (`axis_slices`) still populate the cache
    /// lazily on first use.
    pub(crate) fn prewarm_prepared_plan(
        &self,
        src_handle: crate::manager::LayoutHandle,
        src_layout: &crate::transfer::PhysicalLayout,
        dst_handle: crate::manager::LayoutHandle,
        dst_layout: &crate::transfer::PhysicalLayout,
    ) -> anyhow::Result<()> {
        if !self.prepared_plan_cache.is_enabled() {
            return Ok(());
        }
        let src_kv = src_layout.layout().block_layout();
        let dst_kv = dst_layout.layout().block_layout();
        if !src_kv.requires_transform(&dst_kv) {
            // Same-layout direct copies don't use a prepared plan.
            return Ok(());
        }
        // Transform-pair prewarm is backend-agnostic: `build_transform_invocation`
        // returns pure `kernel_catalog` metadata and `PreparedTransferPlan::build_transform`
        // does not touch a device stream. Both backends share the same cache.
        use crate::transfer::executor::planner::build_transform_invocation;
        use crate::transfer::prepared::{PreparedPlanKey, PreparedTransferPlan};
        use crate::transfer::strategy::{TransferPlan, select_strategy};
        let plan = select_strategy(src_layout, dst_layout, self)?;
        let strategy = match plan {
            TransferPlan::Direct(strategy) => strategy,
            // Two-hop plans bypass the prepared-plan cache (per-call
            // bounce layout). Nothing to prewarm.
            TransferPlan::TwoHop { .. } => return Ok(()),
        };
        let key = PreparedPlanKey::new(src_handle, dst_handle, strategy, &[]);
        self.prepared_plan_cache
            .get_or_insert_with(self.worker_id(), key, || {
                let invocation = build_transform_invocation(src_layout, dst_layout)?;
                PreparedTransferPlan::build_transform(invocation, src_layout, dst_layout)
            })?;
        Ok(())
    }

    /// PR-7.5.1: Benchmark a set of candidates for a given layout-pair key
    /// and record the winner in the cache.
    ///
    /// This is an explicit-API benchmark (Path B): the caller decides when
    /// to benchmark (e.g. at startup with known layout pairs) and provides
    /// the key, pre-decoded candidates, and a device stream to dispatch on.
    ///
    /// Supported variants: `DirectDma`, `TransformKernel`, and
    /// `NixlDirectDma`. All routes measure end-to-end transfer time
    /// including device/network completion. Under `xpu-sycl` the
    /// `DirectDma` and `TransformKernel` routes bail at the
    /// dispatch-site pending the SYCL kernel-dispatch port.
    ///
    /// See [`BenchmarkCache::benchmark_pair`] for timing semantics and
    /// error conditions.
    #[allow(dead_code)]
    pub(crate) fn benchmark_pair(
        &self,
        key: BenchmarkKey,
        candidates: Vec<BenchmarkCandidate>,
        stream: &Arc<DeviceStream>,
    ) -> anyhow::Result<BenchmarkOutcome> {
        self.benchmark_cache.benchmark_pair(key, candidates, stream)
    }

    /// Clone the device-event polling channel sender.
    ///
    /// Used by the planner-driven Staged executor (PR-6.2) to register
    /// device events from inside a `tokio::spawn`-ed chain task without
    /// holding `&TransferContext` across an `.await`.
    pub(crate) fn tx_device_event_clone(
        &self,
    ) -> mpsc::Sender<RegisterPollingNotification<notifications::DeviceEventChecker>> {
        self.tx_device_event.clone()
    }

    /// Clone the NIXL status polling channel sender. Used for the same
    /// reason as [`Self::tx_device_event_clone`] — Staged-task NIXL
    /// completion registration without `&TransferContext`.
    pub(crate) fn tx_nixl_status_clone(
        &self,
    ) -> mpsc::Sender<RegisterPollingNotification<notifications::NixlStatusChecker>> {
        self.tx_nixl_status.clone()
    }

    /// Register a NIXL transfer request for status polling completion.
    ///
    /// This method enqueues the transfer request to be polled for completion
    /// using `agent.get_xfer_status()`. Returns a notification object that
    /// can be awaited for completion.
    pub(crate) fn register_nixl_status(
        &self,
        xfer_req: XferRequest,
        telemetry: Option<notifications::XferTelemetry>,
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
            telemetry,
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
            telemetry: None,
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
