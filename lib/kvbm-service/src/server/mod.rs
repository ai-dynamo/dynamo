// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level service driver.
//!
//! [`KvbmService::start`] is the single entry point. It:
//! 1. Discovers host resources (`dynamo_memory::resources::Resources::discover`)
//!    and uses the CUDA-visible GPU count as the slot capacity.
//! 2. Builds the metrics registry and the registry state machine.
//! 3. Binds the HTTP sidecar first so it can learn the actual port the OS
//!    chose for `127.0.0.1:0`.
//! 4. Derives the UDS path from `<uds_dir>/kvbm-service.<pid>.<http_port>.sock`
//!    (or honors an explicit `uds_path` from config) and binds the gRPC
//!    listener.
//! 5. Publishes the bound UDS path to the HTTP state so `/ready` flips to 200
//!    and `/v1/discovery/socket` resolves.
//!
//! [`KvbmService::shutdown_graceful`] coordinates the multi-phase
//! shutdown: drain the registry, broadcast `ServerShutdownInitiated` to
//! every connected client, drive the container's `on_server_shutdown`,
//! wait until clients detach voluntarily (or the grace period elapses), and
//! only then cancel the gRPC/HTTP serve loops.

pub mod grpc;
pub mod http;
pub mod shutdown;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use dynamo_memory::resources::Resources;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::config::ServiceConfig;
use crate::container::{NoopContainer, ServiceContainer};
use crate::error::{ServiceError, ServiceResult};
use crate::metrics::ServiceMetrics;
use crate::pool::HostMemoryPool;
use crate::proto::v1::kvbm_service_server::KvbmServiceServer;
use crate::registry::Registry;
use crate::server::grpc::KvbmServiceGrpc;
use crate::server::http::{HttpState, serve as http_serve};
use crate::server::shutdown::UdsGuard;

/// Live service instance. Construct via [`KvbmService::start`] (or
/// [`KvbmService::start_with_container`] to plug in a custom container);
/// await [`serve`](Self::serve) or [`shutdown_graceful`](Self::shutdown_graceful)
/// to block until shutdown.
pub struct KvbmService {
    pub registry: Registry,
    pub resources: Arc<Resources>,
    pub metrics: ServiceMetrics,
    pub container: Arc<dyn ServiceContainer>,
    /// Host-memory pool — `None` for shell-only deployments (default
    /// [`Self::start`] / [`Self::start_with_container`]) and `Some` when
    /// constructed via [`Self::start_with_pool`].
    pub pool: Option<Arc<HostMemoryPool>>,
    pub http_addr: SocketAddr,
    pub uds_path: PathBuf,
    cancel: CancellationToken,
    http_handle: tokio::task::JoinHandle<()>,
    grpc_handle: tokio::task::JoinHandle<()>,
    // Held to ensure the UDS file is unlinked when KvbmService is dropped.
    _uds_guard: UdsGuard,
}

impl KvbmService {
    /// Build, bind, and launch the service with the default [`NoopContainer`]
    /// and no host-memory pool. Suitable for shell-only tests.
    pub async fn start(cfg: ServiceConfig) -> ServiceResult<Self> {
        Self::start_with_container(cfg, Arc::new(NoopContainer)).await
    }

    /// Build, bind, and launch the service with a specific
    /// [`ServiceContainer`] and no host-memory pool.
    pub async fn start_with_container(
        cfg: ServiceConfig,
        container: Arc<dyn ServiceContainer>,
    ) -> ServiceResult<Self> {
        Self::start_inner(cfg, container, None).await
    }

    /// Build, bind, and launch with both a [`ServiceContainer`] and a
    /// freshly-allocated [`HostMemoryPool`]. The pool is constructed
    /// before any gRPC traffic is accepted; its single lease is handed to
    /// the container via [`ServiceContainer::on_resources_attached`].
    pub async fn start_with_pool(
        cfg: ServiceConfig,
        container: Arc<dyn ServiceContainer>,
    ) -> ServiceResult<Self> {
        let instance_id = uuid::Uuid::new_v4().to_string();
        let pool = HostMemoryPool::new(&cfg.pool, &instance_id)?;
        let lease = pool.lease_all()?;
        container
            .on_resources_attached(lease)
            .await
            .map_err(|e| ServiceError::Internal(format!("container.on_resources_attached: {e}")))?;
        let svc = Self::start_inner(cfg, container, Some(pool.clone())).await?;
        publish_pool_metrics(&svc.metrics, &pool);
        Ok(svc)
    }

    async fn start_inner(
        cfg: ServiceConfig,
        container: Arc<dyn ServiceContainer>,
        pool: Option<Arc<HostMemoryPool>>,
    ) -> ServiceResult<Self> {
        let resources = Arc::new(Resources::discover());
        let capacity = capacity_from_resources(&resources);
        info!(
            capacity_slots = capacity,
            container = container.name(),
            "discovered CUDA-visible GPUs (slot capacity)"
        );

        let metrics = ServiceMetrics::new();
        let registry = Registry::new(capacity, metrics.clone());

        // 1. HTTP sidecar (binds first so we can learn the port).
        let cancel = CancellationToken::new();
        let mut http_state = HttpState::new(registry.clone(), resources.clone(), metrics.clone());
        if let Some(p) = pool.clone() {
            http_state = http_state.with_pool(p);
        }
        let (http_addr, http_handle) =
            http_serve(cfg.http_addr, http_state.clone(), cancel.clone()).await?;
        info!(%http_addr, "HTTP sidecar listening");

        // 2. Derive UDS path from the resolved HTTP port (or honor explicit).
        let uds_path = cfg.resolved_uds_path(http_addr.port());
        if uds_path.exists() {
            warn!(path = %uds_path.display(), "removing stale UDS socket");
            let _ = std::fs::remove_file(&uds_path);
        }
        if let Some(parent) = uds_path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).map_err(ServiceError::Io)?;
        }

        // 3. Bind UDS and start the gRPC server.
        let listener = UnixListener::bind(&uds_path).map_err(ServiceError::Io)?;
        let _ = set_uds_perms(&uds_path);
        let stream = UnixListenerStream::new(listener);
        info!(path = %uds_path.display(), "gRPC listening on UDS");

        let grpc_svc = KvbmServiceGrpc::new(registry.clone(), container.clone());
        let cancel_grpc = cancel.clone();
        let grpc_handle = tokio::spawn(async move {
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(KvbmServiceServer::new(grpc_svc))
                .serve_with_incoming_shutdown(stream, cancel_grpc.cancelled_owned())
                .await
            {
                tracing::error!("gRPC server error: {e}");
            }
        });

        // 4. Flip /ready to 200 and expose UDS path on /v1/discovery/socket.
        http_state.set_uds_path(uds_path.clone());

        Ok(Self {
            registry,
            resources,
            metrics,
            container,
            pool,
            http_addr,
            uds_path: uds_path.clone(),
            cancel,
            http_handle,
            grpc_handle,
            _uds_guard: UdsGuard::new(uds_path),
        })
    }

    /// Cancellation token for outside-in shutdown. Cancel to skip the
    /// graceful sequence and bring both loops down immediately. Prefer
    /// [`Self::shutdown_graceful`] when correctness matters.
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Run the multi-phase graceful shutdown sequence:
    ///
    /// 1. Flip the registry into `Draining` (rejects new registrations) and
    ///    collect every active stream lifecycle.
    /// 2. Broadcast `ServerShutdownInitiated` to every connected client.
    /// 3. Invoke `container.on_server_shutdown(grace_period)` in parallel
    ///    with awaiting client detach.
    /// 4. Wait for both signals (registry empty + container done) — bounded
    ///    by `grace_period` when `Some`, indefinitely when `None`.
    /// 5. If the grace period elapsed with stragglers, broadcast
    ///    `ServerShutdownTimedOut` and drop their senders.
    /// 6. Cancel the gRPC + HTTP serve loops and await them.
    pub async fn shutdown_graceful(self, grace_period: Option<Duration>) {
        let lifecycles = self.registry.begin_drain();
        info!(
            active_clients = lifecycles.len(),
            grace = ?grace_period,
            container = self.container.name(),
            "graceful shutdown begin"
        );

        // 2. Broadcast initiated. Cheap parallel mpsc sends.
        let initiated_futs: Vec<_> = lifecycles
            .iter()
            .map(|lc| lc.send_shutdown_initiated(grace_period))
            .collect();
        futures::future::join_all(initiated_futs).await;

        // 3+4. Container shutdown + client drain in parallel, bounded by grace.
        let container = self.container.clone();
        let registry = self.registry.clone();
        let combined = async move {
            let _ = tokio::join!(
                container.on_server_shutdown(grace_period),
                registry.wait_until_empty(),
            );
        };

        let timed_out = match grace_period {
            None => {
                combined.await;
                false
            }
            Some(grace) => tokio::time::timeout(grace, combined).await.is_err(),
        };

        if timed_out {
            warn!(
                grace_ms = ?grace_period.map(|d| d.as_millis()),
                "grace period elapsed; forcing remaining clients"
            );
            // Snapshot of whatever lifecycles are still attached. begin_drain
            // is idempotent in Draining and returns the current set.
            let stragglers = self.registry.begin_drain();
            let grace = grace_period.unwrap_or_default();
            let force_futs: Vec<_> = stragglers
                .iter()
                .map(|lc| lc.send_shutdown_timed_out(grace))
                .collect();
            futures::future::join_all(force_futs).await;
        }

        info!("cancelling listener tasks");
        self.cancel.cancel();
        let _ = self.http_handle.await;
        let _ = self.grpc_handle.await;
        info!("graceful shutdown complete");
    }

    /// Convenience: graceful shutdown with no upper bound (waits forever
    /// for clients + container to drain). Equivalent to
    /// `shutdown_graceful(None)`.
    pub async fn shutdown(self) {
        self.shutdown_graceful(None).await
    }

    /// Await until both listener tasks exit (e.g. after `cancel`).
    pub async fn serve(self) {
        let _ = self.http_handle.await;
        let _ = self.grpc_handle.await;
    }
}

fn publish_pool_metrics(metrics: &ServiceMetrics, pool: &HostMemoryPool) {
    metrics.pool_bytes_total.reset();
    metrics.pool_slabs_total.reset();
    for slab in pool.slabs() {
        let node = slab.numa_node().0.to_string();
        let tier = hugepage_tier_label(&slab.hugepage_tier());
        metrics
            .pool_bytes_total
            .with_label_values(&[node.as_str(), tier])
            .set(slab.size_bytes() as i64);
        metrics.pool_slabs_total.with_label_values(&[tier]).inc();
    }
}

fn hugepage_tier_label(tier: &dynamo_memory::HugepageTier) -> &'static str {
    match tier {
        dynamo_memory::HugepageTier::Explicit { .. } => "explicit",
        dynamo_memory::HugepageTier::Thp => "thp",
        dynamo_memory::HugepageTier::None => "none",
    }
}

fn capacity_from_resources(resources: &Resources) -> u32 {
    let count = resources
        .gpus
        .iter()
        .filter(|g| g.cuda_ordinal.is_some())
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

#[cfg(unix)]
fn set_uds_perms(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o660))
}

#[cfg(not(unix))]
fn set_uds_perms(_path: &std::path::Path) -> std::io::Result<()> {
    Ok(())
}
