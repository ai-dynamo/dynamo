// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Service-container abstraction.
//!
//! A `ServiceContainer` is the boundary between the kvbm-service shell (gRPC,
//! HTTP, registry, lifecycle) and the in-process workload it hosts. The
//! eventual kvbm-engine — leader + N workers — will live behind this trait so
//! the shell can route registration callbacks and lifecycle signals without
//! depending on engine internals.
//!
//! Today there is exactly one impl: [`NoopContainer`]. The trait is shaped
//! so other entities (or test doubles) can be plugged in.

use std::time::Duration;

use async_trait::async_trait;

use crate::instance::RegistrationInstance;
use crate::pool::PoolLease;
use crate::registry::RegistrationId;

/// Errors a container may return from [`ServiceContainer::on_register`].
#[derive(Debug, thiserror::Error)]
pub enum ContainerError {
    /// The container is not yet able to accept registrations (e.g. the
    /// engine has not finished attaching).
    #[error("container not ready: {0}")]
    NotReady(String),
    /// The container actively rejected the registration (e.g. policy
    /// violation, incompatible layout).
    #[error("container rejected registration: {0}")]
    Rejected(String),
}

/// In-process workload that the service hosts.
///
/// All hooks must be quick to invoke; long-running work belongs in a spawned
/// task that the container owns. [`Self::on_server_shutdown`] is the one
/// exception — it must drain the container's work and may take as long as
/// the configured grace period.
#[async_trait]
pub trait ServiceContainer: Send + Sync {
    /// Stable identifier for logging / metrics. Used by the shell to tag
    /// log lines and (eventually) Prometheus labels.
    fn name(&self) -> &str;

    /// Invoked immediately after the [`Registry`](crate::registry::Registry)
    /// accepts a new registration. Returning an error tells the shell to
    /// roll the registration back; the gRPC client sees the mapped Status
    /// (`failed_precondition` for [`ContainerError::Rejected`],
    /// `unavailable` for [`ContainerError::NotReady`]).
    async fn on_register(
        &self,
        id: RegistrationId,
        instance: &RegistrationInstance,
    ) -> Result<(), ContainerError>;

    /// Invoked after a registration is released (graceful unregister or
    /// stream drop). Always called exactly once per successful
    /// [`Self::on_register`].
    async fn on_unregister(&self, id: RegistrationId);

    /// Drive container-internal shutdown.
    ///
    /// Implementations must:
    /// - finish any in-flight work that, if interrupted, would corrupt
    ///   client state (e.g. host → CUDA-IPC GPU writes for the engine);
    /// - cancel any work that has not yet started, signalling cancellation
    ///   to clients best-effort;
    /// - return when the above is done, even if `grace_period` is `None`.
    ///
    /// The shell waits up to `grace_period` (indefinitely when `None`) for
    /// this future to resolve before force-closing client streams.
    async fn on_server_shutdown(&self, grace_period: Option<Duration>);

    /// Hand the host-memory pool lease to the container at startup, after
    /// the pool has been allocated and registered with NIXL. Containers
    /// that need the slabs (e.g. the eventual kvbm-engine container) must
    /// hold the [`PoolLease`] for the duration they intend to use the
    /// pinned regions. The default impl drops the lease, which releases
    /// it back to the pool — appropriate for containers that don't manage
    /// host memory directly.
    async fn on_resources_attached(&self, _lease: PoolLease) -> Result<(), ContainerError> {
        Ok(())
    }
}

/// Default container — accepts every registration, does nothing on
/// shutdown. The shell ships with this until the engine container lands.
pub struct NoopContainer;

#[async_trait]
impl ServiceContainer for NoopContainer {
    fn name(&self) -> &str {
        "noop"
    }

    async fn on_register(
        &self,
        _id: RegistrationId,
        _instance: &RegistrationInstance,
    ) -> Result<(), ContainerError> {
        Ok(())
    }

    async fn on_unregister(&self, _id: RegistrationId) {}

    async fn on_server_shutdown(&self, _grace_period: Option<Duration>) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use parking_lot::Mutex;

    use super::*;
    use crate::instance::{KvbmInstance, LayoutShape};
    use crate::mode::ServiceMode;
    use crate::registry::RegistrationId;

    /// Test double that records each callback so suites can assert order
    /// and counts.
    pub(crate) struct RecordingContainer {
        pub registers: AtomicU32,
        pub unregisters: AtomicU32,
        pub shutdowns: AtomicU32,
        pub last_shutdown_grace: Mutex<Option<Option<Duration>>>,
    }

    impl RecordingContainer {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                registers: AtomicU32::new(0),
                unregisters: AtomicU32::new(0),
                shutdowns: AtomicU32::new(0),
                last_shutdown_grace: Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl ServiceContainer for RecordingContainer {
        fn name(&self) -> &str {
            "recording"
        }

        async fn on_register(
            &self,
            _id: RegistrationId,
            _instance: &RegistrationInstance,
        ) -> Result<(), ContainerError> {
            self.registers.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn on_unregister(&self, _id: RegistrationId) {
            self.unregisters.fetch_add(1, Ordering::SeqCst);
        }

        async fn on_server_shutdown(&self, grace: Option<Duration>) {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            *self.last_shutdown_grace.lock() = Some(grace);
        }
    }

    fn instance() -> RegistrationInstance {
        RegistrationInstance::Kvbm(KvbmInstance {
            model_name: "llm".into(),
            layout: LayoutShape::UniversalTp1Canonical { bytes: vec![0xab] },
            tp_size: 2,
            block_size: 64,
            mode: ServiceMode::Kvbm,
        })
    }

    #[tokio::test]
    async fn noop_container_accepts_and_returns() {
        let c = NoopContainer;
        let id = RegistrationId::new();
        c.on_register(id, &instance()).await.unwrap();
        c.on_unregister(id).await;
        c.on_server_shutdown(Some(Duration::from_secs(60))).await;
        c.on_server_shutdown(None).await;
    }

    #[tokio::test]
    async fn recording_container_counts_callbacks() {
        let c = RecordingContainer::new();
        let id = RegistrationId::new();
        c.on_register(id, &instance()).await.unwrap();
        c.on_unregister(id).await;
        c.on_server_shutdown(Some(Duration::from_secs(90))).await;
        assert_eq!(c.registers.load(Ordering::SeqCst), 1);
        assert_eq!(c.unregisters.load(Ordering::SeqCst), 1);
        assert_eq!(c.shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(
            *c.last_shutdown_grace.lock(),
            Some(Some(Duration::from_secs(90)))
        );
    }
}
