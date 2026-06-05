// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbm-service` — gRPC service shell around `kvbm-engine`.
//!
//! The shell exposes a server-streaming `Register` RPC over a unix domain
//! socket and a sidecar HTTP server (axum) for liveness, readiness, metrics,
//! and discovery (UDS path, host topology, active registrations).
//!
//! Registration is split into:
//!
//! - [`instance::RegistrationInstance`] — the typed, per-tenant data the
//!   client sends. Today the only variant is
//!   [`instance::KvbmInstance`]; new tenant kinds plug in alongside.
//! - [`registry::RegistrationKey`] — opaque blake3 hash of the active
//!   instance arm. The registry uses keys for tenancy equality; the rich
//!   instance is what flows to the [`container::ServiceContainer`].
//!
//! The engine itself is not yet attached — [`container::ServiceContainer`]
//! defines the API boundary the eventual kvbm-engine container will plug
//! into. [`container::NoopContainer`] is the default for the shell.
//!
//! Graceful shutdown is coordinated by
//! [`server::KvbmService::shutdown_graceful`]: it broadcasts
//! `ServerShutdownInitiated` to every connected client over their existing
//! stream, drives the container's `on_server_shutdown`, and only force-closes
//! stragglers (with `ServerShutdownTimedOut`) when the configured grace
//! period elapses.

pub mod config;
pub mod container;
pub mod error;
pub mod instance;
pub mod metrics;
pub mod mode;
pub mod pool;
pub mod proto;
pub mod registry;
pub mod server;

pub use crate::config::ServiceConfig;
pub use crate::container::{ContainerError, NoopContainer, ServiceContainer};
pub use crate::error::{ServiceError, ServiceResult};
pub use crate::instance::{KvbmInstance, LayoutShape, RegistrationInstance};
pub use crate::mode::ServiceMode;
pub use crate::pool::{
    HostMemoryPool, NodeSlab, NodeSlabSnapshot, PoolConfig, PoolLease, PoolSizing, PoolSnapshot,
};
pub use crate::registry::{
    NoopLifecycle, RegistrationId, RegistrationKey, Registry, StreamLifecycle,
};
pub use crate::server::KvbmService;
