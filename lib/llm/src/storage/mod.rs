// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage layer for Dynamo's stateful Responses API.
//!
//! Provides pluggable storage backends for persisting API responses across
//! multi-turn conversations. Gated behind the `--enable-stateful-responses`
//! feature flag (env: `DYNAMO_ENABLE_STATEFUL_RESPONSES`); when disabled,
//! no storage overhead is incurred and stateful request fields return 400.
//!
//! # Isolation Model
//!
//! - **Tenant** (`x-tenant-id` header): hard security boundary. Cross-tenant
//!   access is always denied.
//! - **Session** (`x-session-id` header): organizational metadata within a
//!   tenant. Cross-session reads are permitted to support multi-agent workflows.
//!
//! # Storage Backends
//!
//! - [`InMemoryResponseStorage`]: single-instance dev/test use.
//! - [`RedisResponseStorage`] (feature `redis-storage`): horizontally-scaled
//!   production deployments.
//! - Custom: implement the [`ResponseStorage`] trait for Postgres, S3, etc.

pub mod config;
pub mod manager;
pub mod response_storage;
pub mod session_lock;
pub mod trace_replay;

#[cfg(feature = "redis-storage")]
pub mod redis_lock;
#[cfg(feature = "redis-storage")]
pub mod redis_storage;

pub use config::{StorageBackend, StorageConfig};
pub use manager::InMemoryResponseStorage;
pub use response_storage::{ResponseStorage, StorageError, StoredResponse, validate_key_component};
pub use session_lock::{InMemorySessionLock, LockConfig, LockError, LockGuard, SessionLock};
pub use trace_replay::{
    ParsedTrace, ReplayResult, parse_trace_content, parse_trace_file, replay_trace,
};

#[cfg(feature = "redis-storage")]
pub use redis_lock::RedisSessionLock;
#[cfg(feature = "redis-storage")]
pub use redis_storage::RedisResponseStorage;
