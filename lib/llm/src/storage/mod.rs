//! Storage backends for Dynamo
//!
//! This module provides pluggable storage interfaces for stateful responses.
//! Users can implement custom storage backends (Redis, Postgres, S3, etc.)
//! by implementing the provided traits.
//!
//! # Features
//!
//! - `redis-storage`: Enables Redis-based storage and locking implementations
//!   for horizontal scaling across multiple instances.

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
pub use response_storage::{ResponseStorage, StorageError, StoredResponse};
pub use session_lock::{InMemorySessionLock, LockConfig, LockError, LockGuard, SessionLock};
pub use trace_replay::{parse_trace_content, parse_trace_file, replay_trace, ParsedTrace, ReplayResult};

#[cfg(feature = "redis-storage")]
pub use redis_lock::RedisSessionLock;
#[cfg(feature = "redis-storage")]
pub use redis_storage::RedisResponseStorage;
