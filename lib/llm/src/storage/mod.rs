//! Storage backends for Dynamo
//!
//! This module provides pluggable storage interfaces for stateful responses.
//! Users can implement custom storage backends (Redis, Postgres, S3, etc.)
//! by implementing the provided traits.

pub mod manager;
pub mod response_storage;

pub use manager::ResponseStorageManager;
pub use response_storage::{ResponseStorage, StorageError, StoredResponse};
