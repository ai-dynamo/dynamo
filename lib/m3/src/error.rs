// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_memory::{ArenaError, StorageError};
use thiserror::Error;

/// Errors that can occur in M3 operations
#[derive(Debug, Error)]
pub enum M3Error {
    #[error("Key not found: {0}")]
    NotFound(String),

    #[error("Key already exists: {0}")]
    AlreadyExists(String),

    #[error("Buffer too small for operation")]
    BufferTooSmall,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Arena allocation error: {0}")]
    ArenaError(#[from] ArenaError),

    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),

    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),

    #[error("Batch operation failed: {0}")]
    BatchFailed(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<String> for M3Error {
    fn from(s: String) -> Self {
        M3Error::InvalidConfig(s)
    }
}

pub type Result<T> = std::result::Result<T, M3Error>;
