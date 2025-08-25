// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// Result type for Oscar operations.
pub type OscarResult<T> = Result<T, OscarError>;

/// Errors that can occur during Oscar operations.
#[derive(Error, Debug)]
pub enum OscarError {
    #[error("Object too large: {size} bytes (max {max_size} bytes)")]
    ObjectTooLarge { size: usize, max_size: usize },

    #[error("Storage error: {0}")]
    Storage(#[from] dynamo_runtime::storage::key_value_store::StorageError),

    #[error("Hash mismatch: expected {expected}, got {computed}")]
    HashMismatch { expected: String, computed: String },

    #[error("Object not found: {hash}")]
    ObjectNotFound { hash: String },

    #[error("Invalid operation: {reason}")]
    InvalidOperation { reason: String },

    #[error("Concurrency error: {reason}")]
    Concurrency { reason: String },

    #[error("I/O error accessing {path}: {error}")]
    IoError { path: std::path::PathBuf, error: std::io::Error },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}