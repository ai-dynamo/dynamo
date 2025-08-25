// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ContentHash, OscarResult};
use async_trait::async_trait;

/// Trait for storage backends that can store object data.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store object data and return storage key.
    async fn store_object(
        &self,
        hash: &ContentHash,
        data: &[u8],
    ) -> OscarResult<String>;

    /// Retrieve object data by storage key.
    async fn get_object(&self, storage_key: &str) -> OscarResult<bytes::Bytes>;

    /// Delete object data by storage key.
    async fn delete_object(&self, storage_key: &str) -> OscarResult<()>;

    /// Check if object exists by storage key.
    async fn object_exists(&self, storage_key: &str) -> OscarResult<bool>;

    /// Get the name/type of this storage backend.
    fn backend_name(&self) -> &str;
}