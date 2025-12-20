// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote storage transfer abstractions for remote transfers.
//!
//! This module provides the core types for remote storage transfers:
//! - [`RemoteKey`] - Abstract key for remote storage (object or disk)
//! - [`RemoteBlockDescriptor`] - Descriptor for a block in remote storage
//! - [`RemoteTransferPipeline`] - Transfer pipeline configuration
//! - [`RemoteTransferHandle`] - Handle for async transfer operations
//!
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::TransferError;

/// Kind of remote storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemoteStorageKind {
    Object,
    Disk,
}

/// A key that identifies a block in remote storage.
///
/// This is an abstract type that can represent different addressing schemes:
/// - Object storage: bucket + object key
/// - Remote disk: path + offset/key
///
/// The key must be serializable for registry storage and network transmission.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemoteKey {
    /// Object storage (S3, GCS, Azure Blob, MinIO, etc.)
    Object(ObjectKey),
    /// Remote disk (NFS, distributed filesystem, remote NVMe)
    Disk(DiskKey),
}

/// Key for object storage - bucket + object identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectKey {
    /// Bucket/container name
    pub bucket: String,
    /// Object key/path within bucket
    pub key: String,
}

impl ObjectKey {
    /// Create a new object key.
    pub fn new(bucket: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            key: key.into(),
        }
    }

    /// Create from sequence hash (common pattern: hash as hex string).
    pub fn from_hash(bucket: impl Into<String>, hash: u64) -> Self {
        Self {
            bucket: bucket.into(),
            key: format!("{:016x}", hash),
        }
    }

    /// Get the hash if this key was created from a hash.
    pub fn as_hash(&self) -> Option<u64> {
        u64::from_str_radix(&self.key, 16).ok()
    }
}

/// Key for remote disk storage - path + identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DiskKey {
    /// Base path (mount point, share path, etc.)
    pub path: String,
    /// Block identifier within the path (could be filename, offset, etc.)
    pub key: String,
}

impl DiskKey {
    /// Create a new disk key.
    pub fn new(path: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            key: key.into(),
        }
    }

    /// Create from sequence hash.
    pub fn from_hash(path: impl Into<String>, hash: u64) -> Self {
        Self {
            path: path.into(),
            key: format!("{:016x}", hash),
        }
    }

    /// Get full path (path + key).
    pub fn full_path(&self) -> String {
        format!("{}/{}", self.path, self.key)
    }
}

impl RemoteKey {
    /// Get the storage kind.
    pub fn kind(&self) -> RemoteStorageKind {
        match self {
            RemoteKey::Object(_) => RemoteStorageKind::Object,
            RemoteKey::Disk(_) => RemoteStorageKind::Disk,
        }
    }

    /// Get the "raw" key portion (for NIXL device_id).
    /// Returns hash of the key for use in NIXL descriptors.
    pub fn nixl_device_id(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Create object key.
    pub fn object(bucket: impl Into<String>, key: impl Into<String>) -> Self {
        RemoteKey::Object(ObjectKey::new(bucket, key))
    }

    /// Create disk key.
    pub fn disk(path: impl Into<String>, key: impl Into<String>) -> Self {
        RemoteKey::Disk(DiskKey::new(path, key))
    }

    /// Get the sequence hash if this is an object key created from a hash.
    pub fn sequence_hash(&self) -> Option<u64> {
        match self {
            RemoteKey::Object(obj) => obj.as_hash(),
            RemoteKey::Disk(disk) => u64::from_str_radix(&disk.key, 16).ok(),
        }
    }

    /// Get the location (bucket for Object, path for Disk).
    pub fn location(&self) -> &str {
        match self {
            RemoteKey::Object(obj) => &obj.bucket,
            RemoteKey::Disk(disk) => &disk.path,
        }
    }

    /// Get the key portion (object key or disk key).
    pub fn key_str(&self) -> &str {
        match self {
            RemoteKey::Object(obj) => &obj.key,
            RemoteKey::Disk(disk) => &disk.key,
        }
    }

    /// Create object key from sequence hash (common pattern).
    pub fn object_from_hash(bucket: impl Into<String>, hash: u64) -> Self {
        RemoteKey::Object(ObjectKey::from_hash(bucket, hash))
    }

    /// Create disk key from sequence hash.
    pub fn disk_from_hash(path: impl Into<String>, hash: u64) -> Self {
        RemoteKey::Disk(DiskKey::from_hash(path, hash))
    }
}

// =============================================================================
// Remote Block Metadata
// =============================================================================

/// Core metadata associated with a remote block.
///
/// This is the lean, required metadata stored in the registry.
#[derive(Debug, Clone, Copy)]
pub struct RemoteBlockMetadata {
    /// Sequence hash - REQUIRED for deduplication (this is the primary lookup key)
    pub sequence_hash: u64,
    /// Timestamp when block was stored (Unix millis)
    pub stored_at: u64,
}

impl RemoteBlockMetadata {
    /// Create new metadata with current timestamp.
    pub fn new(sequence_hash: u64) -> Self {
        Self {
            sequence_hash,
            stored_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Create metadata with explicit timestamp.
    pub fn with_timestamp(sequence_hash: u64, stored_at: u64) -> Self {
        Self {
            sequence_hash,
            stored_at,
        }
    }
}

// =============================================================================
// Remote Block Descriptor
// =============================================================================

/// Descriptor for a block in remote storage.
///
/// Unlike local blocks which have memory addresses, remote blocks
/// are identified by keys. The descriptor includes both the key
/// and size information needed for transfers.
#[derive(Debug, Clone)]
pub struct RemoteBlockDescriptor {
    /// Key identifying the block in remote storage
    key: RemoteKey,
    /// Size of the block in bytes
    size: usize,
    /// Optional metadata (for registry, debugging, etc.)
    metadata: Option<RemoteBlockMetadata>,
}

impl RemoteBlockDescriptor {
    /// Create a new descriptor with just key and size.
    pub fn new(key: RemoteKey, size: usize) -> Self {
        Self {
            key,
            size,
            metadata: None,
        }
    }

    /// Create with metadata.
    pub fn with_metadata(key: RemoteKey, size: usize, metadata: RemoteBlockMetadata) -> Self {
        Self {
            key,
            size,
            metadata: Some(metadata),
        }
    }

    /// Convenience: create object storage descriptor.
    pub fn object(bucket: impl Into<String>, key: impl Into<String>, size: usize) -> Self {
        Self::new(RemoteKey::object(bucket, key), size)
    }

    /// Convenience: create from sequence hash (common pattern).
    pub fn object_from_hash(bucket: impl Into<String>, hash: u64, size: usize) -> Self {
        let bucket = bucket.into();
        let mut desc = Self::new(RemoteKey::Object(ObjectKey::from_hash(&bucket, hash)), size);
        desc.metadata = Some(RemoteBlockMetadata::new(hash));
        desc
    }

    /// Convenience: create disk storage descriptor.
    pub fn disk(path: impl Into<String>, key: impl Into<String>, size: usize) -> Self {
        Self::new(RemoteKey::disk(path, key), size)
    }

    /// Convenience: create disk descriptor from hash.
    pub fn disk_from_hash(path: impl Into<String>, hash: u64, size: usize) -> Self {
        let path = path.into();
        let mut desc = Self::new(RemoteKey::Disk(DiskKey::from_hash(&path, hash)), size);
        desc.metadata = Some(RemoteBlockMetadata::new(hash));
        desc
    }

    /// Get the remote key.
    pub fn key(&self) -> &RemoteKey {
        &self.key
    }

    /// Get the block size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the storage kind.
    pub fn kind(&self) -> RemoteStorageKind {
        self.key.kind()
    }

    /// Get the metadata if present.
    pub fn metadata(&self) -> Option<&RemoteBlockMetadata> {
        self.metadata.as_ref()
    }

    /// Set the metadata.
    pub fn set_metadata(&mut self, metadata: RemoteBlockMetadata) {
        self.metadata = Some(metadata);
    }

    /// Get sequence hash from metadata (if available).
    pub fn sequence_hash(&self) -> Option<u64> {
        self.metadata.as_ref().map(|m| m.sequence_hash)
    }
}

// =============================================================================
// Remote Transfer Direction
// =============================================================================

/// Direction of remote transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteTransferDirection {
    /// Remote → Local (onboard/restore)
    Onboard,
    /// Local → Remote (offload/persist)
    Offload,
}

impl RemoteTransferDirection {
    /// Check if this is an onboard (read from remote) direction.
    pub fn is_onboard(&self) -> bool {
        matches!(self, Self::Onboard)
    }

    /// Check if this is an offload (write to remote) direction.
    pub fn is_offload(&self) -> bool {
        matches!(self, Self::Offload)
    }
}

/// Defines a complete transfer pipeline for remote storage.
///
/// Supports:
/// - Remote -> Local (onboard to host)
/// - Local -> Remote (offload from host)
/// - Remote -> Bounce -> Device (full onboard pipeline)
/// - Device -> Bounce -> Remote (full offload pipeline)
#[derive(Debug, Clone)]
pub enum RemoteTransferPipeline {
    /// Direct: Remote ↔ Host (no GPU involvement)
    Direct {
        direction: RemoteTransferDirection,
        remote_descriptors: Vec<RemoteBlockDescriptor>,
    },

    /// Full pipeline: Remote ↔ Host ↔ Device
    WithBounce {
        direction: RemoteTransferDirection,
        remote_descriptors: Vec<RemoteBlockDescriptor>,
        /// Host block IDs used as bounce buffers
        bounce_block_ids: Vec<usize>,
        /// Device block IDs (final destination for onboard, source for offload)
        device_block_ids: Vec<usize>,
    },
}

impl RemoteTransferPipeline {
    /// Create a direct Host → Remote offload.
    pub fn offload_direct(descriptors: Vec<RemoteBlockDescriptor>) -> Self {
        Self::Direct {
            direction: RemoteTransferDirection::Offload,
            remote_descriptors: descriptors,
        }
    }

    /// Create a direct Remote → Host onboard.
    pub fn onboard_direct(descriptors: Vec<RemoteBlockDescriptor>) -> Self {
        Self::Direct {
            direction: RemoteTransferDirection::Onboard,
            remote_descriptors: descriptors,
        }
    }

    /// Create a full Device → Bounce → Remote offload.
    pub fn offload_with_bounce(
        descriptors: Vec<RemoteBlockDescriptor>,
        bounce_ids: Vec<usize>,
        device_ids: Vec<usize>,
    ) -> Self {
        Self::WithBounce {
            direction: RemoteTransferDirection::Offload,
            remote_descriptors: descriptors,
            bounce_block_ids: bounce_ids,
            device_block_ids: device_ids,
        }
    }

    /// Create a full Remote → Bounce → Device onboard.
    pub fn onboard_with_bounce(
        descriptors: Vec<RemoteBlockDescriptor>,
        bounce_ids: Vec<usize>,
        device_ids: Vec<usize>,
    ) -> Self {
        Self::WithBounce {
            direction: RemoteTransferDirection::Onboard,
            remote_descriptors: descriptors,
            bounce_block_ids: bounce_ids,
            device_block_ids: device_ids,
        }
    }

    /// Get the transfer direction.
    pub fn direction(&self) -> RemoteTransferDirection {
        match self {
            Self::Direct { direction, .. } => *direction,
            Self::WithBounce { direction, .. } => *direction,
        }
    }

    /// Get the remote descriptors.
    pub fn descriptors(&self) -> &[RemoteBlockDescriptor] {
        match self {
            Self::Direct {
                remote_descriptors, ..
            } => remote_descriptors,
            Self::WithBounce {
                remote_descriptors, ..
            } => remote_descriptors,
        }
    }

    /// Check if this pipeline uses bounce buffers.
    pub fn has_bounce(&self) -> bool {
        matches!(self, Self::WithBounce { .. })
    }

    /// Get the number of blocks in this pipeline.
    pub fn num_blocks(&self) -> usize {
        self.descriptors().len()
    }
}

// =============================================================================
// Remote Transfer Strategy
// =============================================================================

/// Strategy for remote transfers.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteTransferStrategy {
    NixlObjectRead,
    NixlObjectWrite,
    NixlDiskRead,
    NixlDiskWrite,
    Invalid,
}

impl RemoteTransferStrategy {
    /// Check if this is a read operation.
    pub fn is_read(&self) -> bool {
        matches!(self, Self::NixlObjectRead | Self::NixlDiskRead)
    }

    /// Check if this is a write operation.
    pub fn is_write(&self) -> bool {
        matches!(self, Self::NixlObjectWrite | Self::NixlDiskWrite)
    }

    /// Check if this is an object storage operation.
    pub fn is_object(&self) -> bool {
        matches!(self, Self::NixlObjectRead | Self::NixlObjectWrite)
    }

    /// Check if this is a disk operation.
    pub fn is_disk(&self) -> bool {
        matches!(self, Self::NixlDiskRead | Self::NixlDiskWrite)
    }

    /// Determine strategy from direction and storage kind.
    pub fn from_direction_and_kind(
        direction: RemoteTransferDirection,
        kind: RemoteStorageKind,
    ) -> Self {
        match (direction, kind) {
            (RemoteTransferDirection::Onboard, RemoteStorageKind::Object) => Self::NixlObjectRead,
            (RemoteTransferDirection::Offload, RemoteStorageKind::Object) => Self::NixlObjectWrite,
            (RemoteTransferDirection::Onboard, RemoteStorageKind::Disk) => Self::NixlDiskRead,
            (RemoteTransferDirection::Offload, RemoteStorageKind::Disk) => Self::NixlDiskWrite,
        }
    }
}

/// Handle for a remote transfer operation.
///
/// Provides both completion notification and cancellation capability.
/// The transfer runs in the background; the caller can:
/// - `.await` on `wait()` to wait for the transfer to finish
/// - Call `cancel()` to request cooperative cancellation
/// - Drop the handle (transfer continues in background)
#[derive(Debug)]
pub struct RemoteTransferHandle {
    /// Receiver that completes when transfer finishes (Ok) or fails (Err)
    completion: oneshot::Receiver<Result<(), TransferError>>,
    /// Token to request cancellation
    cancel_token: CancellationToken,
}

impl RemoteTransferHandle {
    /// Create a new transfer handle.
    pub(crate) fn new(
        completion: oneshot::Receiver<Result<(), TransferError>>,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            completion,
            cancel_token,
        }
    }

    /// Request cancellation of the transfer.
    ///
    /// This is **cooperative** - the transfer will stop at the next
    /// cancellation check point (typically after current NIXL operation).
    ///
    /// Cancellation is idempotent - calling multiple times is safe.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Get a clone of the cancellation token.
    ///
    /// Useful for passing to child operations or integrating with `select!`.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Wait for the transfer to complete.
    ///
    /// Returns `Ok(())` on success, `Err(TransferError::Cancelled)` if
    /// cancelled, or other `TransferError` variants on failure.
    pub async fn wait(self) -> Result<(), TransferError> {
        self.completion.await.map_err(|_| {
            TransferError::ExecutionError("Transfer task dropped".to_string())
        })?
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_key_from_hash() {
        let key = ObjectKey::from_hash("my-bucket", 0x1234567890abcdef);
        assert_eq!(key.bucket, "my-bucket");
        assert_eq!(key.key, "1234567890abcdef");
        assert_eq!(key.as_hash(), Some(0x1234567890abcdef));
    }

    #[test]
    fn test_disk_key_full_path() {
        let key = DiskKey::new("/mnt/nfs", "block_001");
        assert_eq!(key.full_path(), "/mnt/nfs/block_001");
    }

    #[test]
    fn test_remote_key_kind() {
        let obj_key = RemoteKey::object("bucket", "key");
        assert_eq!(obj_key.kind(), RemoteStorageKind::Object);

        let disk_key = RemoteKey::disk("/path", "key");
        assert_eq!(disk_key.kind(), RemoteStorageKind::Disk);
    }

    #[test]
    fn test_remote_block_descriptor() {
        let desc = RemoteBlockDescriptor::object_from_hash("bucket", 0x1234, 4096);
        assert_eq!(desc.size(), 4096);
        assert_eq!(desc.kind(), RemoteStorageKind::Object);
        assert_eq!(desc.sequence_hash(), Some(0x1234));
    }

    #[test]
    fn test_remote_transfer_pipeline() {
        let descs = vec![
            RemoteBlockDescriptor::object_from_hash("bucket", 0x1234, 4096),
            RemoteBlockDescriptor::object_from_hash("bucket", 0x5678, 4096),
        ];

        // Direct offload
        let pipeline = RemoteTransferPipeline::offload_direct(descs.clone());
        assert_eq!(pipeline.direction(), RemoteTransferDirection::Offload);
        assert!(!pipeline.has_bounce());
        assert_eq!(pipeline.num_blocks(), 2);

        // With bounce
        let pipeline = RemoteTransferPipeline::onboard_with_bounce(
            descs,
            vec![0, 1],
            vec![10, 11],
        );
        assert_eq!(pipeline.direction(), RemoteTransferDirection::Onboard);
        assert!(pipeline.has_bounce());
    }

    #[test]
    fn test_remote_transfer_strategy() {
        assert!(RemoteTransferStrategy::NixlObjectRead.is_read());
        assert!(RemoteTransferStrategy::NixlObjectRead.is_object());
        assert!(!RemoteTransferStrategy::NixlObjectRead.is_write());
        assert!(!RemoteTransferStrategy::NixlObjectRead.is_disk());

        assert!(RemoteTransferStrategy::NixlDiskWrite.is_write());
        assert!(RemoteTransferStrategy::NixlDiskWrite.is_disk());

        // Test from_direction_and_kind
        assert_eq!(
            RemoteTransferStrategy::from_direction_and_kind(
                RemoteTransferDirection::Onboard,
                RemoteStorageKind::Object,
            ),
            RemoteTransferStrategy::NixlObjectRead
        );
        assert_eq!(
            RemoteTransferStrategy::from_direction_and_kind(
                RemoteTransferDirection::Offload,
                RemoteStorageKind::Disk,
            ),
            RemoteTransferStrategy::NixlDiskWrite
        );
    }

    #[test]
    fn test_remote_block_metadata() {
        let meta = RemoteBlockMetadata::new(0x1234);
        assert_eq!(meta.sequence_hash, 0x1234);
        assert!(meta.stored_at > 0);
    }

    #[test]
    fn test_remote_transfer_direction() {
        assert!(RemoteTransferDirection::Onboard.is_onboard());
        assert!(!RemoteTransferDirection::Onboard.is_offload());
        assert!(RemoteTransferDirection::Offload.is_offload());
        assert!(!RemoteTransferDirection::Offload.is_onboard());
    }
}

