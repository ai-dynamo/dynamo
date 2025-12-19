// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder for creating RemoteKeys for all workers from TP config.
//!
//! This module solves the cross-bucket 404 problem by generating candidate
//! keys for all possible workers when looking up sequence hashes.
//!
//! # Example
//!
//! ```ignore
//! let builder = RemoteKeyBuilder::from_object_config(
//!     "kvcache-worker-{worker_id}".to_string(),
//!     4,  // 4 workers
//! );
//!
//! // Generate keys for hash 12345 across all workers
//! let keys = builder.build_all(12345);
//! // keys contains 4 RemoteKey::Object entries for buckets:
//! // - kvcache-worker-0/12345
//! // - kvcache-worker-1/12345
//! // - kvcache-worker-2/12345
//! // - kvcache-worker-3/12345
//! ```

use crate::block_manager::block::transfer::remote::{DiskKey, ObjectKey, RemoteKey};

/// Builder for creating RemoteKeys for all workers from TP config.
#[derive(Debug, Clone)]
pub struct RemoteKeyBuilder {
    template: RemoteKeyTemplate,
    num_workers: u32,
}

/// Template for generating RemoteKeys.
#[derive(Debug, Clone)]
pub enum RemoteKeyTemplate {
    /// Object storage template with bucket name containing {worker_id}.
    Object { bucket_template: String },
    /// Disk storage template with path containing {worker_id}.
    Disk { path_template: String },
}

impl RemoteKeyBuilder {
    /// Create a builder for object storage with a bucket template.
    ///
    /// The template can contain `{worker_id}` which will be substituted
    /// with the worker's ID (0, 1, 2, ...).
    ///
    /// # Arguments
    ///
    /// * `bucket_template` - Bucket name template (e.g., "kvcache-worker-{worker_id}")
    /// * `tp_size` - Number of workers (tensor parallelism size)
    pub fn from_object_config(bucket_template: String, tp_size: u32) -> Self {
        Self {
            template: RemoteKeyTemplate::Object { bucket_template },
            num_workers: tp_size,
        }
    }

    /// Create a builder for disk storage with a path template.
    ///
    /// # Arguments
    ///
    /// * `path_template` - Path template (e.g., "/mnt/nfs/worker-{worker_id}")
    /// * `tp_size` - Number of workers
    pub fn from_disk_config(path_template: String, tp_size: u32) -> Self {
        Self {
            template: RemoteKeyTemplate::Disk { path_template },
            num_workers: tp_size,
        }
    }

    /// Get the number of workers.
    pub fn num_workers(&self) -> u32 {
        self.num_workers
    }

    /// Build RemoteKeys for a sequence hash across all workers.
    ///
    /// Returns one RemoteKey per worker.
    pub fn build_all(&self, sequence_hash: u64) -> Vec<RemoteKey> {
        (0..self.num_workers)
            .map(|id| self.build_for_worker(id, sequence_hash))
            .collect()
    }

    /// Build a RemoteKey for a specific worker.
    pub fn build_for_worker(&self, worker_id: u32, sequence_hash: u64) -> RemoteKey {
        match &self.template {
            RemoteKeyTemplate::Object { bucket_template } => {
                let bucket = bucket_template.replace("{worker_id}", &worker_id.to_string());
                RemoteKey::Object(ObjectKey::from_hash(bucket, sequence_hash))
            }
            RemoteKeyTemplate::Disk { path_template } => {
                let path = path_template.replace("{worker_id}", &worker_id.to_string());
                RemoteKey::Disk(DiskKey::from_hash(path, sequence_hash))
            }
        }
    }

    /// Extract worker ID from a bucket name using the template pattern.
    ///
    /// Returns None if the bucket doesn't match the template pattern.
    pub fn extract_worker_id(&self, bucket: &str) -> Option<u32> {
        match &self.template {
            RemoteKeyTemplate::Object { bucket_template } => {
                // Split template by {worker_id} placeholder
                let parts: Vec<&str> = bucket_template.split("{worker_id}").collect();
                if parts.len() != 2 {
                    return None;
                }

                let prefix = parts[0];
                let suffix = parts[1];

                if !bucket.starts_with(prefix) || !bucket.ends_with(suffix) {
                    return None;
                }

                let worker_id_str = &bucket[prefix.len()..bucket.len() - suffix.len()];
                worker_id_str.parse().ok()
            }
            RemoteKeyTemplate::Disk { path_template } => {
                let parts: Vec<&str> = path_template.split("{worker_id}").collect();
                if parts.len() != 2 {
                    return None;
                }

                let prefix = parts[0];
                let suffix = parts[1];

                if !bucket.starts_with(prefix) || !bucket.ends_with(suffix) {
                    return None;
                }

                let worker_id_str = &bucket[prefix.len()..bucket.len() - suffix.len()];
                worker_id_str.parse().ok()
            }
        }
    }

    /// Get the bucket/path for a specific worker.
    pub fn location_for_worker(&self, worker_id: u32) -> String {
        match &self.template {
            RemoteKeyTemplate::Object { bucket_template } => {
                bucket_template.replace("{worker_id}", &worker_id.to_string())
            }
            RemoteKeyTemplate::Disk { path_template } => {
                path_template.replace("{worker_id}", &worker_id.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_builder() {
        let builder =
            RemoteKeyBuilder::from_object_config("kvcache-worker-{worker_id}".to_string(), 4);

        let key = builder.build_for_worker(2, 0x1234567890abcdef);

        match key {
            RemoteKey::Object(obj) => {
                assert_eq!(obj.bucket, "kvcache-worker-2");
                assert_eq!(obj.key, "1234567890abcdef");
            }
            _ => panic!("Expected Object key"),
        }
    }

    #[test]
    fn test_build_all() {
        let builder =
            RemoteKeyBuilder::from_object_config("kvcache-worker-{worker_id}".to_string(), 3);

        let keys = builder.build_all(0xabcd);

        assert_eq!(keys.len(), 3);

        for (i, key) in keys.iter().enumerate() {
            match key {
                RemoteKey::Object(obj) => {
                    assert_eq!(obj.bucket, format!("kvcache-worker-{}", i));
                    assert_eq!(obj.key, "000000000000abcd");
                }
                _ => panic!("Expected Object key"),
            }
        }
    }

    #[test]
    fn test_disk_builder() {
        let builder = RemoteKeyBuilder::from_disk_config("/mnt/nfs/worker-{worker_id}".to_string(), 2);

        let key = builder.build_for_worker(1, 0x9999);

        match key {
            RemoteKey::Disk(disk) => {
                assert_eq!(disk.path, "/mnt/nfs/worker-1");
                assert_eq!(disk.key, "0000000000009999");
            }
            _ => panic!("Expected Disk key"),
        }
    }

    #[test]
    fn test_extract_worker_id() {
        let builder =
            RemoteKeyBuilder::from_object_config("kvcache-worker-{worker_id}".to_string(), 4);

        assert_eq!(builder.extract_worker_id("kvcache-worker-0"), Some(0));
        assert_eq!(builder.extract_worker_id("kvcache-worker-3"), Some(3));
        assert_eq!(builder.extract_worker_id("kvcache-worker-12"), Some(12));
        assert_eq!(builder.extract_worker_id("other-bucket"), None);
    }

    #[test]
    fn test_location_for_worker() {
        let builder =
            RemoteKeyBuilder::from_object_config("kvcache-worker-{worker_id}".to_string(), 4);

        assert_eq!(builder.location_for_worker(0), "kvcache-worker-0");
        assert_eq!(builder.location_for_worker(3), "kvcache-worker-3");
    }
}



