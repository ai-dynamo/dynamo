// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multimodal content hashing for KV router cache differentiation.
//!
//! This module provides extensible hashing mechanisms to distinguish requests
//! with different multimodal content, preventing false cache hits when token
//! sequences are identical but multimedia content differs.

use crate::kv_router::indexer::compute_hash;
use crate::protocols::common::preprocessor::{MultimodalData, MultimodalDataMap};

/// Trait for computing hashes of multimodal content.
///
/// Implementations can use different strategies (URL-based, content-based, etc.)
/// to generate deterministic hashes that distinguish different multimodal inputs.
pub trait MultimodalHasher {
    /// Compute a hash from multimodal data.
    ///
    /// ### Arguments
    ///
    /// * `data` - A map of multimodal data by type (e.g., "image_url", "video_url")
    ///
    /// ### Returns
    ///
    /// A 64-bit hash value, or None if the data is empty or invalid
    fn hash(&self, data: &MultimodalDataMap) -> Option<u64>;
}

/// Default hasher that computes hashes based on URL strings.
///
/// This implementation is fast and deterministic, suitable for scenarios where
/// the same URL always refers to the same content. It hashes the URLs in a
/// consistent order to ensure deterministic results.
pub struct UrlBasedHasher;

impl UrlBasedHasher {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UrlBasedHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl MultimodalHasher for UrlBasedHasher {
    fn hash(&self, data: &MultimodalDataMap) -> Option<u64> {
        if data.is_empty() {
            return None;
        }

        // Collect all URLs in a deterministic order
        let mut all_urls = Vec::new();

        // Sort keys to ensure deterministic ordering
        let mut keys: Vec<&String> = data.keys().collect();
        keys.sort();

        for key in keys {
            if let Some(media_items) = data.get(key) {
                for item in media_items {
                    match item {
                        MultimodalData::Url(url) => {
                            all_urls.push(url.as_str());
                        }
                    }
                }
            }
        }

        if all_urls.is_empty() {
            return None;
        }

        // Concatenate all URLs and hash the result
        let concatenated = all_urls.join("|");
        Some(compute_hash(concatenated.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use url::Url;

    #[test]
    fn test_url_based_hasher_empty() {
        let hasher = UrlBasedHasher::new();
        let data = HashMap::new();
        assert_eq!(hasher.hash(&data), None);
    }

    #[test]
    fn test_url_based_hasher_single_url() {
        let hasher = UrlBasedHasher::new();
        let mut data = HashMap::new();
        data.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/image1.jpg").unwrap(),
            )],
        );

        let hash = hasher.hash(&data);
        assert!(hash.is_some());
    }

    #[test]
    fn test_url_based_hasher_deterministic() {
        let hasher = UrlBasedHasher::new();
        let mut data = HashMap::new();
        data.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/image1.jpg").unwrap(),
            )],
        );

        let hash1 = hasher.hash(&data);
        let hash2 = hasher.hash(&data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_url_based_hasher_different_urls() {
        let hasher = UrlBasedHasher::new();

        let mut data1 = HashMap::new();
        data1.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/image1.jpg").unwrap(),
            )],
        );

        let mut data2 = HashMap::new();
        data2.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/image2.jpg").unwrap(),
            )],
        );

        let hash1 = hasher.hash(&data1);
        let hash2 = hasher.hash(&data2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_url_based_hasher_multiple_urls_order_independent() {
        let hasher = UrlBasedHasher::new();

        // Create data with keys inserted in different order
        let mut data1 = HashMap::new();
        data1.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/image1.jpg").unwrap(),
            )],
        );
        data1.insert(
            "video_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/video1.mp4").unwrap(),
            )],
        );

        let mut data2 = HashMap::new();
        data2.insert(
            "video_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/video1.mp4").unwrap(),
            )],
        );
        data2.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/image1.jpg").unwrap(),
            )],
        );

        let hash1 = hasher.hash(&data1);
        let hash2 = hasher.hash(&data2);
        assert_eq!(hash1, hash2, "Hash should be order-independent");
    }

    #[test]
    fn test_url_based_hasher_multiple_items_same_type() {
        let hasher = UrlBasedHasher::new();

        let mut data = HashMap::new();
        data.insert(
            "image_url".to_string(),
            vec![
                MultimodalData::Url(Url::parse("http://example.com/image1.jpg").unwrap()),
                MultimodalData::Url(Url::parse("http://example.com/image2.jpg").unwrap()),
            ],
        );

        let hash = hasher.hash(&data);
        assert!(hash.is_some());
    }

    #[test]
    fn test_mm_hash_prevents_false_cache_hits() {
        let hasher = UrlBasedHasher::new();

        // Two requests with same text but different images
        let mut data1 = HashMap::new();
        data1.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/cat.jpg").unwrap(),
            )],
        );

        let mut data2 = HashMap::new();
        data2.insert(
            "image_url".to_string(),
            vec![MultimodalData::Url(
                Url::parse("http://example.com/dog.jpg").unwrap(),
            )],
        );

        let hash1 = hasher.hash(&data1).unwrap();
        let hash2 = hasher.hash(&data2).unwrap();

        // The hashes must be different to prevent false cache hits
        assert_ne!(
            hash1, hash2,
            "Different MM content must produce different hashes"
        );
    }
}
