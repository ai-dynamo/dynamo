// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local registry for self-hosted model metadata files.
//!
//! When a worker registers a model with `self_host_metadata=true`,
//! the runtime inserts one entry per artifact mapping
//! `(model_slug, filename)` to the on-disk path of that file. The
//! `system_status_server`'s `/v1/metadata/{model_slug}/{filename}`
//! handler reads from this registry, opens the file, and serves the
//! bytes; misses return 404.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Process-local map from `(model_slug, filename)` → on-disk path of
/// a metadata artifact registered for self-hosting. Cloning the
/// registry shares the same underlying map.
#[derive(Clone, Default)]
pub struct MetadataArtifactRegistry {
    entries: Arc<RwLock<HashMap<(String, String), PathBuf>>>,
}

impl MetadataArtifactRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register one artifact for self-hosting. Inserts (or replaces)
    /// the on-disk path the route handler will open when serving
    /// `GET /v1/metadata/{model_slug}/{filename}`.
    pub fn register(&self, model_slug: &str, filename: &str, path: PathBuf) {
        let mut entries = self.entries.write().unwrap();
        entries.insert((model_slug.to_string(), filename.to_string()), path);
        tracing::debug!(
            model_slug,
            filename,
            "registered metadata artifact for self-host"
        );
    }

    /// Look up the on-disk path for a `(model_slug, filename)` pair.
    /// Returns `None` if no such entry has been registered.
    pub fn get(&self, model_slug: &str, filename: &str) -> Option<PathBuf> {
        let entries = self.entries.read().unwrap();
        entries
            .get(&(model_slug.to_string(), filename.to_string()))
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_get_roundtrip() {
        let reg = MetadataArtifactRegistry::new();
        let p = PathBuf::from("/tmp/tokenizer.json");
        reg.register("llama-3-8b", "tokenizer.json", p.clone());

        assert_eq!(reg.get("llama-3-8b", "tokenizer.json"), Some(p));
    }

    #[test]
    fn miss_returns_none() {
        let reg = MetadataArtifactRegistry::new();
        assert!(reg.get("unknown", "nope.json").is_none());
    }

    #[test]
    fn miss_on_known_model_unknown_file() {
        let reg = MetadataArtifactRegistry::new();
        reg.register("m", "config.json", PathBuf::from("/x"));
        assert!(reg.get("m", "tokenizer.json").is_none());
        assert_eq!(reg.get("m", "config.json"), Some(PathBuf::from("/x")));
    }

    #[test]
    fn multi_model_no_collision() {
        let reg = MetadataArtifactRegistry::new();
        let p1 = PathBuf::from("/m1/tokenizer.json");
        let p2 = PathBuf::from("/m2/tokenizer.json");
        reg.register("m1", "tokenizer.json", p1.clone());
        reg.register("m2", "tokenizer.json", p2.clone());

        assert_eq!(reg.get("m1", "tokenizer.json"), Some(p1));
        assert_eq!(reg.get("m2", "tokenizer.json"), Some(p2));
    }
}
