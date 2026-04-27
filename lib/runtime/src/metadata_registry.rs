// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local `(model_slug, filename) -> PathBuf` registry backing
//! the system_status_server's `/v1/metadata/{model_slug}/{filename}`
//! handler.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Cloning shares the underlying map.
#[derive(Clone, Default)]
pub struct MetadataArtifactRegistry {
    entries: Arc<RwLock<HashMap<(String, String), PathBuf>>>,
}

impl MetadataArtifactRegistry {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register(&self, model_slug: &str, filename: &str, path: PathBuf) {
        let mut entries = self.entries.write().unwrap();
        entries.insert((model_slug.to_string(), filename.to_string()), path);
        tracing::debug!(model_slug, filename, "registered metadata artifact");
    }

    pub fn get(&self, model_slug: &str, filename: &str) -> Option<PathBuf> {
        let entries = self.entries.read().unwrap();
        entries
            .get(&(model_slug.to_string(), filename.to_string()))
            .cloned()
    }

    /// Drop all entries for a model.
    pub fn unregister_model(&self, model_slug: &str) {
        let mut entries = self.entries.write().unwrap();
        entries.retain(|(slug, _), _| slug != model_slug);
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
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
        assert!(reg.is_empty());
    }

    #[test]
    fn miss_on_known_model_unknown_file() {
        let reg = MetadataArtifactRegistry::new();
        reg.register("m", "config.json", PathBuf::from("/x"));
        assert!(reg.get("m", "tokenizer.json").is_none());
        assert_eq!(reg.len(), 1);
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
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn unregister_model_removes_only_its_entries() {
        let reg = MetadataArtifactRegistry::new();
        reg.register("m1", "config.json", PathBuf::from("/m1/c"));
        reg.register("m1", "tokenizer.json", PathBuf::from("/m1/t"));
        reg.register("m2", "config.json", PathBuf::from("/m2/c"));

        reg.unregister_model("m1");

        assert!(reg.get("m1", "config.json").is_none());
        assert!(reg.get("m1", "tokenizer.json").is_none());
        assert_eq!(reg.get("m2", "config.json"), Some(PathBuf::from("/m2/c")));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn clone_shares_state() {
        let reg = MetadataArtifactRegistry::new();
        let reg2 = reg.clone();
        reg.register("m", "f", PathBuf::from("/p"));
        assert_eq!(reg2.get("m", "f"), Some(PathBuf::from("/p")));
    }
}
