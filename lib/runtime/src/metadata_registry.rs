// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side index from a model metadata file's identity to its
//! on-disk path. When a worker self-hosts metadata, it registers each
//! file here and rewrites the MDC's `CheckedFile.path` to a
//! `/v1/metadata/{slug}/{suffix}/{filename}` URL on its own
//! `system_status_server`. The route handler reads paths back out by
//! the same key and streams the bytes to the frontend, which
//! blake3-verifies them against the MDC.
//!
//! `suffix` is the LoRA slug (or `"_base"` for non-LoRA). It scopes
//! each registration so detaching a LoRA doesn't unregister the base
//! model's files (or vice versa).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;

/// Sentinel `suffix` for non-LoRA registrations. LoRA suffixes are
/// `Slug::slugify` outputs (`[a-z0-9_-]+`); a name that slugifies to
/// `_base` would collide with this sentinel and is not supported.
pub const BASE_SUFFIX: &str = "_base";

/// `(slug, suffix, filename)`.
type Key = (String, String, String);

/// Cloning shares the underlying map.
#[derive(Clone, Debug, Default)]
pub struct MetadataArtifactRegistry {
    entries: Arc<RwLock<HashMap<Key, PathBuf>>>,
}

impl MetadataArtifactRegistry {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register(&self, slug: &str, suffix: &str, filename: &str, path: PathBuf) {
        let mut entries = self.entries.write();
        entries.insert(
            (slug.to_string(), suffix.to_string(), filename.to_string()),
            path,
        );
        tracing::debug!(slug, suffix, filename, "registered metadata artifact");
    }

    pub fn get(&self, slug: &str, suffix: &str, filename: &str) -> Option<PathBuf> {
        let entries = self.entries.read();
        entries
            .get(&(slug.to_string(), suffix.to_string(), filename.to_string()))
            .cloned()
    }

    /// Drop entries for a single registration scoped by `(slug, suffix)`.
    pub fn unregister(&self, slug: &str, suffix: &str) {
        let mut entries = self.entries.write();
        entries.retain(|(s, sx, _), _| !(s == slug && sx == suffix));
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_get_roundtrip() {
        let reg = MetadataArtifactRegistry::new();
        let p = PathBuf::from("/tmp/tokenizer.json");
        reg.register("llama-3-8b", "_base", "tokenizer.json", p.clone());

        assert_eq!(reg.get("llama-3-8b", "_base", "tokenizer.json"), Some(p));
        assert!(reg.get("llama-3-8b", "_base", "missing.json").is_none());
        assert!(reg.get("llama-3-8b", "lora-v1", "tokenizer.json").is_none());
    }

    #[test]
    fn unregister_only_removes_matching_suffix() {
        let reg = MetadataArtifactRegistry::new();
        reg.register("m", "_base", "config.json", PathBuf::from("/m/c"));
        reg.register("m", "_base", "tokenizer.json", PathBuf::from("/m/t"));
        reg.register("m", "lora-v1", "config.json", PathBuf::from("/m/c"));

        reg.unregister("m", "_base");

        assert!(reg.get("m", "_base", "config.json").is_none());
        assert!(reg.get("m", "_base", "tokenizer.json").is_none());
        // LoRA entry on the same slug survives detach of the base.
        assert_eq!(
            reg.get("m", "lora-v1", "config.json"),
            Some(PathBuf::from("/m/c"))
        );
        assert_eq!(reg.len(), 1);
    }
}
