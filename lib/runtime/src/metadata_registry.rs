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
//!
//! Each entry also stores its `Owner = (instance_id, lora_slug)` so
//! `unregister_for_owner` can clean up on detach without the caller
//! threading the model slug. Each `(slug, suffix, filename)` key must
//! have at most one owner — `register` panics on collision with a
//! different owner.

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

/// `(instance_id, lora_slug)`. `None` lora_slug = base model.
pub type Owner = (u64, Option<String>);

/// Cloning shares the underlying map.
#[derive(Clone, Debug, Default)]
pub struct MetadataArtifactRegistry {
    entries: Arc<RwLock<HashMap<Key, (PathBuf, Owner)>>>,
}

impl MetadataArtifactRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Panics if `(slug, suffix, filename)` is already registered by a
    /// different owner — two `LocalModel` instances attaching the same
    /// model + LoRA suffix in one process would let detach-#1 wipe
    /// files that detach-#2 still needs. Re-registering the same
    /// (slug, suffix, filename) by the *same* owner just updates the
    /// path and is fine.
    pub fn register(&self, owner: &Owner, slug: &str, suffix: &str, filename: &str, path: PathBuf) {
        let key = (slug.to_string(), suffix.to_string(), filename.to_string());
        let mut entries = self.entries.write();
        if let Some((_, prior)) = entries.get(&key) {
            assert_eq!(
                prior, owner,
                "metadata-registry collision on {key:?}: prior owner {prior:?} \
                 differs from new owner {owner:?}; two attaches of the same \
                 model+suffix in one process are not supported",
            );
        }
        entries.insert(key, (path, owner.clone()));
        tracing::debug!(slug, suffix, filename, "registered metadata artifact");
    }

    pub fn get(&self, slug: &str, suffix: &str, filename: &str) -> Option<PathBuf> {
        self.entries
            .read()
            .get(&(slug.to_string(), suffix.to_string(), filename.to_string()))
            .map(|(p, _)| p.clone())
    }

    /// Drop entries for a single registration scoped by `(slug, suffix)`.
    pub fn unregister(&self, slug: &str, suffix: &str) {
        self.entries
            .write()
            .retain(|(s, sx, _), _| !(s == slug && sx == suffix));
    }

    /// Drop every entry registered by `owner`. No-op if `owner` never
    /// registered (e.g. self-host was disabled or skipped).
    pub fn unregister_for_owner(&self, owner: &Owner) {
        self.entries.write().retain(|_, (_, o)| o != owner);
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

    fn base() -> Owner {
        (1, None)
    }

    fn lora(slug: &str) -> Owner {
        (1, Some(slug.to_string()))
    }

    #[test]
    fn register_get_roundtrip() {
        let reg = MetadataArtifactRegistry::new();
        let p = PathBuf::from("/tmp/tokenizer.json");
        reg.register(&base(), "llama-3-8b", "_base", "tokenizer.json", p.clone());

        assert_eq!(reg.get("llama-3-8b", "_base", "tokenizer.json"), Some(p));
        assert!(reg.get("llama-3-8b", "_base", "missing.json").is_none());
        assert!(reg.get("llama-3-8b", "lora-v1", "tokenizer.json").is_none());
    }

    #[test]
    fn unregister_only_removes_matching_suffix() {
        let reg = MetadataArtifactRegistry::new();
        reg.register(&base(), "m", "_base", "config.json", PathBuf::from("/m/c"));
        reg.register(
            &base(),
            "m",
            "_base",
            "tokenizer.json",
            PathBuf::from("/m/t"),
        );
        reg.register(
            &lora("lora-v1"),
            "m",
            "lora-v1",
            "config.json",
            PathBuf::from("/m/c"),
        );

        reg.unregister("m", "_base");

        assert!(reg.get("m", "_base", "config.json").is_none());
        assert!(reg.get("m", "_base", "tokenizer.json").is_none());
        assert_eq!(
            reg.get("m", "lora-v1", "config.json"),
            Some(PathBuf::from("/m/c"))
        );
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn unregister_for_owner_clears_only_that_owner() {
        let reg = MetadataArtifactRegistry::new();
        let lora_owner = lora("lora-v1");
        reg.register(&base(), "m", "_base", "config.json", PathBuf::from("/m/c"));
        reg.register(
            &base(),
            "m",
            "_base",
            "tokenizer.json",
            PathBuf::from("/m/t"),
        );
        reg.register(
            &lora_owner,
            "m",
            "lora-v1",
            "adapter.json",
            PathBuf::from("/m/a"),
        );

        reg.unregister_for_owner(&lora_owner);

        assert!(reg.get("m", "lora-v1", "adapter.json").is_none());
        assert_eq!(
            reg.get("m", "_base", "config.json"),
            Some(PathBuf::from("/m/c"))
        );
        // Idempotent — second call is a no-op.
        reg.unregister_for_owner(&lora_owner);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    #[should_panic(expected = "metadata-registry collision")]
    fn register_panics_on_owner_collision() {
        let reg = MetadataArtifactRegistry::new();
        let owner_a = (1, None);
        let owner_b = (2, None);
        reg.register(&owner_a, "m", "_base", "config.json", PathBuf::from("/a"));
        reg.register(&owner_b, "m", "_base", "config.json", PathBuf::from("/b"));
    }

    #[test]
    fn register_same_owner_updates_path() {
        let reg = MetadataArtifactRegistry::new();
        reg.register(&base(), "m", "_base", "config.json", PathBuf::from("/a"));
        reg.register(&base(), "m", "_base", "config.json", PathBuf::from("/b"));
        assert_eq!(
            reg.get("m", "_base", "config.json"),
            Some(PathBuf::from("/b"))
        );
    }
}
