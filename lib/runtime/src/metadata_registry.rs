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

/// `(instance_id, lora_slug)`. `None` lora_slug = base model.
pub type Owner = (u64, Option<String>);

/// Cloning shares the underlying map.
#[derive(Clone, Debug, Default)]
pub struct MetadataArtifactRegistry {
    // Key is a single concatenated `{slug}\0{suffix}\0{filename}` string —
    // one allocation per lookup instead of a 3-`String` tuple.
    entries: Arc<RwLock<HashMap<String, (PathBuf, Owner)>>>,
}

fn make_key(slug: &str, suffix: &str, filename: &str) -> String {
    let mut s = String::with_capacity(slug.len() + suffix.len() + filename.len() + 2);
    s.push_str(slug);
    s.push('\0');
    s.push_str(suffix);
    s.push('\0');
    s.push_str(filename);
    s
}

impl MetadataArtifactRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Panics on owner collision — two `LocalModel` instances attaching the
    /// same (slug, suffix, filename) in one process would let detach-#1
    /// wipe files detach-#2 still needs. Same-owner re-register is fine
    /// (path update).
    pub fn register(&self, owner: &Owner, slug: &str, suffix: &str, filename: &str, path: PathBuf) {
        let key = make_key(slug, suffix, filename);
        let mut entries = self.entries.write();
        if let Some((_, prior)) = entries.get(&key) {
            assert_eq!(
                prior, owner,
                "metadata-registry collision on {key:?}: prior={prior:?}, new={owner:?}",
            );
        }
        entries.insert(key, (path, owner.clone()));
        tracing::debug!(slug, suffix, filename, "registered metadata artifact");
    }

    pub fn get(&self, slug: &str, suffix: &str, filename: &str) -> Option<PathBuf> {
        self.entries
            .read()
            .get(&make_key(slug, suffix, filename))
            .map(|(p, _)| p.clone())
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
