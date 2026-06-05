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
//! Detach uses [`MetadataArtifactRegistry::unregister_for_owner`]: each
//! registration tags itself with an `Owner = (connection_id, lora_slug)`
//! so the static `LocalModel::detach_from_endpoint` (which has those
//! two values but not the model slug) can clean up without an extra
//! parameter on the Python `unregister_model` API.

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

/// `(connection_id, lora_slug)` — identifies what `detach_from_endpoint`
/// has on hand. `None` lora_slug = base model.
pub type Owner = (u64, Option<String>);

/// Cloning shares the underlying maps.
#[derive(Clone, Debug, Default)]
pub struct MetadataArtifactRegistry {
    entries: Arc<RwLock<HashMap<Key, PathBuf>>>,
    owners: Arc<RwLock<HashMap<Owner, (String, String)>>>,
}

impl MetadataArtifactRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one file and tag the `(slug, suffix)` pair with `owner`
    /// so `unregister_for_owner` can find it on detach.
    pub fn register(&self, owner: Owner, slug: &str, suffix: &str, filename: &str, path: PathBuf) {
        self.entries.write().insert(
            (slug.to_string(), suffix.to_string(), filename.to_string()),
            path,
        );
        self.owners
            .write()
            .insert(owner, (slug.to_string(), suffix.to_string()));
        tracing::debug!(slug, suffix, filename, "registered metadata artifact");
    }

    pub fn get(&self, slug: &str, suffix: &str, filename: &str) -> Option<PathBuf> {
        self.entries
            .read()
            .get(&(slug.to_string(), suffix.to_string(), filename.to_string()))
            .cloned()
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
        if let Some((slug, suffix)) = self.owners.write().remove(owner) {
            self.unregister(&slug, &suffix);
        }
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

    const BASE: Owner = (1, None);

    #[test]
    fn register_get_roundtrip() {
        let reg = MetadataArtifactRegistry::new();
        let p = PathBuf::from("/tmp/tokenizer.json");
        reg.register(BASE, "llama-3-8b", "_base", "tokenizer.json", p.clone());

        assert_eq!(reg.get("llama-3-8b", "_base", "tokenizer.json"), Some(p));
        assert!(reg.get("llama-3-8b", "_base", "missing.json").is_none());
        assert!(reg.get("llama-3-8b", "lora-v1", "tokenizer.json").is_none());
    }

    #[test]
    fn unregister_only_removes_matching_suffix() {
        let reg = MetadataArtifactRegistry::new();
        let lora = (1, Some("lora-v1".to_string()));
        reg.register(BASE, "m", "_base", "config.json", PathBuf::from("/m/c"));
        reg.register(BASE, "m", "_base", "tokenizer.json", PathBuf::from("/m/t"));
        reg.register(lora, "m", "lora-v1", "config.json", PathBuf::from("/m/c"));

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

    #[test]
    fn unregister_for_owner_clears_only_that_owner() {
        // Two attaches in the same process (same connection_id): one base
        // plus one LoRA. Detaching the LoRA must leave the base intact —
        // the auto-cleanup hook on `LocalModel::detach_from_endpoint`
        // depends on this scoping.
        let reg = MetadataArtifactRegistry::new();
        let lora_owner = (1, Some("lora-v1".to_string()));
        reg.register(BASE, "m", "_base", "config.json", PathBuf::from("/m/c"));
        reg.register(BASE, "m", "_base", "tokenizer.json", PathBuf::from("/m/t"));
        reg.register(
            lora_owner.clone(),
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
}
