// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side index from a model metadata file's identity to its
//! on-disk path. When a worker self-hosts metadata, it registers each
//! file here and rewrites the MDC's `CheckedFile.path` to a
//! `/v1/metadata/{namespace}/{component}/{endpoint}/{slug}/{suffix}/{filename}`
//! URL on its own `system_status_server`. The route handler reads paths
//! back out by the same key and streams the bytes to the frontend, which
//! blake3-verifies them against the MDC.
//!
//! The endpoint triple in the key disambiguates multiple `LocalModel`
//! instances coexisting on one DRT (e.g. different roles attaching the
//! same base model). `suffix` is the LoRA slug (or `"_base"` for
//! non-LoRA), scoping each registration so detaching a LoRA doesn't
//! unregister the base model's files.
//!
//! Each entry also stores its `Owner = (instance_id, lora_slug)` so
//! `unregister_for_owner` can clean up on detach without the caller
//! threading the model slug. `register` returns
//! `Err(CollisionError)` on conflict — the caller propagates and the
//! worker fails to start.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use thiserror::Error;

/// Sentinel `suffix` for non-LoRA registrations. LoRA suffixes are
/// `Slug::slugify` outputs (`[a-z0-9_-]+`); a name that slugifies to
/// `_base` would collide with this sentinel and is not supported.
pub const BASE_SUFFIX: &str = "_base";

/// `(instance_id, lora_slug)`. `None` lora_slug = base model.
pub type Owner = (u64, Option<String>);

/// `(namespace, component, endpoint, slug, suffix, filename)`.
type Key = (String, String, String, String, String, String);

#[allow(clippy::too_many_arguments)]
fn make_key(
    namespace: &str,
    component: &str,
    endpoint: &str,
    slug: &str,
    suffix: &str,
    filename: &str,
) -> Key {
    (
        namespace.to_string(),
        component.to_string(),
        endpoint.to_string(),
        slug.to_string(),
        suffix.to_string(),
        filename.to_string(),
    )
}

/// Registration collided with a different owner — programmer error.
#[derive(Debug, Error)]
#[error("metadata-registry collision on key {key:?}: prior_owner={prior:?}, new_owner={new:?}")]
pub struct CollisionError {
    pub key: Key,
    pub prior: Owner,
    pub new: Owner,
}

/// Cloning shares the underlying map.
#[derive(Clone, Debug, Default)]
pub struct MetadataArtifactRegistry {
    entries: Arc<RwLock<HashMap<Key, (PathBuf, Owner)>>>,
}

impl MetadataArtifactRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `Err(CollisionError)` if a different owner already
    /// registered this key. Same-owner re-register updates the path.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        owner: &Owner,
        namespace: &str,
        component: &str,
        endpoint: &str,
        slug: &str,
        suffix: &str,
        filename: &str,
        path: PathBuf,
    ) -> Result<(), Box<CollisionError>> {
        let key = make_key(namespace, component, endpoint, slug, suffix, filename);
        let mut entries = self.entries.write();
        if let Some((_, prior)) = entries.get(&key)
            && prior != owner
        {
            return Err(Box::new(CollisionError {
                key,
                prior: prior.clone(),
                new: owner.clone(),
            }));
        }
        entries.insert(key, (path, owner.clone()));
        tracing::debug!(
            namespace,
            component,
            endpoint,
            slug,
            suffix,
            filename,
            "registered metadata artifact",
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get(
        &self,
        namespace: &str,
        component: &str,
        endpoint: &str,
        slug: &str,
        suffix: &str,
        filename: &str,
    ) -> Option<PathBuf> {
        let key = make_key(namespace, component, endpoint, slug, suffix, filename);
        self.entries.read().get(&key).map(|(p, _)| p.clone())
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
        reg.register(
            &base(),
            "ns",
            "comp",
            "ep",
            "llama-3-8b",
            "_base",
            "tokenizer.json",
            p.clone(),
        )
        .unwrap();

        assert_eq!(
            reg.get("ns", "comp", "ep", "llama-3-8b", "_base", "tokenizer.json"),
            Some(p)
        );
        assert!(
            reg.get("ns", "comp", "ep", "llama-3-8b", "_base", "missing.json")
                .is_none()
        );
        assert!(
            reg.get(
                "ns",
                "comp",
                "ep",
                "llama-3-8b",
                "lora-v1",
                "tokenizer.json"
            )
            .is_none()
        );
    }

    #[test]
    fn unregister_for_owner_clears_only_that_owner() {
        let reg = MetadataArtifactRegistry::new();
        let lora_owner = lora("lora-v1");
        reg.register(
            &base(),
            "ns",
            "comp",
            "ep",
            "m",
            "_base",
            "config.json",
            PathBuf::from("/m/c"),
        )
        .unwrap();
        reg.register(
            &base(),
            "ns",
            "comp",
            "ep",
            "m",
            "_base",
            "tokenizer.json",
            PathBuf::from("/m/t"),
        )
        .unwrap();
        reg.register(
            &lora_owner,
            "ns",
            "comp",
            "ep",
            "m",
            "lora-v1",
            "adapter.json",
            PathBuf::from("/m/a"),
        )
        .unwrap();

        reg.unregister_for_owner(&lora_owner);

        assert!(
            reg.get("ns", "comp", "ep", "m", "lora-v1", "adapter.json")
                .is_none()
        );
        assert_eq!(
            reg.get("ns", "comp", "ep", "m", "_base", "config.json"),
            Some(PathBuf::from("/m/c"))
        );
        // Idempotent — second call is a no-op.
        reg.unregister_for_owner(&lora_owner);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn register_returns_err_on_owner_collision() {
        let reg = MetadataArtifactRegistry::new();
        let owner_a = (1, None);
        let owner_b = (2, None);
        reg.register(
            &owner_a,
            "ns",
            "comp",
            "ep",
            "m",
            "_base",
            "config.json",
            PathBuf::from("/a"),
        )
        .unwrap();
        let err = reg
            .register(
                &owner_b,
                "ns",
                "comp",
                "ep",
                "m",
                "_base",
                "config.json",
                PathBuf::from("/b"),
            )
            .unwrap_err();
        assert_eq!(err.prior, owner_a);
        assert_eq!(err.new, owner_b);
    }

    #[test]
    fn register_same_owner_updates_path() {
        let reg = MetadataArtifactRegistry::new();
        reg.register(
            &base(),
            "ns",
            "comp",
            "ep",
            "m",
            "_base",
            "config.json",
            PathBuf::from("/a"),
        )
        .unwrap();
        reg.register(
            &base(),
            "ns",
            "comp",
            "ep",
            "m",
            "_base",
            "config.json",
            PathBuf::from("/b"),
        )
        .unwrap();
        assert_eq!(
            reg.get("ns", "comp", "ep", "m", "_base", "config.json"),
            Some(PathBuf::from("/b"))
        );
    }

    #[test]
    fn different_endpoints_coexist() {
        let reg = MetadataArtifactRegistry::new();
        let owner_a = (1, None);
        let owner_b = (2, None);
        reg.register(
            &owner_a,
            "ns",
            "comp",
            "ep-a",
            "m",
            "_base",
            "config.json",
            PathBuf::from("/a"),
        )
        .unwrap();
        // Same (slug, suffix, filename) but different endpoint → no collision.
        reg.register(
            &owner_b,
            "ns",
            "comp",
            "ep-b",
            "m",
            "_base",
            "config.json",
            PathBuf::from("/b"),
        )
        .unwrap();
        assert_eq!(
            reg.get("ns", "comp", "ep-a", "m", "_base", "config.json"),
            Some(PathBuf::from("/a"))
        );
        assert_eq!(
            reg.get("ns", "comp", "ep-b", "m", "_base", "config.json"),
            Some(PathBuf::from("/b"))
        );
    }
}
