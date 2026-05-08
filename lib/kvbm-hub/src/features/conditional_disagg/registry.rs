// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CD-feature peer state, separated from the orchestrator.
//!
//! `CdPeerRegistry` holds per-peer state that the conditional-disagg feature
//! needs (role + engine URL). It is held by both [`ConditionalDisaggManager`]
//! (which writes into it on register/unregister) and by the manager's
//! prefill drainer (which reads from it via [`PrefillPeerSource`]). The
//! registry holds no back-references, so the manager → drainer-task →
//! registry chain has no cycles.
//!
//! Mirrors dynamo's split between `Client` (discovery) and `PushRouter`
//! (policy): a focused state type that policy layers can consume without
//! reaching into orchestration.
//!
//! [`ConditionalDisaggManager`]: super::manager::ConditionalDisaggManager

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;
use velo_common::InstanceId;

use super::selector::{PrefillPeerEntry, PrefillPeerSource};
use crate::protocol::{ConditionalDisaggInstancesResponse, ConditionalDisaggRole};

/// Errors returned by [`CdPeerRegistry::insert`].
#[derive(Debug, thiserror::Error)]
pub enum CdRegistryError {
    /// Tried to register an instance under a different role than its
    /// existing entry.
    #[error("instance {instance_id} already registered as {existing:?}, cannot switch to {new:?}")]
    RoleConflict {
        instance_id: InstanceId,
        existing: ConditionalDisaggRole,
        new: ConditionalDisaggRole,
    },
}

/// CD-specific peer state. Holds nothing back; safe to share via `Arc` to
/// any number of readers.
pub struct CdPeerRegistry {
    inner: RwLock<CdRegistryInner>,
}

struct CdRegistryInner {
    prefill: HashSet<InstanceId>,
    decode: HashSet<InstanceId>,
    by_instance: HashMap<InstanceId, ConditionalDisaggRole>,
    /// Engine HTTP URLs for prefill peers (e.g. `http://10.0.0.42:8000`).
    /// Decode peers do not have entries here.
    engine_urls: HashMap<InstanceId, String>,
}

impl Default for CdPeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CdPeerRegistry {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(CdRegistryInner {
                prefill: HashSet::new(),
                decode: HashSet::new(),
                by_instance: HashMap::new(),
                engine_urls: HashMap::new(),
            }),
        }
    }

    /// Register a peer under a role. Idempotent for same-role re-registrations.
    /// Returns [`CdRegistryError::RoleConflict`] if the instance is already
    /// registered under a different role.
    ///
    /// `engine_url` is stored only when `role == Prefill` and the value is
    /// `Some`. Decode peers and prefills without an advertised URL get no
    /// entry in the URL map (and are filtered out of [`Self::prefill_peers`]).
    pub fn insert(
        &self,
        instance_id: InstanceId,
        role: ConditionalDisaggRole,
        engine_url: Option<String>,
    ) -> Result<(), CdRegistryError> {
        let mut inner = self.inner.write();
        if let Some(existing) = inner.by_instance.get(&instance_id).copied() {
            if existing != role {
                return Err(CdRegistryError::RoleConflict {
                    instance_id,
                    existing,
                    new: role,
                });
            }
            // Same-role re-registration is idempotent; leave URL untouched.
            return Ok(());
        }
        inner.by_instance.insert(instance_id, role);
        match role {
            ConditionalDisaggRole::Prefill => {
                inner.prefill.insert(instance_id);
                if let Some(url) = engine_url {
                    inner.engine_urls.insert(instance_id, url);
                }
            }
            ConditionalDisaggRole::Decode => {
                inner.decode.insert(instance_id);
            }
        }
        Ok(())
    }

    /// Remove a peer. No-op if not present.
    pub fn remove(&self, instance_id: InstanceId) {
        let mut inner = self.inner.write();
        if let Some(role) = inner.by_instance.remove(&instance_id) {
            match role {
                ConditionalDisaggRole::Prefill => {
                    inner.prefill.remove(&instance_id);
                }
                ConditionalDisaggRole::Decode => {
                    inner.decode.remove(&instance_id);
                }
            }
        }
        inner.engine_urls.remove(&instance_id);
    }

    /// Snapshot the role split.
    pub fn snapshot(&self) -> ConditionalDisaggInstancesResponse {
        let inner = self.inner.read();
        ConditionalDisaggInstancesResponse {
            prefill: inner.prefill.iter().copied().collect(),
            decode: inner.decode.iter().copied().collect(),
        }
    }
}

/// Snapshot the live prefill peers as `PrefillPeerEntry`s, dropping any
/// that have no `engine_url` registered (and are therefore not dispatchable).
impl PrefillPeerSource for CdPeerRegistry {
    fn prefill_peers(&self) -> Vec<PrefillPeerEntry> {
        let inner = self.inner.read();
        inner
            .prefill
            .iter()
            .filter_map(|id| {
                let engine_url = inner.engine_urls.get(id)?.clone();
                Some(PrefillPeerEntry {
                    instance_id: *id,
                    engine_url,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> InstanceId {
        InstanceId::new_v4()
    }

    #[test]
    fn insert_prefill_with_url_makes_dispatchable() {
        let reg = CdPeerRegistry::new();
        let pid = id();
        reg.insert(
            pid,
            ConditionalDisaggRole::Prefill,
            Some("http://a:8000".into()),
        )
        .unwrap();
        let peers = reg.prefill_peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].instance_id, pid);
        assert_eq!(peers[0].engine_url, "http://a:8000");
    }

    #[test]
    fn insert_prefill_without_url_filtered_from_prefill_peers() {
        let reg = CdPeerRegistry::new();
        reg.insert(id(), ConditionalDisaggRole::Prefill, None)
            .unwrap();
        assert!(reg.prefill_peers().is_empty());
    }

    #[test]
    fn decode_peers_never_appear_in_prefill_peers() {
        let reg = CdPeerRegistry::new();
        reg.insert(
            id(),
            ConditionalDisaggRole::Decode,
            Some("http://shouldnt-be-here".into()),
        )
        .unwrap();
        assert!(reg.prefill_peers().is_empty());
    }

    #[test]
    fn role_conflict_rejected() {
        let reg = CdPeerRegistry::new();
        let pid = id();
        reg.insert(pid, ConditionalDisaggRole::Prefill, Some("http://a".into()))
            .unwrap();
        let err = reg
            .insert(pid, ConditionalDisaggRole::Decode, None)
            .unwrap_err();
        assert!(matches!(err, CdRegistryError::RoleConflict { .. }));
    }

    #[test]
    fn same_role_re_registration_is_idempotent() {
        let reg = CdPeerRegistry::new();
        let pid = id();
        reg.insert(pid, ConditionalDisaggRole::Prefill, Some("http://a".into()))
            .unwrap();
        reg.insert(pid, ConditionalDisaggRole::Prefill, Some("http://b".into()))
            .unwrap();
        let peers = reg.prefill_peers();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].engine_url, "http://a");
    }

    #[test]
    fn remove_clears_role_set_and_engine_url() {
        let reg = CdPeerRegistry::new();
        let pid = id();
        reg.insert(pid, ConditionalDisaggRole::Prefill, Some("http://a".into()))
            .unwrap();
        reg.remove(pid);
        assert!(reg.prefill_peers().is_empty());
        assert!(reg.snapshot().prefill.is_empty());
    }

    #[test]
    fn snapshot_returns_role_split() {
        let reg = CdPeerRegistry::new();
        let p1 = id();
        let p2 = id();
        let d1 = id();
        reg.insert(p1, ConditionalDisaggRole::Prefill, Some("http://a".into()))
            .unwrap();
        reg.insert(p2, ConditionalDisaggRole::Prefill, Some("http://b".into()))
            .unwrap();
        reg.insert(d1, ConditionalDisaggRole::Decode, None).unwrap();
        let snap = reg.snapshot();
        assert_eq!(snap.prefill.len(), 2);
        assert_eq!(snap.decode.len(), 1);
    }
}
