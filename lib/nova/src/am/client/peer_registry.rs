// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer registry for tracking remote peer state and handler availability.

use crate::am::InstanceId;
use dashmap::DashMap;
use std::{collections::HashSet, sync::Arc, time::Instant};

/// State information for a registered peer.
#[derive(Clone, Debug)]
pub(crate) struct PeerState {
    /// Known handlers on this peer (None = haven't queried yet)
    pub handlers: Option<HashSet<String>>,
    /// Last time we communicated with this peer
    pub last_seen: Instant,
}

impl PeerState {
    pub fn new() -> Self {
        Self {
            handlers: None,
            last_seen: Instant::now(),
        }
    }

    pub fn update_last_seen(&mut self) {
        self.last_seen = Instant::now();
    }

    pub fn set_handlers(&mut self, handlers: Vec<String>) {
        self.handlers = Some(handlers.into_iter().collect());
        self.update_last_seen();
    }

    pub fn has_handler(&self, handler: &str) -> bool {
        self.handlers
            .as_ref()
            .map(|h| h.contains(handler))
            .unwrap_or(false)
    }
}

/// Registry for tracking remote peers and their state.
#[derive(Clone)]
pub(crate) struct PeerRegistry {
    peers: Arc<DashMap<InstanceId, PeerState>>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(DashMap::new()),
        }
    }

    /// Register a peer without handler information.
    pub fn register_peer(&self, instance_id: InstanceId) {
        self.peers.insert(instance_id, PeerState::new());
    }

    /// Check if we have handler information for a peer.
    pub fn has_handler_info(&self, instance_id: InstanceId) -> bool {
        self.peers
            .get(&instance_id)
            .map(|state| state.handlers.is_some())
            .unwrap_or(false)
    }

    /// Get the list of handlers for a peer.
    pub fn get_handlers(&self, instance_id: InstanceId) -> Option<Vec<String>> {
        self.peers
            .get(&instance_id)
            .and_then(|state| state.handlers.as_ref().map(|h| h.iter().cloned().collect()))
    }

    /// Update the handler list for a peer.
    pub fn update_handlers(&self, instance_id: InstanceId, handlers: Vec<String>) {
        if let Some(mut state) = self.peers.get_mut(&instance_id) {
            state.set_handlers(handlers);
        }
    }

    /// Check if a specific handler exists on a peer.
    pub fn handler_exists(&self, instance_id: InstanceId, handler: &str) -> bool {
        self.peers
            .get(&instance_id)
            .map(|state| state.has_handler(handler))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_registry_basic() {
        let registry = PeerRegistry::new();
        let instance_id = InstanceId::new_v4();

        // Register peer
        registry.register_peer(instance_id);
        assert!(!registry.has_handler_info(instance_id));

        // Update handlers
        registry.update_handlers(
            instance_id,
            vec!["handler1".to_string(), "handler2".to_string()],
        );
        assert!(registry.has_handler_info(instance_id));
        assert!(registry.handler_exists(instance_id, "handler1"));
        assert!(registry.handler_exists(instance_id, "handler2"));
        assert!(!registry.handler_exists(instance_id, "handler3"));

        // Get handlers
        let handlers = registry.get_handlers(instance_id).unwrap();
        assert_eq!(handlers.len(), 2);
        assert!(handlers.contains(&"handler1".to_string()));
        assert!(handlers.contains(&"handler2".to_string()));
    }

    #[test]
    fn test_peer_state_new() {
        let state = PeerState::new();
        assert!(state.handlers.is_none());
    }

    #[test]
    fn test_peer_state_set_handlers() {
        let mut state = PeerState::new();
        state.set_handlers(vec!["test".to_string()]);
        assert!(state.handlers.is_some());
        assert!(state.has_handler("test"));
        assert!(!state.has_handler("missing"));
    }
}
