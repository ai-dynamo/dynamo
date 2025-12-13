// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Handler registration manager for the Nova active message system.

use anyhow::Result;

use super::{ActiveMessageDispatcher, ControlMessage, NovaHandler};

/// Manager for registering handlers with the dispatcher.
///
/// Provides both public and internal registration methods with appropriate validation.
#[derive(Clone)]
pub(crate) struct HandlerManager {
    control_tx: flume::Sender<ControlMessage>,
}

impl HandlerManager {
    /// Create a new handler manager.
    pub(crate) fn new(control_tx: flume::Sender<ControlMessage>) -> Self {
        Self { control_tx }
    }

    /// Register a public handler (handler names cannot start with `_`).
    ///
    /// This is the public API for users to register their own handlers.
    pub fn register_handler(&self, handler: NovaHandler) -> Result<()> {
        let name = handler.name();

        // Validate: no leading underscore
        if name.starts_with('_') {
            anyhow::bail!(
                "Handler name '{}' cannot start with '_'. System handlers must be registered via internal APIs.",
                name
            );
        }

        self.register_dispatcher(handler.dispatcher)
    }

    /// Register a system/internal handler (allows names starting with `_`).
    ///
    /// This is for internal use only - registers system handlers like `_hello`, `_list_handlers`, etc.
    pub(crate) fn register_internal_handler(&self, handler: NovaHandler) -> Result<()> {
        self.register_dispatcher(handler.dispatcher)
    }

    /// Internal method to send registration to dispatcher hub.
    fn register_dispatcher(
        &self,
        dispatcher: std::sync::Arc<dyn ActiveMessageDispatcher>,
    ) -> Result<()> {
        self.control_tx
            .send(ControlMessage::Register { dispatcher })
            .map_err(|_| anyhow::anyhow!("Failed to register handler: control channel closed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::am::server::dispatcher::{ControlMessage, HandlerContext};
    use std::sync::Arc;

    // Mock dispatcher for testing
    struct MockDispatcher {
        name: String,
    }

    impl ActiveMessageDispatcher for MockDispatcher {
        fn name(&self) -> &str {
            &self.name
        }

        fn dispatch(&self, _ctx: HandlerContext) {
            // No-op for testing
        }
    }

    #[test]
    fn test_register_handler_blocks_underscore() {
        let (tx, _rx) = flume::bounded(10);
        let manager = HandlerManager::new(tx);

        let dispatcher = Arc::new(MockDispatcher {
            name: "_system".to_string(),
        });

        let handler = NovaHandler { dispatcher };

        let result = manager.register_handler(handler);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("cannot start with '_'")
        );
    }

    #[test]
    fn test_register_handler_allows_normal_names() {
        let (tx, rx) = flume::bounded(10);
        let manager = HandlerManager::new(tx);

        let dispatcher = Arc::new(MockDispatcher {
            name: "my_handler".to_string(),
        });

        let handler = NovaHandler { dispatcher };

        let result = manager.register_handler(handler);
        assert!(result.is_ok());

        // Verify message was sent
        let msg = rx.try_recv().unwrap();
        match msg {
            ControlMessage::Register { .. } => {
                // No-op for testing
            }
            _ => panic!("Expected Register message"),
        }
    }

    #[test]
    fn test_register_internal_handler_allows_underscore() {
        let (tx, rx) = flume::bounded(10);
        let manager = HandlerManager::new(tx);

        let dispatcher = Arc::new(MockDispatcher {
            name: "_hello".to_string(),
        });

        let handler = NovaHandler { dispatcher };

        let result = manager.register_internal_handler(handler);
        assert!(result.is_ok());

        // Verify message was sent
        let msg = rx.try_recv().unwrap();
        match msg {
            ControlMessage::Register { .. } => {
                // No-op for testing
            }
            _ => panic!("Expected Register message"),
        }
    }

    #[test]
    fn test_register_internal_handler_allows_normal_names() {
        let (tx, rx) = flume::bounded(10);
        let manager = HandlerManager::new(tx);

        let dispatcher = Arc::new(MockDispatcher {
            name: "internal_util".to_string(),
        });

        let handler = NovaHandler { dispatcher };

        let result = manager.register_internal_handler(handler);
        assert!(result.is_ok());

        // Verify message was sent
        let msg = rx.try_recv().unwrap();
        match msg {
            ControlMessage::Register { .. } => {
                // No-op for testing
            }
            _ => panic!("Expected Register message"),
        }
    }
}
