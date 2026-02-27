// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Messenger instance setup utilities for testing.

use anyhow::Result;
use std::net::TcpListener;
use std::sync::Arc;
use tokio::time::Duration;
use velo::Messenger;
use velo_backend::tcp::TcpTransportBuilder;
use velo_backend::Transport;

/// Create a single Messenger instance with TCP transport on a random port.
///
/// # Returns
/// Messenger instance
///
/// # Example
/// ```ignore
/// let messenger = create_messenger_tcp().await?;
/// ```
pub async fn create_messenger_tcp() -> Result<Arc<Messenger>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;

    let transport: Arc<dyn Transport> = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );

    let messenger = Messenger::new(vec![transport], None).await?;

    // Give transport a moment to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(messenger)
}

/// Container for a pair of connected Messenger instances.
pub struct MessengerPair {
    pub messenger_a: Arc<Messenger>,
    pub messenger_b: Arc<Messenger>,
}

/// Create a pair of Messenger instances with bidirectional peer registration.
///
/// Both instances:
/// - Use TCP transport on random ports
/// - Are registered as peers of each other
/// - Ready for communication
///
/// # Example
/// ```ignore
/// let pair = create_messenger_pair_tcp().await?;
///
/// // Can now send messages between messenger_a and messenger_b
/// pair.messenger_a.unary("handler")?
///     .instance(pair.messenger_b.instance_id())
///     .send().await?;
/// ```
pub async fn create_messenger_pair_tcp() -> Result<MessengerPair> {
    // Create first Messenger instance
    let messenger_a = create_messenger_tcp().await?;

    // Create second Messenger instance
    let messenger_b = create_messenger_tcp().await?;

    // Register each as peer of the other
    messenger_a.register_peer(messenger_b.peer_info())?;
    messenger_b.register_peer(messenger_a.peer_info())?;

    // Give time for peer registration to propagate
    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok(MessengerPair {
        messenger_a,
        messenger_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_messenger_instance() {
        let messenger = create_messenger_tcp()
            .await
            .expect("Should create Messenger");

        let peer_info = messenger.peer_info();
        assert_eq!(
            peer_info.instance_id().worker_id(),
            messenger.instance_id().worker_id()
        );
        assert!(!peer_info.worker_address().as_bytes().is_empty());

        // Local handlers should include system entries
        let handlers = messenger.list_local_handlers();
        assert!(
            handlers.contains(&"_list_handlers".to_string()),
            "Expected _list_handlers in local handler list: {:?}",
            handlers
        );
        assert!(
            handlers.contains(&"_hello".to_string()),
            "Expected _hello in local handler list: {:?}",
            handlers
        );
    }

    #[tokio::test]
    async fn test_create_messenger_pair() {
        let pair = create_messenger_pair_tcp()
            .await
            .expect("Should create pair");

        // Verify both instances have different IDs
        assert_ne!(
            pair.messenger_a.instance_id(),
            pair.messenger_b.instance_id()
        );

        // Verify worker addresses differ
        assert_ne!(
            pair.messenger_a.peer_info().worker_address().checksum(),
            pair.messenger_b.peer_info().worker_address().checksum()
        );

        // Verify system handlers are discoverable across peers
        let handlers_from_a = pair
            .messenger_a
            .available_handlers(pair.messenger_b.instance_id())
            .await
            .expect("Handlers from messenger_b should be available");
        assert!(
            handlers_from_a.contains(&"_list_handlers".to_string()),
            "messenger_a should see _list_handlers on messenger_b: {:?}",
            handlers_from_a
        );
        assert!(
            handlers_from_a.contains(&"_hello".to_string()),
            "messenger_a should see _hello on messenger_b: {:?}",
            handlers_from_a
        );

        let handlers_from_b = pair
            .messenger_b
            .available_handlers(pair.messenger_a.instance_id())
            .await
            .expect("Handlers from messenger_a should be available");
        assert!(
            handlers_from_b.contains(&"_list_handlers".to_string()),
            "messenger_b should see _list_handlers on messenger_a: {:?}",
            handlers_from_b
        );
        assert!(
            handlers_from_b.contains(&"_hello".to_string()),
            "messenger_b should see _hello on messenger_a: {:?}",
            handlers_from_b
        );
    }
}
