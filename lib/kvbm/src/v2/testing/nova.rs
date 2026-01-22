// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nova instance setup utilities for testing.

use anyhow::Result;
use dynamo_nova::am::Nova;
use dynamo_nova_backend::{Transport, tcp::TcpTransportBuilder};
use std::net::TcpListener;
use std::sync::Arc;
use tokio::time::Duration;

/// Create a single Nova instance with TCP transport on a random port.
///
/// # Returns
/// Nova instance
///
/// # Example
/// ```ignore
/// let nova = create_nova_instance_tcp().await?;
/// ```
pub async fn create_nova_instance_tcp() -> Result<Arc<Nova>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;

    let transport: Arc<dyn Transport> = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );

    let nova = Nova::new(vec![transport], vec![]).await?;

    // Give transport a moment to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(nova)
}

/// Container for a pair of connected Nova instances.
pub struct NovaPair {
    pub nova_a: Arc<Nova>,
    pub nova_b: Arc<Nova>,
}

/// Create a pair of Nova instances with bidirectional peer registration.
///
/// Both instances:
/// - Use TCP transport on random ports
/// - Are registered as peers of each other
/// - Ready for communication
///
/// # Example
/// ```ignore
/// let pair = create_nova_pair_tcp().await?;
///
/// // Can now send messages between nova_a and nova_b
/// pair.nova_a.unary("handler")?
///     .instance(pair.nova_b.instance_id())
///     .send().await?;
/// ```
pub async fn create_nova_pair_tcp() -> Result<NovaPair> {
    // Create first Nova instance
    let nova_a = create_nova_instance_tcp().await?;

    // Create second Nova instance
    let nova_b = create_nova_instance_tcp().await?;

    // Register each as peer of the other
    nova_a.register_peer(nova_b.peer_info())?;
    nova_b.register_peer(nova_a.peer_info())?;

    // Give time for peer registration to propagate
    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok(NovaPair { nova_a, nova_b })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_nova_instance() {
        let nova = create_nova_instance_tcp()
            .await
            .expect("Should create Nova");

        let peer_info = nova.peer_info();
        assert_eq!(
            peer_info.instance_id().worker_id(),
            nova.instance_id().worker_id()
        );
        assert!(!peer_info.worker_address().as_bytes().is_empty());

        // Local handlers should include system entries
        let handlers = nova.list_local_handlers();
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
    async fn test_create_nova_pair() {
        let pair = create_nova_pair_tcp().await.expect("Should create pair");

        // Verify both instances have different IDs
        assert_ne!(pair.nova_a.instance_id(), pair.nova_b.instance_id());

        // Verify worker addresses differ
        assert_ne!(
            pair.nova_a.peer_info().worker_address().checksum(),
            pair.nova_b.peer_info().worker_address().checksum()
        );

        // Verify system handlers are discoverable across peers
        let handlers_from_a = pair
            .nova_a
            .available_handlers(pair.nova_b.instance_id())
            .await
            .expect("Handlers from nova_b should be available");
        assert!(
            handlers_from_a.contains(&"_list_handlers".to_string()),
            "nova_a should see _list_handlers on nova_b: {:?}",
            handlers_from_a
        );
        assert!(
            handlers_from_a.contains(&"_hello".to_string()),
            "nova_a should see _hello on nova_b: {:?}",
            handlers_from_a
        );

        let handlers_from_b = pair
            .nova_b
            .available_handlers(pair.nova_a.instance_id())
            .await
            .expect("Handlers from nova_a should be available");
        assert!(
            handlers_from_b.contains(&"_list_handlers".to_string()),
            "nova_b should see _list_handlers on nova_a: {:?}",
            handlers_from_b
        );
        assert!(
            handlers_from_b.contains(&"_hello".to_string()),
            "nova_b should see _hello on nova_a: {:?}",
            handlers_from_b
        );
    }
}
