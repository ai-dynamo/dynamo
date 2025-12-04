// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System handlers for Nova active message infrastructure.
//!
//! These handlers implement the handshake protocol and peer discovery mechanisms.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::am::{
    NovaHandler, PeerInfo,
    handlers::{HandlerManager, TypedContext},
};

/// Request payload for the _hello handshake handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloRequest {
    /// The peer information of the sender
    pub peer_info: PeerInfo,
}

/// Response payload for handler list queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlersResponse {
    /// List of available handler names
    pub handlers: Vec<String>,
}

/// Create the _hello system handler.
///
/// This handler receives peer information from remote instances and returns
/// our local handler list as part of the handshake protocol.
fn create_hello_handler() -> NovaHandler {
    NovaHandler::typed_unary_async("_hello", |ctx: TypedContext<HelloRequest>| async move {
        let peer_info = ctx.input.peer_info;

        tracing::debug!(
            target: "dynamo_nova::system",
            instance_id = %peer_info.instance_id(),
            "Received _hello handshake from peer"
        );

        // Register peer with backend (makes it available for sending messages)
        if let Err(e) = ctx.nova.register_peer(peer_info.clone()) {
            tracing::warn!(
                target: "dynamo_nova::system",
                instance_id = %peer_info.instance_id(),
                error = %e,
                "Failed to register peer from _hello"
            );
        }

        let handlers = ctx.nova.list_local_handlers();

        tracing::debug!(
            target: "dynamo_nova::system",
            instance_id = %peer_info.instance_id(),
            handler_count = handlers.len(),
            "Completed _hello handshake"
        );

        Ok(HandlersResponse { handlers })
    })
    .spawn()
    .build()
}

/// Create the _list_handlers system handler.
///
/// This handler returns the current list of registered handlers on this instance.
fn create_list_handlers_handler() -> NovaHandler {
    NovaHandler::typed_unary_async("_list_handlers", |ctx: TypedContext<()>| async move {
        let handlers = ctx.nova.list_local_handlers();

        tracing::debug!(
            target: "dynamo_nova::system",
            handler_count = handlers.len(),
            "Responding to _list_handlers query"
        );

        Ok(HandlersResponse { handlers })
    })
    .spawn()
    .build()
}

/// Register all system handlers with the dispatcher.
///
/// This should be called during Nova initialization to install the built-in
/// system handlers that manage peer discovery and handshake protocol.
pub(crate) fn register_system_handlers(manager: &HandlerManager) -> Result<()> {
    manager.register_internal_handler(create_hello_handler())?;
    manager.register_internal_handler(create_list_handlers_handler())?;

    tracing::info!(
        target: "dynamo_nova::system",
        "Registered system handlers: _hello, _list_handlers"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_request_serialization() {
        use crate::am::InstanceId;
        use dynamo_nova_backend::WorkerAddress;

        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::builder().build().unwrap();
        let peer_info = PeerInfo::new(instance_id, address);

        let request = HelloRequest {
            peer_info: peer_info.clone(),
        };

        // Serialize and deserialize
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: HelloRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(
            request.peer_info.instance_id(),
            deserialized.peer_info.instance_id()
        );
    }

    #[test]
    fn test_handlers_response_serialization() {
        let response = HandlersResponse {
            handlers: vec![
                "handler1".to_string(),
                "handler2".to_string(),
                "_system".to_string(),
            ],
        };

        // Serialize and deserialize
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: HandlersResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(response.handlers.len(), deserialized.handlers.len());
        assert_eq!(response.handlers, deserialized.handlers);
    }
}
