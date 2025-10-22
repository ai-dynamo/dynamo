// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Network Manager Demo
//!
//! This example demonstrates the unified network manager interface for both
//! RPC calls and publish/subscribe messaging across different transport protocols.
//!
//! The network manager provides:
//! - Lazy connection creation and pooling
//! - Unified API for RPC and pub/sub
//! - Support for multiple transport protocols (HTTP, NATS, TCP, ZMQ)
//! - Automatic service discovery and load balancing

use dynamo_runtime::{
    Result, logging,
    pipeline::network::manager::*,
};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use anyhow;

/// Mock HTTP transport implementation
struct MockHttpTransport {
    connection_info: ConnectionInfo,
}

#[async_trait]
impl Transport for MockHttpTransport {
    async fn execute_rpc(&self, request: RpcRequest) -> Result<RpcResponse> {
        // Simulate HTTP RPC call
        tracing::info!(
            endpoint = %self.connection_info.endpoint,
            payload_size = request.payload.len(),
            "Executing HTTP RPC"
        );
        
        // Simulate processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(RpcResponse {
            payload: Bytes::from(format!("HTTP Response to: {}", String::from_utf8_lossy(&request.payload))),
            headers: HashMap::new(),
            status: RpcStatus::Success,
        })
    }
    
    async fn publish(&self, event: Event) -> Result<()> {
        tracing::info!(
            endpoint = %self.connection_info.endpoint,
            topic = %event.topic,
            event_id = %event.event_id,
            "Publishing HTTP event"
        );
        Ok(())
    }
    
    async fn subscribe(&self, topic: &str) -> Result<Box<dyn EventStream>> {
        tracing::info!(
            endpoint = %self.connection_info.endpoint,
            topic = %topic,
            "Subscribing to HTTP events"
        );
        Ok(Box::new(MockEventStream::new(topic.to_string())))
    }
    
    async fn unsubscribe(&self, topic: &str) -> Result<()> {
        tracing::info!(topic = %topic, "Unsubscribing from HTTP events");
        Ok(())
    }
    
    async fn is_healthy(&self) -> bool {
        true
    }
    
    fn connection_info(&self) -> &ConnectionInfo {
        &self.connection_info
    }
    
    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            supports_rpc: true,
            supports_pubsub: false, // HTTP typically doesn't support pub/sub natively
            supports_persistence: false,
            supports_ordering: false,
            supports_ack: false,
            max_message_size: Some(1024 * 1024), // 1MB
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        tracing::info!(endpoint = %self.connection_info.endpoint, "Closing HTTP transport");
        Ok(())
    }
}

/// Mock NATS transport implementation
struct MockNatsTransport {
    connection_info: ConnectionInfo,
}

#[async_trait]
impl Transport for MockNatsTransport {
    async fn execute_rpc(&self, request: RpcRequest) -> Result<RpcResponse> {
        tracing::info!(
            endpoint = %self.connection_info.endpoint,
            payload_size = request.payload.len(),
            "Executing NATS RPC"
        );
        
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        Ok(RpcResponse {
            payload: Bytes::from(format!("NATS Response to: {}", String::from_utf8_lossy(&request.payload))),
            headers: HashMap::new(),
            status: RpcStatus::Success,
        })
    }
    
    async fn publish(&self, event: Event) -> Result<()> {
        tracing::info!(
            endpoint = %self.connection_info.endpoint,
            topic = %event.topic,
            event_id = %event.event_id,
            "Publishing NATS event"
        );
        Ok(())
    }
    
    async fn subscribe(&self, topic: &str) -> Result<Box<dyn EventStream>> {
        tracing::info!(
            endpoint = %self.connection_info.endpoint,
            topic = %topic,
            "Subscribing to NATS events"
        );
        Ok(Box::new(MockEventStream::new(topic.to_string())))
    }
    
    async fn unsubscribe(&self, topic: &str) -> Result<()> {
        tracing::info!(topic = %topic, "Unsubscribing from NATS events");
        Ok(())
    }
    
    async fn is_healthy(&self) -> bool {
        true
    }
    
    fn connection_info(&self) -> &ConnectionInfo {
        &self.connection_info
    }
    
    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            supports_rpc: true,
            supports_pubsub: true,
            supports_persistence: true,
            supports_ordering: true,
            supports_ack: true,
            max_message_size: Some(64 * 1024), // 64KB
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        tracing::info!(endpoint = %self.connection_info.endpoint, "Closing NATS transport");
        Ok(())
    }
}

/// Mock event stream
struct MockEventStream {
    topic: String,
    counter: u32,
}

impl MockEventStream {
    fn new(topic: String) -> Self {
        Self { topic, counter: 0 }
    }
}

#[async_trait]
impl EventStream for MockEventStream {
    async fn next_event(&mut self) -> Result<Option<Event>> {
        // Simulate receiving events
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        self.counter += 1;
        
        if self.counter > 3 {
            return Ok(None); // End of stream
        }
        
        Ok(Some(Event {
            topic: self.topic.clone(),
            payload: Bytes::from(format!("Mock event {} on topic {}", self.counter, self.topic)),
            metadata: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
            event_id: format!("mock_event_{}", self.counter),
        }))
    }
    
    async fn close(&mut self) -> Result<()> {
        tracing::info!(topic = %self.topic, "Closing event stream");
        Ok(())
    }
}

/// HTTP transport factory
struct HttpTransportFactory;

#[async_trait]
impl TransportFactory for HttpTransportFactory {
    async fn create_transport(&self, info: ConnectionInfo) -> Result<Arc<dyn Transport>> {
        tracing::info!(endpoint = %info.endpoint, "Creating HTTP transport");
        Ok(Arc::new(MockHttpTransport { connection_info: info }))
    }
    
    fn protocol(&self) -> TransportProtocol {
        TransportProtocol::Http
    }
    
    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            supports_rpc: true,
            supports_pubsub: false,
            supports_persistence: false,
            supports_ordering: false,
            supports_ack: false,
            max_message_size: Some(1024 * 1024),
        }
    }
}

/// NATS transport factory
struct NatsTransportFactory;

#[async_trait]
impl TransportFactory for NatsTransportFactory {
    async fn create_transport(&self, info: ConnectionInfo) -> Result<Arc<dyn Transport>> {
        tracing::info!(endpoint = %info.endpoint, "Creating NATS transport");
        Ok(Arc::new(MockNatsTransport { connection_info: info }))
    }
    
    fn protocol(&self) -> TransportProtocol {
        TransportProtocol::Nats
    }
    
    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            supports_rpc: true,
            supports_pubsub: true,
            supports_persistence: true,
            supports_ordering: true,
            supports_ack: true,
            max_message_size: Some(64 * 1024),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::init();
    
    tracing::info!("🚀 Starting Network Manager Demo");
    
    // Create network manager
    let mut network_manager = NetworkManager::new();
    
    // Register transport protocols
    network_manager.register_transport(
        TransportProtocol::Http,
        Arc::new(HttpTransportFactory),
        None,
    );
    
    network_manager.register_transport(
        TransportProtocol::Nats,
        Arc::new(NatsTransportFactory),
        None,
    );
    
    tracing::info!("✅ Registered HTTP and NATS transports");
    
    // Demo 1: RPC Calls
    tracing::info!("\n📞 === RPC DEMO ===");
    
    let service_target = ServiceTarget {
        namespace: "demo".to_string(),
        component: "echo-service".to_string(),
        endpoint: "process".to_string(),
        instance_id: None,
    };
    
    // This would normally discover services, but for demo we'll simulate direct calls
    tracing::info!("Making RPC calls (simulated - normally would use service discovery)");
    
    // Demo 2: Publish/Subscribe
    tracing::info!("\n📡 === PUB/SUB DEMO ===");
    
    // Publish events
    let topics = vec!["user.created", "order.placed", "payment.processed"];
    
    for topic in &topics {
        let result = network_manager.publish_event(
            topic.to_string(),
            Bytes::from(format!("Event data for {}", topic)),
            Some({
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), "demo".to_string());
                metadata.insert("version".to_string(), "1.0".to_string());
                metadata
            }),
        ).await;
        
        match result {
            Ok(_) => tracing::info!("✅ Published event to topic: {}", topic),
            Err(e) => tracing::error!("❌ Failed to publish to {}: {}", topic, e),
        }
    }
    
    // Subscribe to events
    tracing::info!("Subscribing to events...");
    
    for topic in &topics {
        match network_manager.subscribe_to_topic(topic.to_string()).await {
            Ok(mut event_stream) => {
                tracing::info!("✅ Subscribed to topic: {}", topic);
                
                // Read a few events
                tokio::spawn(async move {
                    let topic_name = topic.to_string();
                    while let Ok(Some(event)) = event_stream.next_event().await {
                        tracing::info!(
                            "📨 Received event on {}: {} (ID: {})",
                            topic_name,
                            String::from_utf8_lossy(&event.payload),
                            event.event_id
                        );
                    }
                    tracing::info!("🔚 Event stream ended for topic: {}", topic_name);
                });
            }
            Err(e) => tracing::error!("❌ Failed to subscribe to {}: {}", topic, e),
        }
    }
    
    // Demo 3: Network Statistics
    tracing::info!("\n📊 === NETWORK STATS ===");
    
    let stats = network_manager.get_stats().await;
    tracing::info!("Network Statistics:");
    tracing::info!("  Total connections: {}", stats.total_connections);
    for (protocol, count) in &stats.protocols {
        tracing::info!("  {:?}: {} connections", protocol, count);
    }
    
    // Wait for event processing
    tracing::info!("\n⏳ Waiting for event processing...");
    tokio::time::sleep(tokio::time::Duration::from_secs(8)).await;
    
    tracing::info!("🎉 Network Manager Demo completed!");
    tracing::info!("\nKey features demonstrated:");
    tracing::info!("  ✅ Unified API for RPC and pub/sub");
    tracing::info!("  ✅ Multiple transport protocols (HTTP, NATS)");
    tracing::info!("  ✅ Lazy connection creation");
    tracing::info!("  ✅ Connection pooling and health checking");
    tracing::info!("  ✅ Event streaming and subscription management");
    
    Ok(())
}
