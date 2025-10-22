// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Network Manager Layer
//!
//! This module provides a high-level network manager that abstracts connection management
//! and RPC execution across different transport protocols (NATS, TCP, ZMQ, HTTP, etc.).
//!
//! The network manager is responsible for:
//! - Managing connections to upstream services
//! - Connection pooling and lifecycle management
//! - Executing RPCs over the wire with appropriate transport
//! - Load balancing and failover
//! - Metrics and monitoring

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::RwLock;
use bytes::Bytes;
use serde::{Serialize, Deserialize};
use tracing;

use crate::component::{Instance, TransportType, Client as ComponentClient};
use crate::pipeline::network::request_plane::Headers;

/// Event for publish/subscribe messaging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event topic/subject
    pub topic: String,
    /// Event payload
    pub payload: Bytes,
    /// Event metadata/headers
    pub metadata: HashMap<String, String>,
    /// Event timestamp
    pub timestamp: std::time::SystemTime,
    /// Event ID for deduplication
    pub event_id: String,
}

/// Supported transport protocols
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportProtocol {
    /// HTTP/HTTPS transport
    Http,
    /// NATS messaging
    Nats,
    /// TCP direct connection
    Tcp,
    /// ZeroMQ messaging
    Zmq,
    /// gRPC over HTTP/2
    Grpc,
}

impl TransportProtocol {
    /// Get protocol from transport type
    pub fn from_transport_type(transport: &TransportType) -> Self {
        match transport {
            TransportType::HttpTcp { .. } => Self::Http,
            TransportType::NatsTcp(_) => Self::Nats,
        }
    }

    /// Get default port for protocol
    pub fn default_port(&self) -> u16 {
        match self {
            Self::Http => 80,
            Self::Nats => 4222,
            Self::Tcp => 8080,
            Self::Zmq => 5555,
            Self::Grpc => 9090,
        }
    }
}

/// Connection information for a specific transport
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Transport protocol
    pub protocol: TransportProtocol,
    /// Connection endpoint (URL, address, etc.)
    pub endpoint: String,
    /// Additional connection metadata
    pub metadata: HashMap<String, String>,
}

/// RPC request context
#[derive(Debug, Clone)]
pub struct RpcRequest {
    /// Request payload
    pub payload: Bytes,
    /// Request headers/metadata
    pub headers: Headers,
    /// Target service information
    pub target: ServiceTarget,
    /// Request timeout
    pub timeout: Option<std::time::Duration>,
}

/// RPC response
#[derive(Debug, Clone)]
pub struct RpcResponse {
    /// Response payload
    pub payload: Bytes,
    /// Response headers/metadata
    pub headers: Headers,
    /// Response status/error information
    pub status: RpcStatus,
}

/// RPC status information
#[derive(Debug, Clone)]
pub enum RpcStatus {
    /// Successful response
    Success,
    /// Error response with details
    Error { code: u32, message: String },
    /// Timeout occurred
    Timeout,
    /// Connection failed
    ConnectionFailed,
}

/// Service target information
#[derive(Debug, Clone)]
pub struct ServiceTarget {
    /// Namespace
    pub namespace: String,
    /// Component name
    pub component: String,
    /// Endpoint name
    pub endpoint: String,
    /// Specific instance ID (optional)
    pub instance_id: Option<i64>,
}

/// Common transport interface for both RPC and pub/sub operations
/// 
/// This trait provides a unified interface for all transport protocols,
/// supporting both request/response (RPC) and publish/subscribe patterns.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Execute an RPC call (request/response pattern)
    async fn execute_rpc(&self, request: RpcRequest) -> Result<RpcResponse>;
    
    /// Publish an event (fire-and-forget)
    async fn publish(&self, event: Event) -> Result<()>;
    
    /// Subscribe to events on a topic
    async fn subscribe(&self, topic: &str) -> Result<Box<dyn EventStream>>;
    
    /// Unsubscribe from a topic
    async fn unsubscribe(&self, topic: &str) -> Result<()>;
    
    /// Check if connection is healthy
    async fn is_healthy(&self) -> bool;
    
    /// Get connection info
    fn connection_info(&self) -> &ConnectionInfo;
    
    /// Get supported capabilities
    fn capabilities(&self) -> TransportCapabilities;
    
    /// Close the connection
    async fn close(&mut self) -> Result<()>;
}

/// Legacy alias for backward compatibility
pub type TransportConnection = dyn Transport;

/// Event stream for subscriptions
#[async_trait]
pub trait EventStream: Send + Sync {
    /// Get the next event from the stream
    async fn next_event(&mut self) -> Result<Option<Event>>;
    
    /// Close the event stream
    async fn close(&mut self) -> Result<()>;
}

/// Transport capabilities
#[derive(Debug, Clone)]
pub struct TransportCapabilities {
    /// Supports RPC (request/response)
    pub supports_rpc: bool,
    /// Supports publish/subscribe
    pub supports_pubsub: bool,
    /// Supports persistent connections
    pub supports_persistence: bool,
    /// Supports message ordering
    pub supports_ordering: bool,
    /// Supports message acknowledgment
    pub supports_ack: bool,
    /// Maximum message size (None = unlimited)
    pub max_message_size: Option<usize>,
}

/// Connection pool for a specific transport protocol
/// 
/// Connections are created lazily - only when first needed for an endpoint.
/// This avoids creating unnecessary connections and reduces resource usage.
pub struct ConnectionPool {
    /// Protocol this pool manages
    protocol: TransportProtocol,
    /// Active connections (endpoint -> connection)
    connections: Arc<RwLock<HashMap<String, Arc<dyn Transport>>>>,
    /// Pending connection creation (endpoint -> future)
    /// Prevents multiple concurrent connection attempts to the same endpoint
    pending_connections: Arc<RwLock<HashMap<String, Arc<tokio::sync::Notify>>>>,
    /// Transport factory
    factory: Arc<dyn TransportFactory>,
    /// Pool configuration
    config: PoolConfig,
}

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum connections per endpoint
    pub max_connections_per_endpoint: usize,
    /// Connection timeout
    pub connection_timeout: std::time::Duration,
    /// Idle timeout before closing connections
    pub idle_timeout: std::time::Duration,
    /// Health check interval
    pub health_check_interval: std::time::Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_endpoint: 10,
            connection_timeout: std::time::Duration::from_secs(30),
            idle_timeout: std::time::Duration::from_secs(300),
            health_check_interval: std::time::Duration::from_secs(60),
        }
    }
}

/// Factory for creating transport connections
#[async_trait]
pub trait TransportFactory: Send + Sync {
    /// Create a new transport connection for the given endpoint
    async fn create_transport(&self, info: ConnectionInfo) -> Result<Arc<dyn Transport>>;
    
    /// Get supported protocol
    fn protocol(&self) -> TransportProtocol;
    
    /// Get transport capabilities
    fn capabilities(&self) -> TransportCapabilities;
}

/// Legacy alias for backward compatibility
pub type ConnectionFactory = dyn TransportFactory;

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(
        protocol: TransportProtocol,
        factory: Arc<dyn TransportFactory>,
        config: PoolConfig,
    ) -> Self {
        Self {
            protocol,
            connections: Arc::new(RwLock::new(HashMap::new())),
            pending_connections: Arc::new(RwLock::new(HashMap::new())),
            factory,
            config,
        }
    }

    /// Get or create a connection for the endpoint (lazy creation)
    /// 
    /// This method implements lazy connection creation with the following benefits:
    /// - Connections are only created when actually needed
    /// - Prevents thundering herd - only one connection attempt per endpoint at a time
    /// - Reuses existing healthy connections
    /// - Automatically removes and recreates unhealthy connections
    pub async fn get_connection(&self, endpoint: &str) -> Result<Arc<dyn Transport>> {
        // Fast path: check if we already have a healthy connection
        {
            let connections = self.connections.read().await;
            if let Some(conn) = connections.get(endpoint) {
                if conn.is_healthy().await {
                    tracing::trace!(
                        protocol = ?self.protocol,
                        endpoint = %endpoint,
                        "Reusing existing connection"
                    );
                    return Ok(conn.clone());
                }
                // Connection is unhealthy, we'll need to create a new one
                tracing::debug!(
                    protocol = ?self.protocol,
                    endpoint = %endpoint,
                    "Existing connection unhealthy, will recreate"
                );
            }
        }

        // Check if another task is already creating a connection for this endpoint
        let notify = {
            let mut pending = self.pending_connections.write().await;
            if let Some(existing_notify) = pending.get(endpoint) {
                // Another task is creating the connection, wait for it
                let notify = existing_notify.clone();
                drop(pending);
                
                tracing::trace!(
                    protocol = ?self.protocol,
                    endpoint = %endpoint,
                    "Waiting for concurrent connection creation"
                );
                
                notify.notified().await;
                
                // Try again after the other task completes
                let connections = self.connections.read().await;
                if let Some(conn) = connections.get(endpoint) {
                    if conn.is_healthy().await {
                        return Ok(conn.clone());
                    }
                }
                
                // Still no healthy connection, need to create one ourselves
                // Re-acquire the lock to add our notify
                let mut pending = self.pending_connections.write().await;
                let notify = Arc::new(tokio::sync::Notify::new());
                pending.insert(endpoint.to_string(), notify.clone());
                notify
            } else {
                // We'll create the connection, add a notify for other waiters
                let notify = Arc::new(tokio::sync::Notify::new());
                pending.insert(endpoint.to_string(), notify.clone());
                notify
            }
        };

        // Create the connection (this is where lazy creation happens)
        let result = self.create_connection_internal(endpoint).await;
        
        // Notify waiting tasks and clean up
        {
            let mut pending_guard = self.pending_connections.write().await;
            pending_guard.remove(endpoint);
        }
        notify.notify_waiters();
        
        result
    }

    /// Internal method to actually create the connection
    async fn create_connection_internal(&self, endpoint: &str) -> Result<Arc<dyn Transport>> {
        tracing::info!(
            protocol = ?self.protocol,
            endpoint = %endpoint,
            "Lazily creating new connection"
        );

        let connection_info = ConnectionInfo {
            protocol: self.protocol.clone(),
            endpoint: endpoint.to_string(),
            metadata: HashMap::new(),
        };

        let start_time = std::time::Instant::now();
        let connection = tokio::time::timeout(
            self.config.connection_timeout,
            self.factory.create_transport(connection_info)
        ).await??;
        
        let creation_time = start_time.elapsed();
        
        // Store in pool
        {
            let mut connections = self.connections.write().await;
            // Remove old unhealthy connection if it exists
            connections.remove(endpoint);
            connections.insert(endpoint.to_string(), connection.clone());
        }

        tracing::info!(
            protocol = ?self.protocol,
            endpoint = %endpoint,
            creation_time_ms = creation_time.as_millis(),
            "Successfully created lazy connection"
        );

        Ok(connection)
    }

    /// Execute RPC using pooled connection
    pub async fn execute_rpc(&self, endpoint: &str, request: RpcRequest) -> Result<RpcResponse> {
        let connection = self.get_connection(endpoint).await?;
        connection.execute_rpc(request).await
    }

    /// Publish event using pooled connection
    pub async fn publish_event(&self, endpoint: &str, event: Event) -> Result<()> {
        let connection = self.get_connection(endpoint).await?;
        connection.publish(event).await
    }

    /// Subscribe to events using pooled connection
    pub async fn subscribe_to_events(&self, endpoint: &str, topic: &str) -> Result<Box<dyn EventStream>> {
        let connection = self.get_connection(endpoint).await?;
        connection.subscribe(topic).await
    }

    /// Start background health checking and idle connection cleanup
    /// 
    /// This background task:
    /// - Removes unhealthy connections (they'll be lazily recreated when needed)
    /// - Removes idle connections after the configured timeout
    /// - Logs connection pool statistics
    pub async fn start_health_checker(&self) {
        let connections = self.connections.clone();
        let interval = self.config.health_check_interval;
        let _idle_timeout = self.config.idle_timeout; // TODO: implement idle timeout
        let protocol = self.protocol.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                let mut to_remove = Vec::new();
                let mut healthy_count = 0;
                let total_count;
                
                {
                    let connections_guard = connections.read().await;
                    total_count = connections_guard.len();
                    
                    for (endpoint, connection) in connections_guard.iter() {
                        if !connection.is_healthy().await {
                            to_remove.push((endpoint.clone(), "unhealthy".to_string()));
                        } else {
                            healthy_count += 1;
                            
                            // TODO: Add idle timeout checking
                            // This would require tracking last used time in the connection
                        }
                    }
                }

                // Remove unhealthy connections (they'll be lazily recreated when needed)
                if !to_remove.is_empty() {
                    let mut connections_guard = connections.write().await;
                    for (endpoint, reason) in &to_remove {
                        tracing::warn!(
                            protocol = ?protocol,
                            endpoint = %endpoint,
                            reason = %reason,
                            "Removing connection (will be lazily recreated if needed)"
                        );
                        connections_guard.remove(endpoint);
                    }
                }

                // Log pool statistics
                if total_count > 0 {
                    tracing::debug!(
                        protocol = ?protocol,
                        total_connections = total_count,
                        healthy_connections = healthy_count,
                        removed_connections = to_remove.len(),
                        "Connection pool health check completed"
                    );
                }
            }
        });
    }

    /// Get connection pool statistics
    pub async fn get_pool_stats(&self) -> PoolStats {
        let connections = self.connections.read().await;
        let pending = self.pending_connections.read().await;
        
        let mut healthy_count = 0;
        for connection in connections.values() {
            if connection.is_healthy().await {
                healthy_count += 1;
            }
        }

        PoolStats {
            protocol: self.protocol.clone(),
            total_connections: connections.len(),
            healthy_connections: healthy_count,
            pending_connections: pending.len(),
        }
    }

    /// Preemptively close all connections (useful for shutdown)
    pub async fn close_all_connections(&self) {
        let mut connections = self.connections.write().await;
        
        for (endpoint, _connection) in connections.drain() {
            // Note: Proper connection closing would require the Transport trait
            // to have a method that doesn't require &mut self, or use interior mutability
            tracing::info!(
                protocol = ?self.protocol,
                endpoint = %endpoint,
                "Removed connection from pool"
            );
        }
        
        tracing::info!(protocol = ?self.protocol, "Closed all connections");
    }
}

/// Main network manager that coordinates all transport protocols
pub struct NetworkManager {
    /// Connection pools by protocol
    pools: HashMap<TransportProtocol, Arc<ConnectionPool>>,
    /// Service discovery client
    discovery_clients: Arc<RwLock<HashMap<ServiceTarget, ComponentClient>>>,
    /// Load balancing strategy
    load_balancer: Arc<dyn LoadBalancer>,
}

/// Load balancing strategy
#[async_trait]
pub trait LoadBalancer: Send + Sync {
    /// Select an instance for the request
    async fn select_instance(&self, instances: &[Instance], request: &RpcRequest) -> Option<Instance>;
}

/// Round-robin load balancer
pub struct RoundRobinBalancer {
    counters: Arc<RwLock<HashMap<String, std::sync::atomic::AtomicUsize>>>,
}

impl RoundRobinBalancer {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl LoadBalancer for RoundRobinBalancer {
    async fn select_instance(&self, instances: &[Instance], _request: &RpcRequest) -> Option<Instance> {
        if instances.is_empty() {
            return None;
        }

        let service_key = format!("{}/{}/{}", 
            instances[0].namespace, 
            instances[0].component, 
            instances[0].endpoint
        );

        let mut counters = self.counters.write().await;
        let counter = counters
            .entry(service_key)
            .or_insert_with(|| std::sync::atomic::AtomicUsize::new(0));

        let index = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % instances.len();
        Some(instances[index].clone())
    }
}

impl NetworkManager {
    /// Create a new network manager
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            discovery_clients: Arc::new(RwLock::new(HashMap::new())),
            load_balancer: Arc::new(RoundRobinBalancer::new()),
        }
    }

    /// Register a transport protocol with its factory
    pub fn register_transport(
        &mut self,
        protocol: TransportProtocol,
        factory: Arc<dyn TransportFactory>,
        config: Option<PoolConfig>,
    ) {
        let pool = Arc::new(ConnectionPool::new(
            protocol.clone(),
            factory,
            config.unwrap_or_default(),
        ));

        // Start health checker
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            pool_clone.start_health_checker().await;
        });

        let protocol_name = format!("{:?}", protocol);
        self.pools.insert(protocol, pool);
        tracing::info!(protocol = %protocol_name, "Registered transport protocol");
    }

    /// Execute an RPC call to a specific service endpoint
    pub async fn call_service(
        &self,
        target: ServiceTarget,
        payload: Bytes,
        headers: Option<Headers>,
        timeout: Option<std::time::Duration>,
    ) -> Result<RpcResponse> {
        let request = RpcRequest {
            payload,
            headers: headers.unwrap_or_default(),
            target,
            timeout,
        };
        self.execute_rpc(request).await
    }

    /// Publish an event to a topic
    pub async fn publish_event(
        &self,
        topic: String,
        payload: Bytes,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        let event = Event {
            topic: topic.clone(),
            payload,
            metadata: metadata.unwrap_or_default(),
            timestamp: std::time::SystemTime::now(),
            event_id: format!("evt_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
        };

        // For pub/sub, we typically use a message broker endpoint
        // This could be configured or discovered
        let broker_endpoint = self.get_pubsub_broker_endpoint(&topic).await?;
        
        // Get transport protocol for the broker
        let protocol = self.get_protocol_for_endpoint(&broker_endpoint).await?;
        
        // Get connection pool for protocol
        let pool = self.pools.get(&protocol)
            .ok_or_else(|| anyhow::anyhow!("No connection pool for protocol: {:?}", protocol))?;

        pool.publish_event(&broker_endpoint, event).await
    }

    /// Subscribe to events on a topic
    pub async fn subscribe_to_topic(&self, topic: String) -> Result<Box<dyn EventStream>> {
        // Get broker endpoint for the topic
        let broker_endpoint = self.get_pubsub_broker_endpoint(&topic).await?;
        
        // Get transport protocol for the broker
        let protocol = self.get_protocol_for_endpoint(&broker_endpoint).await?;
        
        // Get connection pool for protocol
        let pool = self.pools.get(&protocol)
            .ok_or_else(|| anyhow::anyhow!("No connection pool for protocol: {:?}", protocol))?;

        pool.subscribe_to_events(&broker_endpoint, &topic).await
    }

    /// Get broker endpoint for pub/sub operations
    async fn get_pubsub_broker_endpoint(&self, _topic: &str) -> Result<String> {
        // This would typically discover or configure the message broker
        // For now, return a default NATS endpoint
        Ok("nats://localhost:4222".to_string())
    }

    /// Get transport protocol for an endpoint
    async fn get_protocol_for_endpoint(&self, endpoint: &str) -> Result<TransportProtocol> {
        // Parse endpoint to determine protocol
        if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            Ok(TransportProtocol::Http)
        } else if endpoint.starts_with("nats://") {
            Ok(TransportProtocol::Nats)
        } else if endpoint.starts_with("tcp://") {
            Ok(TransportProtocol::Tcp)
        } else if endpoint.starts_with("zmq://") {
            Ok(TransportProtocol::Zmq)
        } else {
            // Default to NATS for backward compatibility
            Ok(TransportProtocol::Nats)
        }
    }

    /// Execute an RPC call to a service
    pub async fn execute_rpc(&self, mut request: RpcRequest) -> Result<RpcResponse> {
        // Discover service instances
        let instances = self.discover_service(&request.target).await?;
        
        if instances.is_empty() {
            return Ok(RpcResponse {
                payload: Bytes::new(),
                headers: HashMap::new(),
                status: RpcStatus::Error {
                    code: 404,
                    message: "No instances found for service".to_string(),
                },
            });
        }

        // Select instance using load balancer
        let instance = match self.load_balancer.select_instance(&instances, &request).await {
            Some(instance) => instance,
            None => return Err(anyhow::anyhow!("Load balancer failed to select instance")),
        };

        // Get transport protocol
        let protocol = TransportProtocol::from_transport_type(&instance.transport);
        
        // Get connection pool for protocol
        let pool = self.pools.get(&protocol)
            .ok_or_else(|| anyhow::anyhow!("No connection pool for protocol: {:?}", protocol))?;

        // Get endpoint from instance
        let endpoint = match &instance.transport {
            TransportType::HttpTcp { http_endpoint } => http_endpoint.clone(),
            TransportType::NatsTcp(subject) => subject.clone(),
        };

        // Add instance ID to request metadata
        request.headers.insert("x-instance-id".to_string(), instance.instance_id.to_string());

        // Execute RPC
        tracing::debug!(
            protocol = ?protocol,
            endpoint = %endpoint,
            instance_id = instance.instance_id,
            "Executing RPC"
        );

        pool.execute_rpc(&endpoint, request).await
    }

    /// Discover service instances
    async fn discover_service(&self, _target: &ServiceTarget) -> Result<Vec<Instance>> {
        // This would integrate with the existing service discovery
        // For now, return empty vec - this would be implemented to use ComponentClient
        Ok(Vec::new())
    }

    /// Get connection statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let mut stats = NetworkStats {
            protocols: HashMap::new(),
            total_connections: 0,
        };

        for (protocol, pool) in &self.pools {
            let connections = pool.connections.read().await;
            let count = connections.len();
            stats.protocols.insert(protocol.clone(), count);
            stats.total_connections += count;
        }

        stats
    }
}

/// Network manager statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// Connections per protocol
    pub protocols: HashMap<TransportProtocol, usize>,
    /// Total active connections
    pub total_connections: usize,
}

/// Connection pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Protocol this pool manages
    pub protocol: TransportProtocol,
    /// Total connections in pool
    pub total_connections: usize,
    /// Healthy connections
    pub healthy_connections: usize,
    /// Pending connection attempts
    pub pending_connections: usize,
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_protocol_from_transport_type() {
        let http_transport = TransportType::HttpTcp {
            http_endpoint: "http://example.com".to_string(),
        };
        assert_eq!(
            TransportProtocol::from_transport_type(&http_transport),
            TransportProtocol::Http
        );

        let nats_transport = TransportType::NatsTcp("test.subject".to_string());
        assert_eq!(
            TransportProtocol::from_transport_type(&nats_transport),
            TransportProtocol::Nats
        );
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections_per_endpoint, 10);
        assert_eq!(config.connection_timeout.as_secs(), 30);
    }
}
