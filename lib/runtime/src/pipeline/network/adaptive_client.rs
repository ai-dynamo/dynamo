// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive request plane client that automatically discovers and adapts to available transports
//!
//! This module provides an adaptive client that:
//! - Discovers services from etcd dynamically
//! - Inspects the transport field in discovered instances
//! - Automatically creates the appropriate client (HTTP or NATS) based on what it finds
//! - Does not rely on DYN_REQUEST_PLANE environment variable
//! - Assumes all instances of a component use the same transport type

use std::sync::Arc;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing;

use crate::component::{Instance, TransportType, Client as ComponentClient, InstanceSource};
use crate::pipeline::network::request_plane::{RequestPlaneClient, Headers};
use crate::pipeline::network::egress::http_router::HttpRequestClient;
use async_nats::Client as NatsClient;
use bytes::Bytes;

/// Transport client wrapper that can be either HTTP or NATS
#[derive(Clone)]
pub enum AdaptiveTransportClient {
    Http(Arc<HttpRequestClient>),
    Nats(NatsClient),
}

/// Detected transport type for a component
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentTransportType {
    Http,
    Nats,
}

/// Adaptive request plane client that discovers services and adapts to their transports
/// 
/// Assumes all instances of a particular component use the same transport type,
/// so we only need to check the first discovered instance to determine the transport.
/// 
/// This assumption provides several benefits:
/// - Efficient transport detection (check only first instance)
/// - Consistent behavior across all instances of a component  
/// - Deadlock prevention in bidirectional communication scenarios (A↔B)
/// - Simplified connection management and resource pooling
pub struct AdaptiveRequestPlaneClient {
    /// Component client for service discovery
    component_client: ComponentClient,
    /// The detected transport type for this component
    transport_type: Arc<RwLock<Option<ComponentTransportType>>>,
    /// The transport client (created once transport type is detected)
    transport_client: Arc<RwLock<Option<AdaptiveTransportClient>>>,
    /// NATS client (if available)
    nats_client: Option<NatsClient>,
}

impl AdaptiveRequestPlaneClient {
    /// Create a new adaptive client
    pub async fn new(
        component_client: ComponentClient,
        nats_client: Option<NatsClient>,
    ) -> Result<Self> {
        let client = Self {
            component_client,
            transport_type: Arc::new(RwLock::new(None)),
            transport_client: Arc::new(RwLock::new(None)),
            nats_client,
        };

        // Start monitoring for instance changes
        client.start_instance_monitoring().await?;

        Ok(client)
    }

    /// Start monitoring instance changes and detect transport type
    async fn start_instance_monitoring(&self) -> Result<()> {
        let transport_type = self.transport_type.clone();
        let transport_client = self.transport_client.clone();
        let nats_client = self.nats_client.clone();
        let component_client = self.component_client.clone();

        tokio::spawn(async move {
            // Only monitor if we have dynamic instance discovery
            if let InstanceSource::Dynamic(mut rx) = component_client.instance_source.as_ref().clone() {
                loop {
                    // Wait for instance changes
                    if rx.changed().await.is_err() {
                        tracing::error!("Instance source watch channel closed");
                        break;
                    }

                    let instances = rx.borrow().clone();
                    
                    // If we have instances and haven't detected transport type yet, detect it now
                    if !instances.is_empty() {
                        let current_transport_type = transport_type.read().await.clone();
                        
                        if current_transport_type.is_none() {
                            // Detect transport type from first instance (all instances use same transport)
                            let first_instance = &instances[0];
                            let detected_type = match &first_instance.transport {
                                TransportType::HttpTcp { .. } => ComponentTransportType::Http,
                                TransportType::NatsTcp(_) => ComponentTransportType::Nats,
                            };

                            tracing::info!(
                                transport_type = ?detected_type,
                                instance_count = instances.len(),
                                "Detected component transport type"
                            );

                            // Create the appropriate transport client
                            let client = match Self::create_transport_client(&detected_type, &nats_client) {
                                Ok(client) => client,
                                Err(e) => {
                                    tracing::error!(
                                        error = %e,
                                        "Failed to create transport client"
                                    );
                                    continue;
                                }
                            };

                            // Update both transport type and client
                            *transport_type.write().await = Some(detected_type);
                            *transport_client.write().await = Some(client);

                            tracing::info!("Adaptive transport client ready");
                        }
                    } else {
                        // No instances available, reset transport detection
                        let mut transport_type_guard = transport_type.write().await;
                        let mut transport_client_guard = transport_client.write().await;
                        
                        if transport_type_guard.is_some() {
                            tracing::info!("No instances available, resetting transport detection");
                            *transport_type_guard = None;
                            *transport_client_guard = None;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Create a transport client for the detected transport type
    fn create_transport_client(
        transport_type: &ComponentTransportType,
        nats_client: &Option<NatsClient>,
    ) -> Result<AdaptiveTransportClient> {
        match transport_type {
            ComponentTransportType::Http => {
                let http_client = Arc::new(HttpRequestClient::from_env()?);
                Ok(AdaptiveTransportClient::Http(http_client))
            }
            ComponentTransportType::Nats => {
                let nats_client = nats_client
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("NATS client not available for NATS transport"))?;
                Ok(AdaptiveTransportClient::Nats(nats_client.clone()))
            }
        }
    }

    /// Get the detected transport type for this component
    pub async fn get_transport_type(&self) -> Option<ComponentTransportType> {
        self.transport_type.read().await.clone()
    }

    /// Check if HTTP transport is detected
    pub async fn is_http_transport(&self) -> bool {
        matches!(self.get_transport_type().await, Some(ComponentTransportType::Http))
    }

    /// Check if NATS transport is detected
    pub async fn is_nats_transport(&self) -> bool {
        matches!(self.get_transport_type().await, Some(ComponentTransportType::Nats))
    }

    /// Wait for transport type to be detected
    pub async fn wait_for_transport_detection(&self) -> Result<ComponentTransportType> {
        // First wait for instances to be available
        self.component_client.wait_for_instances().await?;
        
        // Then wait for transport type to be detected
        loop {
            if let Some(transport_type) = self.get_transport_type().await {
                return Ok(transport_type);
            }
            
            // Small delay before checking again
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    /// Send a request to a specific instance using the detected transport
    pub async fn send_request_to_instance(
        &self,
        instance_id: i64,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes> {
        // Get the transport client
        let transport_client = {
            let guard = self.transport_client.read().await;
            guard.clone().ok_or_else(|| {
                anyhow::anyhow!("Transport client not ready. Call wait_for_transport_detection() first.")
            })?
        };

        // Get the instance details to determine the address
        let instances = self.component_client.instances();
        let instance = instances
            .iter()
            .find(|i| i.instance_id == instance_id)
            .ok_or_else(|| anyhow::anyhow!("Instance {} not found", instance_id))?;

        // Determine the address based on transport type
        let address = match &instance.transport {
            TransportType::HttpTcp { http_endpoint } => http_endpoint.clone(),
            TransportType::NatsTcp(subject) => subject.clone(),
        };

        // Send the request using the appropriate transport
        match transport_client {
            AdaptiveTransportClient::Http(http_client) => {
                http_client.send_request(address, payload, headers).await
            }
            AdaptiveTransportClient::Nats(nats_client) => {
                // For NATS, we'll send without custom headers for now
                // TODO: Implement proper header conversion when needed
                let response = nats_client
                    .request(address, payload)
                    .await?;

                Ok(response.payload)
            }
        }
    }

    /// Get all available instances
    pub fn get_instances(&self) -> Vec<Instance> {
        self.component_client.instances()
    }

    /// Get transport detection status for debugging
    pub async fn get_transport_status(&self) -> String {
        let transport_type = self.transport_type.read().await;
        let instances = self.get_instances();
        
        match transport_type.as_ref() {
            Some(ComponentTransportType::Http) => {
                format!("HTTP transport detected ({} instances)", instances.len())
            }
            Some(ComponentTransportType::Nats) => {
                format!("NATS transport detected ({} instances)", instances.len())
            }
            None => {
                if instances.is_empty() {
                    "No instances discovered yet".to_string()
                } else {
                    "Transport detection in progress...".to_string()
                }
            }
        }
    }
}

#[async_trait]
impl RequestPlaneClient for AdaptiveRequestPlaneClient {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        headers: Headers,
    ) -> Result<Bytes> {
        // Try to find an instance that matches the address
        let instances = self.component_client.instances();
        
        for instance in &instances {
            let instance_address = match &instance.transport {
                TransportType::HttpTcp { http_endpoint } => http_endpoint.clone(),
                TransportType::NatsTcp(subject) => subject.clone(),
            };

            if instance_address == address {
                return self.send_request_to_instance(instance.instance_id, payload, headers).await;
            }
        }

        // If no matching instance found, try to use the first available instance
        if let Some(instance) = instances.first() {
            tracing::warn!(
                address = %address,
                instance_id = instance.instance_id,
                "No exact address match found, using first available instance"
            );
            return self.send_request_to_instance(instance.instance_id, payload, headers).await;
        }

        Err(anyhow::anyhow!("No instances available for address: {}", address))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_component_transport_type() {
        assert_eq!(ComponentTransportType::Http, ComponentTransportType::Http);
        assert_ne!(ComponentTransportType::Http, ComponentTransportType::Nats);
    }
}
