// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local client implementation for direct engine access without network overhead

use crate::engine::{
    AnyAsyncEngine, AsyncEngine, AsyncEngineContextProvider, Data, DowncastAnyAsyncEngine,
    StreamData,
};
use crate::traits::DistributedRuntimeProvider;
use crate::v2::entity::{ComponentDescriptor, EndpointDescriptor, NamespaceDescriptor};
use crate::{Result, error};
use std::sync::Arc;

use super::Endpoint;

/// A client that directly invokes local engines without network overhead
#[derive(Clone)]
pub struct LocalClient<Req, Resp, E>
where
    Req: Data,
    Resp: StreamData + AsyncEngineContextProvider,
    E: Data,
{
    engine: Arc<dyn AsyncEngine<Req, Resp, E>>,
    descriptor: EndpointDescriptor,
}

impl<Req, Resp, E> LocalClient<Req, Resp, E>
where
    Req: Data,
    Resp: StreamData + AsyncEngineContextProvider,
    E: Data,
{
    /// Create a LocalClient from an endpoint descriptor
    pub async fn from_descriptor(
        endpoint: &Endpoint,
        descriptor: EndpointDescriptor,
    ) -> Result<Self> {
        let key = descriptor.to_string();

        // Get the engine from the local registry
        let any_engine = endpoint
            .drt()
            .get_local_engine(&key)
            .await
            .ok_or_else(|| error!("No local engine registered for endpoint: {}", key))?;

        // Downcast to the specific types
        let engine = any_engine
            .downcast_stream::<Req, Resp, E>()
            .ok_or_else(|| error!("Type mismatch when downcasting local engine for: {}", key))?;

        Ok(Self { engine, descriptor })
    }

    /// Create a LocalClient from an endpoint
    pub async fn from_endpoint(endpoint: &Endpoint) -> Result<Self> {
        // Extract the full namespace hierarchy
        let namespace_segments = Self::get_namespace_hierarchy(&endpoint.component.namespace);

        // Create the descriptor for this endpoint
        let namespace_desc = NamespaceDescriptor::new(
            &namespace_segments
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| error!("Invalid namespace: {}", e))?;

        let component_desc = namespace_desc
            .component(&endpoint.component.name)
            .map_err(|e| error!("Invalid component: {}", e))?;
        let endpoint_desc = component_desc
            .endpoint(&endpoint.name)
            .map_err(|e| error!("Invalid endpoint: {}", e))?;

        Self::from_descriptor(endpoint, endpoint_desc).await
    }

    /// Extract the full namespace hierarchy from a Namespace
    fn get_namespace_hierarchy(namespace: &super::Namespace) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current: Option<&super::Namespace> = Some(namespace);

        // Walk up the parent chain to collect all namespace segments
        while let Some(ns) = current {
            segments.push(ns.name.clone());
            current = ns.parent.as_deref();
        }

        // Reverse to get root-to-leaf order
        segments.reverse();
        segments
    }

    /// Generate a response using the local engine directly
    pub async fn generate(&self, request: Req) -> Result<Resp, E> {
        self.engine.generate(request).await
    }

    /// Get the descriptor for this client
    pub fn descriptor(&self) -> &EndpointDescriptor {
        &self.descriptor
    }

    /// Get the underlying engine
    pub fn engine(&self) -> &Arc<dyn AsyncEngine<Req, Resp, E>> {
        &self.engine
    }
}

/// Extract the full namespace hierarchy from a Namespace
fn get_namespace_hierarchy(namespace: &super::Namespace) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current: Option<&super::Namespace> = Some(namespace);

    // Walk up the parent chain to collect all namespace segments
    while let Some(ns) = current {
        segments.push(ns.name.clone());
        current = ns.parent.as_deref();
    }

    // Reverse to get root-to-leaf order
    segments.reverse();
    segments
}
