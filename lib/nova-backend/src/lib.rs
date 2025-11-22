// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nova Backend: transport-specific address types and helpers.

mod address;

pub mod tcp;

#[cfg(feature = "ucx")]
pub mod ucx;

#[cfg(feature = "http")]
pub mod http;

#[cfg(feature = "nats")]
pub mod nats;

#[cfg(feature = "grpc")]
pub mod grpc;

mod transport;

use std::{collections::HashMap, sync::Arc};

use dashmap::DashMap;
use parking_lot::Mutex;

// Public re-exports from dynamo-identity
pub use dynamo_identity::{InstanceId, WorkerId};

// Re-export identity types
pub use address::{PeerInfo, WorkerAddress};
pub use transport::{
    DataStreams, HealthCheckError, MessageType, Transport, TransportAdapter, TransportError,
    TransportErrorHandler, TransportKey, make_channels,
};

#[derive(Debug, thiserror::Error)]
pub enum NovaBackendError {
    #[error("No compatible transports found")]
    NoCompatibleTransports,

    #[error("Transport not found for instance: {0}")]
    InstanceNotRegistered(InstanceId),

    #[error("Worker not found: {0}")]
    WorkerNotRegistered(WorkerId),

    #[error("Transport not found: {0}")]
    TransportNotFound(TransportKey),

    #[error("Invalid transport priority: {0}")]
    InvalidTransportPriority(String),
}

pub struct NovaBackend {
    instance_id: InstanceId,
    address: WorkerAddress,
    priorities: Mutex<Vec<TransportKey>>,
    transports: HashMap<TransportKey, Arc<dyn Transport>>,
    primary_transport: DashMap<InstanceId, Arc<dyn Transport>>,
    alternative_transports: DashMap<InstanceId, Vec<TransportKey>>,
    workers: DashMap<WorkerId, InstanceId>,

    #[allow(dead_code)]
    runtime: tokio::runtime::Handle,
}

impl NovaBackend {
    pub async fn new(
        backend_transports: Vec<Arc<dyn Transport>>,
    ) -> anyhow::Result<(Self, DataStreams)> {
        let instance_id = InstanceId::new_v4();

        // build worker address
        let mut priorities = Vec::new();
        let mut builder = WorkerAddress::builder();
        let mut transports = HashMap::new();

        let (adapter, data_streams) = transport::make_channels();

        let runtime = tokio::runtime::Handle::current();

        for transport in backend_transports {
            transport
                .start(instance_id, adapter.clone(), runtime.clone())
                .await?;
            builder.merge(&transport.address())?;
            priorities.push(transport.key());
            transports.insert(transport.key(), transport);
        }
        let address = builder.build()?;

        Ok((
            Self {
                instance_id,
                address,
                transports,
                priorities: Mutex::new(priorities),
                primary_transport: DashMap::new(),
                alternative_transports: DashMap::new(),
                workers: DashMap::new(),
                runtime,
            },
            data_streams,
        ))
    }

    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    pub fn peer_info(&self) -> PeerInfo {
        PeerInfo::new(self.instance_id, self.address.clone())
    }

    pub fn is_registered(&self, instance_id: InstanceId) -> bool {
        self.primary_transport.contains_key(&instance_id)
    }

    /// Fast-path lookup of worker_id -> instance_id from cache.
    ///
    /// Returns `WorkerNotRegistered` if the worker is not in the cache.
    /// Higher layers (Nova, NovaEvents, ActiveMessageClient) should handle
    /// discovery fallback when this returns an error.
    ///
    /// # Example
    /// ```ignore
    /// match backend.try_translate_worker_id(worker_id) {
    ///     Ok(instance_id) => { /* fast path: send immediately */ }
    ///     Err(NovaBackendError::WorkerNotRegistered(_)) => {
    ///         /* slow path: query discovery, then register_peer() */
    ///     }
    /// }
    /// ```
    pub fn try_translate_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> Result<InstanceId, NovaBackendError> {
        self.workers
            .get(&worker_id)
            .map(|entry| *entry)
            .ok_or(NovaBackendError::WorkerNotRegistered(worker_id))
    }

    /// Deprecated: Use `try_translate_worker_id()` for explicit fast-path semantics.
    #[deprecated(since = "0.7.0", note = "Use try_translate_worker_id() instead")]
    pub fn translate_worker_id(&self, worker_id: WorkerId) -> Result<InstanceId, NovaBackendError> {
        self.try_translate_worker_id(worker_id)
    }

    /// Check if an instance_id is registered.
    pub fn has_instance(&self, instance_id: InstanceId) -> bool {
        self.primary_transport.contains_key(&instance_id)
    }

    pub fn send_message(
        &self,
        target: InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: Arc<dyn TransportErrorHandler>,
    ) -> anyhow::Result<()> {
        let transport = self
            .primary_transport
            .get(&target)
            .ok_or(NovaBackendError::InstanceNotRegistered(target))?;

        transport.send_message(target, header, payload, message_type, on_error);

        Ok(())
    }

    pub fn send_message_with_transport(
        &self,
        target: InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: Arc<dyn TransportErrorHandler>,
        transport_key: TransportKey,
    ) -> anyhow::Result<()> {
        let transport = self
            .primary_transport
            .get(&target)
            .ok_or(NovaBackendError::InstanceNotRegistered(target))?;

        if transport.value().key() == transport_key {
            transport.send_message(target, header, payload, message_type, on_error);
        } else {
            // if we got here, we can unwrap because there is an entry in the alternative_transports map
            let alternative_transports = self
                .alternative_transports
                .get(&target)
                .ok_or(NovaBackendError::InstanceNotRegistered(target))?;

            for alternative_transport in alternative_transports.iter() {
                if *alternative_transport == transport_key
                    && let Some(transport) = self.transports.get(alternative_transport)
                {
                    transport.send_message(target, header, payload, message_type, on_error);
                    return Ok(());
                }
            }
        }

        Err(NovaBackendError::NoCompatibleTransports)?
    }

    /// Send message to a worker (fast-path only).
    ///
    /// This method uses `try_translate_worker_id()` for fast-path lookup.
    /// Returns `WorkerNotRegistered` error if the worker is not in the cache.
    ///
    /// For automatic discovery, use the two-phase pattern:
    /// ```ignore
    /// match backend.send_message_to_worker(...) {
    ///     Ok(()) => { /* success */ }
    ///     Err(e) if matches_worker_not_registered(&e) => {
    ///         tokio::spawn(async move {
    ///             let instance_id = backend.resolve_and_register_worker(worker_id).await?;
    ///             backend.send_message(instance_id, ...)?;
    ///         });
    ///     }
    /// }
    /// ```
    pub fn send_message_to_worker(
        &self,
        worker_id: WorkerId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: Arc<dyn TransportErrorHandler>,
    ) -> anyhow::Result<()> {
        let instance_id = self.try_translate_worker_id(worker_id)?;
        self.send_message(instance_id, header, payload, message_type, on_error)
    }

    pub fn register_peer(&self, peer: PeerInfo) -> Result<(), NovaBackendError> {
        // try to register the peer with each transport
        // we must have at least one compatible transport; otherwise, return an error
        let instance_id = peer.instance_id();
        let mut compatible_transports = Vec::new();
        for (key, transport) in self.transports.iter() {
            if transport.register(peer.clone()).is_ok() {
                compatible_transports.push(key.clone());
            }
        }
        if compatible_transports.is_empty() {
            return Err(NovaBackendError::NoCompatibleTransports);
        }

        // sort against the preferred transports
        let sorted_transports = self
            .priorities
            .lock()
            .iter()
            .filter(|key| compatible_transports.contains(key))
            .cloned()
            .collect::<Vec<TransportKey>>();

        let primary_transport_key = sorted_transports[0].clone();
        let alternative_transport_keys = sorted_transports[1..].to_vec();

        let primary_transport = self.transports.get(&primary_transport_key).unwrap();

        self.primary_transport
            .insert(instance_id, primary_transport.clone());
        self.alternative_transports
            .insert(instance_id, alternative_transport_keys);
        self.workers.insert(instance_id.worker_id(), instance_id);

        Ok(())
    }

    /// Get the available transports.
    pub fn available_transports(&self) -> Vec<TransportKey> {
        self.transports.keys().cloned().collect()
    }

    /// Set the priority of the transports.
    ///
    /// The list of [`TransportKey`]s must be an order set of the available transports.
    pub fn set_transport_priority(
        &self,
        priorities: Vec<TransportKey>,
    ) -> Result<(), NovaBackendError> {
        let required_transports = self.available_transports();
        if required_transports.len() != priorities.len() {
            return Err(NovaBackendError::InvalidTransportPriority(format!(
                "Required transports: {:?}, provided priorities: {:?}",
                required_transports, priorities
            )));
        }

        for priority in &priorities {
            if !required_transports.contains(priority) {
                return Err(NovaBackendError::InvalidTransportPriority(format!(
                    "Priority transport not found: {:?}",
                    priority
                )));
            }
        }

        let mut guard = self.priorities.lock();
        *guard = priorities;
        Ok(())
    }
}
