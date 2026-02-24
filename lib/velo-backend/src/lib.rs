// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! Multi-transport active message routing framework.
//!
//! `velo-backend` abstracts TCP, HTTP, NATS, gRPC, and UCX behind a unified
//! [`Transport`] trait with zero-copy [`bytes::Bytes`], fire-and-forget error
//! callbacks, priority-based peer routing, and 3-phase graceful shutdown.
//!
//! # Architecture
//!
//! [`VeloBackend`] is the central orchestrator. It holds a set of transports,
//! each identified by a [`TransportKey`]. When a peer registers, the backend
//! selects a *primary* transport (highest-priority compatible transport) and
//! records any alternatives. Outbound messages are routed through the primary
//! transport by default, or through an explicit alternative.
//!
//! Inbound messages arrive via [`DataStreams`] — three independent channels
//! for messages, responses, and events.
//!
//! # Shutdown
//!
//! Graceful shutdown follows three phases:
//! 1. **Gate** — flip the draining flag; transports reject new inbound requests.
//! 2. **Drain** — wait for all in-flight requests to complete.
//! 3. **Teardown** — cancel listeners/writers and call `shutdown()` on each transport.

mod address;

pub mod tcp;

// #[cfg(feature = "ucx")]
// pub mod ucx;

// #[cfg(feature = "http")]
// pub mod http;

// #[cfg(feature = "nats")]
// pub mod nats;

// #[cfg(feature = "grpc")]
// pub mod grpc;

mod transport;

use std::{collections::HashMap, sync::Arc};

use dashmap::DashMap;
use parking_lot::Mutex;

// Public re-exports from velo-common
pub use velo_common::{
    InstanceId, PeerInfo, TransportKey, WorkerAddress, WorkerAddressError, WorkerId,
};

// Internal builder for address construction
use address::WorkerAddressBuilder;

// Re-export transport types
pub use transport::{
    DataStreams, HealthCheckError, InFlightGuard, MessageType, ShutdownPolicy, ShutdownState,
    Transport, TransportAdapter, TransportError, TransportErrorHandler, make_channels,
};

/// Errors returned by [`VeloBackend`] operations.
#[derive(Debug, thiserror::Error)]
pub enum VeloBackendError {
    /// No transport could accept the peer's address.
    #[error("No compatible transports found")]
    NoCompatibleTransports,

    /// The target instance was never registered via [`VeloBackend::register_peer`].
    #[error("Transport not found for instance: {0}")]
    InstanceNotRegistered(InstanceId),

    /// The worker ID is not in the fast-path cache.
    #[error("Worker not found: {0}")]
    WorkerNotRegistered(WorkerId),

    /// The requested [`TransportKey`] does not match any loaded transport.
    #[error("Transport not found: {0}")]
    TransportNotFound(TransportKey),

    /// The priority list does not match the set of available transports.
    #[error("Invalid transport priority: {0}")]
    InvalidTransportPriority(String),
}

/// Central orchestrator that aggregates multiple transports and routes messages
/// to peers via priority-based transport selection.
///
/// Each peer is registered with all compatible transports; the highest-priority
/// compatible transport becomes the *primary* for that peer. Worker IDs are
/// cached for fast-path routing without discovery lookups.
pub struct VeloBackend {
    instance_id: InstanceId,
    address: WorkerAddress,
    priorities: Mutex<Vec<TransportKey>>,
    transports: HashMap<TransportKey, Arc<dyn Transport>>,
    primary_transport: DashMap<InstanceId, Arc<dyn Transport>>,
    alternative_transports: DashMap<InstanceId, Vec<TransportKey>>,
    workers: DashMap<WorkerId, InstanceId>,
    shutdown_state: ShutdownState,

    #[allow(dead_code)]
    runtime: tokio::runtime::Handle,
}

impl VeloBackend {
    /// Create a new backend from a list of transports.
    ///
    /// Each transport is started (bound, listening) and its address is merged
    /// into a composite [`WorkerAddress`]. Returns the backend and the
    /// [`DataStreams`] receivers for inbound messages.
    pub async fn new(
        backend_transports: Vec<Arc<dyn Transport>>,
    ) -> anyhow::Result<(Self, DataStreams)> {
        let instance_id = InstanceId::new_v4();

        // build worker address
        let mut priorities = Vec::new();
        let mut builder = WorkerAddressBuilder::new();
        let mut transports = HashMap::new();

        let (adapter, data_streams) = transport::make_channels();
        let shutdown_state = adapter.shutdown_state.clone();

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
                shutdown_state,
                runtime,
            },
            data_streams,
        ))
    }

    /// Returns this backend's unique instance identifier.
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Returns a [`PeerInfo`] describing this backend (instance ID + composite address).
    pub fn peer_info(&self) -> PeerInfo {
        PeerInfo::new(self.instance_id, self.address.clone())
    }

    /// Returns `true` if the given instance has been registered via [`register_peer`](Self::register_peer).
    pub fn is_registered(&self, instance_id: InstanceId) -> bool {
        self.primary_transport.contains_key(&instance_id)
    }

    /// Fast-path lookup of worker_id -> instance_id from cache.
    ///
    /// Returns `WorkerNotRegistered` if the worker is not in the cache.
    /// Higher layers (Velo, VeloEvents, ActiveMessageClient) should handle
    /// discovery fallback when this returns an error.
    ///
    /// # Example
    /// ```ignore
    /// match backend.try_translate_worker_id(worker_id) {
    ///     Ok(instance_id) => { /* fast path: send immediately */ }
    ///     Err(VeloBackendError::WorkerNotRegistered(_)) => {
    ///         /* slow path: query discovery, then register_peer() */
    ///     }
    /// }
    /// ```
    pub fn try_translate_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> Result<InstanceId, VeloBackendError> {
        self.workers
            .get(&worker_id)
            .map(|entry| *entry)
            .ok_or(VeloBackendError::WorkerNotRegistered(worker_id))
    }

    /// Deprecated: Use `try_translate_worker_id()` for explicit fast-path semantics.
    #[deprecated(since = "0.7.0", note = "Use try_translate_worker_id() instead")]
    pub fn translate_worker_id(&self, worker_id: WorkerId) -> Result<InstanceId, VeloBackendError> {
        self.try_translate_worker_id(worker_id)
    }

    /// Check if an instance_id is registered.
    pub fn has_instance(&self, instance_id: InstanceId) -> bool {
        self.primary_transport.contains_key(&instance_id)
    }

    /// Send a message to a registered peer via its primary transport.
    ///
    /// Returns [`VeloBackendError::InstanceNotRegistered`] if the peer has not
    /// been registered with [`register_peer`](Self::register_peer).
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
            .ok_or(VeloBackendError::InstanceNotRegistered(target))?;

        transport.send_message(target, header, payload, message_type, on_error);

        Ok(())
    }

    /// Send a message to a registered peer via a specific transport.
    ///
    /// If `transport_key` matches the peer's primary transport, the message is
    /// sent directly. Otherwise, the alternative transports are searched.
    /// Returns [`VeloBackendError::NoCompatibleTransports`] if the requested
    /// transport is not available for this peer.
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
            .ok_or(VeloBackendError::InstanceNotRegistered(target))?;

        if transport.value().key() == transport_key {
            transport.send_message(target, header, payload, message_type, on_error);
            return Ok(());
        } else {
            // if we got here, we can unwrap because there is an entry in the alternative_transports map
            let alternative_transports = self
                .alternative_transports
                .get(&target)
                .ok_or(VeloBackendError::InstanceNotRegistered(target))?;

            for alternative_transport in alternative_transports.iter() {
                if *alternative_transport == transport_key
                    && let Some(transport) = self.transports.get(alternative_transport)
                {
                    transport.send_message(target, header, payload, message_type, on_error);
                    return Ok(());
                }
            }
        }

        Err(VeloBackendError::NoCompatibleTransports)?
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

    /// Register a remote peer with all compatible transports.
    ///
    /// The highest-priority compatible transport becomes the peer's *primary*.
    /// Returns [`VeloBackendError::NoCompatibleTransports`] if no transport
    /// can accept the peer's address.
    pub fn register_peer(&self, peer: PeerInfo) -> Result<(), VeloBackendError> {
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
            return Err(VeloBackendError::NoCompatibleTransports);
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
    ) -> Result<(), VeloBackendError> {
        let required_transports = self.available_transports();
        if required_transports.len() != priorities.len() {
            return Err(VeloBackendError::InvalidTransportPriority(format!(
                "Required transports: {:?}, provided priorities: {:?}",
                required_transports, priorities
            )));
        }

        for priority in &priorities {
            if !required_transports.contains(priority) {
                return Err(VeloBackendError::InvalidTransportPriority(format!(
                    "Priority transport not found: {:?}",
                    priority
                )));
            }
        }

        let mut guard = self.priorities.lock();
        *guard = priorities;
        Ok(())
    }

    /// Get the shared shutdown state.
    pub fn shutdown_state(&self) -> &ShutdownState {
        &self.shutdown_state
    }

    /// Perform a graceful 3-phase shutdown.
    ///
    /// 1. **Gate**: Flip the draining flag and notify each transport via `begin_drain()`.
    /// 2. **Drain**: Wait for all in-flight requests to complete (per `policy`).
    /// 3. **Teardown**: Cancel the teardown token and call `shutdown()` on each transport.
    pub async fn graceful_shutdown(&self, policy: ShutdownPolicy) {
        // Phase 1: Gate
        self.shutdown_state.begin_drain();
        for transport in self.transports.values() {
            transport.begin_drain();
        }

        // Phase 2: Drain
        match policy {
            ShutdownPolicy::WaitForever => {
                self.shutdown_state.wait_for_drain().await;
            }
            ShutdownPolicy::Timeout(duration) => {
                let _ = tokio::time::timeout(duration, self.shutdown_state.wait_for_drain()).await;
            }
        }

        // Phase 3: Teardown
        self.shutdown_state.teardown_token().cancel();
        for transport in self.transports.values() {
            transport.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::future::BoxFuture;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    /// Mock transport for testing VeloBackend logic without real networking.
    struct MockTransport {
        key: TransportKey,
        address: WorkerAddress,
        accept_register: bool,
        started: AtomicBool,
        drained: AtomicBool,
        shut_down: AtomicBool,
        send_count: AtomicUsize,
    }

    impl MockTransport {
        fn new(key: &str, accept_register: bool) -> Arc<Self> {
            let mut builder = WorkerAddressBuilder::new();
            builder
                .add_entry(key, format!("mock://{}", key).into_bytes())
                .unwrap();
            let address = builder.build().unwrap();

            Arc::new(Self {
                key: TransportKey::from(key),
                address,
                accept_register,
                started: AtomicBool::new(false),
                drained: AtomicBool::new(false),
                shut_down: AtomicBool::new(false),
                send_count: AtomicUsize::new(0),
            })
        }
    }

    impl Transport for MockTransport {
        fn key(&self) -> TransportKey {
            self.key.clone()
        }
        fn address(&self) -> WorkerAddress {
            self.address.clone()
        }
        fn register(&self, _peer_info: PeerInfo) -> Result<(), TransportError> {
            if self.accept_register {
                Ok(())
            } else {
                Err(TransportError::NoEndpoint)
            }
        }
        fn send_message(
            &self,
            _instance_id: InstanceId,
            _header: Vec<u8>,
            _payload: Vec<u8>,
            _message_type: MessageType,
            _on_error: Arc<dyn TransportErrorHandler>,
        ) {
            self.send_count.fetch_add(1, Ordering::Relaxed);
        }
        fn start(
            &self,
            _instance_id: InstanceId,
            _channels: TransportAdapter,
            _rt: tokio::runtime::Handle,
        ) -> BoxFuture<'_, anyhow::Result<()>> {
            self.started.store(true, Ordering::Relaxed);
            Box::pin(async { Ok(()) })
        }
        fn shutdown(&self) {
            self.shut_down.store(true, Ordering::Relaxed);
        }
        fn begin_drain(&self) {
            self.drained.store(true, Ordering::Relaxed);
        }
        fn check_health(
            &self,
            _instance_id: InstanceId,
            _timeout: Duration,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<(), transport::HealthCheckError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(()) })
        }
    }

    struct NoopErrorHandler;
    impl TransportErrorHandler for NoopErrorHandler {
        fn on_error(&self, _header: Bytes, _payload: Bytes, _error: String) {}
    }

    /// Helper: build a PeerInfo with entries for specified transport keys.
    fn make_peer_info(keys: &[&str]) -> PeerInfo {
        let instance_id = InstanceId::new_v4();
        let mut builder = WorkerAddressBuilder::new();
        for key in keys {
            builder
                .add_entry(*key, format!("mock://{}", key).into_bytes())
                .unwrap();
        }
        let address = builder.build().unwrap();
        PeerInfo::new(instance_id, address)
    }

    #[tokio::test]
    async fn test_new_single_transport() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t.clone() as Arc<dyn Transport>])
            .await
            .unwrap();

        assert!(t.started.load(Ordering::Relaxed));
        // instance_id should be a valid v4 UUID (non-zero)
        assert!(!backend.instance_id().as_bytes().iter().all(|&b| b == 0));
        assert_eq!(backend.available_transports().len(), 1);
    }

    #[tokio::test]
    async fn test_new_multiple_transports() {
        let t1 = MockTransport::new("tcp", true);
        let t2 = MockTransport::new("http", true);
        let (backend, _streams) = VeloBackend::new(vec![
            t1.clone() as Arc<dyn Transport>,
            t2.clone() as Arc<dyn Transport>,
        ])
        .await
        .unwrap();

        assert!(t1.started.load(Ordering::Relaxed));
        assert!(t2.started.load(Ordering::Relaxed));
        assert_eq!(backend.available_transports().len(), 2);
    }

    #[tokio::test]
    async fn test_register_peer_selects_primary_by_priority() {
        let t1 = MockTransport::new("tcp", true);
        let t2 = MockTransport::new("http", true);
        let (backend, _streams) = VeloBackend::new(vec![
            t1.clone() as Arc<dyn Transport>,
            t2.clone() as Arc<dyn Transport>,
        ])
        .await
        .unwrap();

        let peer = make_peer_info(&["tcp", "http"]);
        let peer_id = peer.instance_id();
        backend.register_peer(peer).unwrap();

        assert!(backend.is_registered(peer_id));
        // Primary should be "tcp" (first in priority)
        let primary = backend.primary_transport.get(&peer_id).unwrap();
        assert_eq!(primary.value().key(), TransportKey::from("tcp"));
    }

    #[tokio::test]
    async fn test_register_peer_no_compatible_transports() {
        // Transport rejects all registrations
        let t = MockTransport::new("tcp", false);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let peer = make_peer_info(&["tcp"]);
        let result = backend.register_peer(peer);
        assert!(matches!(
            result,
            Err(VeloBackendError::NoCompatibleTransports)
        ));
    }

    #[tokio::test]
    async fn test_register_peer_stores_worker_mapping() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let peer = make_peer_info(&["tcp"]);
        let peer_id = peer.instance_id();
        let worker_id = peer_id.worker_id();
        backend.register_peer(peer).unwrap();

        let resolved = backend.try_translate_worker_id(worker_id).unwrap();
        assert_eq!(resolved, peer_id);
    }

    #[tokio::test]
    async fn test_send_message_routes_to_primary() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t.clone() as Arc<dyn Transport>])
            .await
            .unwrap();

        let peer = make_peer_info(&["tcp"]);
        let peer_id = peer.instance_id();
        backend.register_peer(peer).unwrap();

        backend
            .send_message(
                peer_id,
                vec![1],
                vec![2],
                MessageType::Message,
                Arc::new(NoopErrorHandler),
            )
            .unwrap();

        assert_eq!(t.send_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_send_message_unregistered_peer() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let result = backend.send_message(
            InstanceId::new_v4(),
            vec![],
            vec![],
            MessageType::Message,
            Arc::new(NoopErrorHandler),
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_message_with_transport_primary_match() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t.clone() as Arc<dyn Transport>])
            .await
            .unwrap();

        let peer = make_peer_info(&["tcp"]);
        let peer_id = peer.instance_id();
        backend.register_peer(peer).unwrap();

        backend
            .send_message_with_transport(
                peer_id,
                vec![1],
                vec![2],
                MessageType::Message,
                Arc::new(NoopErrorHandler),
                TransportKey::from("tcp"),
            )
            .unwrap();

        assert_eq!(t.send_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_send_message_with_transport_alternative() {
        let t1 = MockTransport::new("tcp", true);
        let t2 = MockTransport::new("http", true);
        let (backend, _streams) = VeloBackend::new(vec![
            t1.clone() as Arc<dyn Transport>,
            t2.clone() as Arc<dyn Transport>,
        ])
        .await
        .unwrap();

        let peer = make_peer_info(&["tcp", "http"]);
        let peer_id = peer.instance_id();
        backend.register_peer(peer).unwrap();

        // Send via "http" (the alternative transport)
        backend
            .send_message_with_transport(
                peer_id,
                vec![1],
                vec![2],
                MessageType::Message,
                Arc::new(NoopErrorHandler),
                TransportKey::from("http"),
            )
            .unwrap();

        assert_eq!(t2.send_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_send_message_with_transport_not_found() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let peer = make_peer_info(&["tcp"]);
        let peer_id = peer.instance_id();
        backend.register_peer(peer).unwrap();

        let result = backend.send_message_with_transport(
            peer_id,
            vec![],
            vec![],
            MessageType::Message,
            Arc::new(NoopErrorHandler),
            TransportKey::from("grpc"),
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_try_translate_worker_id_not_found() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let result = backend.try_translate_worker_id(InstanceId::new_v4().worker_id());
        assert!(matches!(
            result,
            Err(VeloBackendError::WorkerNotRegistered(_))
        ));
    }

    #[tokio::test]
    async fn test_set_transport_priority_valid() {
        let t1 = MockTransport::new("tcp", true);
        let t2 = MockTransport::new("http", true);
        let (backend, _streams) =
            VeloBackend::new(vec![t1 as Arc<dyn Transport>, t2 as Arc<dyn Transport>])
                .await
                .unwrap();

        // Reverse the priority
        backend
            .set_transport_priority(vec![TransportKey::from("http"), TransportKey::from("tcp")])
            .unwrap();
    }

    #[tokio::test]
    async fn test_set_transport_priority_wrong_length() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let result = backend
            .set_transport_priority(vec![TransportKey::from("tcp"), TransportKey::from("http")]);
        assert!(matches!(
            result,
            Err(VeloBackendError::InvalidTransportPriority(_))
        ));
    }

    #[tokio::test]
    async fn test_set_transport_priority_unknown_key() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let result = backend.set_transport_priority(vec![TransportKey::from("unknown")]);
        assert!(matches!(
            result,
            Err(VeloBackendError::InvalidTransportPriority(_))
        ));
    }

    #[tokio::test]
    async fn test_graceful_shutdown_calls_all_transports() {
        let t1 = MockTransport::new("tcp", true);
        let t2 = MockTransport::new("http", true);
        let (backend, _streams) = VeloBackend::new(vec![
            t1.clone() as Arc<dyn Transport>,
            t2.clone() as Arc<dyn Transport>,
        ])
        .await
        .unwrap();

        backend
            .graceful_shutdown(ShutdownPolicy::Timeout(Duration::from_millis(100)))
            .await;

        assert!(t1.drained.load(Ordering::Relaxed));
        assert!(t2.drained.load(Ordering::Relaxed));
        assert!(t1.shut_down.load(Ordering::Relaxed));
        assert!(t2.shut_down.load(Ordering::Relaxed));
        assert!(backend.shutdown_state().is_draining());
        assert!(backend.shutdown_state().teardown_token().is_cancelled());
    }

    #[tokio::test]
    async fn test_peer_info_roundtrip() {
        let t = MockTransport::new("tcp", true);
        let (backend, _streams) = VeloBackend::new(vec![t as Arc<dyn Transport>])
            .await
            .unwrap();

        let info = backend.peer_info();
        assert_eq!(info.instance_id(), backend.instance_id());
    }
}
