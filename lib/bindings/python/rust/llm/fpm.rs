// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Forward Pass Metrics (FPM) event plane integration.
//!
//! - `FpmEventRelay`: thin wrapper around `dynamo_llm::fpm_publisher::FpmEventRelay`
//! - `FpmEventSubscriber`: wraps `EventSubscriber::for_component` for the consumer side

use std::sync::Arc;

use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::Endpoint;
use crate::to_pyerr;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;

const FPM_TOPIC: &str = "forward-pass-metrics";

// ---------------------------------------------------------------------------
// Relay: raw ZMQ (child process) -> event plane
// ---------------------------------------------------------------------------

/// Relay that bridges ForwardPassMetrics from a local raw ZMQ PUB socket
/// (InstrumentedScheduler in EngineCore child process) to the Dynamo event
/// plane with automatic discovery registration.
#[pyclass]
pub(crate) struct FpmEventRelay {
    inner: llm_rs::fpm_publisher::FpmEventRelay,
}

#[pymethods]
impl FpmEventRelay {
    /// Create a relay that bridges raw ZMQ to the event plane.
    ///
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    ///     zmq_endpoint: Local ZMQ PUB address to subscribe to
    ///         (e.g., "tcp://127.0.0.1:20380").
    #[new]
    #[pyo3(signature = (endpoint, zmq_endpoint))]
    fn new(endpoint: Endpoint, zmq_endpoint: String) -> PyResult<Self> {
        let component = endpoint.inner.component().clone();
        let inner =
            llm_rs::fpm_publisher::FpmEventRelay::new(component, zmq_endpoint).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Shut down the relay task.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Subscriber: event plane -> consumer
// ---------------------------------------------------------------------------

/// Subscriber for ForwardPassMetrics from the event plane.
///
/// Auto-discovers engine publishers via the discovery plane (K8s CRD / etcd / file).
/// Returns raw msgspec-serialized bytes that Python decodes with
/// `forward_pass_metrics.decode()`.
#[pyclass]
pub(crate) struct FpmEventSubscriber {
    rx: Arc<std::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>,
    cancel: CancellationToken,
}

#[pymethods]
impl FpmEventSubscriber {
    /// Create a subscriber that auto-discovers FPM publishers.
    ///
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    #[new]
    #[pyo3(signature = (endpoint,))]
    fn new(endpoint: Endpoint) -> PyResult<Self> {
        let component = endpoint.inner.component().clone();
        let rt = component.drt().runtime().secondary();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();

        rt.spawn(async move {
            let mut subscriber = match EventSubscriber::for_component(&component, FPM_TOPIC).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("FPM subscriber: failed to create: {e}");
                    return;
                }
            };

            tracing::info!("FPM subscriber: listening for forward-pass-metrics events");

            loop {
                tokio::select! {
                    biased;
                    _ = cancel_clone.cancelled() => {
                        tracing::info!("FPM subscriber: shutting down");
                        break;
                    }
                    event = subscriber.next() => {
                        match event {
                            Some(Ok(envelope)) => {
                                if tx.send(envelope.payload.to_vec()).is_err() {
                                    tracing::info!("FPM subscriber: receiver dropped, exiting");
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                tracing::warn!("FPM subscriber: event error: {e}");
                            }
                            None => {
                                tracing::info!("FPM subscriber: stream ended");
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            rx: Arc::new(std::sync::Mutex::new(rx)),
            cancel,
        })
    }

    /// Blocking receive of next message bytes. Releases the GIL while waiting.
    ///
    /// Returns the raw msgspec payload, or None if the stream is closed.
    fn recv(&self, py: Python) -> PyResult<Option<Vec<u8>>> {
        let rx = self.rx.clone();
        py.allow_threads(move || {
            let mut guard = rx
                .lock()
                .map_err(|e| to_pyerr(format!("lock poisoned: {e}")))?;
            Ok(guard.blocking_recv())
        })
    }

    /// Shut down the subscriber.
    fn shutdown(&self) {
        self.cancel.cancel();
    }
}

impl Drop for FpmEventSubscriber {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
