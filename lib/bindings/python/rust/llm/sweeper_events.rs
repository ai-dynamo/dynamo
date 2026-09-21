// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PyO3 bindings for DEP #15073 (Sweeper event emission over Dynamo's event
// plane). Modeled directly on the real FpmDirectPublisher/FpmEventSubscriber
// in fpm_bindings.rs -- same `crate::Endpoint` argument convention, same
// tokio::runtime::Handle acquisition
// (`endpoint.inner.component().drt().runtime().secondary()`), same
// spawn-a-background-task-and-blocking_recv-under-allow_threads pattern for
// the subscriber side. Confirmed real, not guessed:
//
//   EventPublisher::for_endpoint(endpoint: &Endpoint, topic) -> Result<Self>
//   EventPublisher::publish_bytes_ref(&self, &[u8]) -> Result<()>
//   EventSubscriber::for_endpoint(endpoint: &Endpoint, topic) -> Result<Self>
//   EventSubscriber::next(&mut self) -> Option<Result<EventEnvelope>>
//   EventEnvelope { publisher_id: u64, sequence: u64, published_at: u64,
//                   topic: String, payload: Bytes }
//
// where `Endpoint` here is `dynamo_runtime::component::Endpoint` -- the same
// type already inside `crate::Endpoint.inner`. No DistributedRuntime/
// EndpointId plumbing is needed: we already hold a live Endpoint handle.
//
// One subject per event type (DEP's "Subject naming"): SweeperEventPublisher
// lazily creates and caches one EventPublisher per distinct `subject` string
// it's asked to publish to.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use super::*;
use crate::Endpoint;
use crate::to_pyerr;
use dynamo_runtime::component::Endpoint as RuntimeEndpoint;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};

// ---------------------------------------------------------------------------
// Publisher: Sweeper process -> event plane
// ---------------------------------------------------------------------------

/// Publishes Sweeper progress/outcome events onto the Dynamo event plane.
/// Lazily creates one EventPublisher per distinct subject passed to
/// `publish_subject` (one per Sweeper event type, per DEP #15073's subject
/// naming: `sweeper.<run_uid>.<event_name>`, computed on the Python side).
#[pyclass]
pub(crate) struct SweeperEventPublisher {
    endpoint: RuntimeEndpoint,
    runtime_handle: tokio::runtime::Handle,
    publishers: Mutex<HashMap<String, EventPublisher>>,
}

#[pymethods]
impl SweeperEventPublisher {
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    #[new]
    #[pyo3(signature = (endpoint,))]
    fn new(endpoint: Endpoint) -> PyResult<Self> {
        let runtime_handle = endpoint.inner.component().drt().runtime().secondary();
        Ok(Self {
            endpoint: endpoint.inner,
            runtime_handle,
            publishers: Mutex::new(HashMap::new()),
        })
    }

    /// Blocking publish of one pre-serialized envelope onto `subject`.
    /// Intended to be called from a background thread (the drain loop in
    /// sweeper_event_plane.py), never from a latency-sensitive caller.
    /// Releases the GIL while the underlying publish awaits.
    fn publish_subject(&self, py: Python, subject: String, payload: Vec<u8>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut publishers = self
                .publishers
                .lock()
                .map_err(|e| to_pyerr(format!("SweeperEventPublisher lock poisoned: {e}")))?;

            if !publishers.contains_key(&subject) {
                let publisher = self
                    .runtime_handle
                    .block_on(EventPublisher::for_endpoint(&self.endpoint, subject.clone()))
                    .map_err(to_pyerr)?;
                publishers.insert(subject.clone(), publisher);
            }

            let publisher = publishers.get(&subject).expect("just inserted");
            self.runtime_handle
                .block_on(publisher.publish_bytes_ref(&payload))
                .map_err(to_pyerr)
        })
    }

    /// Drop all underlying publishers. Each one unregisters itself from
    /// discovery on drop (EventPublisher's own Drop impl) -- nothing extra
    /// to do here.
    fn close(&self) -> PyResult<()> {
        let mut publishers = self
            .publishers
            .lock()
            .map_err(|e| to_pyerr(format!("SweeperEventPublisher lock poisoned: {e}")))?;
        publishers.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Subscriber: event plane -> consumer. Not required for item 4's scope
// (emission only -- see DEP #15073's Non-goals), included because it's the
// natural pair and is useful for integration-testing the publisher above
// without a separate NATS/ZMQ inspection tool.
// ---------------------------------------------------------------------------

#[pyclass]
pub(crate) struct SweeperEventSubscriber {
    endpoint: RuntimeEndpoint,
    runtime_handle: tokio::runtime::Handle,
    channels: Mutex<HashMap<String, Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>>>,
}

#[pymethods]
impl SweeperEventSubscriber {
    #[new]
    #[pyo3(signature = (endpoint,))]
    fn new(endpoint: Endpoint) -> PyResult<Self> {
        let runtime_handle = endpoint.inner.component().drt().runtime().secondary();
        Ok(Self {
            endpoint: endpoint.inner,
            runtime_handle,
            channels: Mutex::new(HashMap::new()),
        })
    }

    /// Blocking receive of the next envelope payload on `subject`, or None if
    /// the stream ended. Releases the GIL while waiting. Spawns one
    /// background task per distinct subject, lazily, on first call --
    /// mirrors FpmEventSubscriber's recv-mode pattern.
    fn recv(&self, py: Python, subject: String) -> PyResult<Option<Vec<u8>>> {
        let rx_arc = {
            let mut channels = self
                .channels
                .lock()
                .map_err(|e| to_pyerr(format!("SweeperEventSubscriber lock poisoned: {e}")))?;

            if !channels.contains_key(&subject) {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
                let endpoint = self.endpoint.clone();
                let topic = subject.clone();

                self.runtime_handle.spawn(async move {
                    let mut subscriber =
                        match EventSubscriber::for_endpoint(&endpoint, topic.clone()).await {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::error!(
                                    "Sweeper subscriber ({topic}): failed to create: {e}"
                                );
                                return;
                            }
                        };

                    loop {
                        match subscriber.next().await {
                            Some(Ok(envelope)) => {
                                if tx.send(envelope.payload.to_vec()).is_err() {
                                    tracing::info!(
                                        "Sweeper subscriber ({topic}): receiver dropped, exiting"
                                    );
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                tracing::warn!("Sweeper subscriber ({topic}): event error: {e}");
                            }
                            None => {
                                tracing::info!("Sweeper subscriber ({topic}): stream ended");
                                break;
                            }
                        }
                    }
                });

                channels.insert(subject.clone(), Arc::new(Mutex::new(rx)));
            }

            channels
                .get(&subject)
                .expect("just inserted or already present")
                .clone()
        };

        py.allow_threads(move || {
            let mut rx = rx_arc
                .lock()
                .map_err(|e| to_pyerr(format!("channel lock poisoned: {e}")))?;
            Ok(rx.blocking_recv())
        })
    }
}

// Registration: add both classes wherever FpmDirectPublisher/FpmEventRelay/
// FpmEventSubscriber are registered today (same crate, so almost certainly
// the same `m.add_class::<...>()` block as those three):
//
//   m.add_class::<SweeperEventPublisher>()?;
//   m.add_class::<SweeperEventSubscriber>()?;
