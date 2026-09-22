// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PyO3 bindings for DEP #15073 (Sweeper event emission over Dynamo's event
// plane).

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

#[pyclass]
pub(crate) struct SweeperEventSubscriber {
    endpoint: RuntimeEndpoint,
    runtime_handle: tokio::runtime::Handle,
    channels: Mutex<HashMap<String, Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>>>,
    cancel_txs: Mutex<HashMap<String, tokio::sync::oneshot::Sender<()>>>,
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
            cancel_txs: Mutex::new(HashMap::new()),
        })
    }

    /// Blocking receive of the next envelope payload on `subject`, or None if
    /// the stream ended. Releases the GIL while waiting. Spawns one
    /// background task per distinct subject, lazily, on first call --
    /// mirrors FpmEventSubscriber's recv-mode pattern. Each spawned task
    /// selects between a per-subject cancellation signal and
    /// `subscriber.next()`, so it exits promptly on `close()` or Drop even
    /// while idle (no events arriving) on that subject.
    fn recv(&self, py: Python, subject: String) -> PyResult<Option<Vec<u8>>> {
        let rx_arc = {
            let mut channels = self
                .channels
                .lock()
                .map_err(|e| to_pyerr(format!("SweeperEventSubscriber lock poisoned: {e}")))?;

            if !channels.contains_key(&subject) {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
                let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel::<()>();
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
                        tokio::select! {
                            // Fires on close()/Drop sending, or the sender
                            // being dropped -- either way, stop.
                            _ = &mut cancel_rx => {
                                tracing::info!(
                                    "Sweeper subscriber ({topic}): cancelled, exiting"
                                );
                                break;
                            }
                            next = subscriber.next() => {
                                match next {
                                    Some(Ok(envelope)) => {
                                        if tx.send(envelope.payload.to_vec()).is_err() {
                                            tracing::info!(
                                                "Sweeper subscriber ({topic}): receiver dropped, exiting"
                                            );
                                            break;
                                        }
                                    }
                                    Some(Err(e)) => {
                                        tracing::warn!(
                                            "Sweeper subscriber ({topic}): event error: {e}"
                                        );
                                    }
                                    None => {
                                        tracing::info!(
                                            "Sweeper subscriber ({topic}): stream ended"
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                    }
                });

                channels.insert(subject.clone(), Arc::new(Mutex::new(rx)));
                self.cancel_txs
                    .lock()
                    .map_err(|e| to_pyerr(format!("SweeperEventSubscriber lock poisoned: {e}")))?
                    .insert(subject.clone(), cancel_tx);
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

    /// Cancel every outstanding per-subject subscriber task. Idempotent --
    /// safe to call more than once, and Drop calls this too so an explicit
    /// close() is not required for cleanup.
    fn close(&self) -> PyResult<()> {
        let mut cancel_txs = self
            .cancel_txs
            .lock()
            .map_err(|e| to_pyerr(format!("SweeperEventSubscriber lock poisoned: {e}")))?;
        for (_, tx) in cancel_txs.drain() {
            // Err means the task already exited on its own (e.g. stream
            // ended); nothing to do in that case either.
            let _ = tx.send(());
        }
        Ok(())
    }
}

impl Drop for SweeperEventSubscriber {
    fn drop(&mut self) {
        if let Ok(mut cancel_txs) = self.cancel_txs.lock() {
            for (_, tx) in cancel_txs.drain() {
                let _ = tx.send(());
            }
        }
    }
}
