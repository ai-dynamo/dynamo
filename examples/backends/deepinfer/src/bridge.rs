// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process PyO3 calls and callbacks; Python owns engine execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use parking_lot::Mutex;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct Ready {
    pub vocab_size: u32,
    pub eos_token_id: u32,
}

pub struct Generate {
    pub token_ids: Vec<u32>,
    pub stop_token_ids: Vec<u32>,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i32,
    pub max_tokens: u32,
    pub ignore_eos: bool,
    pub seed: Option<u64>,
}

pub struct Output {
    pub token_id: Option<u32>,
    pub finished: bool,
    pub finish_reason: String,
    pub logprob: Option<f64>,
    pub stop_token_id: Option<u32>,
    pub error: Option<String>,
}

#[derive(Clone)]
struct Pending {
    wire_id: String,
    sender: mpsc::Sender<Output>,
    cancelled: CancellationToken,
    completed: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
}

type Requests = Arc<Mutex<HashMap<String, Pending>>>;

pub struct Bridge {
    pub ready: Ready,
    pub failed: CancellationToken,
    runtime: Py<PyAny>,
    pending: Requests,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    engine_active: Arc<AtomicU32>,
    limit: usize,
}

/// Lives inside the response stream, including before its first poll.
pub struct Session {
    bridge: Arc<Bridge>,
    id: String,
    pub receiver: mpsc::Receiver<Output>,
    pub cancelled: CancellationToken,
    pub failure: Arc<Mutex<Option<String>>>,
    completed: Arc<AtomicBool>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.bridge.pending.lock().remove(&self.id);
        if !self.completed.load(Ordering::Acquire) {
            self.bridge.abort_engine(&self.id);
        }
    }
}

#[pyclass]
struct OutputCallback {
    id: String,
    pending: Requests,
}

#[pymethods]
impl OutputCallback {
    #[pyo3(signature = (token_id=None, finished=false, finish_reason="", logprob=None, stop_token_id=None, error=None))]
    fn __call__(
        &self,
        token_id: Option<u32>,
        finished: bool,
        finish_reason: &str,
        logprob: Option<f64>,
        stop_token_id: Option<u32>,
        error: Option<String>,
    ) -> bool {
        let request = self.pending.lock().get(&self.id).cloned();
        let Some(request) = request else { return false };
        let terminal = finished || error.is_some();
        let output = Output {
            token_id,
            finished,
            finish_reason: finish_reason.to_owned(),
            logprob,
            stop_token_id,
            error,
        };
        if request.sender.try_send(output).is_err() {
            *request.failure.lock() =
                Some("response consumer exceeded the bounded output queue".into());
            request.cancelled.cancel();
            return false;
        }
        if terminal {
            request.completed.store(true, Ordering::Release);
        }
        true
    }
}

#[pyclass]
struct StatusCallback {
    pending: Requests,
    alive: Arc<AtomicBool>,
    active: Arc<AtomicU32>,
    failed: CancellationToken,
}

#[pymethods]
impl StatusCallback {
    #[pyo3(signature = (running_requests, error=None))]
    fn __call__(&self, running_requests: u32, error: Option<String>) {
        self.active.store(running_requests, Ordering::Release);
        if let Some(error) = error {
            self.alive.store(false, Ordering::Release);
            for request in self.pending.lock().values() {
                *request.failure.lock() = Some(error.clone());
                request.cancelled.cancel();
            }
            self.failed.cancel();
        }
    }
}

impl Bridge {
    pub fn new(
        py: Python<'_>,
        runtime: Py<PyAny>,
        ready: Ready,
        limit: usize,
    ) -> PyResult<Arc<Self>> {
        let bridge = Arc::new(Self {
            ready,
            failed: CancellationToken::new(),
            runtime,
            pending: Arc::default(),
            next_id: AtomicU64::new(1),
            alive: Arc::new(AtomicBool::new(true)),
            engine_active: Arc::new(AtomicU32::new(0)),
            limit,
        });
        let callback = Py::new(
            py,
            StatusCallback {
                pending: bridge.pending.clone(),
                alive: bridge.alive.clone(),
                active: bridge.engine_active.clone(),
                failed: bridge.failed.clone(),
            },
        )?;
        bridge
            .runtime
            .call_method1(py, "set_status_callback", (callback,))?;
        Ok(bridge)
    }

    #[allow(clippy::result_large_err)]
    pub fn submit(
        self: &Arc<Self>,
        wire_id: &str,
        command: Generate,
    ) -> Result<Session, tonic::Status> {
        if !self.alive() {
            return Err(tonic::Status::unavailable("engine is not alive"));
        }
        let mut pending = self.pending.lock();
        if pending.len() >= self.limit {
            return Err(tonic::Status::resource_exhausted(
                "engine request capacity reached",
            ));
        }
        if pending.values().any(|request| request.wire_id == wire_id) {
            return Err(tonic::Status::already_exists(
                "request_id is already active",
            ));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let (sender, receiver) = mpsc::channel(128);
        let request = Pending {
            wire_id: wire_id.to_owned(),
            sender,
            cancelled: CancellationToken::new(),
            completed: Arc::new(AtomicBool::new(false)),
            failure: Arc::default(),
        };
        let session = Session {
            bridge: self.clone(),
            id: id.clone(),
            receiver,
            cancelled: request.cancelled.clone(),
            failure: request.failure.clone(),
            completed: request.completed.clone(),
        };
        pending.insert(id.clone(), request);
        // Python callbacks also lock this map; never hold it while acquiring the GIL.
        drop(pending);
        // GIL acquisition can wait behind an engine step; let Tokio hand off this worker.
        tokio::task::block_in_place(|| {
            Python::with_gil(|py| -> Result<(), tonic::Status> {
                let result = (|| -> PyResult<()> {
                    let sampling = PyDict::new(py);
                    sampling.set_item("temperature", command.temperature)?;
                    sampling.set_item("top_p", command.top_p)?;
                    sampling.set_item("top_k", command.top_k)?;
                    sampling.set_item("max_tokens", command.max_tokens)?;
                    sampling.set_item("ignore_eos", command.ignore_eos)?;
                    sampling.set_item("seq_seed", command.seed)?;
                    let callback = Py::new(
                        py,
                        OutputCallback {
                            id: id.clone(),
                            pending: self.pending.clone(),
                        },
                    )?;
                    self.runtime.call_method1(
                        py,
                        "submit",
                        (
                            id,
                            command.token_ids,
                            command.stop_token_ids,
                            sampling,
                            callback,
                        ),
                    )?;
                    Ok(())
                })();
                result.map_err(|error| {
                    if error.is_instance_of::<PyOverflowError>(py) {
                        tonic::Status::resource_exhausted(error.to_string())
                    } else {
                        tonic::Status::unavailable(error.to_string())
                    }
                })
            })
        })?;
        Ok(session)
    }

    fn abort_engine(&self, id: &str) {
        let result = tokio::task::block_in_place(|| {
            Python::with_gil(|py| self.runtime.call_method1(py, "abort", (id,)))
        });
        if let Err(error) = result {
            tracing::error!(%error, "failed to signal engine cancellation");
            self.alive.store(false, Ordering::Release);
            self.failed.cancel();
        }
    }

    pub fn abort(&self, wire_id: Option<&str>) -> bool {
        let ids: Vec<_> = self
            .pending
            .lock()
            .iter()
            .filter_map(|(id, request)| {
                if wire_id.is_none_or(|wire_id| request.wire_id == wire_id)
                    && !request.completed.load(Ordering::Acquire)
                    && !request.cancelled.is_cancelled()
                {
                    request.cancelled.cancel();
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        for id in &ids {
            self.abort_engine(id);
        }
        !ids.is_empty()
    }

    pub fn active(&self) -> u32 {
        self.engine_active.load(Ordering::Acquire)
    }
    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
}

pub fn python_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
