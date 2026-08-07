// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-owned drain/resume lifecycle control.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::engine_routes::{EngineRouteCallback, EngineRouteRegistry};
use parking_lot::RwLock;
use serde::Serialize;
use tokio::sync::{Mutex, Notify};

use crate::disagg::DisaggregationMode;
use crate::worker::EngineKind;

const QUIESCENCE_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkerLifecycleState {
    Serving = 0,
    Draining = 1,
    Drained = 2,
    Stopping = 3,
}

impl WorkerLifecycleState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Serving,
            1 => Self::Draining,
            2 => Self::Drained,
            3 => Self::Stopping,
            _ => unreachable!("invalid worker lifecycle state {value}"),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum KvTransferState {
    NotApplicable = 0,
    Unknown = 1,
    Pending = 2,
    Complete = 3,
}

impl KvTransferState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::NotApplicable,
            1 => Self::Unknown,
            2 => Self::Pending,
            3 => Self::Complete,
            _ => unreachable!("invalid KV transfer state {value}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct WorkerLifecycleStatus {
    state: WorkerLifecycleState,
    inflight_requests: u64,
    kv_transfers: KvTransferState,
    safe_to_delete: bool,
    discovery_registered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

/// Shared request-admission and in-flight tracker.
///
/// The double-check around `fetch_add` closes the drain/admission race: a
/// request is either rejected after draining starts or counted so the drain
/// monitor waits for its response stream to be dropped.
#[derive(Debug)]
pub(crate) struct RequestTracker {
    accepting: AtomicBool,
    inflight: AtomicU64,
    changed: Notify,
}

impl RequestTracker {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            accepting: AtomicBool::new(true),
            inflight: AtomicU64::new(0),
            changed: Notify::new(),
        })
    }

    pub(crate) fn try_acquire(self: &Arc<Self>) -> Result<RequestGuard> {
        if !self.accepting.load(Ordering::Acquire) {
            bail!("worker is not accepting new requests");
        }

        self.inflight.fetch_add(1, Ordering::AcqRel);
        if !self.accepting.load(Ordering::Acquire) {
            self.release();
            bail!("worker started draining before request admission completed");
        }

        Ok(RequestGuard {
            tracker: Arc::clone(self),
        })
    }

    pub(crate) fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::Release);
        self.changed.notify_waiters();
    }

    fn start_accepting(&self) {
        self.accepting.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }

    pub(crate) fn inflight(&self) -> u64 {
        self.inflight.load(Ordering::Acquire)
    }

    fn release(&self) {
        let previous = self.inflight.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "request tracker underflow");
        if previous == 1 {
            self.changed.notify_waiters();
        }
    }
}

pub(crate) struct RequestGuard {
    tracker: Arc<RequestTracker>,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.tracker.release();
    }
}

/// Coordinates the Admin API, request admission, discovery, and SIGTERM.
pub(crate) struct WorkerLifecycleController {
    state: AtomicU8,
    kv_transfers: AtomicU8,
    discovery_registered: AtomicBool,
    last_error: RwLock<Option<String>>,
    operation_lock: Mutex<()>,
    drain_generation: AtomicU64,
    tracker: Arc<RequestTracker>,
    endpoint: Endpoint,
    engine: EngineKind,
    mode: DisaggregationMode,
}

impl WorkerLifecycleController {
    pub(crate) fn new(
        tracker: Arc<RequestTracker>,
        endpoint: Endpoint,
        engine: EngineKind,
        mode: DisaggregationMode,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(WorkerLifecycleState::Serving as u8),
            kv_transfers: AtomicU8::new(Self::initial_kv_state(mode) as u8),
            discovery_registered: AtomicBool::new(true),
            last_error: RwLock::new(None),
            operation_lock: Mutex::new(()),
            drain_generation: AtomicU64::new(0),
            tracker,
            endpoint,
            engine,
            mode,
        })
    }

    fn initial_kv_state(mode: DisaggregationMode) -> KvTransferState {
        if mode.is_prefill() {
            KvTransferState::Unknown
        } else {
            KvTransferState::NotApplicable
        }
    }

    fn state(&self) -> WorkerLifecycleState {
        WorkerLifecycleState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub(crate) fn status(&self) -> WorkerLifecycleStatus {
        let state = self.state();
        WorkerLifecycleStatus {
            state,
            inflight_requests: self.tracker.inflight(),
            kv_transfers: KvTransferState::from_u8(self.kv_transfers.load(Ordering::Acquire)),
            safe_to_delete: state == WorkerLifecycleState::Drained,
            discovery_registered: self.discovery_registered.load(Ordering::Acquire),
            last_error: self.last_error.read().clone(),
        }
    }

    pub(crate) fn register_admin_routes(self: &Arc<Self>, registry: &EngineRouteRegistry) {
        registry.register(
            "drain",
            lifecycle_callback(Arc::clone(self), LifecycleAction::Drain),
        );
        registry.register(
            "resume",
            lifecycle_callback(Arc::clone(self), LifecycleAction::Resume),
        );
        registry.register(
            "status",
            lifecycle_callback(Arc::clone(self), LifecycleAction::Status),
        );
        tracing::info!("registered worker Admin API routes under /engine");
    }

    pub(crate) async fn drain(self: &Arc<Self>) -> Result<WorkerLifecycleStatus> {
        let operation = self.operation_lock.lock().await;
        let generation = match self.state() {
            WorkerLifecycleState::Serving => {
                self.last_error.write().take();
                self.tracker.stop_accepting();
                self.kv_transfers
                    .store(Self::initial_kv_state(self.mode) as u8, Ordering::Release);
                self.state
                    .store(WorkerLifecycleState::Draining as u8, Ordering::Release);
                Some(self.drain_generation.fetch_add(1, Ordering::AcqRel) + 1)
            }
            WorkerLifecycleState::Draining => None,
            WorkerLifecycleState::Drained => return Ok(self.status()),
            WorkerLifecycleState::Stopping => bail!("worker shutdown is already in progress"),
        };

        if self.discovery_registered.load(Ordering::Acquire) {
            if let Err(error) = self.endpoint.unregister_endpoint_instance().await {
                let message = format!("failed to unregister worker from discovery: {error}");
                *self.last_error.write() = Some(message.clone());
                self.drain_generation.fetch_add(1, Ordering::AcqRel);
                self.state
                    .store(WorkerLifecycleState::Serving as u8, Ordering::Release);
                self.tracker.start_accepting();
                return Err(error).context("failed to unregister worker from discovery");
            }
            self.discovery_registered.store(false, Ordering::Release);
        }

        drop(operation);
        if let Some(generation) = generation {
            let controller = Arc::clone(self);
            tokio::spawn(async move { controller.monitor_drain(generation).await });
        }
        Ok(self.status())
    }

    pub(crate) async fn resume(self: &Arc<Self>) -> Result<WorkerLifecycleStatus> {
        let _operation = self.operation_lock.lock().await;
        let previous_state = match self.state() {
            WorkerLifecycleState::Serving => return Ok(self.status()),
            WorkerLifecycleState::Stopping => bail!("worker shutdown is already in progress"),
            state @ (WorkerLifecycleState::Draining | WorkerLifecycleState::Drained) => state,
        };

        let generation = self.drain_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.state
            .store(WorkerLifecycleState::Draining as u8, Ordering::Release);

        if !self.discovery_registered.load(Ordering::Acquire) {
            if let Err(error) = self.endpoint.register_endpoint_instance().await {
                let message = format!("failed to re-register worker in discovery: {error}");
                *self.last_error.write() = Some(message);
                self.state.store(previous_state as u8, Ordering::Release);
                if previous_state == WorkerLifecycleState::Draining {
                    let controller = Arc::clone(self);
                    tokio::spawn(async move { controller.monitor_drain(generation).await });
                }
                return Err(error).context("failed to re-register worker in discovery");
            }
            self.discovery_registered.store(true, Ordering::Release);
        }

        self.last_error.write().take();
        self.kv_transfers
            .store(Self::initial_kv_state(self.mode) as u8, Ordering::Release);
        self.state
            .store(WorkerLifecycleState::Serving as u8, Ordering::Release);
        self.tracker.start_accepting();
        Ok(self.status())
    }

    /// Move into the irreversible SIGTERM path and ensure discovery is down.
    pub(crate) async fn begin_shutdown(&self) {
        let _operation = self.operation_lock.lock().await;
        self.tracker.stop_accepting();
        self.drain_generation.fetch_add(1, Ordering::AcqRel);
        self.state
            .store(WorkerLifecycleState::Stopping as u8, Ordering::Release);

        if self.discovery_registered.load(Ordering::Acquire) {
            if let Err(error) = self.endpoint.unregister_endpoint_instance().await {
                tracing::warn!(%error, "discovery unregister failed during shutdown");
                *self.last_error.write() = Some(error.to_string());
            } else {
                self.discovery_registered.store(false, Ordering::Release);
            }
        }
    }

    async fn monitor_drain(self: Arc<Self>, generation: u64) {
        loop {
            if self.state() != WorkerLifecycleState::Draining
                || self.drain_generation.load(Ordering::Acquire) != generation
            {
                return;
            }

            if self.tracker.inflight() == 0 {
                let kv_complete = if self.mode.is_prefill() {
                    match self.engine.is_quiescent().await {
                        Ok(Some(true)) => {
                            self.kv_transfers
                                .store(KvTransferState::Complete as u8, Ordering::Release);
                            true
                        }
                        Ok(Some(false)) => {
                            self.kv_transfers
                                .store(KvTransferState::Pending as u8, Ordering::Release);
                            false
                        }
                        Ok(None) => {
                            self.kv_transfers
                                .store(KvTransferState::Unknown as u8, Ordering::Release);
                            false
                        }
                        Err(error) => {
                            self.kv_transfers
                                .store(KvTransferState::Unknown as u8, Ordering::Release);
                            *self.last_error.write() = Some(error.to_string());
                            false
                        }
                    }
                } else {
                    true
                };

                if kv_complete
                    && self.drain_generation.load(Ordering::Acquire) == generation
                    && self
                        .state
                        .compare_exchange(
                            WorkerLifecycleState::Draining as u8,
                            WorkerLifecycleState::Drained as u8,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                {
                    tracing::info!("worker drained and is safe to delete");
                    return;
                }
            }

            tokio::select! {
                _ = self.tracker.changed.notified() => {}
                _ = tokio::time::sleep(QUIESCENCE_POLL_INTERVAL) => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
enum LifecycleAction {
    Drain,
    Resume,
    Status,
}

fn lifecycle_callback(
    controller: Arc<WorkerLifecycleController>,
    action: LifecycleAction,
) -> EngineRouteCallback {
    Arc::new(move |body| {
        let controller = Arc::clone(&controller);
        Box::pin(async move {
            if !body.is_object() {
                bail!("worker Admin API request body must be a JSON object");
            }
            let status = match action {
                LifecycleAction::Drain => controller.drain().await?,
                LifecycleAction::Resume => controller.resume().await?,
                LifecycleAction::Status => controller.status(),
            };
            serde_json::to_value(status).context("serialize worker lifecycle status")
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_tracker_rejects_after_drain_starts() {
        let tracker = RequestTracker::new();
        let guard = tracker.try_acquire().unwrap();
        assert_eq!(tracker.inflight(), 1);

        tracker.stop_accepting();
        assert!(tracker.try_acquire().is_err());
        assert_eq!(tracker.inflight(), 1);

        drop(guard);
        assert_eq!(tracker.inflight(), 0);
    }

    #[test]
    fn request_tracker_accepts_again_after_resume() {
        let tracker = RequestTracker::new();
        tracker.stop_accepting();
        tracker.start_accepting();
        let guard = tracker.try_acquire().unwrap();
        assert_eq!(tracker.inflight(), 1);
        drop(guard);
    }
}
