// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, ensure};
use bytes::Bytes;
use parking_lot::Mutex;
use prometheus::IntGauge;
use tokio::{
    runtime::Handle,
    sync::{Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError},
    time::Instant,
};
use tokio_util::{
    sync::CancellationToken,
    task::{TaskTracker, task_tracker::TaskTrackerToken},
};
use tracing::Instrument;

use crate::metrics::{
    prometheus_names::clamp_u64_to_i64,
    work_handler_perf::WORK_HANDLER_NETWORK_TRANSIT_SECONDS,
    work_handler_pool::{
        WORK_HANDLER_ENQUEUE_REJECTED_TOTAL, WORK_HANDLER_PERMIT_WAIT_SECONDS,
        WORK_HANDLER_POOL_ACTIVE_TASKS, WORK_HANDLER_POOL_CAPACITY, WORK_HANDLER_QUEUE_CAPACITY,
        WORK_HANDLER_QUEUE_DEPTH,
    },
};

use super::server::InflightGuard;

#[derive(Default)]
pub(in crate::pipeline::network) struct RequestWork {
    gate: Mutex<()>,
    closed: AtomicBool,
    outstanding: AtomicUsize,
    idle: Notify,
    pub(super) cancel: CancellationToken,
    pub(in crate::pipeline::network) tasks: TaskTracker,
    pub(in crate::pipeline::network) finalizers: TaskTracker,
}

impl RequestWork {
    pub(in crate::pipeline::network) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub(in crate::pipeline::network) fn close(&self) {
        let _gate = self.gate.lock();
        self.closed.store(true, Ordering::Release);
        self.tasks.close();
    }

    pub(in crate::pipeline::network) fn cancel(&self) {
        self.cancel.cancel();
    }

    pub(super) fn reserve(self: &Arc<Self>) -> Option<WorkReservation> {
        let _gate = self.gate.lock();
        if self.is_closed() {
            return None;
        }
        let token = self.tasks.token();
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        Some(WorkReservation {
            work: self.clone(),
            _token: token,
        })
    }

    pub(in crate::pipeline::network) fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }

    pub(in crate::pipeline::network) async fn wait_idle(&self) {
        loop {
            let idle = self.idle.notified();
            if self.outstanding() == 0 {
                return;
            }
            idle.await;
        }
    }
}

pub(super) struct WorkReservation {
    work: Arc<RequestWork>,
    _token: TaskTrackerToken,
}

impl Drop for WorkReservation {
    fn drop(&mut self) {
        if self.work.outstanding.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.work.idle.notify_waiters();
        }
    }
}

pub(super) struct Admission {
    admitted: Arc<Semaphore>,
    running: Arc<Semaphore>,
    capacity: usize,
    runtime: Handle,
}

enum RunningPermit {
    Ready(OwnedSemaphorePermit, GaugeGuard),
    Waiting(Arc<Semaphore>, GaugeGuard, Instant),
}

impl Admission {
    pub(super) fn new(pool: usize, queue: usize, runtime: Handle) -> Result<Self> {
        ensure!(pool > 0, "DYN_TCP_WORKER_POOL_SIZE must be positive");
        let capacity = pool
            .checked_add(queue)
            .filter(|capacity| *capacity <= Semaphore::MAX_PERMITS)
            .ok_or_else(|| {
                anyhow::anyhow!("Velo worker pool and queue exceed semaphore capacity")
            })?;
        WORK_HANDLER_POOL_CAPACITY.set(clamp_u64_to_i64(pool as u64));
        WORK_HANDLER_QUEUE_CAPACITY.set(clamp_u64_to_i64(queue as u64));
        Ok(Self {
            admitted: Arc::new(Semaphore::new(capacity)),
            running: Arc::new(Semaphore::new(pool)),
            capacity,
            runtime,
        })
    }

    pub(super) fn close(&self) {
        self.admitted.close();
    }

    pub(super) fn is_closed(&self) -> bool {
        self.admitted.is_closed()
    }

    pub(super) async fn wait_idle(&self) {
        while self.admitted.available_permits() != self.capacity {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    pub(super) fn try_admit(
        &self,
        guard: InflightGuard,
        payload: Bytes,
        headers: HashMap<String, String>,
    ) -> Bytes {
        let admitted = match self.admitted.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(error) => {
                WORK_HANDLER_ENQUEUE_REJECTED_TOTAL.inc();
                return Bytes::from_static(match error {
                    TryAcquireError::Closed => b"Server unavailable: worker pool closed",
                    TryAcquireError::NoPermits => b"Server overloaded: worker at capacity",
                });
            }
        };
        let running = match self.running.clone().try_acquire_owned() {
            Ok(permit) => {
                RunningPermit::Ready(permit, GaugeGuard::new(&WORK_HANDLER_POOL_ACTIVE_TASKS))
            }
            Err(_) => RunningPermit::Waiting(
                self.running.clone(),
                GaugeGuard::new(&WORK_HANDLER_QUEUE_DEPTH),
                Instant::now(),
            ),
        };
        let cancel = guard.endpoint.work.cancel.clone();

        // Acquisition already holds the tracker token. It follows the task through
        // cancellation and unwind, including a delay between admission and spawn.
        self.runtime.spawn(async move {
            let inflight = guard;
            let endpoint = &inflight.endpoint;
            let _admitted = admitted;
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {}
                _ = async {
                    let (_running, _active) = match running {
                        RunningPermit::Ready(permit, active) => (permit, active),
                        RunningPermit::Waiting(pool, queued, since) => {
                            let Ok(permit) = pool.acquire_owned().await else { return };
                            drop(queued);
                            WORK_HANDLER_PERMIT_WAIT_SECONDS.observe(since.elapsed().as_secs_f64());
                            (permit, GaugeGuard::new(&WORK_HANDLER_POOL_ACTIVE_TASKS))
                        }
                    };
                    record_transit(&headers);
                    let span = crate::logging::make_handle_payload_span_from_tcp_headers(
                        &headers,
                        &endpoint.id.component,
                        &endpoint.id.name,
                        &endpoint.id.namespace,
                        endpoint.instance_id,
                    );
                    let request_id = headers.get("request-id").or_else(|| headers.get("x-dynamo-request-id")).cloned();
                    if let Err(error) = endpoint.handler.handle_payload(payload, request_id).instrument(span).await {
                        tracing::warn!(instance_id = endpoint.instance_id, %error, "Velo worker failed to handle request");
                    }
                } => {}
            }
        });
        Bytes::new()
    }
}

struct GaugeGuard(&'static IntGauge);

impl GaugeGuard {
    fn new(gauge: &'static IntGauge) -> Self {
        gauge.inc();
        Self(gauge)
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        self.0.dec();
    }
}

fn record_transit(headers: &HashMap<String, String>) {
    let Some(sent) = headers
        .get("x-frontend-send-ts-ns")
        .and_then(|sent| sent.parse::<u64>().ok())
    else {
        return;
    };
    let received = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    WORK_HANDLER_NETWORK_TRANSIT_SECONDS
        .observe(received.saturating_sub(sent) as f64 / 1_000_000_000.0);
}
