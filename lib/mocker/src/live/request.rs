// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use tokio::sync::{mpsc, watch};
use uuid::Uuid;

use crate::common::protocols::OutputSignal;

use super::handoff::DestinationCancellation;

#[derive(Default)]
pub(super) struct RequestRoutes {
    pub(super) by_client: DashMap<Uuid, Arc<RequestRoute>>,
    pub(super) by_scheduler: DashMap<Uuid, Arc<RequestRoute>>,
}

pub(super) type Routes = Arc<RequestRoutes>;

pub(crate) struct ObservedOutput {
    pub(crate) event: OutputSignal,
    pub(crate) observed_at: tokio::time::Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestState {
    Submitting,
    Active,
    Cancelling,
    Closed,
}

#[derive(Clone)]
pub(super) enum RequestCancellation {
    Request,
    Destination(DestinationCancellation),
}

#[derive(Clone)]
struct RequestLifecycle {
    state: RequestState,
    cancellation: RequestCancellation,
    stream_abandoned: bool,
    terminal_seen: bool,
}

pub(super) struct RequestRoute {
    pub(super) client_id: Uuid,
    pub(super) scheduler_id: Uuid,
    output_tx: Mutex<Option<mpsc::Sender<ObservedOutput>>>,
    lifecycle_tx: watch::Sender<RequestLifecycle>,
    pub(super) cancel_lock: tokio::sync::Mutex<()>,
}

impl RequestRoute {
    pub(super) fn new(
        client_id: Uuid,
        scheduler_id: Uuid,
        output_tx: mpsc::Sender<ObservedOutput>,
    ) -> Self {
        let (lifecycle_tx, _) = watch::channel(RequestLifecycle {
            state: RequestState::Submitting,
            cancellation: RequestCancellation::Request,
            stream_abandoned: false,
            terminal_seen: false,
        });
        Self {
            client_id,
            scheduler_id,
            output_tx: Mutex::new(Some(output_tx)),
            lifecycle_tx,
            cancel_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub(super) fn activate(&self, cancellation: RequestCancellation) {
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state != RequestState::Submitting {
                return false;
            }
            lifecycle.state = RequestState::Active;
            lifecycle.cancellation = cancellation;
            true
        });
    }

    pub(super) fn abandon_stream(&self) -> bool {
        self.close_output();
        let mut abandoned = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.stream_abandoned {
                return false;
            }
            lifecycle.stream_abandoned = true;
            abandoned = true;
            true
        });
        abandoned
    }

    pub(super) async fn wait_for_admission(&self) -> bool {
        let mut lifecycle_rx = self.lifecycle_tx.subscribe();
        loop {
            match lifecycle_rx.borrow_and_update().state {
                RequestState::Submitting | RequestState::Cancelling => {}
                RequestState::Active => return true,
                RequestState::Closed => return false,
            }
            if lifecycle_rx.changed().await.is_err() {
                return false;
            }
        }
    }

    pub(super) fn begin_cancellation(&self) -> Option<RequestCancellation> {
        let mut cancellation = None;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state == RequestState::Active {
                lifecycle.state = RequestState::Cancelling;
                cancellation = Some(lifecycle.cancellation.clone());
                return true;
            }
            false
        });
        cancellation
    }

    pub(super) fn finish_cancellation(&self, result: &anyhow::Result<bool>) -> bool {
        let mut remove = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state != RequestState::Cancelling {
                return false;
            }
            remove = match result {
                Ok(true) => true,
                Ok(false) => lifecycle.stream_abandoned || lifecycle.terminal_seen,
                Err(_) => lifecycle.terminal_seen,
            };
            lifecycle.state = if remove {
                RequestState::Closed
            } else {
                RequestState::Active
            };
            true
        });
        if remove {
            self.close_output();
        }
        remove
    }

    pub(super) fn send_output(&self, output: ObservedOutput) -> OutputDelivery {
        let output_tx = self.output_tx.lock().unwrap();
        let Some(output_tx) = output_tx.as_ref() else {
            return OutputDelivery::Closed;
        };
        match output_tx.try_send(output) {
            Ok(()) => OutputDelivery::Delivered,
            Err(mpsc::error::TrySendError::Full(_)) => OutputDelivery::Full,
            Err(mpsc::error::TrySendError::Closed(_)) => OutputDelivery::Closed,
        }
    }

    /// Record a terminal signal and return whether the route can be removed.
    /// An in-flight cancellation retains it until the scheduler acknowledges
    /// cleanup; its scheduler ID is never reused by a replacement request.
    pub(super) fn observe_terminal(&self) -> bool {
        self.close_output();
        let mut remove = false;
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            lifecycle.terminal_seen = true;
            if lifecycle.state != RequestState::Cancelling {
                lifecycle.state = RequestState::Closed;
                remove = true;
            }
            true
        });
        remove
    }

    pub(super) fn shutdown(&self) {
        self.close_output();
        self.lifecycle_tx.send_if_modified(|lifecycle| {
            if lifecycle.state == RequestState::Closed {
                return false;
            }
            lifecycle.state = RequestState::Closed;
            true
        });
    }

    fn close_output(&self) {
        self.output_tx.lock().unwrap().take();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OutputDelivery {
    Delivered,
    Full,
    Closed,
}

pub(super) fn remove_route(routes: &RequestRoutes, route: &Arc<RequestRoute>) -> bool {
    let removed = routes
        .by_client
        .remove_if(&route.client_id, |_, current| Arc::ptr_eq(current, route))
        .is_some();
    routes
        .by_scheduler
        .remove_if(&route.scheduler_id, |_, current| {
            Arc::ptr_eq(current, route)
        });
    removed
}

pub(super) fn route_is_registered(routes: &RequestRoutes, route: &Arc<RequestRoute>) -> bool {
    routes
        .by_client
        .get(&route.client_id)
        .is_some_and(|current| Arc::ptr_eq(current.value(), route))
        && routes
            .by_scheduler
            .get(&route.scheduler_id)
            .is_some_and(|current| Arc::ptr_eq(current.value(), route))
}

pub(super) fn shutdown_routes(routes: &RequestRoutes) {
    let active_routes = routes
        .by_client
        .iter()
        .map(|entry| Arc::clone(entry.value()))
        .collect::<Vec<_>>();
    for route in active_routes {
        route.shutdown();
    }
    routes.by_client.clear();
    routes.by_scheduler.clear();
}
