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
    #[cfg(test)]
    output_gate_bypass_tx: watch::Sender<bool>,
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
        #[cfg(test)]
        let (output_gate_bypass_tx, _) = watch::channel(false);
        Self {
            client_id,
            scheduler_id,
            output_tx: Mutex::new(Some(output_tx)),
            lifecycle_tx,
            #[cfg(test)]
            output_gate_bypass_tx,
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
            return OutputDelivery::Closed(output.event);
        };
        match output_tx.try_send(output) {
            Ok(()) => OutputDelivery::Delivered,
            Err(mpsc::error::TrySendError::Full(output)) => OutputDelivery::Full(output.event),
            Err(mpsc::error::TrySendError::Closed(output)) => OutputDelivery::Closed(output.event),
        }
    }

    #[cfg(test)]
    pub(super) fn request_output_gate_bypass(&self) {
        self.output_gate_bypass_tx.send_replace(true);
    }

    #[cfg(test)]
    pub(super) fn output_gate_bypass_requested(&self) -> bool {
        *self.output_gate_bypass_tx.borrow()
    }

    #[cfg(test)]
    pub(super) async fn wait_for_output_gate_bypass(&self) {
        let mut bypass = self.output_gate_bypass_tx.subscribe();
        if *bypass.borrow_and_update() {
            return;
        }
        let _ = bypass.wait_for(|requested| *requested).await;
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

pub(super) enum OutputDelivery {
    Delivered,
    Full(OutputSignal),
    Closed(OutputSignal),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(uuid: Uuid, token_id: u32) -> OutputSignal {
        OutputSignal {
            uuid,
            token_id: Some(token_id),
            completed: false,
            rejected: false,
            handoff_delay_ms: None,
            cached_tokens: None,
        }
    }

    fn observed(event: OutputSignal) -> ObservedOutput {
        ObservedOutput {
            event,
            observed_at: tokio::time::Instant::now(),
        }
    }

    #[test]
    fn failed_output_delivery_returns_the_original_signal() {
        let client_id = Uuid::from_u128(1);
        let scheduler_id = Uuid::from_u128(2);
        let (output_tx, mut output_rx) = mpsc::channel(1);
        let route = RequestRoute::new(client_id, scheduler_id, output_tx);

        assert!(matches!(
            route.send_output(observed(signal(client_id, 10))),
            OutputDelivery::Delivered
        ));
        let full = match route.send_output(observed(signal(client_id, 20))) {
            OutputDelivery::Full(signal) => signal,
            _ => panic!("expected full output delivery"),
        };
        assert_eq!(full.uuid, client_id);
        assert_eq!(full.token_id, Some(20));

        assert_eq!(output_rx.try_recv().unwrap().event.token_id, Some(10));
        drop(output_rx);
        let closed = match route.send_output(observed(signal(client_id, 30))) {
            OutputDelivery::Closed(signal) => signal,
            _ => panic!("expected closed output delivery"),
        };
        assert_eq!(closed.uuid, client_id);
        assert_eq!(closed.token_id, Some(30));
    }
}
