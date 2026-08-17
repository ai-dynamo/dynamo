// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex, Weak};

use anyhow::{anyhow, bail};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::handoff::{HandoffId, HandoffTransferTiming};
use crate::scheduler::{SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent};

use super::request::{Routes, shutdown_routes};
use super::{
    LiveEngine, LiveEngineInner, LiveRequestRegistration, PreparedSubmission, send_command,
};

const HANDOFF_EVENT_CAPACITY: usize = 8;

#[derive(Default)]
pub(super) struct HandoffRoutes {
    by_id: DashMap<HandoffId, Arc<HandoffRoute>>,
    last_by_id: DashMap<HandoffId, (Uuid, Weak<HandoffRoute>)>,
}

pub(super) type SharedHandoffRoutes = Arc<HandoffRoutes>;

impl HandoffRoutes {
    pub(super) fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

struct HandoffRoute {
    handoff_id: HandoffId,
    generation: Uuid,
    routes: Weak<HandoffRoutes>,
    event_tx: Mutex<Option<mpsc::Sender<LiveHandoffEvent>>>,
    command_lock: Arc<tokio::sync::Mutex<()>>,
}

impl HandoffRoute {
    fn new(
        handoff_id: HandoffId,
        routes: Weak<HandoffRoutes>,
        event_tx: mpsc::Sender<LiveHandoffEvent>,
    ) -> Self {
        Self {
            handoff_id,
            generation: Uuid::new_v4(),
            routes,
            event_tx: Mutex::new(Some(event_tx)),
            command_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    async fn send(&self, event: LiveHandoffEvent, cancel: &CancellationToken) -> bool {
        let event_tx = self.event_tx.lock().unwrap().as_ref().cloned();
        let Some(event_tx) = event_tx else {
            return false;
        };
        let delivered = tokio::select! {
            biased;
            _ = cancel.cancelled() => false,
            result = event_tx.send(event) => result.is_ok(),
        };
        if !delivered {
            self.shutdown();
        }
        delivered
    }

    fn shutdown(&self) {
        self.event_tx.lock().unwrap().take();
    }

    fn is_latest_generation(&self, routes: &HandoffRoutes) -> bool {
        match routes.last_by_id.get(&self.handoff_id) {
            Some(current) => current.value().0 == self.generation,
            None => false,
        }
    }
}

impl Drop for HandoffRoute {
    fn drop(&mut self) {
        let Some(routes) = self.routes.upgrade() else {
            return;
        };
        routes
            .last_by_id
            .remove_if(&self.handoff_id, |_, (generation, _)| {
                *generation == self.generation
            });
    }
}

/// Scheduler lifecycle events normalized for a registered disaggregated handoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LiveHandoffEvent {
    SourceHeld {
        transfer_timing: HandoffTransferTiming,
    },
    DestinationReserved {
        transferable_prompt_tokens: usize,
    },
}

impl LiveEngine {
    /// Register one disaggregated handoff and its normalized lifecycle stream.
    pub fn register_handoff(
        &self,
        handoff_id: HandoffId,
    ) -> anyhow::Result<(LiveHandoffControl, LiveHandoffEvents)> {
        anyhow::ensure!(
            !self.inner.cancel.is_cancelled(),
            "live Mocker engine is not running"
        );
        let (event_tx, event_rx) = mpsc::channel(HANDOFF_EVENT_CAPACITY);
        let previous_route = self
            .inner
            .handoff_routes
            .last_by_id
            .get(&handoff_id)
            .and_then(|entry| entry.value().1.upgrade());
        let _previous_command = match previous_route {
            Some(route) => Some(route.command_lock.clone().try_lock_owned().map_err(|_| {
                anyhow!("handoff {handoff_id:?} still has a scheduler command in progress")
            })?),
            None => None,
        };
        let route = Arc::new(HandoffRoute::new(
            handoff_id,
            Arc::downgrade(&self.inner.handoff_routes),
            event_tx,
        ));
        match self.inner.handoff_routes.by_id.entry(handoff_id) {
            Entry::Occupied(_) => {
                bail!("handoff {handoff_id:?} already has a lifecycle route")
            }
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&route));
            }
        }
        self.inner
            .handoff_routes
            .last_by_id
            .insert(handoff_id, (route.generation, Arc::downgrade(&route)));
        if self.inner.cancel.is_cancelled() {
            route.shutdown();
            remove_handoff_route(&self.inner.handoff_routes, &route);
            bail!("live Mocker engine is not running");
        }
        Ok((
            LiveHandoffControl {
                engine: Arc::downgrade(&self.inner),
                route: Arc::clone(&route),
            },
            LiveHandoffEvents {
                route,
                routes: Arc::clone(&self.inner.handoff_routes),
                event_rx,
            },
        ))
    }
}

/// Typed scheduler controls for one disaggregated handoff.
#[derive(Clone)]
pub struct LiveHandoffControl {
    engine: Weak<LiveEngineInner>,
    route: Arc<HandoffRoute>,
}

impl LiveHandoffControl {
    pub fn handoff_id(&self) -> HandoffId {
        self.route.handoff_id
    }

    pub async fn submit_prefill(
        &self,
        registration: LiveRequestRegistration,
    ) -> anyhow::Result<()> {
        let command_guard = self.route.command_lock.clone().lock_owned().await;
        let engine = self.engine()?;
        self.ensure_generation(&engine, false)?;
        LiveEngine { inner: engine }
            .submit_prepared(
                registration,
                PreparedSubmission::Source(self.route.handoff_id),
                Some(command_guard),
            )
            .await
    }

    pub async fn reserve_destination(
        &self,
        registration: LiveRequestRegistration,
    ) -> anyhow::Result<()> {
        let command_guard = self.route.command_lock.clone().lock_owned().await;
        let engine = self.engine()?;
        self.ensure_generation(&engine, false)?;
        LiveEngine { inner: engine }
            .submit_prepared(
                registration,
                PreparedSubmission::Destination(DestinationCancellation {
                    route: Arc::clone(&self.route),
                }),
                Some(command_guard),
            )
            .await
    }

    pub async fn release_source(&self) -> anyhow::Result<()> {
        self.send_handoff_command(
            SchedulerCommand::ReleaseSource {
                handoff_id: self.route.handoff_id,
            },
            HandoffCommandResult::AppliedOrNoop,
        )
        .await
    }

    pub async fn cancel_source(&self) -> anyhow::Result<()> {
        self.send_handoff_command(
            SchedulerCommand::CancelSource {
                handoff_id: self.route.handoff_id,
            },
            HandoffCommandResult::AppliedOrNoop,
        )
        .await
    }

    pub async fn activate_destination(&self) -> anyhow::Result<()> {
        self.send_handoff_command(
            SchedulerCommand::ActivateDestination {
                handoff_id: self.route.handoff_id,
            },
            HandoffCommandResult::Applied,
        )
        .await
    }

    pub async fn cancel_destination(&self) -> anyhow::Result<()> {
        self.send_handoff_command(
            SchedulerCommand::CancelDestination {
                handoff_id: self.route.handoff_id,
            },
            HandoffCommandResult::AppliedOrNoop,
        )
        .await
    }

    async fn send_handoff_command(
        &self,
        command: SchedulerCommand,
        expected: HandoffCommandResult,
    ) -> anyhow::Result<()> {
        let _command_guard = self.route.command_lock.clone().lock_owned().await;
        let engine = self.engine()?;
        self.ensure_generation(&engine, true)?;
        anyhow::ensure!(
            !engine.cancel.is_cancelled(),
            "live Mocker engine is not running"
        );
        let result = send_command(&engine.command_tx, command).await?;
        match (expected, result) {
            (HandoffCommandResult::Applied, SchedulerCommandResult::Applied)
            | (
                HandoffCommandResult::AppliedOrNoop,
                SchedulerCommandResult::Applied | SchedulerCommandResult::Noop,
            ) => Ok(()),
            (_, result) => Err(anyhow!(
                "unexpected scheduler handoff result for {:?}: {result:?}",
                self.route.handoff_id
            )),
        }
    }

    fn engine(&self) -> anyhow::Result<Arc<LiveEngineInner>> {
        self.engine
            .upgrade()
            .ok_or_else(|| anyhow!("live Mocker engine no longer exists"))
    }

    fn ensure_generation(
        &self,
        engine: &LiveEngineInner,
        allow_unregistered: bool,
    ) -> anyhow::Result<()> {
        match engine.handoff_routes.by_id.get(&self.route.handoff_id) {
            Some(current) if Arc::ptr_eq(current.value(), &self.route) => Ok(()),
            Some(_) => bail!(
                "handoff {:?} control belongs to an earlier registration",
                self.route.handoff_id
            ),
            None if allow_unregistered
                && self.route.is_latest_generation(&engine.handoff_routes) =>
            {
                Ok(())
            }
            None if allow_unregistered => bail!(
                "handoff {:?} control belongs to an earlier registration",
                self.route.handoff_id
            ),
            None => bail!(
                "handoff {:?} lifecycle route is no longer registered",
                self.route.handoff_id
            ),
        }
    }
}

#[derive(Clone)]
pub(super) struct DestinationCancellation {
    route: Arc<HandoffRoute>,
}

impl DestinationCancellation {
    pub(super) fn handoff_id(&self) -> HandoffId {
        self.route.handoff_id
    }

    pub(super) async fn cancel(
        &self,
        command_tx: &mpsc::Sender<crate::scheduler::SchedulerCommandEnvelope>,
    ) -> anyhow::Result<bool> {
        let _command_guard = self.route.command_lock.clone().lock_owned().await;
        let Some(routes) = self.route.routes.upgrade() else {
            return Ok(false);
        };
        if !self.route.is_latest_generation(&routes) {
            return Ok(false);
        }
        super::cancel_destination(command_tx, self.route.handoff_id).await
    }
}

#[derive(Clone, Copy)]
enum HandoffCommandResult {
    Applied,
    AppliedOrNoop,
}

/// Request-owned stream of normalized handoff lifecycle events.
pub struct LiveHandoffEvents {
    route: Arc<HandoffRoute>,
    routes: SharedHandoffRoutes,
    event_rx: mpsc::Receiver<LiveHandoffEvent>,
}

impl LiveHandoffEvents {
    pub fn handoff_id(&self) -> HandoffId {
        self.route.handoff_id
    }

    pub async fn recv(&mut self) -> Option<LiveHandoffEvent> {
        self.event_rx.recv().await
    }
}

impl Drop for LiveHandoffEvents {
    fn drop(&mut self) {
        self.route.shutdown();
        remove_handoff_route(&self.routes, &self.route);
    }
}

fn remove_handoff_route(routes: &HandoffRoutes, route: &Arc<HandoffRoute>) -> bool {
    routes
        .by_id
        .remove_if(&route.handoff_id, |_, current| Arc::ptr_eq(current, route))
        .is_some()
}

pub(super) async fn run_lifecycle_dispatcher(
    mut lifecycle_rx: mpsc::Receiver<SchedulerLifecycleEvent>,
    routes: SharedHandoffRoutes,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    loop {
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            event = lifecycle_rx.recv() => {
                let Some(event) = event else {
                    if cancel.is_cancelled() {
                        return Ok(());
                    }
                    bail!("live Mocker lifecycle lane closed unexpectedly");
                };
                event
            }
        };
        let (handoff_id, event) = match event {
            SchedulerLifecycleEvent::SourceHeld {
                handoff_id,
                transfer_timing,
                ..
            } => (handoff_id, LiveHandoffEvent::SourceHeld { transfer_timing }),
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens,
                ..
            } => (
                handoff_id,
                LiveHandoffEvent::DestinationReserved {
                    transferable_prompt_tokens,
                },
            ),
        };
        let route = routes
            .by_id
            .get(&handoff_id)
            .map(|entry| Arc::clone(entry.value()));
        if let Some(route) = route
            && !route.send(event, &cancel).await
        {
            remove_handoff_route(&routes, &route);
        }
    }
}

pub(super) async fn supervise_lifecycle_dispatcher(
    dispatcher: tokio::task::JoinHandle<anyhow::Result<()>>,
    routes: Routes,
    handoff_routes: SharedHandoffRoutes,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let result = match dispatcher.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error),
        Err(error) => Err(anyhow!(
            "live Mocker lifecycle dispatcher task failed: {error}"
        )),
    };
    if let Err(error) = &result {
        tracing::error!(%error, "live Mocker lifecycle dispatcher failed");
    } else if !cancel.is_cancelled() {
        tracing::error!("live Mocker lifecycle dispatcher exited unexpectedly");
    }
    cancel.cancel();
    shutdown_routes(&routes);
    shutdown_handoff_routes(&handoff_routes);
    result
}

pub(super) fn shutdown_handoff_routes(routes: &HandoffRoutes) {
    let active_routes = routes
        .by_id
        .iter()
        .map(|entry| Arc::clone(entry.value()))
        .collect::<Vec<_>>();
    for route in active_routes {
        route.shutdown();
    }
    routes.by_id.clear();
}
