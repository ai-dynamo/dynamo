// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use dynamo_backend_common::{BackendError, EngineConfig, EngineRecovery, ErrorType};
use dynamo_llm::kv_router::publisher::{KvStreamCommand, KvStreamStatus, ZmqRecoveryConfig};
use futures::StreamExt;
use futures::future::try_join_all;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::timeout_at;
use tonic_health_v14::pb::health_check_response::ServingStatus;

#[derive(Default)]
pub(super) struct Lifecycle {
    pub(super) is_recovering: AtomicBool,
    current: Mutex<Option<Arc<Connection>>>,
    deadline: Mutex<Option<Instant>>,
}

pub(super) struct Connection {
    pub(super) client: Arc<VllmClient>,
    model: DiscoveredModel,
    pub(super) cancel: CancellationToken,
    health: watch::Receiver<EngineHealth>,
    ranks: Mutex<Vec<RankRecovery>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EngineHealth {
    Serving,
    NotServing,
    /// The watch stream failed or closed; the engine may have been replaced.
    Lost,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Clone)]
struct RankRecovery {
    rank: u32,
    status: watch::Receiver<KvStreamStatus>,
    commands: mpsc::Sender<KvStreamCommand>,
}

impl VllmSidecarEngine {
    pub(super) fn started_state(&self) -> Result<Arc<Connection>, DynamoError> {
        self.state
            .current
            .lock()
            .clone()
            .ok_or_else(|| client::engine_shutdown("vLLM sidecar is not started"))
    }

    /// Starts the startup budget before metadata bootstrap; registration clears it.
    pub(super) fn start_deadline(&self, deadline: Instant) {
        *self.state.deadline.lock() = Some(deadline);
    }

    fn recovery_deadline(&self) -> Result<Instant, DynamoError> {
        let mut deadline = self.state.deadline.lock();
        if let Some(deadline) = *deadline {
            Ok(deadline)
        } else {
            let value = client::startup_deadline(self.transport.startup_deadline)?;
            *deadline = Some(value);
            Ok(value)
        }
    }

    async fn connect_generation(&self, deadline: Instant) -> Result<Arc<Connection>, DynamoError> {
        let mut last_error = None;
        let connected = timeout_at(deadline, async {
            loop {
                match self.connect_once(deadline).await {
                    Ok(connection) => return Ok(connection),
                    Err(error)
                        if matches!(
                            error.error_type(),
                            ErrorType::Backend(
                                BackendError::EngineShutdown
                                    | BackendError::CannotConnect
                                    | BackendError::ConnectionTimeout
                            )
                        ) =>
                    {
                        tracing::debug!(%error, "Waiting for healthy vLLM services");
                        last_error = Some(error.to_string());
                        tokio::time::sleep(self.transport.retry_interval).await;
                    }
                    Err(error) => return Err(error),
                }
            }
        })
        .await;
        connected.map_err(|_| {
            client::engine_shutdown(format!(
                "vLLM services did not recover before the configured startup deadline: {}",
                last_error
                    .as_deref()
                    .unwrap_or("no connection attempt finished")
            ))
        })?
    }

    async fn connect_once(&self, deadline: Instant) -> Result<Arc<Connection>, DynamoError> {
        let client =
            Arc::new(VllmClient::connect(&self.endpoint, self.transport, deadline, false).await?);
        client
            .wait_for_services(
                &[CONTROL_SERVICE, INFERENCE_SERVICE],
                deadline,
                self.transport.retry_interval,
            )
            .await?;
        let (model, server) = client.discover(deadline).await?;
        if server.instance_id.trim().is_empty() {
            return Err(client::protocol_error(
                "vLLM recovery requires GetServerInfo.instance_id",
            ));
        }
        let model = DiscoveredModel::from_proto(model, server)?;
        self.model.ensure_startup_compatible(&model)?;
        let cancel = self.cancel.child_token();
        let health = monitor_health(
            client.clone(),
            cancel.clone(),
            self.transport.connect_attempt_timeout,
        );
        let connection = Arc::new(Connection {
            client,
            model,
            cancel,
            health,
            ranks: Mutex::new(Vec::new()),
        });
        timeout_at(deadline, serving(connection.health.clone()))
            .await
            .map_err(|_| {
                client::engine_shutdown("vLLM Health.Watch startup deadline exceeded")
            })??;
        let (_, observed) = connection.client.discover(deadline).await?;
        if observed.instance_id != connection.model.instance_id()
            || *connection.health.borrow() != EngineHealth::Serving
        {
            return Err(client::engine_shutdown(
                "vLLM engine changed while establishing Health.Watch",
            ));
        }
        Ok(connection)
    }

    pub(super) async fn initialize(&self) -> Result<EngineConfig, DynamoError> {
        if self.state.current.lock().is_some() {
            return Err(client::engine_shutdown("vLLM sidecar has already started"));
        }
        let deadline = self.recovery_deadline()?;
        let state = self.connect_generation(deadline).await?;
        let config = self.engine_config(&state.model)?;
        self.routing_image_token_id
            .set(resolve_routing_image_token_id(&state.model, deadline).await)
            .map_err(|_| client::engine_shutdown("vLLM sidecar has already started"))?;
        tracing::info!(instance_id = %state.model.instance_id(), connections = state.client.connection_count(), "Connected to vLLM engine");
        *self.state.current.lock() = Some(state);
        Ok(config)
    }

    pub(super) async fn recovery_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        if self.mode.is_encode() {
            return Ok(Vec::new());
        }
        let state = self.started_state()?;
        let deadline = self.recovery_deadline()?;
        let reported = timeout_at(deadline, state.client.kv_event_sources())
            .await
            .map_err(|_| {
                client::engine_shutdown("GetKvEventSources recovery deadline exceeded")
            })??;
        if reported.is_empty() {
            return Ok(Vec::new());
        }
        let expected = state.model.data_parallel_range();
        let mut ranks = HashSet::new();
        let mut sources = Vec::new();
        let mut recovery = Vec::new();
        for source in reported {
            let rank = source
                .data_parallel_rank
                .ok_or_else(|| client::protocol_error("KV source has no data_parallel_rank"))?;
            if !expected.contains(&rank) || !ranks.insert(rank) {
                return Err(client::protocol_error(format!(
                    "KV source rank {rank} is duplicate or outside {expected:?}"
                )));
            }
            if source.transport != "zmq"
                || source.encoding != "msgpack"
                || source.schema_version != 1
            {
                return Err(client::protocol_error(format!(
                    "unsupported KV event source: transport={}, encoding={}, schema={}",
                    source.transport, source.encoding, source.schema_version
                )));
            }
            if source.endpoint.trim().is_empty()
                || source.replay_endpoint.trim().is_empty()
                || source.buffer_steps == 0
            {
                return Err(client::protocol_error(
                    "vLLM KV recovery requires a live endpoint, replay_endpoint and positive buffer_steps",
                ));
            }
            let (status_tx, status) = watch::channel(KvStreamStatus::Recovering);
            let (commands, command_rx) = mpsc::channel(8);
            recovery.push(RankRecovery {
                rank,
                status,
                commands,
            });
            sources.push(KvEventSource::Zmq {
                heartbeat_timeout: None,
                endpoint: zmq_connect_endpoint(&source.endpoint, &self.endpoint),
                topic: source.topic,
                dp_rank: rank,
                image_token_id: self.routing_image_token_id.get().copied().flatten(),
                recovery: Some(ZmqRecoveryConfig {
                    replay_endpoint: zmq_connect_endpoint(&source.replay_endpoint, &self.endpoint),
                    dp_rank: rank,
                    deadline,
                    status: status_tx,
                    commands: command_rx,
                }),
            });
        }
        if ranks.len() != expected.len() {
            return Err(client::protocol_error(format!(
                "KV routing requires sources for every local rank in {expected:?}; received {}",
                ranks.len()
            )));
        }
        *state.ranks.lock() = recovery;
        Ok(sources)
    }

    pub(super) async fn wait_for_recovery(&self) -> Result<(), DynamoError> {
        let state = self.started_state()?;
        let deadline = self.recovery_deadline()?;
        let ranks = state.ranks.lock().clone();
        let result = timeout_at(deadline, async {
            let gate = try_join_all(ranks.into_iter().map(Self::wait_for_rank));
            let gate = tokio::select! {
                biased;
                error = unhealthy(state.health.clone()) => return Err(error),
                result = gate => result.map(|_| ()),
            };
            // Check identity even when a rank failed: the failure may belong to a replaced engine.
            let (_, server) = state.client.discover(deadline).await?;
            let is_replaced = server.instance_id != state.model.instance_id();
            if is_replaced {
                // This connection now reaches another engine; only a reconnect can recover.
                state.cancel.cancel();
            }
            gate?;
            if is_replaced || *state.health.borrow() != EngineHealth::Serving {
                return Err(client::engine_shutdown(
                    "vLLM engine changed or became unhealthy during KV recovery",
                ));
            }
            self.ensure_ready(&state)
        })
        .await
        .map_err(|_| {
            client::engine_shutdown(
                "vLLM KV recovery deadline exceeded; history availability is unknown",
            )
        })?;
        if result.is_ok() && self.runtime_endpoint.get().is_none() {
            self.state.is_recovering.store(false, Ordering::Release);
        }
        result
    }

    fn ensure_ready(&self, state: &Connection) -> Result<(), DynamoError> {
        if *state.health.borrow() != EngineHealth::Serving
            || state
                .ranks
                .lock()
                .iter()
                .any(|rank| !matches!(*rank.status.borrow(), KvStreamStatus::Success))
        {
            return Err(client::engine_shutdown(
                "vLLM health or KV continuity changed before registration",
            ));
        }
        Ok(())
    }

    pub(super) async fn finish_recovery(&self) -> Result<(), DynamoError> {
        timeout_at(self.recovery_deadline()?, async {
            if self.is_lora_enabled() {
                self.reconcile_loaded_loras().await?;
            }
            self.ensure_ready(self.started_state()?.as_ref())?;
            *self.state.deadline.lock() = None;
            self.state.is_recovering.store(false, Ordering::Release);
            Ok(())
        })
        .await
        .map_err(|_| {
            client::engine_shutdown("vLLM recovery deadline exceeded before registration")
        })?
    }

    async fn wait_for_rank(mut rank: RankRecovery) -> Result<(), DynamoError> {
        loop {
            let status = rank.status.borrow_and_update().clone();
            match status {
                KvStreamStatus::Success => return Ok(()),
                KvStreamStatus::MissingHistory { expected, got } => {
                    return Err(client::engine_shutdown(format!(
                        "rank {} is missing KV sequence {expected}; next retained sequence {got}",
                        rank.rank
                    )));
                }
                KvStreamStatus::Failed { reason } => {
                    return Err(client::engine_shutdown(format!(
                        "rank {} KV recovery failed: {reason}",
                        rank.rank
                    )));
                }
                KvStreamStatus::Recovering => {}
            }
            rank.status
                .changed()
                .await
                .map_err(|_| client::engine_shutdown("KV listener ended before readiness"))?;
        }
    }

    pub(super) async fn monitor_availability(&self) -> Result<(), DynamoError> {
        let state = self.started_state()?;
        let ranks = state.ranks.lock().clone();
        let has_ranks = !ranks.is_empty();
        let streams = async {
            futures::future::select_all(ranks.into_iter().map(|mut rank| {
                Box::pin(async move {
                    loop {
                        if !matches!(*rank.status.borrow_and_update(), KvStreamStatus::Success) {
                            return;
                        }
                        if rank.status.changed().await.is_err() {
                            return;
                        }
                    }
                })
            }))
            .await;
        };
        let health = tokio::select! {
            health = unavailable(state.health.clone()) => {
                tracing::warn!(?health, "vLLM became unavailable");
                health
            }
            _ = streams, if has_ranks => {
                tracing::warn!("vLLM KV continuity requires recovery");
                EngineHealth::Serving
            }
        };
        // Stop admitting requests. KV ingestion continues, and so do in-flight
        // requests unless the connection itself was lost.
        self.state.is_recovering.store(true, Ordering::Release);
        if health == EngineHealth::Lost {
            state.cancel.cancel();
        }
        Ok(())
    }

    /// Recover a paused engine or a KV-only gap on the existing connection.
    async fn recover_in_place(
        &self,
        state: &Connection,
        deadline: Instant,
    ) -> Result<(), DynamoError> {
        loop {
            timeout_at(deadline, serving(state.health.clone()))
                .await
                .map_err(|_| {
                    client::engine_shutdown("vLLM did not resume before the recovery deadline")
                })??;
            let error = match self.wait_for_recovery().await {
                Ok(()) => return Ok(()),
                Err(error) => error,
            };
            let is_retryable = !state.cancel.is_cancelled()
                && *state.health.borrow() != EngineHealth::Lost
                && Instant::now() < deadline
                && !state.ranks.lock().iter().any(|rank| {
                    matches!(
                        *rank.status.borrow(),
                        KvStreamStatus::MissingHistory { .. } | KvStreamStatus::Failed { .. }
                    )
                });
            if !is_retryable {
                return Err(error);
            }
            tracing::debug!(%error, "Retrying vLLM recovery on the existing connection");
            tokio::time::sleep(self.transport.retry_interval).await;
        }
    }

    pub(super) async fn reconnect(&self) -> Result<EngineRecovery, DynamoError> {
        self.state.is_recovering.store(true, Ordering::Release);
        let old = self.started_state()?;
        let deadline = self.recovery_deadline()?;
        if *old.health.borrow() != EngineHealth::Lost {
            match self.recover_in_place(&old, deadline).await {
                Ok(()) => return Ok(EngineRecovery::SameInstance),
                Err(error) => {
                    tracing::warn!(%error, "vLLM recovery on the existing connection failed; reconnecting")
                }
            }
        }
        let ranks = old.ranks.lock().clone();
        if is_failure_final(&old, &ranks) {
            return Err(client::protocol_error(
                "KV recovery failed; engine shutdown is not justified",
            ));
        }
        old.cancel.cancel();
        timeout_at(deadline, self.replace_connection(&old, ranks, deadline))
            .await
            .map_err(|_| {
                client::engine_shutdown(
                    "vLLM recovery deadline exceeded while waiting for an engine replacement",
                )
            })?
    }

    async fn replace_connection(
        &self,
        old: &Connection,
        ranks: Vec<RankRecovery>,
        deadline: Instant,
    ) -> Result<EngineRecovery, DynamoError> {
        let has_missing_history = |ranks: &[RankRecovery]| {
            ranks
                .iter()
                .any(|rank| matches!(*rank.status.borrow(), KvStreamStatus::MissingHistory { .. }))
        };
        let mut is_history_missing = false;
        let mut is_shutdown_requested = false;
        self.unpublish_all_loras().await;
        loop {
            let connection = self.connect_generation(deadline).await?;
            let is_same = connection.model.instance_id() == old.model.instance_id();
            // A listener may confirm the loss after a failed recovery attempt.
            is_history_missing |= has_missing_history(&ranks);
            if is_history_missing && is_same {
                if !is_shutdown_requested {
                    is_shutdown_requested = self.request_shutdown(old.model.instance_id()).await?;
                }
                connection.cancel.cancel();
                tokio::time::sleep(self.transport.retry_interval).await;
                continue;
            }
            tracing::info!(instance_id = %connection.model.instance_id(), is_replacement = !is_same, "Reconnected to vLLM engine");
            if is_same {
                *connection.ranks.lock() = ranks.clone();
                for rank in &ranks {
                    if matches!(*rank.status.borrow(), KvStreamStatus::Failed { .. }) {
                        restart_rank(rank, deadline).await?;
                    }
                }
            }
            let config = self.engine_config(&connection.model)?;
            *self.state.current.lock() = Some(connection);
            if !is_same {
                return Ok(EngineRecovery::Replacement(Box::new(config)));
            }
            let error = match self.wait_for_recovery().await {
                Ok(()) => return Ok(EngineRecovery::SameInstance),
                Err(error) => error,
            };
            let state = self.started_state()?;
            if is_failure_final(&state, &ranks) {
                return Err(error);
            }
            state.cancel.cancel();
            tracing::warn!(%error, "vLLM recovery remains incomplete");
            tokio::time::sleep(self.transport.retry_interval).await;
        }
    }

    /// Returns whether the engine whose identity was verified accepted Shutdown.
    async fn request_shutdown(&self, instance_id: &str) -> Result<bool, DynamoError> {
        tracing::warn!(
            instance_id,
            "Required KV history is unavailable; requesting configured vLLM shutdown"
        );
        let result = tokio::time::timeout(
            self.transport.connect_attempt_timeout,
            client::shutdown_instance(&self.endpoint, instance_id),
        )
        .await;
        match result {
            Ok(Ok(is_accepted)) => Ok(is_accepted),
            Ok(Err(status))
                if matches!(
                    status.code(),
                    tonic::Code::FailedPrecondition
                        | tonic::Code::Unimplemented
                        | tonic::Code::PermissionDenied
                        | tonic::Code::Unauthenticated
                ) =>
            {
                Err(client::status_to_dynamo("Shutdown", status))
            }
            result => {
                tracing::warn!(?result, "Shutdown delivery uncertain; awaiting replacement");
                Ok(false)
            }
        }
    }
}

/// Failed replay is final while the engine's health is verified on this connection.
fn is_failure_final(state: &Connection, ranks: &[RankRecovery]) -> bool {
    !state.cancel.is_cancelled()
        && *state.health.borrow() != EngineHealth::Lost
        && ranks
            .iter()
            .any(|rank| matches!(*rank.status.borrow(), KvStreamStatus::Failed { .. }))
}

/// Restarts replay that ran out of transport retries; a final failure refuses.
async fn restart_rank(rank: &RankRecovery, deadline: Instant) -> Result<(), DynamoError> {
    let (accepted, response) = oneshot::channel();
    let refused = || {
        client::protocol_error(format!(
            "rank {} KV recovery failed; engine shutdown is not justified",
            rank.rank
        ))
    };
    rank.commands
        .send(KvStreamCommand::Restart { deadline, accepted })
        .await
        .map_err(|_| refused())?;
    response.await.map_err(|_| refused())
}

async fn serving(mut health: watch::Receiver<EngineHealth>) -> Result<(), DynamoError> {
    loop {
        match *health.borrow_and_update() {
            EngineHealth::Serving => return Ok(()),
            EngineHealth::Lost => return Err(client::engine_shutdown("vLLM Health.Watch ended")),
            EngineHealth::NotServing => {}
        }
        health
            .changed()
            .await
            .map_err(|_| client::engine_shutdown("vLLM Health.Watch ended"))?;
    }
}

async fn unavailable(mut health: watch::Receiver<EngineHealth>) -> EngineHealth {
    loop {
        let current = *health.borrow_and_update();
        if current != EngineHealth::Serving {
            return current;
        }
        if health.changed().await.is_err() {
            return EngineHealth::Lost;
        }
    }
}

async fn unhealthy(health: watch::Receiver<EngineHealth>) -> DynamoError {
    match unavailable(health).await {
        EngineHealth::NotServing => client::engine_shutdown("vLLM Health.Watch is not SERVING"),
        _ => client::engine_shutdown("vLLM Health.Watch ended"),
    }
}

fn monitor_health(
    client: Arc<VllmClient>,
    cancel: CancellationToken,
    attempt_timeout: Duration,
) -> watch::Receiver<EngineHealth> {
    let (status, receiver) = watch::channel(EngineHealth::NotServing);
    tokio::spawn(async move {
        let monitor = async {
            let client = client.as_ref();
            let streams = tokio::time::timeout(
                attempt_timeout,
                try_join_all(
                    [CONTROL_SERVICE, INFERENCE_SERVICE]
                        .into_iter()
                        .map(|service| client.watch_service(service)),
                ),
            )
            .await
            .map_err(|_| client::engine_shutdown("Health.Watch connection timed out"))??;
            let mut is_serving = vec![false; streams.len()];
            let mut updates = futures::stream::select_all(
                streams
                    .into_iter()
                    .enumerate()
                    .map(|(index, stream)| stream.map(move |update| (index, update)).boxed()),
            );
            // A healthy stream may stay silent; only a status change or stream end matters.
            while let Some((index, update)) = updates.next().await {
                let response =
                    update.map_err(|error| client::status_to_dynamo("Health.Watch", error))?;
                is_serving[index] = response.status == ServingStatus::Serving as i32;
                status.send_replace(if is_serving.iter().all(|serving| *serving) {
                    EngineHealth::Serving
                } else {
                    EngineHealth::NotServing
                });
            }
            Ok::<(), DynamoError>(())
        };
        tokio::select! {
            result = monitor => match result {
                Ok(()) => tracing::warn!("vLLM Health.Watch ended"),
                Err(error) => tracing::warn!(%error, "vLLM health monitor ended"),
            },
            _ = cancel.cancelled() => {},
        }
        status.send_replace(EngineHealth::Lost);
    });
    receiver
}
