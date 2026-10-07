// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dynamo_backend_common::{BackendError, EngineConfig, EngineRecovery, ErrorType};
use dynamo_llm::kv_router::publisher::{
    KvStreamCommand, KvStreamStatus, ZmqBootstrapConfig, ZmqRecoveryControl,
};
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
    health: watch::Receiver<bool>,
    ranks: Mutex<Vec<RankRecovery>>,
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
        timeout_at(deadline, async {
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
                        tokio::time::sleep(self.transport.retry_interval).await;
                    }
                    Err(error) => return Err(error),
                }
            }
        })
        .await
        .map_err(|_| {
            client::engine_shutdown(
                "vLLM services did not recover before the configured startup deadline",
            )
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
        let mut health = connection.health.clone();
        timeout_at(deadline, async {
            loop {
                if *health.borrow_and_update() {
                    return Ok::<(), DynamoError>(());
                }
                health.changed().await.map_err(|_| {
                    client::engine_shutdown("vLLM Health.Watch ended before SERVING")
                })?;
            }
        })
        .await
        .map_err(|_| client::engine_shutdown("vLLM Health.Watch startup deadline exceeded"))??;
        let (_, observed) = connection.client.discover(deadline).await?;
        if observed.instance_id != connection.model.instance_id() || !*connection.health.borrow() {
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
        let config = state.model.engine_config(!self.mode.is_encode())?;
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
            let (completion, _) = oneshot::channel();
            recovery.push(RankRecovery {
                rank,
                status,
                commands,
            });
            sources.push(KvEventSource::Zmq {
                endpoint: zmq_connect_endpoint(&source.endpoint, &self.endpoint),
                topic: source.topic,
                dp_rank: rank,
                image_token_id: self.routing_image_token_id.get().copied().flatten(),
                bootstrap: Some(ZmqBootstrapConfig {
                    endpoint: zmq_connect_endpoint(&source.replay_endpoint, &self.endpoint),
                    dp_rank: rank,
                    timeout: deadline.saturating_duration_since(Instant::now()),
                    completion,
                    recovery: Some(ZmqRecoveryControl {
                        status: status_tx,
                        commands: command_rx,
                    }),
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
            let gate = try_join_all(
                ranks
                    .into_iter()
                    .map(|rank| self.wait_for_rank(&state, rank, deadline)),
            );
            tokio::select! {
                biased;
                error = unhealthy(state.health.clone()) => Err(error),
                result = gate => result.map(|_| ()),
            }?;
            let (_, server) = state.client.discover(deadline).await?;
            if server.instance_id != state.model.instance_id() || !*state.health.borrow() {
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
            *self.state.deadline.lock() = None;
            self.state.is_recovering.store(false, Ordering::Release);
        }
        result
    }

    fn ensure_ready(&self, state: &Connection) -> Result<(), DynamoError> {
        if !*state.health.borrow()
            || state
                .ranks
                .lock()
                .iter()
                .any(|rank| !matches!(*rank.status.borrow(), KvStreamStatus::Ready))
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

    async fn wait_for_rank(
        &self,
        state: &Connection,
        mut rank: RankRecovery,
        deadline: Instant,
    ) -> Result<(), DynamoError> {
        let mut probes = 0;
        loop {
            let status = rank.status.borrow_and_update().clone();
            match status {
                KvStreamStatus::Ready => return Ok(()),
                KvStreamStatus::MissingHistory { expected, got } => {
                    return Err(client::engine_shutdown(format!(
                        "rank {} is missing KV sequence {expected}; next retained sequence {got}",
                        rank.rank
                    )));
                }
                KvStreamStatus::Uncertain { reason, .. } => {
                    return Err(client::engine_shutdown(format!(
                        "rank {} KV recovery uncertain: {reason}",
                        rank.rank
                    )));
                }
                KvStreamStatus::ProbeRequired => {
                    if probes >= 3 {
                        return Err(client::engine_shutdown(format!(
                            "rank {} did not produce a verifiable KV stream after three probes",
                            rank.rank
                        )));
                    }
                    probes += 1;
                    self.probe(state, rank.rank, deadline).await?;
                    let (accepted, response) = oneshot::channel();
                    rank.commands
                        .send(KvStreamCommand::RetryReplay(accepted))
                        .await
                        .map_err(|_| client::engine_shutdown("KV listener stopped during probe"))?;
                    response.await.map_err(|_| {
                        client::engine_shutdown("KV listener ended before replay retry")
                    })?;
                }
                KvStreamStatus::Recovering => {}
            }
            rank.status
                .changed()
                .await
                .map_err(|_| client::engine_shutdown("KV listener ended before readiness"))?;
        }
    }

    async fn probe(
        &self,
        state: &Connection,
        rank: u32,
        deadline: Instant,
    ) -> Result<(), DynamoError> {
        let block_size = state.model.kv_cache_block_size()?.ok_or_else(|| {
            client::protocol_error("KV probe requires a reported cache block size")
        })?;
        let prompt_len = block_size
            .checked_add(1)
            .filter(|len| *len < state.model.max_model_len())
            .ok_or_else(|| {
                client::protocol_error("model context is too short for a full-block KV probe")
            })?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let id = format!("dynamo-kv-probe-{rank}-{nonce}");
        let request = pb::GenerateRequest {
            request_id: id.clone(),
            model: state.model.served_name.clone(),
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![0; prompt_len as usize],
            })),
            temperature: Some(0.0),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 1,
                min_new_tokens: 1,
                ignore_eos: true,
                ..Default::default()
            }),
            kv: Some(pb::KvCacheParameters {
                cache_salt: id,
                ..Default::default()
            }),
            ..Default::default()
        };
        tracing::info!(rank, "Probing vLLM to establish KV replay/live continuity");
        timeout_at(
            deadline.min(Instant::now() + Duration::from_secs(30)),
            async {
                let mut stream = state.client.generate_stream(request, Some(rank)).await?;
                while let Some(response) = stream
                    .message()
                    .await
                    .map_err(|status| client::status_to_dynamo("KV probe", status))?
                {
                    if response
                        .outputs
                        .as_ref()
                        .is_some_and(|output| output.finish_info.is_some())
                    {
                        return Ok(());
                    }
                }
                Err(client::protocol_error(
                    "KV probe ended without a terminal response",
                ))
            },
        )
        .await
        .map_err(|_| {
            client::engine_shutdown(
                "KV inference probe timed out; history availability remains unknown",
            )
        })?
    }

    pub(super) async fn monitor_availability(&self) -> Result<(), DynamoError> {
        let state = self.started_state()?;
        let ranks = state.ranks.lock().clone();
        let has_ranks = !ranks.is_empty();
        let streams = async {
            futures::future::select_all(ranks.into_iter().map(|mut rank| {
                Box::pin(async move {
                    loop {
                        if !matches!(*rank.status.borrow_and_update(), KvStreamStatus::Ready) {
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
        tokio::select! {
            error = unhealthy(state.health.clone()) => tracing::warn!(%error, "vLLM became unavailable"),
            _ = streams, if has_ranks => tracing::warn!("vLLM KV continuity requires recovery"),
        }
        self.state.is_recovering.store(true, Ordering::Release);
        state.cancel.cancel();
        for rank in state.ranks.lock().clone() {
            let _ = rank.commands.try_send(KvStreamCommand::Suspend);
        }
        Ok(())
    }

    pub(super) async fn reconnect(&self, is_startup: bool) -> Result<EngineRecovery, DynamoError> {
        self.state.is_recovering.store(true, Ordering::Release);
        let old = self.started_state()?;
        old.cancel.cancel();
        let deadline = self.recovery_deadline()?;
        let ranks = old.ranks.lock().clone();
        let mut has_missing_history = ranks
            .iter()
            .any(|rank| matches!(*rank.status.borrow(), KvStreamStatus::MissingHistory { .. }));
        if ranks.iter().any(|rank| {
            matches!(
                *rank.status.borrow(),
                KvStreamStatus::Uncertain {
                    is_terminal: true,
                    ..
                }
            )
        }) {
            return Err(client::protocol_error(
                "KV recovery failed decoding or applying events; engine shutdown is not justified",
            ));
        }
        let mut is_shutdown_requested = false;
        timeout_at(deadline, async {
            self.unpublish_all_loras().await;
            loop {
                let connection = self.connect_generation(deadline).await?;
                let is_same = connection.model.instance_id() == old.model.instance_id();
                if has_missing_history && is_same {
                    if !is_shutdown_requested {
                        tracing::warn!(instance_id = %old.model.instance_id(), "Required KV history is unavailable; requesting configured vLLM shutdown");
                        let result = tokio::time::timeout(self.transport.connect_attempt_timeout, client::shutdown_instance(&self.endpoint, old.model.instance_id())).await;
                        match result {
                            Ok(Ok(is_accepted)) => is_shutdown_requested = is_accepted,
                            Ok(Err(status)) if matches!(status.code(), tonic::Code::FailedPrecondition | tonic::Code::Unimplemented | tonic::Code::PermissionDenied | tonic::Code::Unauthenticated) => return Err(client::status_to_dynamo("Shutdown", status)),
                            result => tracing::warn!(?result, "Shutdown delivery uncertain; awaiting replacement"),
                        }
                    }
                    connection.cancel.cancel();
                    tokio::time::sleep(self.transport.retry_interval).await;
                    continue;
                }
                if is_same && !is_startup {
                    *connection.ranks.lock() = ranks.clone();
                    for rank in &ranks {
                        let (accepted, response) = oneshot::channel();
                        rank.commands.send(KvStreamCommand::ReconnectSameInstance(accepted)).await.map_err(|_| client::engine_shutdown("KV listener ended; retained state cannot resume"))?;
                        response.await.map_err(|_| client::engine_shutdown("KV listener ended before reconnect"))?;
                    }
                }
                let config = connection.model.engine_config(!self.mode.is_encode())?;
                *self.state.current.lock() = Some(connection);
                if is_same && !is_startup {
                    match self.wait_for_recovery().await {
                        Ok(()) => return Ok(EngineRecovery::SameInstance),
                        Err(error) => {
                            let current = self.started_state()?;
                            current.cancel.cancel();
                            has_missing_history = ranks.iter().any(|rank| matches!(*rank.status.borrow(), KvStreamStatus::MissingHistory { .. }));
                            if ranks.iter().any(|rank| matches!(*rank.status.borrow(), KvStreamStatus::Uncertain { is_terminal: true, .. })) {
                                return Err(error);
                            }
                            for rank in &ranks { let _ = rank.commands.try_send(KvStreamCommand::Suspend); }
                            tracing::warn!(%error, "vLLM recovery remains incomplete");
                            tokio::time::sleep(self.transport.retry_interval).await;
                            continue;
                        }
                    }
                }
                return Ok(EngineRecovery::Replacement(Box::new(config)));
            }
        }).await.map_err(|_| client::engine_shutdown("vLLM recovery deadline exceeded while waiting for an engine replacement"))?
    }
}

async fn unhealthy(mut health: watch::Receiver<bool>) -> DynamoError {
    loop {
        if !*health.borrow_and_update() {
            return client::engine_shutdown("vLLM Health.Watch is not SERVING");
        }
        if health.changed().await.is_err() {
            return client::engine_shutdown("vLLM Health.Watch ended");
        }
    }
}

fn monitor_health(
    client: Arc<VllmClient>,
    cancel: CancellationToken,
    attempt_timeout: Duration,
) -> watch::Receiver<bool> {
    let (status, receiver) = watch::channel(false);
    tokio::spawn(async move {
        let monitor = async {
            let client = client.as_ref();
            let streams = try_join_all([CONTROL_SERVICE, INFERENCE_SERVICE].into_iter().map(
                |service| async move {
                    let mut stream = client.watch_service(service).await?;
                    let response = stream
                        .message()
                        .await
                        .map_err(|error| client::status_to_dynamo("Health.Watch", error))?
                        .ok_or_else(|| {
                            client::engine_shutdown("Health.Watch ended before initial status")
                        })?;
                    if response.status != ServingStatus::Serving as i32 {
                        return Err(client::engine_shutdown(format!("{service} is not SERVING")));
                    }
                    Ok(stream)
                },
            ));
            let initial = tokio::time::timeout(attempt_timeout, streams)
                .await
                .map_err(|_| client::engine_shutdown("Health.Watch initial status timed out"))??;
            status.send_replace(true);
            let result = futures::future::select_all(initial.into_iter().map(|mut stream| {
                Box::pin(async move {
                    loop {
                        match stream.message().await {
                            Ok(Some(response))
                                if response.status == ServingStatus::Serving as i32 => {}
                            result => return result,
                        }
                    }
                })
            }))
            .await;
            tracing::debug!(status = ?result.0, "Health.Watch stopped serving");
            Ok::<(), DynamoError>(())
        };
        tokio::select! {
            result = monitor => if let Err(error) = result { tracing::warn!(%error, "vLLM health monitor ended"); },
            _ = cancel.cancelled() => {},
        }
        status.send_replace(false);
    });
    receiver
}
