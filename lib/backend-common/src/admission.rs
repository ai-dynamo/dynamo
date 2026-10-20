// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller permission is distinct from engine readiness. Only the membership
//! task publishes discovery; these routes change intent and report local progress.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::engine_routes::EngineRouteCallback;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::serving::{EngineServingState, EngineServingStates};
use crate::worker::EngineKind;

/// An approval is scoped to one engine, policy version and uninterrupted state
/// subscription. Engines must change observation_epoch after any stream loss.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionIdentity {
    pub engine_instance_id: u64,
    pub observation_epoch: u64,
    pub weight_version: String,
}

impl AdmissionIdentity {
    pub(crate) fn from_observation(state: &EngineServingState) -> Option<Self> {
        Some(Self {
            engine_instance_id: state.instance_id,
            observation_epoch: state.observation_epoch,
            weight_version: state
                .weight_version
                .as_ref()
                .filter(|v| !v.trim().is_empty())?
                .clone(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Mutation {
    session: String,
    revision: u64,
    #[serde(default)]
    expected: Option<AdmissionIdentity>,
}

struct State {
    session: String,
    revision: u64,
    last: Option<(bool, Mutation)>,
    grant: Option<AdmissionIdentity>,
    published: Option<bool>,
    applied_revision: u64,
    last_error: Option<String>,
}

pub(crate) struct ControllerAdmission {
    state: Mutex<State>,
    changed: watch::Sender<u64>,
}

impl ControllerAdmission {
    pub(crate) fn new(endpoint: &Endpoint) -> Arc<Self> {
        static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
        Arc::new(Self {
            state: Mutex::new(State {
                session: format!(
                    "{:x}-{}",
                    endpoint.drt().connection_id(),
                    NEXT_SESSION.fetch_add(1, Ordering::Relaxed)
                ),
                revision: 0,
                last: None,
                grant: None,
                published: Some(false),
                applied_revision: 0,
                last_error: None,
            }),
            changed: watch::channel(0).0,
        })
    }

    pub(crate) fn changes(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    /// Called with the worker mutation lock held, before reconciling discovery.
    pub(crate) fn permits(&self, observation: Option<&EngineServingState>) -> bool {
        let identity = observation.and_then(AdmissionIdentity::from_observation);
        let mut state = self.state.lock().unwrap();
        if state.grant.is_some() && state.grant != identity {
            state.grant = None;
            state.last_error = Some(
                "engine identity/version/state continuity changed; fresh approval required".into(),
            );
        }
        state.grant.is_some()
    }

    pub(crate) fn applied(&self, published: Option<bool>, error: Option<String>) {
        let mut state = self.state.lock().unwrap();
        state.published = published;
        if let Some(error) = error {
            state.last_error = Some(error);
        } else {
            state.applied_revision = state.revision;
        }
    }

    fn status(&self, observation: Option<&EngineServingState>, shutdown: bool) -> Value {
        let state = self.state.lock().unwrap();
        json!({
            "status": "ok", "session": state.session,
            "revision": state.revision, "applied_revision": state.applied_revision,
            "admitted": !shutdown && state.grant.is_some(),
            "published": if shutdown { None } else { state.published },
            "shutting_down": shutdown, "expected": state.grant,
            "observed": observation.and_then(AdmissionIdentity::from_observation),
            "engine_ready": observation.is_some_and(|s| s.ready),
            "last_error": state.last_error,
        })
    }

    fn begin(&self, admit: bool, command: &Mutation) -> anyhow::Result<bool> {
        let mut state = self.state.lock().unwrap();
        anyhow::ensure!(
            command.session == state.session,
            "sidecar session mismatch; read status again"
        );
        anyhow::ensure!(command.revision > 0, "revision must be positive");
        if command.revision == state.revision {
            anyhow::ensure!(
                state.last.as_ref() == Some(&(admit, command.clone())),
                "conflicting command at the same revision"
            );
            return Ok(false); // Exact retry reports status; never resurrects a revoked grant.
        }
        anyhow::ensure!(
            command.revision > state.revision,
            "stale admission revision"
        );
        state.revision = command.revision;
        state.last = Some((admit, command.clone()));
        state.grant = None; // A failed/cancelled replacement cannot retain prior permission.
        state.last_error = None;
        self.changed
            .send_modify(|revision| *revision = command.revision);
        Ok(true)
    }

    pub(crate) fn register_routes(
        self: &Arc<Self>,
        endpoint: &Endpoint,
        engine: EngineKind,
        observations: EngineServingStates,
        mutation: Arc<tokio::sync::Mutex<()>>,
        shutdown: CancellationToken,
    ) {
        for operation in ["status", "admit", "withdraw"] {
            let gate = self.clone();
            let engine = engine.clone();
            let observations = observations.clone();
            let mutation = mutation.clone();
            let shutdown = shutdown.clone();
            let callback: EngineRouteCallback = Arc::new(move |body| {
                let gate = gate.clone();
                let engine = engine.clone();
                let observations = observations.clone();
                let mutation = mutation.clone();
                let shutdown = shutdown.clone();
                Box::pin(async move {
                    if operation == "status" {
                        anyhow::ensure!(
                            body.as_object().is_some_and(|v| v.is_empty()),
                            "status body must be an empty object"
                        );
                        return Ok(
                            gate.status(observations.borrow().as_ref(), shutdown.is_cancelled())
                        );
                    }
                    let command: Mutation = serde_json::from_value(body)?;
                    let admit = operation == "admit";
                    anyhow::ensure!(
                        admit == command.expected.is_some(),
                        "admit requires expected identity; withdraw must omit it"
                    );
                    if let Some(identity) = &command.expected {
                        anyhow::ensure!(
                            identity.engine_instance_id != 0
                                && identity.observation_epoch != 0
                                && !identity.weight_version.trim().is_empty(),
                            "invalid expected identity"
                        );
                    }
                    let _guard = tokio::select! {
                        biased;
                        _ = shutdown.cancelled() => anyhow::bail!("worker is shutting down"),
                        guard = mutation.lock() => guard,
                    };
                    anyhow::ensure!(!shutdown.is_cancelled(), "worker is shutting down");
                    if gate.begin(admit, &command)?
                        && let Some(expected) = &command.expected
                    {
                        let verification = async {
                            let observation = observations.borrow().clone();
                            anyhow::ensure!(
                                observation.as_ref().is_some_and(|s| s.ready)
                                    && observation
                                        .as_ref()
                                        .and_then(AdmissionIdentity::from_observation)
                                        .as_ref()
                                        == Some(expected),
                                "engine observation is not ready or does not match expected identity"
                            );
                            engine
                                .verify_serving_admission(expected)
                                .await
                                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                            let latest = observations.borrow().clone();
                            anyhow::ensure!(
                                latest.as_ref().is_some_and(|s| s.ready)
                                    && latest
                                        .as_ref()
                                        .and_then(AdmissionIdentity::from_observation)
                                        .as_ref()
                                        == Some(expected),
                                "engine changed during admission verification"
                            );
                            anyhow::Ok(())
                        };
                        let result = tokio::select! {
                            biased;
                            _ = shutdown.cancelled() => Err(anyhow::anyhow!("worker is shutting down")),
                            result = tokio::time::timeout(Duration::from_secs(10), verification) => {
                                result.unwrap_or_else(|_| Err(anyhow::anyhow!("admission verification timed out")))
                            }
                        };
                        match result {
                            Ok(()) => gate.state.lock().unwrap().grant = Some(expected.clone()),
                            Err(error) => {
                                gate.state.lock().unwrap().last_error = Some(error.to_string());
                                return Err(error);
                            }
                        }
                    }
                    Ok(gate.status(observations.borrow().as_ref(), false))
                })
            });
            endpoint
                .drt()
                .engine_routes()
                .register(&format!("admission/{operation}"), callback);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DynamoError, EngineConfig, GenerateContext, LLMEngine, LLMEngineOutput, PreprocessedRequest,
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::DiscoveryQuery,
        distributed::DistributedConfig,
        pipeline::{ManyOut, SingleIn, network::Ingress},
        protocols::annotated::Annotated,
        system_health::ReadinessHold,
    };
    use std::sync::atomic::AtomicBool;

    struct Engine {
        fail: AtomicBool,
        block: AtomicBool,
        entered: tokio::sync::Notify,
    }
    #[async_trait::async_trait]
    impl LLMEngine for Engine {
        async fn start(&self, _: u64) -> Result<EngineConfig, DynamoError> {
            unreachable!()
        }
        async fn generate(
            &self,
            _: PreprocessedRequest,
            _: GenerateContext,
        ) -> Result<
            futures::stream::BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
            DynamoError,
        > {
            unreachable!()
        }
        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
        async fn verify_serving_admission(&self, _: &AdmissionIdentity) -> Result<(), DynamoError> {
            self.entered.notify_one();
            if self.block.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            if self.fail.load(Ordering::Relaxed) {
                return Err(DynamoError::builder()
                    .error_type(crate::ErrorType::Backend(
                        crate::BackendError::InvalidArgument,
                    ))
                    .message("live version mismatch")
                    .build());
            }
            Ok(())
        }
    }

    fn observation() -> EngineServingState {
        EngineServingState {
            instance_id: 42,
            revision: 1,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("17".into()),
            taints: Default::default(),
        }
    }

    async fn count(endpoint: &Endpoint) -> usize {
        let id = endpoint.id();
        endpoint
            .drt()
            .discovery()
            .list(DiscoveryQuery::Endpoint {
                namespace: id.namespace,
                component: id.component,
                endpoint: id.name,
            })
            .await
            .unwrap()
            .len()
    }

    async fn wait_published(gate: &ControllerAdmission, expected: bool, revision: u64) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let status = gate.status(None, false);
                if status["published"] == expected && status["applied_revision"] == revision {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn controller_owns_membership_across_versions_reconnects_and_shutdown() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("controller_admission")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let started = endpoint
            .endpoint_builder()
            .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
            .start_without_registration()
            .await
            .unwrap();
        let gate = ControllerAdmission::new(&endpoint);
        let engine = Arc::new(Engine {
            fail: AtomicBool::new(false),
            block: AtomicBool::new(false),
            entered: tokio::sync::Notify::new(),
        });
        let mut state = observation();
        let (tx, rx) = watch::channel(Some(state.clone()));
        let mutation = Arc::new(tokio::sync::Mutex::new(()));
        let shutdown = CancellationToken::new();
        gate.register_routes(
            &endpoint,
            EngineKind::Llm(engine.clone()),
            rx.clone(),
            mutation.clone(),
            shutdown.clone(),
        );
        let routes = endpoint.drt().engine_routes();
        let admit = routes.get("admission/admit").unwrap();
        let withdraw = routes.get("admission/withdraw").unwrap();
        let status = routes.get("admission/status").unwrap();
        let session = status(json!({})).await.unwrap()["session"].clone();
        let command = |revision, state: &EngineServingState| {
            json!({"session": session, "revision": revision,
            "expected": AdmissionIdentity::from_observation(state).unwrap() })
        };
        let task = tokio::spawn(crate::serving::follow_engine_state(
            crate::serving::EndpointMembership::new(
                endpoint.clone(),
                started.instance(),
                Duration::from_secs(1),
            ),
            rx,
            mutation,
            shutdown.clone(),
            ReadinessHold::take(endpoint.drt().system_health(), endpoint.name()),
            Some(gate.clone()),
            None,
        ));

        // A healthy, unpaused engine must NOT auto-join. A withdrawn barrier lets
        // this assertion distinguish the managed path from an unscheduled task.
        withdraw(json!({"session": session, "revision": 1}))
            .await
            .unwrap();
        wait_published(&gate, false, 1).await;
        assert_eq!(count(&endpoint).await, 0);
        let first = command(2, &state);
        let accepted = admit(first.clone()).await.unwrap();
        assert_eq!(accepted["admitted"], true);
        assert_eq!(accepted["published"], false);
        wait_published(&gate, true, 2).await;
        assert_eq!(count(&endpoint).await, 1);
        withdraw(json!({"session": session, "revision": 3}))
            .await
            .unwrap();
        wait_published(&gate, false, 3).await;
        assert!(admit(first).await.is_err());
        assert!(admit(command(3, &state)).await.is_err());
        assert!(admit(json!({"session":"old-process", "revision":99, "expected": AdmissionIdentity::from_observation(&state)})).await.is_err());
        assert_eq!(status(json!({})).await.unwrap()["revision"], 3);

        engine.fail.store(true, Ordering::Relaxed);
        assert!(admit(command(4, &state)).await.is_err());
        wait_published(&gate, false, 4).await;
        assert_eq!(status(json!({})).await.unwrap()["admitted"], false);
        engine.fail.store(false, Ordering::Relaxed);
        // Retrying the failed revision cannot silently turn it into permission.
        assert_eq!(admit(command(4, &state)).await.unwrap()["admitted"], false);
        admit(command(5, &state)).await.unwrap();
        wait_published(&gate, true, 5).await;
        state.ready = false;
        state.revision += 1;
        tx.send_replace(Some(state.clone()));
        wait_published(&gate, false, 5).await;
        state.ready = true;
        state.revision += 1;
        tx.send_replace(Some(state.clone()));
        wait_published(&gate, true, 5).await;

        state.weight_version = Some("18".into());
        state.revision += 1;
        tx.send_replace(Some(state.clone()));
        wait_published(&gate, false, 5).await;
        assert_eq!(status(json!({})).await.unwrap()["admitted"], false);
        admit(command(6, &state)).await.unwrap();
        wait_published(&gate, true, 6).await;
        // Even a coalesced disconnect/reconnect invalidates the old approval.
        tx.send_replace(None);
        state.observation_epoch += 1;
        tx.send_replace(Some(state.clone()));
        wait_published(&gate, false, 6).await;
        assert!(admit(command(6, &observation())).await.is_err());
        admit(command(7, &state)).await.unwrap();
        wait_published(&gate, true, 7).await;
        state.instance_id += 1;
        state.revision = 1;
        tx.send_replace(Some(state.clone()));
        wait_published(&gate, false, 7).await;
        // A replacement sidecar creates a new session even in the same runtime.
        assert_ne!(
            ControllerAdmission::new(&endpoint).status(None, false)["session"],
            session
        );

        engine.block.store(true, Ordering::Relaxed);
        let _ = engine.entered.notified().now_or_never();
        let pending = tokio::spawn(admit(command(8, &state)));
        engine.entered.notified().await;
        shutdown.cancel();
        assert!(pending.await.unwrap().is_err());
        task.await.unwrap();
        assert!(admit(command(9, &state)).await.is_err());
        assert_eq!(status(json!({})).await.unwrap()["shutting_down"], true);
        assert_eq!(count(&endpoint).await, 0);
        started.shutdown().await.unwrap();
        runtime.shutdown();
    }

    use futures::FutureExt;
}
