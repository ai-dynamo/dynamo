// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use dynamo_backend_common::{DynamoError, model_taints::ModelTaints};
use dynamo_sidecar_common::GrpcTransportConfig;
use tokio::sync::{Mutex, OnceCell, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, timeout_at};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::{
    client::{self, Client},
    proto as pb,
};

const PREFIX: &str = "dynamo.policy/";

pub(crate) struct PolicyTaints {
    namespace: String,
    subscription: OnceCell<ModelTaints>,
    task: Mutex<Option<JoinHandle<()>>>,
    cancel: CancellationToken,
}

impl PolicyTaints {
    pub(crate) fn new(namespace: String) -> Self {
        Self {
            namespace,
            subscription: OnceCell::new(),
            task: Mutex::new(None),
            cancel: CancellationToken::new(),
        }
    }

    pub(crate) fn subscription(&self) -> Option<ModelTaints> {
        self.subscription.get().cloned()
    }

    pub(crate) async fn start(
        &self,
        mut client: Client,
        transport: GrpcTransportConfig,
        deadline: Instant,
        expected_model: &str,
    ) -> Result<(), DynamoError> {
        let mut task = self.task.lock().await;
        let (mut stream, first) = subscribe(&mut client, deadline)
            .await
            .map_err(|error| client::engine_shutdown(format!("policy-version taints: {error}")))?;
        let model_path = first
            .model_info
            .as_ref()
            .map(|info| info.model_path.clone())
            .filter(|path| !path.is_empty())
            .ok_or_else(|| {
                client::engine_shutdown("policy-version taints require model identity")
            })?;
        if model_path != expected_model {
            return Err(client::engine_shutdown(
                "engine model changed during sidecar startup",
            ));
        }
        let (tx, rx) = watch::channel(snapshot_taints(&first, &self.namespace));
        self.subscription
            .set(ModelTaints {
                prefix: PREFIX,
                values: rx,
            })
            .map_err(|_| client::engine_shutdown("policy taint observer already started"))?;
        let namespace = self.namespace.clone();
        let cancel = self.cancel.clone();
        *task = Some(tokio::spawn(async move {
            let instance = first.instance_id;
            let mut revision = first.revision;
            loop {
                let result = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return,
                    result = stream.message() => result,
                };
                match result {
                    Ok(Some(snapshot)) if snapshot.instance_id == instance
                        && snapshot.revision > revision
                        && snapshot.model_info.as_ref().is_some_and(|info| info.model_path == model_path) => {
                        revision = snapshot.revision;
                        let taints = snapshot_taints(&snapshot, &namespace);
                        tx.send_if_modified(|current| {
                            if *current == taints { false } else { *current = taints; true }
                        });
                        continue;
                    }
                    Ok(Some(_)) => tracing::warn!("policy taint observer received inconsistent engine identity or revision"),
                    Ok(None) => tracing::warn!("policy taint engine stream ended"),
                    Err(error) => tracing::warn!(%error, "policy taint engine stream failed"),
                }
                tx.send_replace(HashSet::new());
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        _ = tokio::time::sleep(transport.retry_interval) => {},
                    }
                    let result = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        result = subscribe(&mut client, Instant::now() + transport.connect_attempt_timeout) => result,
                    };
                    match result {
                        Ok((new_stream, snapshot)) if snapshot.instance_id == instance
                            && snapshot.revision >= revision
                            && snapshot.model_info.as_ref().is_some_and(|info| info.model_path == model_path) => {
                            revision = snapshot.revision;
                            tx.send_replace(snapshot_taints(&snapshot, &namespace));
                            stream = new_stream;
                            break;
                        }
                        Ok(_) => {
                            tracing::warn!("engine identity changed; restart sidecar to refresh registration metadata");
                            return;
                        }
                        Err(error) => tracing::warn!(%error, "policy taint stream reconnect failed"),
                    }
                }
            }
        }.in_current_span()));
        Ok(())
    }

    pub(crate) async fn stop(&self) {
        self.cancel.cancel();
        if let Some(task) = self.task.lock().await.take()
            && let Err(error) = task.await
        {
            tracing::warn!(%error, "policy taint observer failed");
        }
    }
}

impl Drop for PolicyTaints {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.get_mut().take() {
            task.abort();
        }
    }
}

async fn subscribe(
    client: &mut Client,
    deadline: Instant,
) -> anyhow::Result<(
    tonic::Streaming<pb::EngineStateSnapshot>,
    pb::EngineStateSnapshot,
)> {
    timeout_at(deadline, async {
        let mut stream = client
            .watch_engine_state(pb::WatchEngineStateRequest {})
            .await?
            .into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| anyhow::anyhow!("engine state stream ended before first snapshot"))?;
        anyhow::ensure!(
            first.instance_id != 0 && first.revision != 0,
            "missing engine identity/revision"
        );
        Ok((stream, first))
    })
    .await?
}

fn snapshot_taints(snapshot: &pb::EngineStateSnapshot, namespace: &str) -> HashSet<String> {
    if !snapshot.healthy || snapshot.is_pause {
        return HashSet::new();
    }
    let version = snapshot
        .model_info
        .as_ref()
        .and_then(|info| serde_json::from_str::<serde_json::Value>(&info.json_info).ok())
        .and_then(|info| {
            info.get("weight_version")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .filter(|version| !version.trim().is_empty());
    version
        .map(|version| {
            let encode = |value: &str| {
                url::form_urlencoded::byte_serialize(value.as_bytes()).collect::<String>()
            };
            HashSet::from([format!(
                "{PREFIX}version={}:{}",
                encode(namespace),
                encode(&version)
            )])
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::task::{Context, Poll};
    use std::time::Duration;
    use tonic::body::Body;
    use tonic::codegen::{BoxFuture, Service, http};

    #[derive(Clone)]
    struct StateService(watch::Receiver<Result<pb::EngineStateSnapshot, tonic::Status>>);

    impl tonic::server::NamedService for StateService {
        const NAME: &'static str = "sglang.runtime.v1.SglangService";
    }

    impl tonic::server::ServerStreamingService<pb::WatchEngineStateRequest> for StateService {
        type Response = pb::EngineStateSnapshot;
        type ResponseStream =
            futures::stream::BoxStream<'static, Result<Self::Response, tonic::Status>>;
        type Future = BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;

        fn call(&mut self, _: tonic::Request<pb::WatchEngineStateRequest>) -> Self::Future {
            let mut rx = self.0.clone();
            Box::pin(async move {
                let stream = async_stream::stream! {
                    loop {
                        let state = rx.borrow_and_update().clone();
                        let failed = state.is_err();
                        yield state;
                        if failed || rx.changed().await.is_err() { break; }
                    }
                };
                Ok(tonic::Response::new(
                    Box::pin(stream) as Self::ResponseStream
                ))
            })
        }
    }

    impl Service<http::Request<Body>> for StateService {
        type Response = http::Response<Body>;
        type Error = Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, request: http::Request<Body>) -> Self::Future {
            let service = self.clone();
            Box::pin(async move {
                assert_eq!(
                    request.uri().path(),
                    "/sglang.runtime.v1.SglangService/WatchEngineState"
                );
                let mut grpc = tonic::server::Grpc::new(tonic::codec::ProstCodec::default());
                Ok(grpc.server_streaming(service, request).await)
            })
        }
    }

    #[tokio::test]
    async fn watch_stream_tracks_versions_disconnects_and_reconnects() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut snapshot = pb::EngineStateSnapshot {
                instance_id: 1,
                revision: 1,
                healthy: true,
                model_info: Some(pb::GetModelInfoResponse {
                    model_path: "model".into(),
                    json_info: r#"{"weight_version":"1"}"#.into(),
                }),
                ..Default::default()
            };
            let (tx, rx) = watch::channel(Ok(snapshot.clone()));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = format!("http://{}", listener.local_addr().unwrap());
            let incoming = futures::stream::unfold(listener, |listener| async move {
                Some((listener.accept().await.map(|(stream, _)| stream), listener))
            });
            let server = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_service(StateService(rx))
                    .serve_with_incoming(incoming)
                    .await
                    .unwrap();
            }));
            let client = Client::connect(address).await.unwrap();
            let observer = PolicyTaints::new("run".into());
            let transport = GrpcTransportConfig {
                retry_interval: Duration::from_millis(10),
                ..Default::default()
            };
            observer
                .start(
                    client.clone(),
                    transport,
                    Instant::now() + Duration::from_secs(1),
                    "model",
                )
                .await
                .unwrap();
            let mut values = observer.subscription().unwrap().values;
            assert_eq!(
                *values.borrow_and_update(),
                HashSet::from(["dynamo.policy/version=run:1".into()])
            );
            snapshot.revision = 2;
            snapshot.model_info.as_mut().unwrap().json_info = r#"{"weight_version":"2"}"#.into();
            let _ = tx.send_replace(Ok(snapshot.clone()));
            values.changed().await.unwrap();
            assert_eq!(
                *values.borrow_and_update(),
                HashSet::from(["dynamo.policy/version=run:2".into()])
            );
            let _ = tx.send_replace(Err(tonic::Status::unavailable("disconnected")));
            values.changed().await.unwrap();
            assert!(values.borrow_and_update().is_empty());
            let _ = tx.send_replace(Ok(snapshot.clone()));
            loop {
                values.changed().await.unwrap();
                if !values.borrow_and_update().is_empty() {
                    break;
                }
            }
            assert_eq!(
                *values.borrow(),
                HashSet::from(["dynamo.policy/version=run:2".into()])
            );
            snapshot.instance_id = 2;
            let _ = tx.send_replace(Ok(snapshot));
            while values.changed().await.is_ok() {
                assert!(values.borrow_and_update().is_empty());
            }
            observer.stop().await;
            let unavailable = PolicyTaints::new("run".into());
            let _ = tx.send_replace(Err(tonic::Status::unimplemented("unsupported")));
            assert!(
                unavailable
                    .start(
                        client.clone(),
                        transport,
                        Instant::now() + Duration::from_secs(1),
                        "model"
                    )
                    .await
                    .is_err()
            );
            assert!(unavailable.subscription().is_none());
            let resumed = pb::EngineStateSnapshot {
                instance_id: 2,
                revision: 1,
                healthy: true,
                model_info: Some(pb::GetModelInfoResponse {
                    model_path: "model".into(),
                    json_info: r#"{"weight_version":"3"}"#.into(),
                }),
                ..Default::default()
            };
            let _ = tx.send_replace(Ok(resumed.clone()));
            let idle = PolicyTaints::new("run".into());
            idle.start(
                client,
                transport,
                Instant::now() + Duration::from_secs(1),
                "model",
            )
            .await
            .unwrap();
            let values = idle.subscription().unwrap().values;
            idle.stop().await;
            assert!(values.has_changed().is_err());
            drop(server);
        })
        .await
        .expect("policy taint stream lifecycle timed out");
    }

    #[test]
    fn versions_are_scoped_and_unavailable_snapshots_clear_taints() {
        let mut state = pb::EngineStateSnapshot {
            instance_id: 1,
            revision: 1,
            healthy: true,
            model_info: Some(pb::GetModelInfoResponse {
                model_path: "model".into(),
                json_info: r#"{"weight_version":"v:1"}"#.into(),
            }),
            ..Default::default()
        };
        assert_eq!(
            snapshot_taints(&state, "run/a"),
            HashSet::from(["dynamo.policy/version=run%2Fa:v%3A1".into()])
        );
        state.model_info.as_mut().unwrap().json_info = r#"{"weight_version":"2"}"#.into();
        assert_eq!(
            snapshot_taints(&state, "run/a"),
            HashSet::from(["dynamo.policy/version=run%2Fa:2".into()])
        );
        state.is_pause = true;
        assert!(snapshot_taints(&state, "run/a").is_empty());
        state.is_pause = false;
        state.healthy = false;
        assert!(snapshot_taints(&state, "run/a").is_empty());
        state.healthy = true;
        for info in ["{}", "invalid", r#"{"weight_version":" "}"#] {
            state.model_info.as_mut().unwrap().json_info = info.into();
            assert!(snapshot_taints(&state, "run/a").is_empty());
        }
    }
}
