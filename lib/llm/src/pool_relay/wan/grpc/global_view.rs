// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reverse transport for the same snapshots served by the Relay's subscription RPCs.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use dynamo_global_common::{PoolId, PoolIdDeriver, PoolKey, RouterReplica, V1PoolIdDeriver, grpc};
use dynamo_global_view_broadcaster::{Frame, PoolViewBroadcaster, RouterConnection};
use prost::Message;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;

use super::load::LoadUpdateHub;
use super::protocol::{KvPoolLoadUpdate, ServingLoadUpdate};
use super::service::{catalog_to_wire, readiness_to_wire};
use super::source::GrpcPublicationSource;

struct PublisherLifetime {
    cancel: CancellationToken,
    host: CancellationToken,
}

impl Drop for PublisherLifetime {
    fn drop(&mut self) {
        if !self.cancel.is_cancelled() && !self.host.is_cancelled() {
            tracing::error!("Global Router publisher stopped unexpectedly; stopping the Relay");
            self.host.cancel();
        }
    }
}

pub(super) fn spawn(
    key: PoolKey,
    routers: watch::Receiver<HashSet<RouterReplica>>,
    source: GrpcPublicationSource,
    kv: LoadUpdateHub<KvPoolLoadUpdate>,
    serving: LoadUpdateHub<ServingLoadUpdate>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let _lifetime = PublisherLifetime {
            cancel: cancel.clone(),
            host: source.lifecycle().clone(),
        };
        let pool = V1PoolIdDeriver.derive(&key);
        let identity = source.relay_identity();
        let hello = grpc::Hello {
            pool_key: Some(grpc::PoolKey {
                cluster_id: key.cluster_id().into(),
                namespace: key.namespace().into(),
                dgd_name: key.dgd_name().into(),
            }),
            logical_service: pool.to_string(),
            replica_id: identity.drt_instance_id().to_string(),
            incarnation: identity.relay_incarnation(),
        };
        let mut catalog = source.watch_catalog();
        let mut readiness = source.watch_readiness();
        let mut kv_windows = kv.subscribe();
        let mut serving_windows = serving.subscribe();
        let mut state = grpc::PoolState {
            catalog: catalog_to_wire(catalog.borrow_and_update().clone(), identity)
                .encode_to_vec()
                .into(),
            readiness: readiness_to_wire(&readiness.borrow_and_update(), identity)
                .encode_to_vec()
                .into(),
            kv_load: kv
                .current()
                .map(|window| window.encode_to_vec())
                .unwrap_or_default()
                .into(),
            serving_load: serving
                .current()
                .map(|window| window.encode_to_vec())
                .unwrap_or_default()
                .into(),
        };
        let broadcaster = PoolViewBroadcaster::<PoolId, Arc<grpc::PoolState>>::new();
        // Install the initial state before any router can attach.
        broadcaster.upsert(pool.clone(), Arc::new(state.clone()));
        let connections_cancel = cancel.clone();
        let fanout = broadcaster.run(routers, move |router| {
            attach(router, hello.clone(), connections_cancel.clone())
        });
        let publish = async {
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    changed = catalog.changed() => {
                        if changed.is_err() { break; }
                        state.catalog = catalog_to_wire(catalog.borrow_and_update().clone(), identity).encode_to_vec().into();
                    },
                    changed = readiness.changed() => {
                        if changed.is_err() { break; }
                        state.readiness = readiness_to_wire(&readiness.borrow_and_update(), identity).encode_to_vec().into();
                    },
                    window = kv_windows.recv() => {
                        if matches!(window, Err(tokio::sync::broadcast::error::RecvError::Closed)) { break; }
                        state.kv_load = kv.current().map(|window| window.encode_to_vec()).unwrap_or_default().into();
                    },
                    window = serving_windows.recv() => {
                        if matches!(window, Err(tokio::sync::broadcast::error::RecvError::Closed)) { break; }
                        state.serving_load = serving.current().map(|window| window.encode_to_vec()).unwrap_or_default().into();
                    },
                }
                // Lag is safe: each notification reads the complete current window.
                broadcaster.upsert(pool.clone(), Arc::new(state.clone()));
            }
        };
        tokio::select! {
            _ = cancel.cancelled() => {},
            _ = fanout => {},
            _ = publish => {},
        }
    })
}

async fn attach(
    router: RouterReplica,
    hello: grpc::Hello,
    cancel: CancellationToken,
) -> Result<RouterConnection<PoolId, Arc<grpc::PoolState>>> {
    ensure!(
        router.url.scheme() == "http",
        "Relay receiving endpoint must use http"
    );
    let channel = Endpoint::from_shared(router.url.to_string())?
        .connect_timeout(Duration::from_secs(3))
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_timeout(Duration::from_secs(5))
        .connect()
        .await
        .context("connect to Global Router")?;
    let mut client = grpc::pool_view_receiver_client::PoolViewReceiverClient::new(channel);
    let (sender, mut frames) = mpsc::channel::<Frame<PoolId, Arc<grpc::PoolState>>>(16);
    let transport = tokio::spawn(async move {
        let stream = async_stream::stream! {
            yield grpc::PoolViewFrame { event: Some(grpc::pool_view_frame::Event::Hello(hello)) };
            while let Some(frame) = frames.recv().await {
                use grpc::pool_view_frame::Event;
                let event = match frame {
                    Frame::Snapshot { revision, pools } => {
                        let state = pools.into_values().next().map(|state| (*state).clone());
                        Event::Snapshot(grpc::Snapshot { revision, pool_present: state.is_some(), state })
                    }
                    Frame::Upsert { revision, view, .. } =>
                        Event::Upsert(grpc::Upsert { revision, state: Some((*view).clone()) }),
                    Frame::Remove { revision, .. } => Event::Remove(grpc::Remove { revision }),
                };
                yield grpc::PoolViewFrame { event: Some(event) };
            }
        };
        tokio::select! {
            _ = cancel.cancelled() => {},
            result = client.publish(stream) => if let Err(error) = result {
                tracing::warn!(replica = %router.replica.0, %error, "Global Router receiving stream ended");
            },
        }
    });
    Ok(RouterConnection::with_transport(sender, transport))
}
