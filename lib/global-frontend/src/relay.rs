// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool load from each pool's PoolRelay.
//!
//! One task per pool follows its Relay's serving-load and KV-load streams over
//! the same connection. Reports stay in [`PoolSignals`] until they age out.
//! The Relay address comes from the catalog; when it changes or the pool
//! leaves, the subscription ends and the pool's load is dropped.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use dynamo_global_common::PoolId;
use dynamo_global_router::{LoadReport, PoolSignals, RelaySource};
use dynamo_llm::pool_relay::wan::grpc::protocol::{
    KvEventRelayClient, KvPoolLoadUpdate, RELAY_CONTRACT_MARKER, ServingLoadUpdate,
    SubscribeKvPoolLoadRequest, SubscribeServingLoadRequest, validate_protocol_envelope,
};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};
use url::Url;

use crate::PoolSnapshot;

const SUBSCRIBER_ID: &str = "global-frontend";
const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(5);
/// The Relay's report window when an update leaves it unset.
const DEFAULT_WINDOW: Duration = Duration::from_secs(1);
const FIRST_RETRY: Duration = Duration::from_millis(500);
const LAST_RETRY: Duration = Duration::from_secs(30);
/// A stream that lasted this long resets the retry delay.
const HEALTHY_STREAM: Duration = Duration::from_secs(30);
/// Keeps the Relay subscriptions of every pool in `pools` that names a
/// Relay, until `cancel` fires.
pub async fn follow_relays(
    mut pools: watch::Receiver<Arc<PoolSnapshot>>,
    signals: Arc<PoolSignals>,
    cancel: CancellationToken,
) {
    let mut subscriptions =
        HashMap::<PoolId, (Url, String, CancellationToken, JoinHandle<()>)>::new();
    loop {
        let snapshot = pools.borrow_and_update().clone();
        let wanted = snapshot
            .pools()
            .iter()
            .filter_map(|pool| Some((pool.id.clone(), (pool.relay.clone()?, pool.model.clone()))))
            .collect::<HashMap<_, _>>();
        let removed = subscriptions
            .iter()
            .filter(|(pool, (relay, model, _, _))| {
                !wanted.get(*pool).is_some_and(|(want_relay, want_model)| {
                    want_relay == relay && want_model == model
                })
            })
            .map(|(pool, _)| pool.clone())
            .collect::<Vec<_>>();
        for pool in removed {
            let (_, _, subscription, task) = subscriptions.remove(&pool).unwrap();
            subscription.cancel();
            let _ = task.await;
            signals.forget(&pool);
        }
        for (pool, (relay, model)) in wanted {
            if subscriptions.contains_key(&pool) {
                continue;
            }
            let subscription = cancel.child_token();
            let task = tokio::spawn(subscribe(
                pool.clone(),
                model.clone(),
                relay.clone(),
                signals.clone(),
                subscription.clone(),
            ));
            subscriptions.insert(pool, (relay, model, subscription, task));
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            changed = pools.changed() => if changed.is_err() { return },
        }
    }
}

async fn subscribe(
    pool: PoolId,
    model: String,
    relay: Url,
    signals: Arc<PoolSignals>,
    cancel: CancellationToken,
) {
    let mut retry = FIRST_RETRY;
    loop {
        let started = Instant::now();
        let stream = stream_pool(&pool, &model, &relay, &signals);
        // Stop as soon as the subscription is cancelled, even while
        // connecting, so a replaced subscription never records over the next.
        let Err(error) = tokio::select! {
            _ = cancel.cancelled() => return,
            result = stream => result,
        };
        tracing::warn!(%pool, %relay, error = format!("{error:#}"), "Relay stream ended");
        if started.elapsed() >= HEALTHY_STREAM {
            retry = FIRST_RETRY;
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(retry) => {}
        }
        retry = (retry * 2).min(LAST_RETRY);
    }
}

/// Both RPCs share one HTTP/2 connection, so a Service can select one Relay
/// Pod for the pair. A failed stream ends the pair and triggers a reconnect.
async fn stream_pool(
    pool: &PoolId,
    model: &str,
    relay: &Url,
    signals: &PoolSignals,
) -> Result<Infallible> {
    let channel = connect(relay).await?;
    let mut load_client = KvEventRelayClient::new(channel.clone());
    let mut kv_client = KvEventRelayClient::new(channel);
    let load_request = SubscribeServingLoadRequest {
        subscriber_id: SUBSCRIBER_ID.to_owned(),
        contract_marker: RELAY_CONTRACT_MARKER,
    };
    let kv_request = SubscribeKvPoolLoadRequest {
        subscriber_id: SUBSCRIBER_ID.to_owned(),
        contract_marker: RELAY_CONTRACT_MARKER,
    };
    let (load, kv) = tokio::time::timeout(SUBSCRIBE_TIMEOUT, async {
        tokio::try_join!(
            load_client.subscribe_serving_load(load_request),
            kv_client.subscribe_kv_pool_load(kv_request),
        )
    })
    .await
    .map_err(|_| anyhow!("subscribe timed out"))?
    .context("subscribe")?;
    let mut load_updates = load.into_inner();
    let mut kv_updates = kv.into_inner();
    tracing::info!(%pool, %relay, "following the pool's Relay load and KV load");
    loop {
        tokio::select! {
            update = load_updates.message() => {
                let Some(update) = update.context("read serving load")? else {
                    bail!("the Relay closed the serving-load stream");
                };
                record_serving_update(pool, model, signals, &update)?;
            }
            update = kv_updates.message() => {
                let Some(update) = update.context("read KV load")? else {
                    bail!("the Relay closed the KV-load stream");
                };
                record_kv_update(pool, signals, &update)?;
            }
        }
    }
}

fn record_serving_update(
    pool: &PoolId,
    model: &str,
    signals: &PoolSignals,
    update: &ServingLoadUpdate,
) -> Result<()> {
    validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
    let relay = update.relay.as_ref().context("missing Relay identity")?;
    let source = RelaySource {
        runtime_id: relay.drt_instance_id,
        incarnation: relay.relay_incarnation,
    };
    match pool_report(update, model) {
        Some(report) => signals.record_load(pool, source, report),
        None => tracing::debug!(%pool, "the Relay's load is incomplete"),
    }
    Ok(())
}

fn record_kv_update(pool: &PoolId, signals: &PoolSignals, update: &KvPoolLoadUpdate) -> Result<()> {
    validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
    let relay = update.relay.as_ref().context("missing Relay identity")?;
    let source = RelaySource {
        runtime_id: relay.drt_instance_id,
        incarnation: relay.relay_incarnation,
    };
    match pool_kv(update) {
        Some((used, total)) => {
            signals.record_kv(pool, source, used, total, report_window(update.window_ms))
        }
        None => tracing::debug!(%pool, "the Relay's KV load is incomplete"),
    }
    Ok(())
}

fn report_window(window_ms: u64) -> Duration {
    match window_ms {
        0 => DEFAULT_WINDOW,
        window_ms => Duration::from_millis(window_ms),
    }
}

async fn connect(relay: &Url) -> Result<Channel> {
    Endpoint::from_shared(relay.to_string())?
        .connect_timeout(Duration::from_secs(3))
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_timeout(Duration::from_secs(5))
        .connect()
        .await
        .context("connect")
}

/// The KV blocks the pool's workers hold for running requests, and their
/// total. `None` unless every producer has heard from every rank and knows
/// its capacity: the Relay marks an incomplete view rather than leaving it
/// out, and an incomplete view is not zero load.
fn pool_kv(update: &KvPoolLoadUpdate) -> Option<(u64, u64)> {
    if update.pools.is_empty() {
        return None;
    }
    update
        .pools
        .iter()
        .try_fold((0_u64, 0_u64), |(used, total), entry| {
            let complete = entry.kv_expected_ranks > 0
                && entry.kv_observed_ranks >= entry.kv_expected_ranks
                && entry.total_kv_blocks > 0;
            complete.then(|| {
                (
                    used.saturating_add(entry.kv_used_blocks),
                    total.saturating_add(entry.total_kv_blocks),
                )
            })
        })
}

/// The pool's requests in flight (waiting for a first token or generating)
/// for `model`, those generating, and how many its workers can run at once.
/// `None` unless the request counts are reported: the Relay leaves them
/// unset when its view is incomplete, and unset does not mean zero. A Relay
/// watching one pool reports one model; a model it names differently from
/// the catalog still counts.
fn pool_report(update: &ServingLoadUpdate, model: &str) -> Option<LoadReport> {
    let named = update
        .models
        .iter()
        .filter(|entry| entry.canonical_model_id == model)
        .collect::<Vec<_>>();
    let models = if named.is_empty() {
        update.models.iter().collect()
    } else {
        named
    };
    if models.is_empty() {
        return None;
    }
    let (mut in_flight, mut generating) = (0, 0);
    for entry in models {
        let requests = entry.load.as_ref()?.requests.as_ref()?;
        let running = requests.requests_generating?;
        in_flight += requests.requests_awaiting_first_token? + running;
        generating += running;
    }
    let max_concurrency = update
        .pools
        .iter()
        .map(|pool| pool.deployment.as_ref()?.max_concurrency)
        .sum::<Option<u64>>();
    Some(LoadReport {
        in_flight,
        generating,
        max_concurrency,
        window: report_window(update.window_ms),
    })
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::pin::Pin;
    use std::sync::Mutex;

    use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
    use dynamo_global_router::InFlightTracker;
    use dynamo_llm::pool_relay::wan::grpc::protocol as proto;
    use dynamo_llm::pool_relay::wan::grpc::protocol::{
        KvPoolLoadEntry, LoadView, ModelServingLoad, PoolDeploymentStatus, PoolServingLoad,
        RELAY_PROTOCOL_VERSION, RelayIdentity, RequestLifecycleStats,
    };
    use futures::{Stream, StreamExt};
    use tonic::transport::Server;
    use tonic::{Request, Response, Status};

    use super::*;

    /// An update over a 2 s window from a pool of 8 workers that can run
    /// `capacity` requests in all, with each model's requests waiting and
    /// generating.
    fn update(
        capacity: Option<u64>,
        models: &[(&str, Option<u64>, Option<u64>)],
    ) -> ServingLoadUpdate {
        ServingLoadUpdate {
            window_ms: 2000,
            pools: vec![PoolServingLoad {
                deployment: Some(PoolDeploymentStatus {
                    live_workers: Some(8),
                    max_concurrency: capacity,
                }),
                ..Default::default()
            }],
            models: models
                .iter()
                .map(|(name, awaiting, generating)| ModelServingLoad {
                    canonical_model_id: (*name).to_owned(),
                    load: Some(LoadView {
                        requests: Some(RequestLifecycleStats {
                            requests_awaiting_first_token: *awaiting,
                            requests_generating: *generating,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn pool_id() -> PoolId {
        V1PoolIdDeriver.derive(&PoolKey::new("ohio", "dynamo", "blue").unwrap())
    }

    fn relay_identity(runtime_id: u64) -> RelayIdentity {
        RelayIdentity {
            drt_instance_id: runtime_id,
            relay_incarnation: 1,
        }
    }

    fn serving_update(runtime_id: u64) -> ServingLoadUpdate {
        ServingLoadUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            contract_marker: RELAY_CONTRACT_MARKER,
            relay: Some(relay_identity(runtime_id)),
            window_sequence: 1,
            ..update(Some(128), &[("qwen", Some(3), Some(20))])
        }
    }

    fn kv_update(runtime_id: u64) -> KvPoolLoadUpdate {
        KvPoolLoadUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            contract_marker: RELAY_CONTRACT_MARKER,
            relay: Some(relay_identity(runtime_id)),
            window_sequence: 1,
            window_ms: 2000,
            pools: vec![KvPoolLoadEntry {
                kv_used_blocks: 400,
                total_kv_blocks: 1000,
                kv_observed_ranks: 1,
                kv_expected_ranks: 1,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn rejects_invalid_load_envelopes_before_recording_state() {
        let pool = pool_id();
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()));
        let mut load = serving_update(1);
        load.contract_marker = 0;
        assert!(record_serving_update(&pool, "qwen", &signals, &load).is_err());
        load = serving_update(1);
        load.protocol_version += 1;
        assert!(record_serving_update(&pool, "qwen", &signals, &load).is_err());
        assert_eq!(signals.occupancy(&pool), None);

        record_serving_update(&pool, "qwen", &signals, &serving_update(1)).unwrap();
        let mut kv = kv_update(1);
        kv.contract_marker = 0;
        assert!(record_kv_update(&pool, &signals, &kv).is_err());
        kv = kv_update(1);
        kv.protocol_version += 1;
        assert!(record_kv_update(&pool, &signals, &kv).is_err());
        assert_eq!(signals.occupancy(&pool).unwrap().kv, None);

        record_kv_update(&pool, &signals, &kv_update(1)).unwrap();
        assert!(signals.occupancy(&pool).unwrap().kv.is_some());
    }

    #[test]
    fn keeps_separate_streams_on_the_same_relay_replica() {
        let pool = pool_id();
        let signals = PoolSignals::new(Arc::new(InFlightTracker::new()));
        record_serving_update(&pool, "qwen", &signals, &serving_update(1)).unwrap();
        record_kv_update(&pool, &signals, &kv_update(2)).unwrap();
        assert_eq!(signals.occupancy(&pool).unwrap().kv, None);

        record_kv_update(&pool, &signals, &kv_update(1)).unwrap();
        assert!(signals.occupancy(&pool).unwrap().kv.is_some());
    }

    #[test]
    fn counts_waiting_and_generating_requests_against_capacity() {
        let report = pool_report(&update(Some(128), &[("qwen", Some(3), Some(20))]), "qwen");
        assert_eq!(
            report,
            Some(LoadReport {
                in_flight: 23,
                generating: 20,
                max_concurrency: Some(128),
                window: Duration::from_secs(2),
            })
        );
        // A Relay naming the model differently still reports its pool.
        let report = pool_report(
            &update(Some(128), &[("Qwen/Qwen3", Some(1), Some(1))]),
            "qwen",
        );
        assert_eq!(report.map(|report| report.in_flight), Some(2));
        // Without the workers' capacity the counts still arrive.
        let report = pool_report(&update(None, &[("qwen", Some(1), Some(1))]), "qwen");
        assert_eq!(report.map(|report| report.max_concurrency), Some(None));
    }

    #[test]
    fn incomplete_views_are_unknown_not_zero() {
        assert_eq!(
            pool_report(&update(Some(128), &[("qwen", None, Some(1))]), "qwen"),
            None
        );
        assert_eq!(
            pool_report(&update(Some(128), &[("qwen", Some(1), None)]), "qwen"),
            None
        );
        assert_eq!(pool_report(&update(Some(128), &[]), "qwen"), None);
    }

    #[test]
    fn kv_load_sums_producers_and_is_unknown_when_incomplete() {
        let entry = |used, total, observed, expected| KvPoolLoadEntry {
            kv_used_blocks: used,
            total_kv_blocks: total,
            kv_observed_ranks: observed,
            kv_expected_ranks: expected,
            ..Default::default()
        };
        let update = |pools| KvPoolLoadUpdate {
            pools,
            ..Default::default()
        };
        assert_eq!(
            pool_kv(&update(vec![entry(10, 100, 4, 4), entry(5, 100, 4, 4)])),
            Some((15, 200))
        );
        // A rank not yet heard from, or no capacity, is not zero load.
        assert_eq!(pool_kv(&update(vec![entry(10, 100, 3, 4)])), None);
        assert_eq!(pool_kv(&update(vec![entry(0, 0, 4, 4)])), None);
        assert_eq!(pool_kv(&update(vec![])), None);
    }

    type TestStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

    struct MockRelay {
        serving: ServingLoadUpdate,
        kv: KvPoolLoadUpdate,
        peers: Arc<Mutex<Vec<SocketAddr>>>,
    }

    fn open_stream<T: Send + 'static>(value: T) -> TestStream<T> {
        Box::pin(
            futures::stream::once(async move { Ok(value) })
                .chain(futures::stream::pending::<Result<T, Status>>()),
        )
    }

    #[tonic::async_trait]
    impl proto::KvEventRelay for MockRelay {
        type WatchKvPoolCatalogStream = TestStream<proto::KvPoolCatalogUpdate>;
        type SubscribeKvPoolStream = TestStream<proto::FilterUpdate>;
        type SubscribeServingReadinessStream = TestStream<proto::ServingReadinessUpdate>;
        type SubscribeKvPoolLoadStream = TestStream<KvPoolLoadUpdate>;
        type SubscribeServingLoadStream = TestStream<ServingLoadUpdate>;

        async fn get_relay_info(
            &self,
            _: Request<proto::RelayInfoRequest>,
        ) -> Result<Response<proto::RelayInfo>, Status> {
            Err(Status::unimplemented("unused in this test"))
        }

        async fn watch_kv_pool_catalog(
            &self,
            _: Request<proto::WatchKvPoolCatalogRequest>,
        ) -> Result<Response<Self::WatchKvPoolCatalogStream>, Status> {
            Err(Status::unimplemented("unused in this test"))
        }

        async fn subscribe_kv_pool(
            &self,
            _: Request<proto::SubscribeKvPoolRequest>,
        ) -> Result<Response<Self::SubscribeKvPoolStream>, Status> {
            Err(Status::unimplemented("unused in this test"))
        }

        async fn subscribe_serving_readiness(
            &self,
            _: Request<proto::SubscribeServingReadinessRequest>,
        ) -> Result<Response<Self::SubscribeServingReadinessStream>, Status> {
            Err(Status::unimplemented("unused in this test"))
        }

        async fn subscribe_serving_load(
            &self,
            request: Request<proto::SubscribeServingLoadRequest>,
        ) -> Result<Response<Self::SubscribeServingLoadStream>, Status> {
            self.peers
                .lock()
                .unwrap()
                .push(request.remote_addr().unwrap());
            Ok(Response::new(open_stream(self.serving.clone())))
        }

        async fn subscribe_kv_pool_load(
            &self,
            request: Request<proto::SubscribeKvPoolLoadRequest>,
        ) -> Result<Response<Self::SubscribeKvPoolLoadStream>, Status> {
            self.peers
                .lock()
                .unwrap()
                .push(request.remote_addr().unwrap());
            Ok(Response::new(open_stream(self.kv.clone())))
        }
    }

    #[tokio::test]
    async fn grpc_load_streams_share_a_connection_and_record_matching_replica() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async move {
            listener
                .accept()
                .await
                .ok()
                .map(|(stream, _)| (Ok::<_, std::io::Error>(stream), listener))
        });
        let peers = Arc::new(Mutex::new(Vec::new()));
        let server_cancel = CancellationToken::new();
        let server = {
            let cancel = server_cancel.clone();
            let relay = MockRelay {
                serving: serving_update(7),
                kv: kv_update(7),
                peers: peers.clone(),
            };
            tokio::spawn(async move {
                Server::builder()
                    .add_service(proto::KvEventRelayServer::new(relay))
                    .serve_with_incoming_shutdown(incoming, cancel.cancelled())
                    .await
                    .unwrap();
            })
        };

        let pool = pool_id();
        let signals = Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new())));
        let url = Url::parse(&format!("http://{address}")).unwrap();
        let client = {
            let pool = pool.clone();
            let signals = signals.clone();
            tokio::spawn(async move { stream_pool(&pool, "qwen", &url, &signals).await })
        };
        let occupancy = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(occupancy) = signals.occupancy(&pool)
                    && occupancy.kv.is_some()
                {
                    break occupancy;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(occupancy.slots, 23.0 / 128.0);
        assert_eq!(occupancy.kv, Some(0.46));
        assert_eq!(occupancy.load, 0.46);
        {
            let peers = peers.lock().unwrap();
            assert_eq!(peers.len(), 2);
            assert_eq!(peers[0], peers[1]);
        }

        client.abort();
        let _ = client.await;
        server_cancel.cancel();
        let mut server = server;
        match tokio::time::timeout(Duration::from_secs(5), &mut server).await {
            Ok(result) => result.unwrap(),
            Err(_) => {
                server.abort();
                panic!("mock Relay server did not stop");
            }
        }
    }
}
