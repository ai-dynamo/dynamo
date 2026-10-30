// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-initiated subscriptions to each pool's PoolRelay.
//!
//! One task per pool follows catalog, readiness, and load over the same connection.
//! Both connection directions apply state through the same replica-scoped views.
//! The Relay address comes from the catalog; when it changes or the pool
//! leaves, the subscription ends and the pool's load is dropped.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use dynamo_global_common::PoolId;
use dynamo_global_router::RelaySource;
use dynamo_llm::pool_relay::wan::grpc::protocol::{
    KvEventRelayClient, RELAY_CONTRACT_MARKER, SubscribeKvPoolLoadRequest,
    SubscribeServingLoadRequest, SubscribeServingReadinessRequest, WatchKvPoolCatalogRequest,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tonic::transport::{Channel, Endpoint};
use url::Url;

use crate::{PoolSnapshot, RelayGatedPoolCatalog, RelayPoolState, RelaySourceKey};

const SUBSCRIBER_ID: &str = "global-frontend";
const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(5);
const FIRST_RETRY: Duration = Duration::from_millis(500);
const LAST_RETRY: Duration = Duration::from_secs(30);
/// A stream that lasted this long resets the retry delay.
const HEALTHY_STREAM: Duration = Duration::from_secs(30);
/// Keeps the Relay subscriptions of every pool in `pools` that names a
/// Relay, until `cancel` fires.
pub async fn follow_relays(
    mut pools: watch::Receiver<Arc<PoolSnapshot>>,
    catalog: Arc<RelayGatedPoolCatalog>,
    cancel: CancellationToken,
) {
    let mut subscriptions =
        HashMap::<PoolId, (Url, String, CancellationToken, AbortOnDropHandle<()>)>::new();
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
        }
        for (pool, (relay, model)) in wanted {
            if subscriptions.contains_key(&pool) {
                continue;
            }
            let subscription = cancel.child_token();
            let task = tokio::spawn(subscribe(
                pool.clone(),
                relay.clone(),
                Arc::clone(&catalog),
                subscription.clone(),
            ));
            subscriptions.insert(
                pool,
                (relay, model, subscription, AbortOnDropHandle::new(task)),
            );
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            changed = pools.changed() => if changed.is_err() { return },
        }
    }
}

async fn subscribe(
    pool: PoolId,
    relay: Url,
    catalog: Arc<RelayGatedPoolCatalog>,
    cancel: CancellationToken,
) {
    let mut retry = FIRST_RETRY;
    loop {
        let started = Instant::now();
        let stream = stream_pool(&pool, &relay, &catalog);
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

/// All subscriptions share one connection so a Service selects one Relay Pod.
/// Losing any stream withdraws this replica's view before reconnecting.
async fn stream_pool(
    pool: &PoolId,
    relay: &Url,
    catalog: &RelayGatedPoolCatalog,
) -> Result<Infallible> {
    let channel = connect(relay).await?;
    let mut catalog_client = KvEventRelayClient::new(channel.clone());
    let mut readiness_client = KvEventRelayClient::new(channel.clone());
    let mut load_client = KvEventRelayClient::new(channel.clone());
    let mut kv_client = KvEventRelayClient::new(channel);
    let (catalog_stream, readiness, load, kv) = tokio::time::timeout(SUBSCRIBE_TIMEOUT, async {
        tokio::try_join!(
            catalog_client.watch_kv_pool_catalog(WatchKvPoolCatalogRequest {
                subscriber_id: SUBSCRIBER_ID.into(),
                contract_marker: RELAY_CONTRACT_MARKER
            }),
            readiness_client.subscribe_serving_readiness(SubscribeServingReadinessRequest {
                subscriber_id: SUBSCRIBER_ID.into(),
                contract_marker: RELAY_CONTRACT_MARKER
            }),
            load_client.subscribe_serving_load(SubscribeServingLoadRequest {
                subscriber_id: SUBSCRIBER_ID.into(),
                contract_marker: RELAY_CONTRACT_MARKER
            }),
            kv_client.subscribe_kv_pool_load(SubscribeKvPoolLoadRequest {
                subscriber_id: SUBSCRIBER_ID.into(),
                contract_marker: RELAY_CONTRACT_MARKER
            }),
        )
    })
    .await
    .map_err(|_| anyhow!("subscribe timed out"))?
    .context("subscribe")?;
    let mut catalogs = catalog_stream.into_inner();
    let mut readiness_updates = readiness.into_inner();
    let mut load_updates = load.into_inner();
    let mut kv_updates = kv.into_inner();
    let (initial_catalog, initial_readiness) = tokio::time::timeout(SUBSCRIBE_TIMEOUT, async {
        tokio::try_join!(catalogs.message(), readiness_updates.message())
    })
    .await
    .map_err(|_| anyhow!("initial Relay state timed out"))??;
    let mut current_catalog =
        Arc::new(initial_catalog.context("Relay closed catalog before initial state")?);
    let mut current_readiness =
        initial_readiness.context("Relay closed readiness before initial state")?;
    let identity = current_catalog
        .relay
        .as_ref()
        .context("missing Relay identity")?;
    let source = RelaySource {
        runtime_id: identity.drt_instance_id,
        incarnation: identity.relay_incarnation,
    };
    let mut serving_load = None;
    let mut kv_load = None;
    let initial = RelayPoolState::from_updates(
        Arc::clone(&current_catalog),
        &current_readiness,
        None,
        None,
        source,
    )?;
    let mut view = catalog.open(RelaySourceKey {
        pool: pool.clone(),
        service: pool.to_string(),
        replica: source.runtime_id.to_string(),
    });
    let mut revision = 0_u64;
    view.replace_state_snapshot(revision, [(pool.clone(), Some(Arc::new(initial)))])?;
    tracing::info!(%pool, %relay, "following the pool's Relay state");
    loop {
        tokio::select! {
            update = catalogs.message() => {
                current_catalog = Arc::new(update.context("read catalog")?.context("Relay closed catalog")?);
            }
            update = readiness_updates.message() => {
                current_readiness = update.context("read readiness")?.context("Relay closed readiness")?;
            }
            update = load_updates.message() => {
                serving_load = Some(Arc::new(update.context("read serving load")?.context("Relay closed serving load")?));
            }
            update = kv_updates.message() => {
                kv_load = Some(Arc::new(update.context("read KV load")?.context("Relay closed KV load")?));
            }
        }
        let state = RelayPoolState::from_updates(
            Arc::clone(&current_catalog),
            &current_readiness,
            serving_load.clone(),
            kv_load.clone(),
            source,
        )?;
        revision = revision
            .checked_add(1)
            .context("Relay receiving revision exhausted")?;
        view.upsert_state(revision, pool.clone(), Some(Arc::new(state)))?;
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
