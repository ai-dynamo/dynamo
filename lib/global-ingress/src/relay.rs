// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool load from each pool's Relay.
//!
//! One task per pool subscribes to its Relay's serving-load stream and
//! records the pool's requests in flight, live workers, and the counts its
//! expected wait comes from in [`PoolSignals`]. A pool's Relay address comes from the catalog; when it
//! changes or the pool leaves, the subscription is replaced or ends.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use dynamo_global_router::{LoadReport, PoolLoad, PoolSignals, PoolSnapshot};
use dynamo_global_router_shared::PoolId;
use dynamo_llm::pool_relay::wan::grpc::protocol::{
    KvEventRelayClient, RELAY_CONTRACT_MARKER, ServingLoadUpdate, SubscribeServingLoadRequest,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;

const SUBSCRIBER_ID: &str = "global-ingress";
const FIRST_RETRY: Duration = Duration::from_millis(500);
const LAST_RETRY: Duration = Duration::from_secs(30);
/// A stream that lasted this long resets the retry delay.
const HEALTHY_STREAM: Duration = Duration::from_secs(30);

/// Keeps one Relay subscription per pool in `pools` until `cancel` fires.
pub async fn follow_relays(
    mut pools: watch::Receiver<Arc<PoolSnapshot>>,
    signals: Arc<PoolSignals>,
    port: u16,
    cancel: CancellationToken,
) {
    let mut subscriptions = HashMap::<PoolId, (SocketAddr, CancellationToken)>::new();
    loop {
        let snapshot = pools.borrow_and_update().clone();
        let wanted = snapshot
            .pools()
            .iter()
            .filter_map(|pool| {
                let address = SocketAddr::new(pool.relay?, port);
                Some((
                    pool.id.clone(),
                    (address, pool.name.clone(), pool.model.clone()),
                ))
            })
            .collect::<HashMap<_, _>>();
        subscriptions.retain(|pool, (address, subscription)| {
            let keep = wanted.get(pool).is_some_and(|(want, _, _)| want == address);
            if !keep {
                subscription.cancel();
                signals.forget_load(pool);
            }
            keep
        });
        for (pool, (address, name, model)) in wanted {
            if subscriptions.contains_key(&pool) {
                continue;
            }
            let subscription = cancel.child_token();
            tokio::spawn(subscribe(
                pool.clone(),
                name,
                model,
                address,
                signals.clone(),
                subscription.clone(),
            ));
            subscriptions.insert(pool, (address, subscription));
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            changed = pools.changed() => if changed.is_err() { return },
        }
    }
}

/// Streams one pool's load, reconnecting with a growing delay.
async fn subscribe(
    pool: PoolId,
    name: String,
    model: String,
    address: SocketAddr,
    signals: Arc<PoolSignals>,
    cancel: CancellationToken,
) {
    let mut retry = FIRST_RETRY;
    loop {
        let started = Instant::now();
        match stream_load(&pool, &name, &model, address, &signals, &cancel).await {
            Ok(()) => return,
            Err(error) => {
                tracing::warn!(pool = %name, %address, error = format!("{error:#}"), "Relay load stream ended");
            }
        }
        signals.forget_load(&pool);
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

/// Records each load update until the stream fails, or returns `Ok` when
/// `cancel` fires.
async fn stream_load(
    pool: &PoolId,
    name: &str,
    model: &str,
    address: SocketAddr,
    signals: &PoolSignals,
    cancel: &CancellationToken,
) -> Result<()> {
    let channel = Endpoint::from_shared(format!("http://{address}"))?
        .connect_timeout(Duration::from_secs(3))
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_timeout(Duration::from_secs(5))
        .connect()
        .await
        .context("connect")?;
    let mut updates = KvEventRelayClient::new(channel)
        .subscribe_serving_load(SubscribeServingLoadRequest {
            subscriber_id: SUBSCRIBER_ID.to_owned(),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("subscribe")?
        .into_inner();
    tracing::info!(pool = %name, %address, "following the pool's Relay load");
    loop {
        let update = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            update = updates.message() => update.context("read")?,
        };
        let Some(update) = update else {
            bail!("the Relay closed the stream");
        };
        match pool_report(&update, model) {
            Some(report) => signals.record_load(pool, report),
            None => {
                tracing::debug!(pool = %name, "the Relay's load is incomplete");
                signals.forget_load(pool);
            }
        }
    }
}

/// The pool's requests in flight (waiting for a first token or generating)
/// for `model`, its live workers, the requests waiting and started so far,
/// and when the Relay observed them. `None` unless every part is reported:
/// the Relay leaves a value unset when its view is incomplete, and unset
/// does not mean zero. A Relay watching one pool reports one model; a model
/// it names differently from the catalog still counts.
fn pool_report(update: &ServingLoadUpdate, model: &str) -> Option<LoadReport> {
    let workers = update
        .pools
        .iter()
        .map(|pool| pool.deployment.as_ref()?.live_workers)
        .sum::<Option<u64>>()?;
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
    let (mut in_flight, mut awaiting, mut started_total) = (0, 0, 0);
    for entry in models {
        let requests = entry.load.as_ref()?.requests.as_ref()?;
        let waiting = requests.requests_awaiting_first_token?;
        in_flight += waiting + requests.requests_generating?;
        awaiting += waiting;
        started_total += requests.requests_started_total;
    }
    Some(LoadReport {
        load: PoolLoad { in_flight, workers },
        awaiting,
        started_total,
        observed_ms: update.observed_ms,
    })
}

#[cfg(test)]
mod tests {
    use dynamo_llm::pool_relay::wan::grpc::protocol::{
        LoadView, ModelServingLoad, PoolDeploymentStatus, PoolServingLoad, RequestLifecycleStats,
    };

    use super::*;

    /// An update observed at 5 s, with each model's requests waiting,
    /// generating, and started so far.
    fn update(
        workers: Option<u64>,
        models: &[(&str, Option<u64>, Option<u64>, u64)],
    ) -> ServingLoadUpdate {
        ServingLoadUpdate {
            observed_ms: 5000,
            pools: vec![PoolServingLoad {
                deployment: Some(PoolDeploymentStatus {
                    live_workers: workers,
                    max_concurrency: None,
                }),
                ..Default::default()
            }],
            models: models
                .iter()
                .map(|(name, awaiting, generating, started)| ModelServingLoad {
                    canonical_model_id: (*name).to_owned(),
                    load: Some(LoadView {
                        requests: Some(RequestLifecycleStats {
                            requests_started_total: *started,
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

    #[test]
    fn counts_waiting_and_generating_requests_over_live_workers() {
        let report = pool_report(
            &update(Some(8), &[("qwen", Some(3), Some(20), 900)]),
            "qwen",
        );
        assert_eq!(
            report,
            Some(LoadReport {
                load: PoolLoad {
                    in_flight: 23,
                    workers: 8
                },
                awaiting: 3,
                started_total: 900,
                observed_ms: 5000,
            })
        );
        // A Relay naming the model differently still reports its pool.
        let report = pool_report(
            &update(Some(8), &[("Qwen/Qwen3", Some(1), Some(1), 7)]),
            "qwen",
        );
        assert_eq!(report.map(|report| report.load.in_flight), Some(2));
    }

    #[test]
    fn incomplete_views_are_unknown_not_zero() {
        assert_eq!(
            pool_report(&update(None, &[("qwen", Some(1), Some(1), 0)]), "qwen"),
            None
        );
        assert_eq!(
            pool_report(&update(Some(8), &[("qwen", None, Some(1), 0)]), "qwen"),
            None
        );
        assert_eq!(pool_report(&update(Some(8), &[]), "qwen"), None);
    }
}
