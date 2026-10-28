// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool load from each pool's Relay.
//!
//! Two tasks per pool follow its Relay: one subscribes to the serving-load
//! stream and records the pool's requests in flight, live workers, capacity,
//! and the counts its timing comes from; the other subscribes to the KV-load
//! stream and records the KV blocks its workers hold. Both go to
//! [`PoolSignals`]. A pool's Relay address comes from the catalog; when it
//! changes or the pool leaves, the subscription is replaced or ends.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use dynamo_global_router::{LoadReport, PoolLoad, PoolSignals, PoolSnapshot};
use dynamo_global_router_shared::PoolId;
use dynamo_llm::pool_relay::wan::grpc::protocol::{
    KvEventRelayClient, KvPoolLoadUpdate, RELAY_CONTRACT_MARKER, ServingLoadUpdate,
    SubscribeKvPoolLoadRequest, SubscribeServingLoadRequest,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};

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
                signals.forget_kv(pool);
            }
            keep
        });
        for (pool, (address, name, model)) in wanted {
            if subscriptions.contains_key(&pool) {
                continue;
            }
            let subscription = cancel.child_token();
            for feed in [Feed::Load { model }, Feed::Kv] {
                tokio::spawn(subscribe(
                    pool.clone(),
                    name.clone(),
                    feed,
                    address,
                    signals.clone(),
                    subscription.clone(),
                ));
            }
            subscriptions.insert(pool, (address, subscription));
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            changed = pools.changed() => if changed.is_err() { return },
        }
    }
}

/// Which of a Relay's streams a task follows.
enum Feed {
    /// Serving load, for the catalog's `model`.
    Load { model: String },
    /// The KV blocks the pool's workers hold.
    Kv,
}

/// Streams one pool's `feed`, reconnecting with a growing delay.
async fn subscribe(
    pool: PoolId,
    name: String,
    feed: Feed,
    address: SocketAddr,
    signals: Arc<PoolSignals>,
    cancel: CancellationToken,
) {
    let mut retry = FIRST_RETRY;
    loop {
        let started = Instant::now();
        let result = match &feed {
            Feed::Load { model } => {
                stream_load(&pool, &name, model, address, &signals, &cancel).await
            }
            Feed::Kv => stream_kv(&pool, &name, address, &signals, &cancel).await,
        };
        match result {
            Ok(()) => return,
            Err(error) => {
                tracing::warn!(pool = %name, %address, error = format!("{error:#}"), "Relay stream ended");
            }
        }
        match feed {
            Feed::Load { .. } => signals.forget_load(&pool),
            Feed::Kv => signals.forget_kv(&pool),
        }
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
    let mut updates = KvEventRelayClient::new(connect(address).await?)
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

/// Records each KV-load update until the stream fails, or returns `Ok` when
/// `cancel` fires.
async fn stream_kv(
    pool: &PoolId,
    name: &str,
    address: SocketAddr,
    signals: &PoolSignals,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut updates = KvEventRelayClient::new(connect(address).await?)
        .subscribe_kv_pool_load(SubscribeKvPoolLoadRequest {
            subscriber_id: SUBSCRIBER_ID.to_owned(),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("subscribe")?
        .into_inner();
    tracing::info!(pool = %name, %address, "following the pool's Relay KV load");
    loop {
        let update = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            update = updates.message() => update.context("read")?,
        };
        let Some(update) = update else {
            bail!("the Relay closed the stream");
        };
        match pool_kv(&update) {
            Some((used, total)) => signals.record_kv(pool, used, total),
            None => {
                tracing::debug!(pool = %name, "the Relay's KV load is incomplete");
                signals.forget_kv(pool);
            }
        }
    }
}

async fn connect(address: SocketAddr) -> Result<Channel> {
    Endpoint::from_shared(format!("http://{address}"))?
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
/// for `model`, its live workers, the requests waiting, the requests started
/// and completed and the output tokens produced so far, and when the Relay
/// observed them. `None` unless every part is reported:
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
    let (mut in_flight, mut awaiting) = (0, 0);
    let (mut started_total, mut completed_total, mut output_tokens_total) = (0, 0, 0);
    for entry in models {
        let load = entry.load.as_ref()?;
        let requests = load.requests.as_ref()?;
        let waiting = requests.requests_awaiting_first_token?;
        in_flight += waiting + requests.requests_generating?;
        awaiting += waiting;
        started_total += requests.requests_started_total;
        completed_total += requests.requests_completed_total;
        output_tokens_total += load.tokens.as_ref()?.output_tokens_total?;
    }
    let max_concurrency = update
        .pools
        .iter()
        .map(|pool| pool.deployment.as_ref()?.max_concurrency)
        .sum::<Option<u64>>();
    Some(LoadReport {
        load: PoolLoad { in_flight, workers },
        awaiting,
        started_total,
        completed_total,
        output_tokens_total,
        max_concurrency,
        observed_ms: update.observed_ms,
    })
}

#[cfg(test)]
mod tests {
    use dynamo_llm::pool_relay::wan::grpc::protocol::{
        KvPoolLoadEntry, LoadView, ModelServingLoad, PoolDeploymentStatus, PoolServingLoad,
        RequestLifecycleStats, TokenLoadStats,
    };

    use super::*;

    /// An update observed at 5 s, with each model's requests waiting,
    /// generating, and started so far; half the started requests have
    /// completed, with 10 output tokens each.
    fn update(
        workers: Option<u64>,
        models: &[(&str, Option<u64>, Option<u64>, u64)],
    ) -> ServingLoadUpdate {
        ServingLoadUpdate {
            observed_ms: 5000,
            pools: vec![PoolServingLoad {
                deployment: Some(PoolDeploymentStatus {
                    live_workers: workers,
                    max_concurrency: workers.map(|workers| workers * 16),
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
                            requests_completed_total: started / 2,
                            requests_awaiting_first_token: *awaiting,
                            requests_generating: *generating,
                            ..Default::default()
                        }),
                        tokens: Some(TokenLoadStats {
                            output_tokens_total: Some(started / 2 * 10),
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
                completed_total: 450,
                output_tokens_total: 4500,
                max_concurrency: Some(128),
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
        let mut without_tokens = update(Some(8), &[("qwen", Some(1), Some(1), 0)]);
        without_tokens.models[0].load.as_mut().unwrap().tokens = None;
        assert_eq!(pool_report(&without_tokens, "qwen"), None);
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
}
