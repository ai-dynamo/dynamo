// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool load from each pool's KV DC Relay.
//!
//! Two tasks per pool follow its Relay: one subscribes to the serving-load
//! stream and records the pool's requests in flight and how many it can run
//! at once; the other subscribes to the KV-load stream and records the KV
//! blocks its workers hold. Both go to [`PoolSignals`]. A pool's Relay comes
//! from the catalog; when it changes or the pool leaves, the subscription is
//! replaced or ends.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use dynamo_global_common::PoolId;
use dynamo_global_router::{LoadReport, PoolSignals};
use dynamo_llm::kv_dc_relay::wan::grpc::protocol::{
    KvEventRelayClient, KvPoolLoadUpdate, RELAY_CONTRACT_MARKER, ServingLoadUpdate,
    SubscribeKvPoolLoadRequest, SubscribeServingLoadRequest,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};
use url::Url;

use crate::PoolSnapshot;

const SUBSCRIBER_ID: &str = "global-frontend";
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
    let mut subscriptions = HashMap::<PoolId, (Url, CancellationToken)>::new();
    loop {
        let snapshot = pools.borrow_and_update().clone();
        let wanted = snapshot
            .pools()
            .iter()
            .filter_map(|pool| Some((pool.id.clone(), (pool.relay.clone()?, pool.model.clone()))))
            .collect::<HashMap<_, _>>();
        subscriptions.retain(|pool, (relay, subscription)| {
            let keep = wanted.get(pool).is_some_and(|(want, _)| want == relay);
            if !keep {
                subscription.cancel();
                signals.forget_load(pool);
                signals.forget_kv(pool);
            }
            keep
        });
        for (pool, (relay, model)) in wanted {
            if subscriptions.contains_key(&pool) {
                continue;
            }
            let subscription = cancel.child_token();
            for feed in [Feed::Load { model }, Feed::Kv] {
                tokio::spawn(subscribe(
                    pool.clone(),
                    feed,
                    relay.clone(),
                    signals.clone(),
                    subscription.clone(),
                ));
            }
            subscriptions.insert(pool, (relay, subscription));
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
    feed: Feed,
    relay: Url,
    signals: Arc<PoolSignals>,
    cancel: CancellationToken,
) {
    let mut retry = FIRST_RETRY;
    loop {
        let started = Instant::now();
        let result = match &feed {
            Feed::Load { model } => stream_load(&pool, model, &relay, &signals, &cancel).await,
            Feed::Kv => stream_kv(&pool, &relay, &signals, &cancel).await,
        };
        match result {
            Ok(()) => return,
            Err(error) => {
                tracing::warn!(%pool, %relay, error = format!("{error:#}"), "Relay stream ended");
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
    model: &str,
    relay: &Url,
    signals: &PoolSignals,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut updates = KvEventRelayClient::new(connect(relay).await?)
        .subscribe_serving_load(SubscribeServingLoadRequest {
            subscriber_id: SUBSCRIBER_ID.to_owned(),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("subscribe")?
        .into_inner();
    tracing::info!(%pool, %relay, "following the pool's Relay load");
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
                tracing::debug!(%pool, "the Relay's load is incomplete");
                signals.forget_load(pool);
            }
        }
    }
}

/// Records each KV-load update until the stream fails, or returns `Ok` when
/// `cancel` fires.
async fn stream_kv(
    pool: &PoolId,
    relay: &Url,
    signals: &PoolSignals,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut updates = KvEventRelayClient::new(connect(relay).await?)
        .subscribe_kv_pool_load(SubscribeKvPoolLoadRequest {
            subscriber_id: SUBSCRIBER_ID.to_owned(),
            contract_marker: RELAY_CONTRACT_MARKER,
        })
        .await
        .context("subscribe")?
        .into_inner();
    tracing::info!(%pool, %relay, "following the pool's Relay KV load");
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
                tracing::debug!(%pool, "the Relay's KV load is incomplete");
                signals.forget_kv(pool);
            }
        }
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
    })
}

#[cfg(test)]
mod tests {
    use dynamo_llm::kv_dc_relay::wan::grpc::protocol::{
        KvPoolLoadEntry, LoadView, ModelServingLoad, PoolDeploymentStatus, PoolServingLoad,
        RequestLifecycleStats,
    };

    use super::*;

    /// An update from a pool of 8 workers that can run 16 requests each
    /// (unless `capacity` is `None`), with each model's requests waiting and
    /// generating.
    fn update(
        capacity: Option<u64>,
        models: &[(&str, Option<u64>, Option<u64>)],
    ) -> ServingLoadUpdate {
        ServingLoadUpdate {
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

    #[test]
    fn counts_waiting_and_generating_requests_against_capacity() {
        let report = pool_report(&update(Some(128), &[("qwen", Some(3), Some(20))]), "qwen");
        assert_eq!(
            report,
            Some(LoadReport {
                in_flight: 23,
                generating: 20,
                max_concurrency: Some(128),
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
}
