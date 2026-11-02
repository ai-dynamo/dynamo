// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lists the other Global Frontends from Global Discovery, for the core's
//! replica sync. The service's `/routers` is the set of replicas as the
//! PoolRelays see it; each replica publishes its own API URL there.

use std::collections::BTreeSet;
use std::time::Duration;

use dynamo_global_discovery_client::GlobalDiscoveryClient;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

/// Publishes to `peers` the API URLs of every replica `discovery` lists but
/// `self_url`, polling `GET /routers` every `interval`, until `cancel`
/// fires or every receiver drops. Run it as a task.
///
/// The set is published only when it changes. A failed poll keeps the last
/// published set and is logged once per streak of failures, with the
/// recovery; a replica that cannot reach discovery keeps the peers it has.
pub async fn poll_replica_peers(
    discovery: GlobalDiscoveryClient,
    self_url: Option<Url>,
    interval: Duration,
    peers: watch::Sender<Vec<Url>>,
    cancel: CancellationToken,
) {
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut failing = false;
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = peers.closed() => return,
            _ = ticks.tick() => {}
        }
        let replicas = match discovery.routers().await {
            Ok(replicas) => replicas,
            Err(error) => {
                if !failing {
                    tracing::warn!(
                        error = bounded(format!("{error:#}")),
                        "listing the replicas from Global Discovery failed; keeping the last set"
                    );
                }
                failing = true;
                continue;
            }
        };
        if failing {
            tracing::info!("listing the replicas from Global Discovery succeeds again");
            failing = false;
        }
        // Sorted and deduplicated, so the same set always compares equal.
        let listed: Vec<Url> = replicas
            .into_iter()
            .filter(|replica| self_url.as_ref() != Some(&replica.url))
            .map(|replica| replica.url)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        peers.send_if_modified(|current| {
            if *current == listed {
                return false;
            }
            tracing::info!(
                replicas = ?listed.iter().map(Url::as_str).collect::<Vec<_>>(),
                "Global Discovery lists these other replicas"
            );
            *current = listed;
            true
        });
    }
}

const ERROR_LOG_BYTES: usize = 512;

/// At most `ERROR_LOG_BYTES` of `text`: a discovery error may carry the
/// response body, which the frontend does not control.
fn bounded(mut text: String) -> String {
    if let Some((cut, _)) = text.char_indices().nth(ERROR_LOG_BYTES) {
        text.truncate(cut);
        text.push('\u{2026}');
    }
    text
}
