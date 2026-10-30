// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_global_discovery_client::GlobalDiscoveryClient;
use dynamo_global_router::Pool;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::PoolSnapshot;

/// Candidate pools from complete GDS lists. A failed or partial read keeps the
/// last list; Relay liveness independently decides whether any candidate is routable.
pub fn discover_pools(
    client: GlobalDiscoveryClient,
    interval: Duration,
    cancel: CancellationToken,
) -> watch::Receiver<Arc<PoolSnapshot>> {
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::default()));
    tokio::spawn(async move {
        let mut ticks = tokio::time::interval(interval);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last_error = None;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = ticks.tick() => {},
            }
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = client.pools() => result,
            };
            match result {
                Ok(listed) if listed.complete => {
                    if last_error.take().is_some() {
                        tracing::info!("pool discovery recovered");
                    }
                    let pools = listed
                        .pools
                        .into_iter()
                        .filter_map(|pool| {
                            let model = pool.model?;
                            Some(Pool::new(
                                &pool.key,
                                &model,
                                &pool.location.region,
                                pool.frontends,
                            ))
                        })
                        .collect();
                    let next = Arc::new(PoolSnapshot::new(pools));
                    sender.send_if_modified(|current| {
                        if **current == *next {
                            return false;
                        }
                        *current = next;
                        true
                    });
                }
                result => {
                    let error = match result {
                        Ok(_) => "pool discovery returned a partial list".to_owned(),
                        Err(error) => error.to_string(),
                    };
                    if last_error.as_ref() != Some(&error) {
                        tracing::warn!(%error, "pool discovery failed; keeping the last candidate list");
                        last_error = Some(error);
                    }
                }
            }
        }
    });
    receiver
}
