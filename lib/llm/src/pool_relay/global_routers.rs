// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router replicas listed by the Global Discovery Service.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_global_discovery_client::{GlobalDiscoveryClient, RouterReplica};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

/// How often a Relay reads the Global Routers from the Global Discovery
/// Service. A read that takes longer is abandoned.
pub(crate) const GLOBAL_DISCOVERY_INTERVAL: Duration = Duration::from_secs(5);

/// The stale reason until the first read succeeds or fails.
pub(crate) const NO_SUCCESSFUL_READ: &str = "no successful read yet";

/// Where a PoolRelay learns the current Global Router replicas.
#[async_trait]
pub(crate) trait GlobalRouterSource: Send + Sync {
    /// Every Global Router replica the source knows now.
    async fn routers(&self) -> anyhow::Result<Vec<RouterReplica>>;
}

#[async_trait]
impl GlobalRouterSource for GlobalDiscoveryClient {
    async fn routers(&self) -> anyhow::Result<Vec<RouterReplica>> {
        Ok(GlobalDiscoveryClient::routers(self).await?)
    }
}

/// The Global Router replicas listed by the Global Discovery Service, and
/// whether that list is current.
pub(crate) struct GlobalRouters {
    routers: watch::Receiver<HashSet<RouterReplica>>,
    stale: watch::Receiver<Option<String>>,
}

impl GlobalRouters {
    /// The current set. The channel closes when polling stops.
    pub(crate) fn subscribe(&self) -> watch::Receiver<HashSet<RouterReplica>> {
        self.routers.clone()
    }

    /// Why the set may be out of date: the last read failed, or none has
    /// succeeded yet. The set is kept meanwhile, so router connections stay up.
    pub(crate) fn stale_reason(&self) -> Option<String> {
        self.stale.borrow().clone()
    }
}

/// Reads `source` now and then every `interval` until `cancel` fires. The set
/// starts empty, so nothing waits on the first read.
pub(crate) fn poll_global_routers(
    source: impl GlobalRouterSource + 'static,
    interval: Duration,
    cancel: CancellationToken,
) -> GlobalRouters {
    let (routers, routers_rx) = watch::channel(HashSet::new());
    let (stale, stale_rx) = watch::channel(Some(NO_SUCCESSFUL_READ.to_owned()));
    tokio::spawn(async move {
        let mut ticks = tokio::time::interval(interval);
        ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
        cancel
            .run_until_cancelled(async {
                loop {
                    ticks.tick().await;
                    refresh(&source, &routers, &stale).await;
                }
            })
            .await;
    });
    GlobalRouters {
        routers: routers_rx,
        stale: stale_rx,
    }
}

/// A failed read marks the set stale and keeps it, so a discovery outage does
/// not drop the routers already known. Each new failure reason and the
/// recovery are logged once; health reports the reason meanwhile.
async fn refresh(
    source: &impl GlobalRouterSource,
    routers: &watch::Sender<HashSet<RouterReplica>>,
    stale: &watch::Sender<Option<String>>,
) {
    let listed = match source.routers().await {
        Ok(listed) => listed.into_iter().collect::<HashSet<_>>(),
        Err(error) => {
            let error = format!("{error:#}");
            if stale.borrow().as_deref() != Some(error.as_str()) {
                tracing::warn!(%error, "cannot list Global Routers; keeping the last known routers");
            }
            stale.send_replace(Some(error));
            return;
        }
    };
    if stale
        .send_replace(None)
        .is_some_and(|reason| reason != NO_SUCCESSFUL_READ)
    {
        tracing::info!("listing Global Routers again");
    }
    routers.send_if_modified(|current| {
        if *current == listed {
            return false;
        }
        tracing::info!(routers = listed.len(), "Global Router set changed");
        *current = listed;
        true
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dynamo_global_discovery_client::RouterReplicaId;

    use super::*;

    /// A source whose answer the test sets; `None` is an outage.
    #[derive(Clone, Default)]
    struct FakeSource(Arc<Mutex<Option<Vec<RouterReplica>>>>);

    impl FakeSource {
        fn answer(&self, ids: Option<&[&str]>) {
            *self.0.lock().unwrap() = ids.map(|ids| ids.iter().map(|id| router(id)).collect());
        }
    }

    #[async_trait]
    impl GlobalRouterSource for FakeSource {
        async fn routers(&self) -> anyhow::Result<Vec<RouterReplica>> {
            self.0
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| anyhow::anyhow!("discovery is down"))
        }
    }

    fn router(id: &str) -> RouterReplica {
        RouterReplica {
            replica: RouterReplicaId(id.to_owned()),
            url: format!("http://{id}:9000/").parse().unwrap(),
        }
    }

    fn ids(routers: &HashSet<RouterReplica>) -> Vec<String> {
        let mut ids = routers
            .iter()
            .map(|router| router.replica.0.clone())
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }

    async fn next_set(routers: &mut watch::Receiver<HashSet<RouterReplica>>) -> Vec<String> {
        tokio::time::timeout(Duration::from_secs(5), routers.changed())
            .await
            .expect("the router set did not change")
            .unwrap();
        ids(&routers.borrow_and_update())
    }

    #[tokio::test(start_paused = true)]
    async fn keeps_routers_through_an_outage_and_reports_them_stale() {
        let source = FakeSource::default();
        source.answer(Some(&["a"]));
        let cancel = CancellationToken::new();
        let global_routers =
            poll_global_routers(source.clone(), Duration::from_secs(1), cancel.clone());
        let mut routers = global_routers.subscribe();
        assert_eq!(next_set(&mut routers).await, ["a"]);
        assert_eq!(global_routers.stale_reason(), None);

        source.answer(Some(&["a", "b"]));
        assert_eq!(next_set(&mut routers).await, ["a", "b"]);

        // An outage keeps the routers and marks them stale.
        source.answer(None);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(ids(&routers.borrow()), ["a", "b"]);
        assert_eq!(
            global_routers.stale_reason().as_deref(),
            Some("discovery is down")
        );

        // Recovery clears it, and an unchanged list publishes nothing.
        source.answer(Some(&["b", "a"]));
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(global_routers.stale_reason(), None);
        assert!(!routers.has_changed().unwrap());

        source.answer(Some(&["b"]));
        assert_eq!(next_set(&mut routers).await, ["b"]);

        // Stopping the task closes the channel, which stops the broadcaster.
        cancel.cancel();
        let closed = tokio::time::timeout(Duration::from_secs(5), routers.changed()).await;
        assert!(closed.expect("the channel did not close").is_err());
    }
}
