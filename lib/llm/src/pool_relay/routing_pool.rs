// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registers the Relay's routing pool with Global Discovery while the service
//! takes registrations. Routers learn what the pool serves from the Relay's
//! state streams, so a registration says only where the pool is.

use dynamo_global_common::{PoolKey, PoolLocation, PoolRegistration};
use dynamo_global_discovery_client::{GlobalDiscoveryClient, PoolHeartbeat, RegistrationStatus};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use url::Url;

/// Keeps the routing pool registered while Global Discovery takes
/// registrations, as its `GET /routers` says, until cancelled or dropped. A
/// service that finds pools through MCS takes none, and then nothing is
/// registered and nothing is missing.
pub(super) struct RoutingPoolRegistration {
    heartbeat: PoolHeartbeat,
    plan: watch::Receiver<Plan>,
    _follow: AbortOnDropHandle<()>,
}

/// Whether the pool is registered, as [`follow_service`] decides it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Plan {
    /// The pool should be registered but is not, for this reason.
    Withheld(String),
    /// The service takes no registrations.
    NotTaken,
    /// The heartbeat registers the pool.
    Registering,
}

impl RoutingPoolRegistration {
    /// `accepts` is the service's last answer to whether it takes
    /// registrations, `None` before the first. Without a `location` the
    /// pool cannot be registered.
    pub(super) fn spawn(
        client: GlobalDiscoveryClient,
        key: PoolKey,
        location: Option<PoolLocation>,
        frontends: Vec<Url>,
        accepts: watch::Receiver<Option<bool>>,
        cancel: CancellationToken,
    ) -> Self {
        let (registration, registration_rx) = watch::channel(None);
        let (plan, plan_rx) = watch::channel(Plan::Withheld(UNKNOWN.to_owned()));
        let registration_of = move |location: PoolLocation| PoolRegistration {
            key: key.clone(),
            location,
            frontends: frontends.clone(),
        };
        let follow = tokio::spawn(follow_service(
            location.map(registration_of),
            accepts,
            registration,
            plan,
            cancel.clone(),
        ));
        Self {
            heartbeat: PoolHeartbeat::spawn(client, registration_rx, cancel),
            plan: plan_rx,
            _follow: AbortOnDropHandle::new(follow),
        }
    }

    /// Why the pool is not registered though it should be, or `None` while
    /// it is registered or the service takes no registrations.
    pub(super) fn unregistered_reason(&self) -> Option<String> {
        match self.plan.borrow().clone() {
            Plan::Withheld(reason) => Some(reason),
            Plan::NotTaken => None,
            Plan::Registering => match self.heartbeat.status() {
                RegistrationStatus::Registered(_) => None,
                RegistrationStatus::Unregistered => {
                    Some("no registration has finished yet".to_owned())
                }
                RegistrationStatus::Failing(reason) => Some(reason),
            },
        }
    }
}

const UNKNOWN: &str = "Global Discovery has not said yet whether it takes pool registrations";

/// Sets `registration` to `pool` while the service takes registrations, and
/// to `None` otherwise, with the reason in `plan`. Withdraws the
/// registration when it stops.
async fn follow_service(
    pool: Option<PoolRegistration>,
    mut accepts: watch::Receiver<Option<bool>>,
    registration: watch::Sender<Option<PoolRegistration>>,
    plan: watch::Sender<Plan>,
    cancel: CancellationToken,
) {
    loop {
        let answer = *accepts.borrow_and_update();
        let (next, decided) = match (answer, &pool) {
            (None, _) => (None, Plan::Withheld(UNKNOWN.to_owned())),
            (Some(false), _) => (None, Plan::NotTaken),
            (Some(true), None) => (
                None,
                Plan::Withheld(
                    "Global Discovery takes pool registrations, but the routing pool has no \
                     region to register"
                        .to_owned(),
                ),
            ),
            (Some(true), Some(pool)) => (Some(pool.clone()), Plan::Registering),
        };
        if *plan.borrow() != decided {
            match &decided {
                Plan::NotTaken => tracing::info!(
                    "Global Discovery takes no pool registrations, as when it finds pools \
                     through MCS, so the routing pool is not registered"
                ),
                Plan::Registering => tracing::info!(
                    "Global Discovery takes pool registrations; registering the routing pool"
                ),
                Plan::Withheld(reason) if answer.is_some() => {
                    tracing::warn!(%reason, "the routing pool is not registered")
                }
                Plan::Withheld(_) => {}
            }
        }
        plan.send_replace(decided);
        registration.send_if_modified(|current| {
            if *current == next {
                return false;
            }
            *current = next;
            true
        });
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = registration.closed() => break,
            changed = accepts.changed() => if changed.is_err() {
                break;
            },
        }
    }
    // The heartbeat keeps sending the last value, so leave none.
    plan.send_replace(Plan::Withheld(
        "the Relay stopped registering the pool".to_owned(),
    ));
    registration.send_replace(None);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn pool() -> PoolRegistration {
        PoolRegistration {
            key: PoolKey::new("east", "dynamo", "blue").unwrap(),
            location: PoolLocation {
                region: "us-east-1".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            frontends: vec![Url::parse("http://blue-frontend:8000/").unwrap()],
        }
    }

    async fn changed<T: Clone>(receiver: &mut watch::Receiver<T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), receiver.changed())
            .await
            .expect("no change")
            .unwrap();
        receiver.borrow_and_update().clone()
    }

    #[tokio::test]
    async fn registers_only_while_the_service_takes_registrations() {
        let (accepts, accepts_rx) = watch::channel(None);
        let (registration, mut registration_rx) = watch::channel(None);
        let (plan, mut plan_rx) = watch::channel(Plan::Withheld(UNKNOWN.to_owned()));
        let follow = tokio::spawn(follow_service(
            Some(pool()),
            accepts_rx,
            registration,
            plan,
            CancellationToken::new(),
        ));
        // Nothing is sent before the service has answered.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(*registration_rx.borrow_and_update(), None);
        assert_eq!(
            *plan_rx.borrow_and_update(),
            Plan::Withheld(UNKNOWN.to_owned())
        );

        accepts.send_replace(Some(true));
        assert_eq!(changed(&mut registration_rx).await, Some(pool()));
        assert_eq!(*plan_rx.borrow_and_update(), Plan::Registering);

        // A service that finds pools through MCS takes none: the pool is
        // withdrawn, and nothing is missing.
        accepts.send_replace(Some(false));
        assert_eq!(changed(&mut registration_rx).await, None);
        assert_eq!(*plan_rx.borrow_and_update(), Plan::NotTaken);

        drop(accepts);
        assert_eq!(
            changed(&mut plan_rx).await,
            Plan::Withheld("the Relay stopped registering the pool".to_owned())
        );
        assert_eq!(*registration_rx.borrow(), None);
        follow.await.unwrap();
    }

    #[tokio::test]
    async fn a_pool_without_a_region_registers_nowhere() {
        let (accepts, accepts_rx) = watch::channel(Some(false));
        let (registration, registration_rx) = watch::channel(None);
        let (plan, mut plan_rx) = watch::channel(Plan::Withheld(UNKNOWN.to_owned()));
        let _follow = AbortOnDropHandle::new(tokio::spawn(follow_service(
            None,
            accepts_rx,
            registration,
            plan,
            CancellationToken::new(),
        )));
        assert_eq!(changed(&mut plan_rx).await, Plan::NotTaken);
        accepts.send_replace(Some(true));
        let Plan::Withheld(reason) = changed(&mut plan_rx).await else {
            panic!("expected the registration to be withheld");
        };
        assert!(reason.contains("no region"), "{reason}");
        assert_eq!(*registration_rx.borrow(), None);
    }

    #[tokio::test]
    async fn reports_why_the_pool_is_not_registered() {
        // Nothing serves this URL, so every registration fails.
        let unused = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let client = GlobalDiscoveryClient::new(
            format!("http://{unused}/").parse().unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let (accepts, accepts_rx) = watch::channel(None);
        let pool = pool();
        let registration = RoutingPoolRegistration::spawn(
            client,
            pool.key,
            Some(pool.location),
            pool.frontends,
            accepts_rx,
            CancellationToken::new(),
        );
        let reason = |expected: Option<String>| {
            let registration = &registration;
            async move {
                tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        let reason = registration.unregistered_reason();
                        let matches = match (&reason, &expected) {
                            (Some(reason), Some(expected)) => reason.starts_with(expected),
                            (None, None) => true,
                            _ => false,
                        };
                        if matches {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("the expected reason never showed")
            }
        };
        reason(Some(UNKNOWN.to_owned())).await;
        accepts.send_replace(Some(true));
        reason(Some(format!("request to http://{unused}/pools failed"))).await;
        accepts.send_replace(Some(false));
        reason(None).await;
    }
}
