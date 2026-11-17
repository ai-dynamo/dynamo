// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`PoolHeartbeat`] and [`RouterHeartbeat`] renew registrations on the
//! interval granted by the service's [`Lease`].

use std::future::Future;
use std::time::Duration;

use dynamo_global_common::{Lease, PoolRegistration, RouterRegistration};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use crate::client::error_chain;
use crate::{DiscoveryClientError, GlobalDiscoveryClient};

/// The wait before the first retry. Each failed call doubles it, up to the
/// heartbeat interval.
const FIRST_RETRY: Duration = Duration::from_secs(1);
/// The longest wait between retries before the service has granted a lease.
const RETRY_CAP_WITHOUT_LEASE: Duration = Duration::from_secs(5);
/// How long before the lease ends the last retry inside it starts.
const LEASE_END_MARGIN: Duration = Duration::from_secs(1);

/// Where a [`PoolHeartbeat`] stands. A heartbeat that has stopped keeps its
/// last status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistrationStatus {
    /// Nothing is registered: no call has finished yet, or there is no
    /// registration to send.
    Unregistered,
    /// The last call succeeded, and the service granted this lease.
    Registered(Lease),
    /// The last call failed with this error. The heartbeat keeps retrying.
    Failing(String),
}

/// Registers a pool with `POST /pools` and keeps it registered until
/// cancelled or dropped. Stopping sends nothing: the service drops the pool
/// once its lease passes.
///
/// Every failure is retried, a `501` included. Every Global Discovery mode
/// takes registrations, and with MCS a pool is listed only while its
/// registration is within the lease.
#[derive(Debug)]
#[must_use = "dropping the heartbeat stops registering the pool"]
pub struct PoolHeartbeat {
    status: watch::Receiver<RegistrationStatus>,
    _task: AbortOnDropHandle<()>,
}

impl PoolHeartbeat {
    /// Registers the pool immediately, then again after each lease's
    /// heartbeat interval, sending the latest `registration` each time. If
    /// the sender is dropped, it keeps sending the last value.
    ///
    /// While `registration` is `None` it sends nothing, so the service drops
    /// the pool once its lease passes. The next registration is sent at once.
    ///
    /// A failed call is retried after 1 s, doubling up to the heartbeat
    /// interval (5 s before the first lease), and once more 1 s before the
    /// lease ends, so an outage that ends in time does not lapse it. Waits
    /// count from the start of each call, which the client's request timeout
    /// bounds; keep that well under the lease less the heartbeat interval.
    ///
    /// Runs until `cancel` fires or the heartbeat is dropped. Call it inside a
    /// Tokio runtime.
    pub fn spawn(
        client: GlobalDiscoveryClient,
        registration: watch::Receiver<Option<PoolRegistration>>,
        cancel: CancellationToken,
    ) -> Self {
        Self::spawn_from(client, registration, cancel)
    }

    fn spawn_from<R: Registration + 'static>(
        registrar: impl Registrar<R> + 'static,
        registration: watch::Receiver<Option<R>>,
        cancel: CancellationToken,
    ) -> Self {
        let (status, status_rx) = watch::channel(RegistrationStatus::Unregistered);
        let task = tokio::spawn(async move {
            cancel
                .run_until_cancelled(keep_registered(&registrar, registration, &status))
                .await;
        });
        Self {
            status: status_rx,
            _task: AbortOnDropHandle::new(task),
        }
    }

    /// The result of the last call.
    pub fn status(&self) -> RegistrationStatus {
        self.status.borrow().clone()
    }
}

/// What a [`PoolHeartbeat`] calls. The client implements it; tests use a fake.
trait Registrar<R>: Send + Sync {
    fn register(
        &self,
        registration: &R,
    ) -> impl Future<Output = Result<Lease, DiscoveryClientError>> + Send;
}

impl Registrar<PoolRegistration> for GlobalDiscoveryClient {
    fn register(
        &self,
        registration: &PoolRegistration,
    ) -> impl Future<Output = Result<Lease, DiscoveryClientError>> + Send {
        GlobalDiscoveryClient::register_pool(self, registration)
    }
}

/// Keeps one router incarnation registered until cancelled or dropped.
#[derive(Debug)]
#[must_use = "dropping the heartbeat stops registering the router"]
pub struct RouterHeartbeat(PoolHeartbeat);

impl RouterHeartbeat {
    /// Starts after the advertised Relay receiver is bound. Renewal does not
    /// depend on routable pools, which themselves depend on router discovery.
    pub fn spawn(
        client: GlobalDiscoveryClient,
        registration: RouterRegistration,
        cancel: CancellationToken,
    ) -> Self {
        let (_, registration) = watch::channel(Some(registration));
        Self(PoolHeartbeat::spawn_from(client, registration, cancel))
    }

    pub fn status(&self) -> RegistrationStatus {
        self.0.status()
    }
}

impl Registrar<RouterRegistration> for GlobalDiscoveryClient {
    fn register(
        &self,
        registration: &RouterRegistration,
    ) -> impl Future<Output = Result<Lease, DiscoveryClientError>> + Send {
        GlobalDiscoveryClient::register_router(self, registration)
    }
}

trait Registration: Clone + Send + Sync {
    fn identity(&self) -> impl std::fmt::Display;
}

impl Registration for PoolRegistration {
    fn identity(&self) -> impl std::fmt::Display {
        &self.key
    }
}

impl Registration for RouterRegistration {
    fn identity(&self) -> impl std::fmt::Display {
        &self.replica
    }
}

/// Never returns: it stops when cancelled or aborted.
async fn keep_registered<R: Registration>(
    registrar: &impl Registrar<R>,
    mut registration: watch::Receiver<Option<R>>,
    status: &watch::Sender<RegistrationStatus>,
) {
    // The last lease granted, and when the call that got it started.
    let mut renewed: Option<(Lease, Instant)> = None;
    let mut retry = FIRST_RETRY;
    let mut lapse_logged = false;
    loop {
        // Cloned so the channel is not locked across the call.
        let Some(current) = registration.borrow_and_update().clone() else {
            // Nothing to renew, so the next registration starts over.
            update(status, RegistrationStatus::Unregistered);
            (renewed, retry, lapse_logged) = (None, FIRST_RETRY, false);
            if registration.changed().await.is_err() {
                // The sender is gone and left nothing to register.
                std::future::pending::<()>().await;
            }
            continue;
        };
        let started = Instant::now();
        let wait = match registrar.register(&current).await {
            Ok(lease) => {
                renewed = Some((lease, started));
                retry = FIRST_RETRY;
                lapse_logged = false;
                if update(status, RegistrationStatus::Registered(lease)) {
                    tracing::info!(
                        registration = %current.identity(),
                        heartbeat_interval_s = lease.heartbeat_interval().as_secs(),
                        lease_s = lease.duration().as_secs(),
                        "registered with Global Discovery"
                    );
                }
                lease.heartbeat_interval()
            }
            Err(error) => {
                let reason = error_chain(&error);
                if update(status, RegistrationStatus::Failing(reason.clone())) {
                    tracing::warn!(registration = %current.identity(), error = %reason, "cannot register with Global Discovery; retrying");
                }
                let cap = renewed.map_or(RETRY_CAP_WITHOUT_LEASE, |(lease, _)| {
                    lease.heartbeat_interval()
                });
                let mut wait = retry.min(cap);
                retry = retry.saturating_mul(2).min(cap);
                if let Some((lease, at)) = renewed {
                    // Pulls the last retry inside the lease forward to just
                    // before it ends. Zero once that retry has been made.
                    let before_end = lease
                        .duration()
                        .saturating_sub(started.saturating_duration_since(at))
                        .saturating_sub(LEASE_END_MARGIN);
                    if !before_end.is_zero() {
                        wait = wait.min(before_end);
                    }
                    if !lapse_logged && at.elapsed() >= lease.duration() {
                        lapse_logged = true;
                        tracing::warn!(registration = %current.identity(), lease_s = lease.duration().as_secs(), "registration with Global Discovery lapsed");
                    }
                }
                wait
            }
        };
        // Counted from the start of the call, so a slow call does not push
        // the next one toward the end of the lease. `sleep` takes a wait of
        // any length, as a lease may be.
        tokio::time::sleep(wait.saturating_sub(started.elapsed())).await;
    }
}

/// Publishes `new` unless it is the status already. Returns whether it
/// changed, so a caller logs each change once instead of every call.
fn update(status: &watch::Sender<RegistrationStatus>, new: RegistrationStatus) -> bool {
    status.send_if_modified(|current| {
        if *current == new {
            return false;
        }
        *current = new;
        true
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dynamo_global_common::{Capacity, PoolKey, PoolLocation};
    use reqwest::StatusCode;

    use super::*;

    /// A registrar whose answer and call duration the test sets. It records
    /// the second each call started at and the workers it sent.
    #[derive(Clone)]
    struct FakeRegistrar(Arc<Mutex<Fake>>);

    struct Fake {
        answer: Result<Lease, StatusCode>,
        delay: Duration,
        start: Instant,
        calls: Vec<(u64, u32)>,
    }

    impl FakeRegistrar {
        fn answering(answer: Result<Lease, StatusCode>) -> Self {
            Self(Arc::new(Mutex::new(Fake {
                answer,
                delay: Duration::ZERO,
                start: Instant::now(),
                calls: Vec::new(),
            })))
        }

        fn answer(&self, answer: Result<Lease, StatusCode>) {
            self.0.lock().unwrap().answer = answer;
        }

        /// The second each call started at.
        fn calls(&self) -> Vec<u64> {
            let fake = self.0.lock().unwrap();
            fake.calls.iter().map(|&(at, _)| at).collect()
        }

        /// The workers each call sent.
        fn workers(&self) -> Vec<u32> {
            let fake = self.0.lock().unwrap();
            fake.calls.iter().map(|&(_, workers)| workers).collect()
        }
    }

    impl Registrar<PoolRegistration> for FakeRegistrar {
        async fn register(
            &self,
            registration: &PoolRegistration,
        ) -> Result<Lease, DiscoveryClientError> {
            let (answer, delay) = {
                let mut fake = self.0.lock().unwrap();
                let at = fake.start.elapsed().as_secs();
                fake.calls.push((at, registration.capacity[0].workers));
                (fake.answer, fake.delay)
            };
            tokio::time::sleep(delay).await;
            answer.map_err(|status| DiscoveryClientError::Status {
                url: "http://discovery/pools".parse().unwrap(),
                status,
                body: "discovery is down".to_owned(),
            })
        }
    }

    fn lease(heartbeat_interval_s: u64, lease_duration_s: u64) -> Lease {
        Lease::new(heartbeat_interval_s, lease_duration_s).unwrap()
    }

    fn pool(workers: u32) -> PoolRegistration {
        PoolRegistration {
            key: PoolKey::new("aws-west", "dynamo", "mocker-west-1").unwrap(),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: "Qwen/Qwen3-0.6B".to_owned(),
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: vec![Capacity {
                sku: "H100".to_owned(),
                workers,
            }],
        }
    }

    fn spawn(registrar: &FakeRegistrar, cancel: CancellationToken) -> PoolHeartbeat {
        let (_, registration) = watch::channel(Some(pool(8)));
        PoolHeartbeat::spawn_from(registrar.clone(), registration, cancel)
    }

    async fn sleep_s(seconds: u64) {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }

    #[tokio::test(start_paused = true)]
    async fn renews_on_each_leases_heartbeat_interval() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        let (registration, registration_rx) = watch::channel(Some(pool(8)));
        let heartbeat =
            PoolHeartbeat::spawn_from(registrar.clone(), registration_rx, CancellationToken::new());
        assert_eq!(heartbeat.status(), RegistrationStatus::Unregistered);
        sleep_s(11).await;
        assert_eq!(registrar.calls(), [0, 5, 10]);
        assert_eq!(
            heartbeat.status(),
            RegistrationStatus::Registered(lease(5, 15))
        );

        // The next call sends the latest registration, and the one after it
        // follows the new lease.
        registration.send_replace(Some(pool(16)));
        registrar.answer(Ok(lease(3, 9)));
        sleep_s(8).await;
        assert_eq!(registrar.calls(), [0, 5, 10, 15, 18]);
        assert_eq!(registrar.workers(), [8, 8, 8, 16, 16]);
        assert_eq!(
            heartbeat.status(),
            RegistrationStatus::Registered(lease(3, 9))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn registers_only_while_there_is_a_registration() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        let (registration, registration_rx) = watch::channel(None);
        let heartbeat =
            PoolHeartbeat::spawn_from(registrar.clone(), registration_rx, CancellationToken::new());
        sleep_s(3).await;
        assert!(registrar.calls().is_empty());
        assert_eq!(heartbeat.status(), RegistrationStatus::Unregistered);

        // The first registration is sent at once.
        registration.send_replace(Some(pool(8)));
        sleep_s(6).await;
        assert_eq!(registrar.calls(), [3, 8]);
        assert_eq!(
            heartbeat.status(),
            RegistrationStatus::Registered(lease(5, 15))
        );

        // Withdrawn, it is not renewed at 13 s, and the next one is sent at once.
        registration.send_replace(None);
        sleep_s(10).await;
        assert_eq!(registrar.calls(), [3, 8]);
        assert_eq!(heartbeat.status(), RegistrationStatus::Unregistered);
        registration.send_replace(Some(pool(16)));
        sleep_s(1).await;
        assert_eq!(registrar.calls(), [3, 8, 19]);
        assert_eq!(registrar.workers(), [8, 8, 16]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_slow_call_does_not_delay_the_next() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        registrar.0.lock().unwrap().delay = Duration::from_secs(3);
        let _heartbeat = spawn(&registrar, CancellationToken::new());
        sleep_s(11).await;
        assert_eq!(registrar.calls(), [0, 5, 10]);
    }

    #[tokio::test(start_paused = true)]
    async fn retries_a_failed_call_before_the_lease_runs_out() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        let heartbeat = spawn(&registrar, CancellationToken::new());
        sleep_s(1).await;

        // Retries 1, 2 and 4 s after a failed call, then 1 s before the 15 s
        // lease ends, then every heartbeat interval.
        registrar.answer(Err(StatusCode::SERVICE_UNAVAILABLE));
        sleep_s(24).await;
        assert_eq!(registrar.calls(), [0, 5, 6, 8, 12, 14, 19, 24]);
        assert_eq!(
            heartbeat.status(),
            RegistrationStatus::Failing(
                "http://discovery/pools returned 503 Service Unavailable: discovery is down"
                    .to_owned()
            )
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_outage_that_ends_in_time_keeps_the_lease() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        let heartbeat = spawn(&registrar, CancellationToken::new());
        sleep_s(1).await;
        registrar.answer(Err(StatusCode::SERVICE_UNAVAILABLE));
        sleep_s(12).await;

        // The service is back at 13 s, and the retry at 14 s renews the lease
        // before it ends at 15 s.
        registrar.answer(Ok(lease(5, 15)));
        sleep_s(7).await;
        assert_eq!(registrar.calls(), [0, 5, 6, 8, 12, 14, 19]);
        assert_eq!(
            heartbeat.status(),
            RegistrationStatus::Registered(lease(5, 15))
        );

        // A success restarts the backoff at 1 s.
        registrar.answer(Err(StatusCode::SERVICE_UNAVAILABLE));
        sleep_s(6).await;
        assert_eq!(registrar.calls(), [0, 5, 6, 8, 12, 14, 19, 24, 25]);
    }

    #[tokio::test(start_paused = true)]
    async fn retries_at_least_every_5s_before_the_first_lease() {
        // A 501 too: it does not mean the service takes no registrations.
        let registrar = FakeRegistrar::answering(Err(StatusCode::NOT_IMPLEMENTED));
        let heartbeat = spawn(&registrar, CancellationToken::new());
        sleep_s(18).await;
        assert_eq!(registrar.calls(), [0, 1, 3, 7, 12, 17]);
        assert!(matches!(heartbeat.status(), RegistrationStatus::Failing(_)));
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_stops_the_heartbeat() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        let cancel = CancellationToken::new();
        let heartbeat = spawn(&registrar, cancel.clone());
        sleep_s(1).await;

        cancel.cancel();
        sleep_s(30).await;
        assert_eq!(registrar.calls(), [0]);
        assert_eq!(Arc::strong_count(&registrar.0), 1);
        // A stopped heartbeat keeps its last status.
        assert_eq!(
            heartbeat.status(),
            RegistrationStatus::Registered(lease(5, 15))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_heartbeat_stops_it() {
        let registrar = FakeRegistrar::answering(Ok(lease(5, 15)));
        let heartbeat = spawn(&registrar, CancellationToken::new());
        sleep_s(1).await;

        drop(heartbeat);
        sleep_s(30).await;
        assert_eq!(registrar.calls(), [0]);
        assert_eq!(Arc::strong_count(&registrar.0), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn follows_a_lease_of_any_length() {
        let registrar = FakeRegistrar::answering(Ok(lease(u64::MAX - 1, u64::MAX)));
        let _heartbeat = spawn(&registrar, CancellationToken::new());
        sleep_s(86_400).await;
        assert_eq!(registrar.calls(), [0]);
        // Still waiting, not ended by a panic: the task holds the registrar.
        assert_eq!(Arc::strong_count(&registrar.0), 2);
    }
}
