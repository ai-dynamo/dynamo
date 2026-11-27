// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker-owned hold on this process's endpoint instances.
//!
//! A worker that starts held registers its endpoint instances with `held` set. Routers
//! discover them and prepare for them, but only a request that names the worker reaches it.
//! A controller releases the hold once it has checked the worker. The worker then clears
//! `held` on its own discovery records, which keep their lease.
//!
//! The hold lasts for the life of the process: every new process starts held again, and a
//! released worker stays released.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::environment_names::discovery::DYN_START_HELD;

/// The hold state of one worker process.
#[derive(Debug)]
pub struct RoutingHold {
    held: AtomicBool,
    /// Orders the writers of this worker's endpoint records. An endpoint registers under a
    /// read guard and a release rewrites the records under the write guard, so a release
    /// never misses a record that is being written held, and releases never overlap.
    records: tokio::sync::RwLock<()>,
}

/// Held by an endpoint from reading the hold until its discovery record is written.
pub(crate) struct HoldRegistration<'a> {
    hold: &'a RoutingHold,
    _records: tokio::sync::RwLockReadGuard<'a, ()>,
}

impl HoldRegistration<'_> {
    /// Whether the endpoint registers held.
    pub(crate) fn held(&self) -> bool {
        self.hold.is_held()
    }
}

impl RoutingHold {
    pub(crate) fn new(held: bool) -> Self {
        Self {
            held: AtomicBool::new(held),
            records: tokio::sync::RwLock::new(()),
        }
    }

    /// The hold this worker starts with, from [`DYN_START_HELD`].
    pub(crate) fn from_env(
        kubernetes_discovery: bool,
        system_server_enabled: bool,
    ) -> anyhow::Result<Self> {
        Self::starting(
            crate::config::env_is_truthy(DYN_START_HELD),
            kubernetes_discovery,
            system_server_enabled,
        )
    }

    /// A hold needs a way to end, so a worker that asks for one must run where one exists.
    /// Kubernetes discovery holds a worker Pod through its readiness gate instead, and
    /// without the system status server no controller could release the worker.
    fn starting(
        held: bool,
        kubernetes_discovery: bool,
        system_server_enabled: bool,
    ) -> anyhow::Result<Self> {
        if held {
            anyhow::ensure!(
                !kubernetes_discovery,
                "{DYN_START_HELD} is not supported with Kubernetes discovery, where the \
                 frontend readiness gate holds a worker Pod"
            );
            anyhow::ensure!(
                system_server_enabled,
                "{DYN_START_HELD} needs the system status server to release the hold: \
                 set DYN_SYSTEM_PORT"
            );
            tracing::info!(
                "this worker starts held out of automatic routing until its hold is released"
            );
        }
        Ok(Self::new(held))
    }

    pub fn is_held(&self) -> bool {
        self.held.load(Ordering::Acquire)
    }

    pub(crate) fn set(&self, held: bool) {
        self.held.store(held, Ordering::Release);
    }

    /// Start registering an endpoint. Keep the guard until the discovery record is written.
    pub(crate) async fn registration(&self) -> HoldRegistration<'_> {
        HoldRegistration {
            hold: self,
            _records: self.records.read().await,
        }
    }

    /// Release the hold. `clear_records` clears it on this worker's discovery records and
    /// returns how many changed.
    ///
    /// The hold itself is cleared only once the records are written. A release that fails or
    /// is dropped part-way therefore leaves the worker held, and repeating it completes it:
    /// the records it already cleared are left as they are.
    pub(crate) async fn release(
        &self,
        clear_records: impl Future<Output = anyhow::Result<usize>>,
    ) -> Result<HoldRelease, HoldReleaseError> {
        let _records = self.records.write().await;
        if !self.is_held() {
            return Ok(HoldRelease::AlreadyReleased);
        }
        let endpoints = clear_records.await.map_err(HoldReleaseError::Discovery)?;
        self.set(false);
        tracing::info!(endpoints, "released the worker routing hold");
        Ok(HoldRelease::Released)
    }
}

/// What a release did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HoldRelease {
    /// The worker was held and is now routable.
    Released,
    /// The worker was not held. Repeating a release is safe.
    AlreadyReleased,
}

#[derive(Debug, thiserror::Error)]
pub enum HoldReleaseError {
    /// The request names another worker instance, typically the one this process replaced at
    /// the same address. This worker keeps its hold.
    #[error("instance {requested} is not this worker, which is instance {current}")]
    StaleInstance { requested: u64, current: u64 },
    /// The discovery records could not all be updated. The worker is still held, and
    /// repeating the release completes it.
    #[error("failed to update the worker's discovery records: {0}")]
    Discovery(#[source] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hold_is_refused_where_nothing_could_release_it() {
        assert!(RoutingHold::starting(true, false, true).unwrap().is_held());

        let kubernetes = RoutingHold::starting(true, true, true)
            .unwrap_err()
            .to_string();
        assert!(kubernetes.contains("Kubernetes discovery"), "{kubernetes}");
        let no_server = RoutingHold::starting(true, false, false)
            .unwrap_err()
            .to_string();
        assert!(no_server.contains("DYN_SYSTEM_PORT"), "{no_server}");

        // Without the setting nothing is held and nothing is checked.
        assert!(!RoutingHold::starting(false, true, false).unwrap().is_held());
    }

    #[tokio::test]
    async fn a_failed_release_leaves_the_worker_held_and_a_retry_completes_it() {
        let hold = RoutingHold::new(true);

        let failed = hold
            .release(async { Err(anyhow::anyhow!("discovery is down")) })
            .await;
        assert!(matches!(failed, Err(HoldReleaseError::Discovery(_))));
        assert!(hold.is_held());

        assert_eq!(
            hold.release(async { Ok(1) }).await.unwrap(),
            HoldRelease::Released
        );
        assert!(!hold.is_held());

        // A repeat does not touch the records again.
        let repeat = hold
            .release(async { panic!("a released worker rewrote its records") })
            .await;
        assert_eq!(repeat.unwrap(), HoldRelease::AlreadyReleased);
    }

    /// A controller that gives up closes the connection, which drops the release part-way.
    #[tokio::test]
    async fn an_interrupted_release_leaves_the_worker_held() {
        let hold = RoutingHold::new(true);

        let mut interrupted = Box::pin(hold.release(std::future::pending()));
        assert!(futures::poll!(&mut interrupted).is_pending());
        drop(interrupted);
        assert!(hold.is_held());

        assert_eq!(
            hold.release(async { Ok(1) }).await.unwrap(),
            HoldRelease::Released
        );
    }

    /// An endpoint that read the hold before a release must have its record written before
    /// the release reads the records, or it would stay held with nothing left to release it.
    #[tokio::test]
    async fn a_release_waits_for_an_endpoint_that_is_registering() {
        let hold = RoutingHold::new(true);
        let cleared = std::sync::atomic::AtomicBool::new(false);

        let registration = hold.registration().await;
        assert!(registration.held());
        let mut release = Box::pin(hold.release(async {
            cleared.store(true, Ordering::Release);
            Ok(1)
        }));
        assert!(futures::poll!(&mut release).is_pending());
        assert!(!cleared.load(Ordering::Acquire));

        drop(registration);
        assert_eq!(release.await.unwrap(), HoldRelease::Released);
        assert!(cleared.load(Ordering::Acquire));
        // An endpoint that registers from now on is not held.
        assert!(!hold.registration().await.held());
    }
}
