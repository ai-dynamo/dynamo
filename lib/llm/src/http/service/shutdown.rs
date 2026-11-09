// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's shutdown controller, shared by every listener of a process.
//!
//! One controller drives the lifecycle stages of [`ServiceObserver`] and owns the
//! signals the listeners act on:
//!
//! ```text
//!              shutdown requested                       second signal (any stage) -> force
//!                     |                                            |
//! Ready ------------> Propagating ------------> Draining -------------------> Stopping
//!                     [t0, t0+P)                [t0+P, t0+P+D)      [deadline, +T_close]
//!                     readiness withdrawn       listener closed      abort in-flight exchanges
//!                     listener open             connections retired then drop the sockets
//!                     everything admitted       in-flight finishes
//! ```
//!
//! `P` is [`ShutdownTimings::propagation`], `D` is [`ShutdownTimings::drain`] and
//! `T_close` is [`ShutdownTimings::close`]. Connection tasks run on the controller's
//! [`TaskTracker`], so the drain ends as soon as the last connection has gone, without
//! waiting for `D`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Bytes;
use futures::Stream;
use parking_lot::Mutex;
use tokio::time::Instant;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};
use tokio_util::task::TaskTracker;

use super::service_v2::{InflightPermit, ServiceObserver, ServiceStage};
use dynamo_runtime::config::environment_names::llm as env_llm;

/// Default propagation window: how long every request keeps being served after
/// shutdown is requested, so that endpoint updates reach the load balancers.
pub const DEFAULT_PROPAGATION_DELAY: Duration = Duration::from_secs(5);

/// Default drain timeout, sized so that `P + D + 5 s` fits the operator's default
/// `terminationGracePeriodSeconds` of 60.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(50);

/// How long aborted exchanges get to flush their final frames before their
/// sockets are dropped.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

/// The three durations that bound a frontend shutdown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShutdownTimings {
    /// Propagation window, `DYN_FRONTEND_PROPAGATION_DELAY_SECS`. Zero disables it.
    pub propagation: Duration,
    /// Drain timeout, `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`, counted from the end
    /// of the window.
    pub drain: Duration,
    /// Abort-to-close bound, [`CLOSE_TIMEOUT`].
    pub close: Duration,
}

impl Default for ShutdownTimings {
    fn default() -> Self {
        Self {
            propagation: DEFAULT_PROPAGATION_DELAY,
            drain: DEFAULT_DRAIN_TIMEOUT,
            close: CLOSE_TIMEOUT,
        }
    }
}

impl ShutdownTimings {
    /// Read the timings from the environment. Invalid values fall back to the
    /// defaults with a warning.
    pub fn from_env() -> Self {
        Self::parse(
            std::env::var(env_llm::DYN_FRONTEND_PROPAGATION_DELAY_SECS),
            std::env::var(env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS),
        )
    }

    pub(crate) fn parse(
        propagation: Result<String, std::env::VarError>,
        drain: Result<String, std::env::VarError>,
    ) -> Self {
        Self {
            propagation: parse_seconds(
                env_llm::DYN_FRONTEND_PROPAGATION_DELAY_SECS,
                propagation,
                DEFAULT_PROPAGATION_DELAY,
            ),
            drain: parse_seconds(
                env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
                drain,
                DEFAULT_DRAIN_TIMEOUT,
            ),
            close: CLOSE_TIMEOUT,
        }
    }

    /// The whole budget from the shutdown request to the last socket closing.
    pub fn total(&self) -> Duration {
        self.propagation + self.drain + self.close
    }
}

fn parse_seconds(
    env: &'static str,
    value: Result<String, std::env::VarError>,
    default: Duration,
) -> Duration {
    let value = match value {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return default,
        Err(error) => {
            tracing::warn!(env, %error, default_secs = default.as_secs(), "ignoring invalid shutdown timing");
            return default;
        }
    };
    match value.trim().parse::<u64>() {
        Ok(seconds)
            if Instant::now()
                .checked_add(Duration::from_secs(seconds))
                .is_some() =>
        {
            Duration::from_secs(seconds)
        }
        _ => {
            tracing::warn!(
                env,
                value,
                default_secs = default.as_secs(),
                "ignoring invalid shutdown timing"
            );
            default
        }
    }
}

/// Drives the shutdown stages and owns the signals every listener acts on.
pub struct ShutdownController {
    observer: Arc<ServiceObserver>,
    timings: ShutdownTimings,
    /// Cancelled when Draining starts: listeners close and connections retire.
    stop_accepting: CancellationToken,
    /// Cancelled at the drain deadline: exchanges still running end with an abort.
    abort: CancellationToken,
    /// Cancelled `close` after the abort: the remaining sockets are dropped.
    hard: CancellationToken,
    /// Cancelled once the controller has reached Stopping.
    stopped: CancellationToken,
    /// A second shutdown signal: skip the window and the drain.
    force: CancellationToken,
    /// Every connection task, including upgraded WebSocket sessions.
    tracker: TaskTracker,
    deadline: Mutex<Option<Instant>>,
    driving: AtomicBool,
}

impl std::fmt::Debug for ShutdownController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownController")
            .field("stage", &self.observer.stage())
            .field("timings", &self.timings)
            .field("connections", &self.tracker.len())
            .finish()
    }
}

impl ShutdownController {
    /// `force` is the process-wide force token, or a fresh token when the listener
    /// runs outside a runtime.
    pub fn new(
        observer: Arc<ServiceObserver>,
        timings: ShutdownTimings,
        force: CancellationToken,
    ) -> Self {
        Self {
            observer,
            timings,
            stop_accepting: CancellationToken::new(),
            abort: CancellationToken::new(),
            hard: CancellationToken::new(),
            stopped: CancellationToken::new(),
            force,
            tracker: TaskTracker::new(),
            deadline: Mutex::new(None),
            driving: AtomicBool::new(false),
        }
    }

    pub fn timings(&self) -> ShutdownTimings {
        self.timings
    }

    pub fn observer(&self) -> &Arc<ServiceObserver> {
        &self.observer
    }

    /// Fires when listeners must close and connections must retire.
    pub fn stop_accepting_token(&self) -> CancellationToken {
        self.stop_accepting.child_token()
    }

    /// Fires when exchanges still running must end with an abort.
    pub fn abort_token(&self) -> CancellationToken {
        self.abort.child_token()
    }

    /// Fires when the remaining sockets are dropped.
    pub fn hard_token(&self) -> CancellationToken {
        self.hard.child_token()
    }

    /// Fires once the controller has reached Stopping.
    pub fn stopped_token(&self) -> CancellationToken {
        self.stopped.child_token()
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.is_cancelled()
    }

    pub fn is_aborting(&self) -> bool {
        self.abort.is_cancelled()
    }

    /// Skip the remaining window and drain, abort in-flight work, and stop.
    pub fn force(&self) {
        self.force.cancel();
    }

    /// The tracker every connection task runs on.
    pub fn tracker(&self) -> &TaskTracker {
        &self.tracker
    }

    /// Spawn a connection task: tracked for the early exit and dropped at `hard`.
    pub fn spawn_connection<F>(&self, connection: F) -> tokio::task::JoinHandle<Option<F::Output>>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.tracker.spawn(
            self.hard
                .child_token()
                .run_until_cancelled_owned(connection),
        )
    }

    /// Time left until the abort deadline, once shutdown has been requested.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .lock()
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// Start driving the stages once `trigger` fires. Only the first call starts
    /// the driver; later calls, from other listeners sharing this controller, are
    /// no-ops. `state_cancel` is cancelled when the controller reaches Stopping.
    pub fn start(self: &Arc<Self>, trigger: CancellationToken, state_cancel: CancellationToken) {
        if self.driving.swap(true, Ordering::AcqRel) {
            return;
        }
        tokio::spawn(self.clone().drive(trigger, state_cancel));
    }

    async fn drive(self: Arc<Self>, trigger: CancellationToken, state_cancel: CancellationToken) {
        tokio::select! {
            _ = trigger.cancelled() => {}
            _ = self.force.cancelled() => {}
        }
        let started = Instant::now();
        *self.deadline.lock() = Some(started + self.timings.propagation + self.timings.drain);
        self.observer.start_propagating();
        // Serve the window only if readiness was actually withdrawn into Propagating
        // here. If the stage was already advanced past Ready (readiness withdrawn by
        // other means), there is nothing left to propagate, so drain at once.
        let in_window = self.observer.stage() == ServiceStage::Propagating;
        if in_window && !self.timings.propagation.is_zero() {
            tokio::select! {
                _ = tokio::time::sleep(self.timings.propagation) => {}
                _ = self.force.cancelled() => {
                    tracing::warn!("frontend shutdown forced during the propagation window");
                }
            }
        }

        self.observer.start_draining();
        self.stop_accepting.cancel();
        self.tracker.close();
        let drained = tokio::select! {
            biased;
            _ = self.tracker.wait() => true,
            _ = self.force.cancelled() => {
                tracing::warn!(
                    connections = self.tracker.len(),
                    inflight_requests = self.observer.inflight_count(),
                    "frontend shutdown forced during the drain"
                );
                false
            }
            _ = tokio::time::sleep(self.timings.drain) => {
                tracing::warn!(
                    connections = self.tracker.len(),
                    inflight_requests = self.observer.inflight_count(),
                    drain_timeout_secs = self.timings.drain.as_secs(),
                    "frontend drain timed out; aborting the remaining exchanges"
                );
                false
            }
        };

        if !drained {
            self.abort.cancel();
            let _ = tokio::time::timeout(self.timings.close, self.tracker.wait()).await;
        }
        self.hard.cancel();
        // Dropped connection futures end their tasks at once; the bound is a safety net.
        let _ = tokio::time::timeout(self.timings.close, self.tracker.wait()).await;
        self.observer.start_stopping();
        state_cancel.cancel();
        self.stopped.cancel();
    }
}

/// A response body stream that holds its inflight permit and ends with an error
/// when the controller aborts: hyper closes HTTP/1.1 without the chunk terminator
/// and resets the HTTP/2 stream, so the client cannot mistake it for completion.
pub(super) struct AbortableBodyStream<S> {
    inner: Pin<Box<S>>,
    abort: Pin<Box<WaitForCancellationFutureOwned>>,
    ended: bool,
    _permit: InflightPermit,
}

impl<S> AbortableBodyStream<S> {
    pub(super) fn new(inner: S, abort: CancellationToken, permit: InflightPermit) -> Self {
        Self {
            inner: Box::pin(inner),
            abort: Box::pin(abort.cancelled_owned()),
            ended: false,
            _permit: permit,
        }
    }
}

impl<S> Stream for AbortableBodyStream<S>
where
    S: Stream<Item = Result<Bytes, axum::Error>>,
{
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.ended {
            return Poll::Ready(None);
        }
        if self.abort.as_mut().poll(cx).is_ready() {
            self.ended = true;
            return Poll::Ready(Some(Err(axum::Error::new(shutdown_abort_error()))));
        }
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                self.ended = true;
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

pub(super) fn shutdown_abort_error() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::ConnectionAborted,
        "frontend shutting down",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::env::VarError;

    fn env(value: &str) -> Result<String, VarError> {
        Ok(value.to_string())
    }

    #[test]
    fn timings_parse_and_defaults() {
        let unset = ShutdownTimings::parse(Err(VarError::NotPresent), Err(VarError::NotPresent));
        assert_eq!(unset, ShutdownTimings::default());
        assert_eq!(unset.propagation, Duration::from_secs(5));
        assert_eq!(unset.drain, Duration::from_secs(50));
        assert_eq!(unset.close, Duration::from_secs(1));

        let valid = ShutdownTimings::parse(env("3"), env("15"));
        assert_eq!(valid.propagation, Duration::from_secs(3));
        assert_eq!(valid.drain, Duration::from_secs(15));
        assert_eq!(valid.total(), Duration::from_secs(19));

        let zero = ShutdownTimings::parse(env("0"), env("0"));
        assert!(zero.propagation.is_zero(), "0 disables the window");
        assert!(zero.drain.is_zero());

        for garbage in ["-1", "abc", "1.5", "", "99999999999999999999999"] {
            let parsed = ShutdownTimings::parse(env(garbage), env(garbage));
            assert_eq!(parsed, ShutdownTimings::default(), "{garbage:?}");
        }
    }

    fn controller(timings: ShutdownTimings) -> (Arc<ShutdownController>, CancellationToken) {
        let observer = Arc::new(ServiceObserver::default());
        let force = CancellationToken::new();
        let controller = Arc::new(ShutdownController::new(observer, timings, force.clone()));
        (controller, force)
    }

    const TIMINGS: ShutdownTimings = ShutdownTimings {
        propagation: Duration::from_secs(5),
        drain: Duration::from_secs(50),
        close: Duration::from_secs(1),
    };

    async fn settle() {
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn stages_follow_the_window_and_the_drain_deadline() {
        use super::super::service_v2::ServiceStage;
        let (controller, _force) = controller(TIMINGS);
        let trigger = CancellationToken::new();
        let state_cancel = CancellationToken::new();
        controller.start(trigger.clone(), state_cancel.clone());
        // A connection that never ends holds the drain open until the deadline.
        let _held = controller.spawn_connection(std::future::pending::<()>());
        settle().await;
        assert_eq!(controller.observer().stage(), ServiceStage::Ready);
        assert!(controller.remaining().is_none());

        trigger.cancel();
        settle().await;
        assert_eq!(controller.observer().stage(), ServiceStage::Propagating);
        assert!(controller.observer().is_admitting());
        assert!(!controller.observer().is_ready());
        assert!(!controller.stop_accepting_token().is_cancelled());
        assert_eq!(controller.remaining(), Some(Duration::from_secs(55)));

        tokio::time::advance(Duration::from_secs(5)).await;
        settle().await;
        assert_eq!(controller.observer().stage(), ServiceStage::Draining);
        assert!(!controller.observer().is_admitting());
        assert!(controller.stop_accepting_token().is_cancelled());
        assert!(!controller.is_aborting());

        tokio::time::advance(Duration::from_secs(49)).await;
        settle().await;
        assert!(
            !controller.is_aborting(),
            "the drain runs for the whole timeout"
        );

        tokio::time::advance(Duration::from_secs(1)).await;
        settle().await;
        assert!(controller.is_aborting());
        assert!(!controller.hard_token().is_cancelled());
        assert!(!controller.is_stopped());

        tokio::time::advance(Duration::from_secs(1)).await;
        settle().await;
        assert!(controller.hard_token().is_cancelled());
        assert!(controller.is_stopped());
        assert!(state_cancel.is_cancelled());
        assert_eq!(controller.observer().stage(), ServiceStage::Stopping);
        assert_eq!(
            controller.tracker().len(),
            0,
            "dropped connections leave the tracker"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn drain_ends_early_once_every_connection_has_gone() {
        let (controller, _force) = controller(TIMINGS);
        let trigger = CancellationToken::new();
        controller.start(trigger.clone(), CancellationToken::new());
        let release = CancellationToken::new();
        let _connection = controller.spawn_connection(release.clone().cancelled_owned());
        trigger.cancel();
        // Let the driver observe the trigger and reach the window sleep before time advances.
        settle().await;
        tokio::time::advance(Duration::from_secs(5)).await;
        settle().await;
        assert!(controller.stop_accepting_token().is_cancelled());

        tokio::time::advance(Duration::from_secs(10)).await;
        settle().await;
        assert!(!controller.is_stopped());
        release.cancel();
        settle().await;
        assert!(controller.is_stopped(), "no wait for the drain timeout");
        assert!(!controller.is_aborting(), "nothing was left to abort");
    }

    #[tokio::test(start_paused = true)]
    async fn a_zero_window_drains_at_once() {
        let (controller, _force) = controller(ShutdownTimings {
            propagation: Duration::ZERO,
            ..TIMINGS
        });
        let trigger = CancellationToken::new();
        controller.start(trigger.clone(), CancellationToken::new());
        trigger.cancel();
        settle().await;
        assert!(controller.stop_accepting_token().is_cancelled());
        assert!(
            controller.is_stopped(),
            "no connections: stopped without any timer"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn force_skips_the_window_and_the_drain_from_any_stage() {
        for force_during_window in [true, false] {
            let (controller, force) = controller(TIMINGS);
            let trigger = CancellationToken::new();
            controller.start(trigger.clone(), CancellationToken::new());
            let _held = controller.spawn_connection(std::future::pending::<()>());
            trigger.cancel();
            settle().await;
            if !force_during_window {
                tokio::time::advance(Duration::from_secs(5)).await;
                settle().await;
                assert!(controller.stop_accepting_token().is_cancelled());
            }
            let forced_at = Instant::now();
            force.cancel();
            settle().await;
            assert!(controller.stop_accepting_token().is_cancelled());
            assert!(controller.is_aborting(), "force aborts at once");
            tokio::time::advance(Duration::from_secs(1)).await;
            settle().await;
            assert!(controller.is_stopped());
            assert!(Instant::now() - forced_at <= Duration::from_secs(1));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn force_before_the_trigger_starts_the_shutdown() {
        let (controller, force) = controller(TIMINGS);
        controller.start(CancellationToken::new(), CancellationToken::new());
        force.cancel();
        settle().await;
        assert!(controller.is_stopped());
    }

    #[tokio::test]
    async fn only_the_first_start_drives_the_controller() {
        let (controller, _force) = controller(ShutdownTimings {
            propagation: Duration::ZERO,
            ..TIMINGS
        });
        let first = CancellationToken::new();
        let second = CancellationToken::new();
        controller.start(first.clone(), CancellationToken::new());
        controller.start(second.clone(), CancellationToken::new());
        second.cancel();
        settle().await;
        assert!(!controller.is_stopped(), "the second trigger is ignored");
        first.cancel();
        tokio::time::timeout(
            Duration::from_secs(1),
            controller.stopped_token().cancelled(),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn abortable_body_errors_on_abort_and_releases_its_permit() {
        let observer = Arc::new(ServiceObserver::default());
        let abort = CancellationToken::new();
        let permit = observer.acquire_inflight();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, axum::Error>>(4);
        let inner = tokio_stream::wrappers::ReceiverStream::new(rx);
        let mut body = AbortableBodyStream::new(inner, abort.clone(), permit);
        tx.send(Ok(Bytes::from_static(b"first"))).await.unwrap();
        assert_eq!(
            body.next().await.unwrap().unwrap(),
            Bytes::from_static(b"first")
        );
        assert_eq!(observer.inflight_count(), 1);

        abort.cancel();
        let error = body.next().await.unwrap().unwrap_err();
        assert!(
            error.to_string().contains("frontend shutting down"),
            "{error}"
        );
        assert!(body.next().await.is_none(), "nothing follows the abort");
        drop(body);
        assert_eq!(observer.inflight_count(), 0);
    }

    #[tokio::test]
    async fn abortable_body_passes_a_complete_stream_through() {
        let observer = Arc::new(ServiceObserver::default());
        let permit = observer.acquire_inflight();
        let inner =
            futures::stream::iter([Ok(Bytes::from_static(b"a")), Ok(Bytes::from_static(b"b"))]);
        let body = AbortableBodyStream::new(inner, CancellationToken::new(), permit);
        let chunks: Vec<_> = body.map(|chunk| chunk.unwrap()).collect().await;
        assert_eq!(chunks, [Bytes::from_static(b"a"), Bytes::from_static(b"b")]);
        assert_eq!(observer.inflight_count(), 0);
    }
}
