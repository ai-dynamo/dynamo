// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary: remove when AI gateway ships.
//!
//! Revoked-token check (PN-5102), as go-proxy's `pkg/auth/revoke.go`. After a
//! token passes signature, expiry and the org allowlist, its `token_uuid` is
//! looked up in a per-pod cache and, on a miss, asked of the cluster's
//! authz-gateway over gRPC (`auth.Auth/IsRevoked` at
//! `REVOKE_AUTH_GRPC_TARGET`). Only a "revoked" answer rejects: a token
//! without the claim, a failed or timed-out call, or the pause after one, all
//! let the request through.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_runtime::config::environment_names::llm as env_llm;
use parking_lot::Mutex;
use prometheus::{IntCounterVec, Opts, Registry};
use tokio::sync::watch;
use tonic::Code;
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

mod pb {
    tonic::include_proto!("auth");
}

use pb::IsRevokedRequest;
use pb::auth_client::AuthClient;

/// "Not revoked" answers are reused for this long; "revoked" ones until dropped from the cache.
const TTL: Duration = Duration::from_secs(15 * 60);
/// A "not revoked" entry this old is still used, and refreshed in the background.
const REFRESH_AFTER: Duration = Duration::from_secs(12 * 60);
const CACHE_CAP: usize = 100_000;
const CALL_TIMEOUT: Duration = Duration::from_millis(300);
/// After a failed call no call is made for a while, and cache misses are
/// allowed. The pauses since the last successful call last `PAUSE_MIN` for the
/// first `PAUSES_PER_STEP`, then `PAUSE_STEP`, twice it, three times it… each
/// for `PAUSES_PER_STEP` pauses (1 s ×5, 2 s ×5, 4 s ×5, 6 s ×5…), at most
/// `PAUSE_MAX`; a gateway that comes back after a short outage is used again
/// within seconds.
const PAUSE_MIN: Duration = Duration::from_secs(1);
const PAUSE_STEP: Duration = Duration::from_secs(2);
const PAUSES_PER_STEP: u32 = 5;
const PAUSE_MAX: Duration = Duration::from_secs(10);
/// UNAVAILABLE (the gateway down, or its Valkey lookup failing) comes back at
/// once and costs a request nothing, so calls pause only from this many in a
/// row; any other failure, such as the 300 ms timeout, pauses at once.
const UNAVAILABLE_BEFORE_PAUSE: u32 = 3;
/// The connection is pinged this often, so a dead one is found within about
/// `KEEPALIVE + KEEPALIVE_WAIT`.
const KEEPALIVE: Duration = Duration::from_secs(20);
const KEEPALIVE_WAIT: Duration = Duration::from_secs(5);
/// A connect attempt that hangs (a dropped SYN) is given up after this long.
/// tonic reconnects on the next call, with no back-off of its own; the pauses
/// space the calls.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// The checker's timings and cache size; the constants above in production,
/// shorter ones in tests.
#[derive(Debug, Clone)]
struct Settings {
    ttl: Duration,
    refresh_after: Duration,
    capacity: usize,
    call_timeout: Duration,
    pause_min: Duration,
    pause_step: Duration,
    pauses_per_step: u32,
    pause_max: Duration,
    unavailable_before_pause: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            ttl: TTL,
            refresh_after: REFRESH_AFTER,
            capacity: CACHE_CAP,
            call_timeout: CALL_TIMEOUT,
            pause_min: PAUSE_MIN,
            pause_step: PAUSE_STEP,
            pauses_per_step: PAUSES_PER_STEP,
            pause_max: PAUSE_MAX,
            unavailable_before_pause: UNAVAILABLE_BEFORE_PAUSE,
        }
    }
}

/// One call's result: `None` until it ends, then `Some(None)` for a failed
/// call or `Some(Some(revoked))`.
type CallResult = Option<Option<bool>>;

#[derive(Debug, Clone, Copy)]
struct Entry {
    revoked: bool,
    fetched_at: Instant,
}

#[derive(Default)]
struct CheckerState {
    cache: HashMap<Uuid, Entry>,
    /// One call per token; concurrent misses for the same token wait on it.
    inflight: HashMap<Uuid, watch::Receiver<CallResult>>,
    paused_until: Option<Instant>,
    /// Failed calls since the last successful one, not counting those during a pause.
    failures: u32,
    /// Pauses started since the last successful call.
    pauses: u32,
    last_failure_log: Option<Instant>,
}

impl CheckerState {
    fn paused(&self, now: Instant) -> bool {
        self.paused_until.is_some_and(|until| now < until)
    }
}

/// `proxy_revoke_*`, the names go-proxy uses, so one alert rule covers both.
struct RevokeMetrics {
    checks: IntCounterVec,
    calls: IntCounterVec,
}

impl RevokeMetrics {
    fn new() -> Self {
        Self {
            checks: IntCounterVec::new(
                Opts::new(
                    "proxy_revoke_checks_total",
                    "Revoked-token checks by outcome: revoked, not_revoked, no_token_uuid, paused, error",
                ),
                &["outcome"],
            )
            .expect("valid metric definition"),
            calls: IntCounterVec::new(
                Opts::new(
                    "proxy_revoke_store_calls_total",
                    "Revoke lookups (auth.Auth/IsRevoked on the authz-gateway) by result: ok, error",
                ),
                &["result"],
            )
            .expect("valid metric definition"),
        }
    }

    fn outcome(&self, name: &str) {
        self.checks.with_label_values(&[name]).inc();
    }

    fn lookup(&self, ok: bool) {
        self.calls
            .with_label_values(&[if ok { "ok" } else { "error" }])
            .inc();
    }
}

pub struct RevokeChecker {
    client: AuthClient<Channel>,
    metrics: RevokeMetrics,
    settings: Settings,
    state: Mutex<CheckerState>,
}

/// Whether `target` is `host:port` with a numeric port, the form model-suite-app sets.
fn is_grpc_target(target: &str) -> bool {
    let Some((host, port)) = target.rsplit_once(':') else {
        return false;
    };
    let host = match host.strip_prefix('[') {
        Some(rest) => match rest.strip_suffix(']') {
            Some(ipv6) if !ipv6.is_empty() => ipv6,
            _ => return false,
        },
        // An IPv6 address must be in brackets.
        None if host.contains(':') => return false,
        None => host,
    };
    !host.is_empty()
        && !host.contains(['/', '?', '#', '[', ']', '@', ' '])
        && port.parse::<u16>().is_ok_and(|p| p > 0)
}

impl RevokeChecker {
    /// From `REVOKE_AUTH_GRPC_TARGET`; unset or blank leaves the check off.
    pub fn from_env() -> Option<Arc<Self>> {
        let target = std::env::var(env_llm::REVOKE_AUTH_GRPC_TARGET).unwrap_or_default();
        let target = target.trim();
        if target.is_empty() {
            return None;
        }
        Self::connect(target)
    }

    /// Returns `None`, which leaves the check off, when `target` is not
    /// `host:port` or no Tokio runtime is running. A bad value is logged
    /// rather than failing start-up: the frontend must keep serving.
    pub fn connect(target: &str) -> Option<Arc<Self>> {
        Self::connect_with(target, Settings::default())
    }

    fn connect_with(target: &str, settings: Settings) -> Option<Arc<Self>> {
        if !is_grpc_target(target) {
            tracing::error!(
                grpc_target = target,
                "REVOKE_AUTH_GRPC_TARGET is not host:port; revoked-token check disabled"
            );
            return None;
        }
        // The channel's worker is a Tokio task.
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::error!(
                grpc_target = target,
                "no Tokio runtime to run the revoke client; revoked-token check disabled"
            );
            return None;
        }
        let endpoint = match Endpoint::from_shared(format!("http://{target}")) {
            Ok(endpoint) => endpoint,
            Err(error) => {
                tracing::error!(
                    grpc_target = target,
                    %error,
                    "REVOKE_AUTH_GRPC_TARGET cannot be dialled; revoked-token check disabled"
                );
                return None;
            }
        };
        let channel = endpoint
            .connect_timeout(CONNECT_TIMEOUT)
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(KEEPALIVE_WAIT)
            .keep_alive_while_idle(true)
            .connect_lazy();
        let client = AuthClient::new(channel);

        // Connect now, so the first request does not pay for the handshake
        // inside its timeout. A non-UUID is answered without a lookup, so this
        // costs the gateway nothing; its result is not used or counted.
        let mut warm = client.clone();
        tokio::spawn(async move {
            let _ = tokio::time::timeout(
                CONNECT_TIMEOUT,
                warm.is_revoked(IsRevokedRequest {
                    token_uuid: String::new(),
                }),
            )
            .await;
        });

        tracing::info!(grpc_target = target, "revoked-token check enabled");
        Some(Arc::new(Self {
            client,
            metrics: RevokeMetrics::new(),
            settings,
            state: Mutex::new(CheckerState::default()),
        }))
    }

    /// Adds the `proxy_revoke_*` counters to the frontend's `/metrics`.
    pub fn register_metrics(&self, registry: &Registry) -> prometheus::Result<()> {
        registry.register(Box::new(self.metrics.checks.clone()))?;
        registry.register(Box::new(self.metrics.calls.clone()))
    }

    /// Whether a request carrying this `token_uuid` must be rejected.
    pub async fn revoked(self: &Arc<Self>, token_uuid: &str) -> bool {
        // No claim, or not a UUID: it can never be revoked, so there is nothing to ask.
        let Ok(id) = Uuid::parse_str(token_uuid.trim()) else {
            self.metrics.outcome("no_token_uuid");
            return false;
        };

        let now = Instant::now();
        let mut call = {
            let mut state = self.state.lock();
            if let Some(entry) = state.cache.get(&id).copied() {
                if entry.revoked {
                    drop(state);
                    self.metrics.outcome("revoked");
                    return true;
                }
                let age = now.duration_since(entry.fetched_at);
                if age < self.settings.ttl {
                    if age >= self.settings.refresh_after
                        && !state.paused(now)
                        && !state.inflight.contains_key(&id)
                    {
                        self.start_call_locked(&mut state, id);
                    }
                    drop(state);
                    self.metrics.outcome("not_revoked");
                    return false;
                }
            }
            if state.paused(now) {
                drop(state);
                self.metrics.outcome("paused");
                return false;
            }
            self.start_call_locked(&mut state, id)
        };

        // A closed channel (the call's task gone without an answer) counts as a failure.
        let answer = match call.wait_for(Option::is_some).await {
            Ok(result) => result.flatten(),
            Err(_) => None,
        };
        match answer {
            Some(true) => {
                self.metrics.outcome("revoked");
                true
            }
            Some(false) => {
                self.metrics.outcome("not_revoked");
                false
            }
            None => {
                self.metrics.outcome("error");
                false
            }
        }
    }

    /// Returns the call in flight for `id`, starting one if there is none.
    /// The call runs in its own task, so a client hanging up does not cancel
    /// a call that other requests are waiting on. Caller holds the state lock.
    fn start_call_locked(
        self: &Arc<Self>,
        state: &mut CheckerState,
        id: Uuid,
    ) -> watch::Receiver<CallResult> {
        if let Some(call) = state.inflight.get(&id) {
            return call.clone();
        }
        let (done, call) = watch::channel(None);
        state.inflight.insert(id, call.clone());
        let this = self.clone();
        tokio::spawn(async move {
            let answer = this.lookup(id).await;
            {
                let mut state = this.state.lock();
                if let Some(revoked) = answer {
                    this.put_locked(&mut state, id, revoked, Instant::now());
                }
                state.inflight.remove(&id);
            }
            let _ = done.send(Some(answer));
        });
        call
    }

    /// Makes one call and counts it; a failure may start a pause (see
    /// `PAUSE_MIN`), and is logged at most once per `PAUSE_MAX` however many
    /// calls fail.
    async fn lookup(&self, id: Uuid) -> Option<bool> {
        let mut client = self.client.clone();
        let request = IsRevokedRequest {
            token_uuid: id.to_string(),
        };
        let result = match tokio::time::timeout(
            self.settings.call_timeout,
            client.is_revoked(request),
        )
        .await
        {
            Ok(Ok(response)) => Ok(response.into_inner().revoked),
            Ok(Err(status)) => Err(status),
            Err(_) => Err(tonic::Status::deadline_exceeded(format!(
                "no answer within {:?}",
                self.settings.call_timeout
            ))),
        };
        self.metrics.lookup(result.is_ok());

        let now = Instant::now();
        let mut state = self.state.lock();
        let status = match result {
            Ok(revoked) => {
                state.failures = 0;
                state.pauses = 0;
                return Some(revoked);
            }
            Err(status) => status,
        };
        let pause = self.note_failure_locked(&mut state, status.code(), now);
        let log = state
            .last_failure_log
            .is_none_or(|last| now.duration_since(last) >= self.settings.pause_max);
        if log {
            state.last_failure_log = Some(now);
        }
        drop(state);
        if log {
            if status.code() == Code::Unimplemented {
                tracing::warn!(error = %status, ?pause, "revoke lookup failed: the target has no auth.Auth/IsRevoked (is REVOKE_AUTH_GRPC_TARGET the authz-gateway?); allowing requests");
            } else {
                tracing::warn!(error = %status, ?pause, "revoke lookup failed; allowing requests");
            }
        }
        None
    }

    /// Counts a failed call and starts a pause when one is due, returning its
    /// length (zero for none). A call failing during a pause, having started
    /// before it, is not counted, so calls failing together start one pause.
    fn note_failure_locked(&self, state: &mut CheckerState, code: Code, now: Instant) -> Duration {
        if state.paused(now) {
            return Duration::ZERO;
        }
        state.failures += 1;
        if code == Code::Unavailable && state.failures < self.settings.unavailable_before_pause {
            return Duration::ZERO;
        }
        let pause = self.pause_for(state.pauses);
        state.pauses += 1;
        state.paused_until = Some(now + pause);
        pause
    }

    /// The length of the n-th pause (from 0) since the last successful call; see `PAUSE_MIN`.
    fn pause_for(&self, n: u32) -> Duration {
        match n / self.settings.pauses_per_step {
            0 => self.settings.pause_min,
            step => self
                .settings
                .pause_step
                .saturating_mul(step)
                .min(self.settings.pause_max),
        }
    }

    /// Stores an answer. At the cap, 10% of the entries are dropped at once,
    /// whichever they are: a dropped "revoked" entry is asked again on its
    /// next request, and the gateway still says revoked.
    fn put_locked(&self, state: &mut CheckerState, id: Uuid, revoked: bool, now: Instant) {
        if !state.cache.contains_key(&id) && state.cache.len() >= self.settings.capacity {
            let drop = (self.settings.capacity / 10).max(1);
            let victims: Vec<Uuid> = state.cache.keys().take(drop).copied().collect();
            for victim in victims {
                state.cache.remove(&victim);
            }
        }
        state.cache.insert(
            id,
            Entry {
                revoked,
                fetched_at: now,
            },
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pb::auth_server::{Auth, AuthServer};
    use pb::{CheckRequest, CheckResponse, IsRevokedResponse};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tonic::transport::Server;
    use tonic::transport::server::TcpIncoming;
    use tonic::{Request, Response, Status};

    const R: &str = "5b0c6f8e-3c1a-4e6b-9d2f-0a1b2c3d4e5f";
    const N: &str = "7d1e2f3a-4b5c-4d6e-8f9a-0b1c2d3e4f5a";

    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    pub(crate) enum Mode {
        #[default]
        Answer,
        Unavailable,
        Unimplemented,
        Slow(Duration),
    }

    /// An in-process authz-gateway: `IsRevoked` answers from `revoked`, or fails as `mode` says.
    #[derive(Default)]
    pub(crate) struct FakeGateway {
        pub(crate) revoked: Mutex<HashSet<String>>,
        pub(crate) mode: Mutex<Mode>,
        /// Calls with a token_uuid; the start-up call (empty) is not counted.
        pub(crate) calls: AtomicUsize,
        pub(crate) asked: Mutex<Vec<String>>,
    }

    impl FakeGateway {
        pub(crate) fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    struct Svc(Arc<FakeGateway>);

    #[tonic::async_trait]
    impl Auth for Svc {
        async fn check(&self, _: Request<CheckRequest>) -> Result<Response<CheckResponse>, Status> {
            Err(Status::unimplemented("not used"))
        }

        async fn is_revoked(
            &self,
            request: Request<IsRevokedRequest>,
        ) -> Result<Response<IsRevokedResponse>, Status> {
            let id = request.into_inner().token_uuid;
            if id.is_empty() {
                return Ok(Response::new(IsRevokedResponse { revoked: false }));
            }
            let gateway = &self.0;
            gateway.calls.fetch_add(1, Ordering::SeqCst);
            gateway.asked.lock().push(id.clone());
            let mode = *gateway.mode.lock();
            match mode {
                Mode::Unavailable => return Err(Status::unavailable("valkey lookup failed")),
                Mode::Unimplemented => return Err(Status::unimplemented("unknown method")),
                Mode::Slow(delay) => tokio::time::sleep(delay).await,
                Mode::Answer => {}
            }
            let revoked = gateway.revoked.lock().contains(&id);
            Ok(Response::new(IsRevokedResponse { revoked }))
        }
    }

    /// Serves `gateway` on `listener` until the returned sender is dropped or used.
    pub(crate) fn serve(gateway: Arc<FakeGateway>, listener: TcpListener) -> oneshot::Sender<()> {
        let (stop, stopped) = oneshot::channel::<()>();
        tokio::spawn(
            Server::builder()
                .add_service(AuthServer::new(Svc(gateway)))
                .serve_with_incoming_shutdown(TcpIncoming::from(listener), async {
                    let _ = stopped.await;
                }),
        );
        stop
    }

    pub(crate) async fn start(gateway: Arc<FakeGateway>) -> (SocketAddr, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        (addr, serve(gateway, listener))
    }

    fn checker(addr: SocketAddr, settings: Settings) -> Arc<RevokeChecker> {
        RevokeChecker::connect_with(&addr.to_string(), settings).expect("valid target")
    }

    /// Short pauses, so the schedule runs in milliseconds.
    fn short_pauses() -> Settings {
        Settings {
            pause_min: Duration::from_millis(200),
            pause_step: Duration::from_millis(400),
            pause_max: Duration::from_secs(2),
            ..Settings::default()
        }
    }

    fn outcomes(checker: &RevokeChecker, outcome: &str) -> u64 {
        checker.metrics.checks.with_label_values(&[outcome]).get()
    }

    fn lookups(checker: &RevokeChecker, result: &str) -> u64 {
        checker.metrics.calls.with_label_values(&[result]).get()
    }

    async fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !done() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    #[test]
    fn targets_must_be_host_port() {
        for good in [
            "authz-gateway.authz-gateway.svc.cluster.local:50051",
            "10.0.0.1:1",
            "[::1]:50051",
            "h:65535",
        ] {
            assert!(is_grpc_target(good), "{good}");
        }
        for bad in [
            "",
            " ",
            "authz",
            "authz:",
            ":50051",
            "authz:0",
            "authz:65536",
            "authz:port",
            "http://authz:50051",
            "authz:50051/x",
            "a/b:1",
            "::1:50051",
            "[::1]",
            "[]:1",
            "user@authz:1",
            "authz :1",
        ] {
            assert!(!is_grpc_target(bad), "{bad:?}");
        }
    }

    #[test]
    fn no_runtime_leaves_the_check_off() {
        assert!(RevokeChecker::connect("authz:50051").is_none());
    }

    #[tokio::test]
    async fn from_env_reads_the_target() {
        for (value, on) in [
            (None, false),
            (Some(""), false),
            (Some("  "), false),
            (Some("not a target"), false),
            (Some(" authz:50051 "), true),
        ] {
            let checker = temp_env::with_var(
                env_llm::REVOKE_AUTH_GRPC_TARGET,
                value,
                RevokeChecker::from_env,
            );
            assert_eq!(checker.is_some(), on, "{value:?}");
        }
    }

    #[tokio::test]
    async fn invalid_target_leaves_the_check_off() {
        assert!(RevokeChecker::connect("").is_none());
        assert!(RevokeChecker::connect("http://authz:50051").is_none());
        assert!(RevokeChecker::connect("authz:50051").is_some());
    }

    #[tokio::test]
    async fn answers_and_caches_both_ways() {
        let gateway = Arc::new(FakeGateway::default());
        gateway.revoked.lock().insert(R.to_string());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(addr, Settings::default());

        assert!(checker.revoked(R).await);
        assert!(!checker.revoked(N).await);
        assert_eq!(gateway.calls(), 2);
        assert!(checker.revoked(R).await);
        assert!(!checker.revoked(N).await);
        assert_eq!(gateway.calls(), 2, "both answers are reused");
        assert_eq!(outcomes(&checker, "revoked"), 2);
        assert_eq!(outcomes(&checker, "not_revoked"), 2);
        assert_eq!(lookups(&checker, "ok"), 2);
        assert_eq!(lookups(&checker, "error"), 0);
    }

    #[tokio::test]
    async fn tokens_without_a_uuid_are_allowed_without_a_call() {
        let gateway = Arc::new(FakeGateway::default());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(addr, Settings::default());
        for token_uuid in ["", "  ", "not-a-uuid", "5b0c6f8e-3c1a-4e6b-9d2f"] {
            assert!(!checker.revoked(token_uuid).await);
        }
        assert_eq!(gateway.calls(), 0);
        assert_eq!(outcomes(&checker, "no_token_uuid"), 4);
    }

    #[tokio::test]
    async fn any_uuid_spelling_is_asked_in_canonical_form() {
        let gateway = Arc::new(FakeGateway::default());
        gateway.revoked.lock().insert(R.to_string());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(addr, Settings::default());

        assert!(checker.revoked(&format!(" {} ", R.to_uppercase())).await);
        assert!(
            checker.revoked(&R.replace('-', "")).await,
            "the same id, from the cache"
        );
        assert_eq!(*gateway.asked.lock(), vec![R.to_string()]);
    }

    #[tokio::test]
    async fn a_revoked_answer_is_kept() {
        let gateway = Arc::new(FakeGateway::default());
        gateway.revoked.lock().insert(R.to_string());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(
            addr,
            Settings {
                ttl: Duration::from_millis(50),
                refresh_after: Duration::from_millis(10),
                ..Settings::default()
            },
        );

        assert!(checker.revoked(R).await);
        gateway.revoked.lock().clear();
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(checker.revoked(R).await, "past the TTL, still revoked");
        assert_eq!(gateway.calls(), 1);
    }

    #[tokio::test]
    async fn not_revoked_is_refreshed_in_the_background_then_expires() {
        let gateway = Arc::new(FakeGateway::default());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(
            addr,
            Settings {
                ttl: Duration::from_millis(400),
                refresh_after: Duration::from_millis(150),
                ..Settings::default()
            },
        );

        assert!(!checker.revoked(N).await);
        gateway.revoked.lock().insert(N.to_string());
        assert!(!checker.revoked(N).await, "fresh entry: no call");
        assert_eq!(gateway.calls(), 1);

        tokio::time::sleep(Duration::from_millis(180)).await;
        assert!(
            !checker.revoked(N).await,
            "old entry: still used, refreshed behind it"
        );
        wait_until("the refresh", || {
            gateway.calls() == 2 && checker.state.lock().inflight.is_empty()
        })
        .await;
        assert!(checker.revoked(N).await, "the refresh saw the revoke");
        assert_eq!(gateway.calls(), 2);

        // An entry past the TTL is asked again inline.
        let other = "11111111-2222-4333-8444-555555555555";
        assert!(!checker.revoked(other).await);
        tokio::time::sleep(Duration::from_millis(450)).await;
        let before = gateway.calls();
        assert!(!checker.revoked(other).await);
        assert_eq!(gateway.calls(), before + 1);
    }

    #[tokio::test]
    async fn concurrent_misses_share_one_call() {
        let gateway = Arc::new(FakeGateway::default());
        *gateway.mode.lock() = Mode::Slow(Duration::from_millis(100));
        gateway.revoked.lock().insert(R.to_string());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(addr, Settings::default());

        let answers = futures::future::join_all((0..20).map(|_| checker.revoked(R))).await;
        assert!(answers.into_iter().all(|revoked| revoked));
        assert_eq!(gateway.calls(), 1);
        assert_eq!(lookups(&checker, "ok"), 1);
        assert_eq!(outcomes(&checker, "revoked"), 20);
    }

    #[tokio::test]
    async fn a_call_outlives_a_caller_that_hangs_up() {
        let gateway = Arc::new(FakeGateway::default());
        *gateway.mode.lock() = Mode::Slow(Duration::from_millis(100));
        gateway.revoked.lock().insert(R.to_string());
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(addr, Settings::default());

        let caller = {
            let checker = checker.clone();
            tokio::spawn(async move { checker.revoked(R).await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        caller.abort();
        wait_until("the call to finish", || {
            checker.state.lock().inflight.is_empty()
        })
        .await;
        assert!(checker.revoked(R).await, "answered from the cache");
        assert_eq!(gateway.calls(), 1);
    }

    #[tokio::test]
    async fn pause_schedule() {
        let checker = RevokeChecker::connect("authz:50051").unwrap();
        let seconds: Vec<u64> = (0..40).map(|n| checker.pause_for(n).as_secs()).collect();
        let mut expected = Vec::new();
        for secs in [1, 2, 4, 6, 8, 10, 10, 10] {
            expected.extend([secs; 5]);
        }
        assert_eq!(seconds, expected);
        assert_eq!(checker.pause_for(u32::MAX), PAUSE_MAX);
    }

    #[tokio::test]
    async fn unavailable_pauses_from_the_third_in_a_row() {
        let gateway = Arc::new(FakeGateway::default());
        *gateway.mode.lock() = Mode::Unavailable;
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(addr, short_pauses());
        let ids: Vec<String> = (0..12).map(|_| Uuid::new_v4().to_string()).collect();

        for id in &ids[..3] {
            assert!(!checker.revoked(id).await);
        }
        assert_eq!(gateway.calls(), 3);
        assert!(!checker.revoked(&ids[3]).await);
        assert_eq!(gateway.calls(), 3, "paused after the third");
        assert_eq!(outcomes(&checker, "error"), 3);
        assert_eq!(outcomes(&checker, "paused"), 1);

        // Still failing after the pause: the next failure pauses at once.
        tokio::time::sleep(Duration::from_millis(220)).await;
        assert!(!checker.revoked(&ids[4]).await);
        assert!(!checker.revoked(&ids[5]).await);
        assert_eq!(gateway.calls(), 4);

        // Back: a successful call resets the count, so it takes three failures again.
        *gateway.mode.lock() = Mode::Answer;
        tokio::time::sleep(Duration::from_millis(220)).await;
        assert!(!checker.revoked(&ids[6]).await);
        assert_eq!(lookups(&checker, "ok"), 1);
        *gateway.mode.lock() = Mode::Unavailable;
        for id in &ids[7..10] {
            assert!(!checker.revoked(id).await);
        }
        assert_eq!(gateway.calls(), 8);
        assert!(!checker.revoked(&ids[10]).await);
        assert_eq!(gateway.calls(), 8);
    }

    #[tokio::test]
    async fn a_timeout_or_unimplemented_pauses_at_once() {
        for mode in [Mode::Slow(Duration::from_millis(300)), Mode::Unimplemented] {
            let gateway = Arc::new(FakeGateway::default());
            *gateway.mode.lock() = mode;
            gateway.revoked.lock().insert(R.to_string());
            let (addr, _stop) = start(gateway.clone()).await;
            let checker = checker(
                addr,
                Settings {
                    call_timeout: Duration::from_millis(100),
                    ..short_pauses()
                },
            );

            let started = Instant::now();
            assert!(!checker.revoked(R).await, "{mode:?}: a failed call allows");
            assert!(
                started.elapsed() < Duration::from_millis(250),
                "{mode:?}: bounded by the call timeout"
            );
            assert!(!checker.revoked(R).await, "{mode:?}");
            assert_eq!(gateway.calls(), 1, "{mode:?}: paused after one");
            assert_eq!(outcomes(&checker, "error"), 1);
            assert_eq!(outcomes(&checker, "paused"), 1);
            assert_eq!(lookups(&checker, "error"), 1);
        }
    }

    #[tokio::test]
    async fn gateway_down_then_back() {
        // A free port with nothing listening: connections are refused.
        let addr = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap();
        let checker = checker(addr, short_pauses());
        for _ in 0..3 {
            assert!(!checker.revoked(R).await);
        }
        assert_eq!(
            lookups(&checker, "error"),
            3,
            "refused is UNAVAILABLE: three calls before a pause"
        );
        assert!(!checker.revoked(R).await);
        assert_eq!(outcomes(&checker, "paused"), 1);

        let gateway = Arc::new(FakeGateway::default());
        gateway.revoked.lock().insert(R.to_string());
        let _stop = serve(gateway.clone(), TcpListener::bind(addr).await.unwrap());
        tokio::time::sleep(Duration::from_millis(220)).await;
        assert!(checker.revoked(R).await, "reconnected after the pause");
        assert_eq!(gateway.calls(), 1);
        let state = checker.state.lock();
        assert_eq!((state.failures, state.pauses), (0, 0));
    }

    #[tokio::test]
    async fn failures_are_logged_at_most_once_per_pause_max() {
        let gateway = Arc::new(FakeGateway::default());
        *gateway.mode.lock() = Mode::Unimplemented;
        let (addr, _stop) = start(gateway.clone()).await;
        let checker = checker(
            addr,
            Settings {
                pause_min: Duration::from_millis(20),
                ..Settings::default()
            },
        );
        assert!(!checker.revoked(R).await);
        let first = checker.state.lock().last_failure_log.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(!checker.revoked(N).await);
        assert_eq!(gateway.calls(), 2);
        assert_eq!(checker.state.lock().last_failure_log, Some(first));
    }

    #[tokio::test]
    async fn the_cache_drops_a_tenth_at_the_cap() {
        let checker = RevokeChecker::connect_with(
            "authz:50051",
            Settings {
                capacity: 20,
                ..Settings::default()
            },
        )
        .unwrap();
        let now = Instant::now();
        let mut state = checker.state.lock();
        let ids: Vec<Uuid> = (0..21).map(|_| Uuid::new_v4()).collect();
        for id in &ids[..20] {
            checker.put_locked(&mut state, *id, false, now);
        }
        checker.put_locked(&mut state, ids[0], true, now);
        assert_eq!(state.cache.len(), 20, "replacing an entry drops nothing");
        checker.put_locked(&mut state, ids[20], true, now);
        assert_eq!(state.cache.len(), 19);
        assert!(state.cache[&ids[20]].revoked);
    }

    #[tokio::test]
    async fn metrics_are_registered_under_go_proxys_names() {
        let gateway = Arc::new(FakeGateway::default());
        let (addr, _stop) = start(gateway).await;
        let checker = checker(addr, Settings::default());
        let registry = Registry::new();
        checker.register_metrics(&registry).unwrap();
        checker.revoked(R).await;
        checker.revoked("").await;
        let names: Vec<String> = registry
            .gather()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        assert!(
            names.contains(&"proxy_revoke_checks_total".to_string()),
            "{names:?}"
        );
        assert!(
            names.contains(&"proxy_revoke_store_calls_total".to_string()),
            "{names:?}"
        );
    }
}
