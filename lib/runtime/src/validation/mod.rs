// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker validation gates.
//!
//! A worker configured with `DYN_PUBLISH_GATE` or `DYN_READY_GATE` calls each
//! check URL once its engine is serving, and reports not ready on `/health`
//! until every check passes. `/live` is unaffected, so a held worker is not
//! restarted.
//!
//! The two gates are release points, each called in parallel. [`Gate::Publish`]
//! checks run while the model card's discovery write is held (see
//! `dynamo_llm::validation_hold`); once they all pass the card is published,
//! and only then are the [`Gate::Ready`] checks called.
//!
//! Verdicts live only in this process's memory: a restarted worker container
//! calls every check again. A failed check keeps the worker held for the life
//! of the process; deleting the pod is the retry.

pub mod config;
mod runner;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::future::BoxFuture;
use serde_json::json;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub use config::{Gate, GateSpec, ValidationConfig};

/// Work that must finish after every gate passes and before the worker
/// reports ready, such as writing held model cards. An error keeps the worker
/// held.
pub type ReleaseHook = Box<dyn FnOnce() -> BoxFuture<'static, anyhow::Result<()>> + Send>;

/// Work that must finish after every publish check passes and before any
/// ready check is called, such as writing held model cards. An error
/// keeps the worker held.
pub type PublishHook = ReleaseHook;

/// What the runtime tells every check, as the JSON body of its request.
#[derive(Clone, Debug, Default)]
pub struct GateEnv {
    /// Loopback OpenAI-compatible endpoint serving this worker's model.
    pub openai_base_url: Option<String>,
    /// Model name to request from that endpoint.
    pub model: Option<String>,
}

impl GateEnv {
    fn body(&self, gate: Gate) -> serde_json::Value {
        json!({
            "gate": gate.as_str(),
            "base_url": self.openai_base_url,
            "models": self.model.iter().collect::<Vec<_>>(),
        })
    }
}

enum HookSlot {
    Empty,
    Set(ReleaseHook),
    Taken,
}

/// One check's verdict so far.
#[derive(Clone, Debug, PartialEq, Eq)]
enum GateState {
    Pending,
    Running,
    Passed,
    Failed(String),
}

/// Validation state for one worker process, owned by the `DistributedRuntime`.
pub struct WorkerValidation {
    config: ValidationConfig,
    /// One per configured gate, in the same order.
    states: parking_lot::Mutex<Vec<GateState>>,
    publish_hook: parking_lot::Mutex<HookSlot>,
    release_hook: parking_lot::Mutex<HookSlot>,
    released: watch::Sender<bool>,
    started: AtomicBool,
    local_endpoint_claimed: AtomicBool,
}

impl std::fmt::Debug for WorkerValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerValidation")
            .field("config", &self.config)
            .field("states", &self.states.lock())
            .finish_non_exhaustive()
    }
}

impl WorkerValidation {
    pub fn new(config: ValidationConfig) -> Arc<Self> {
        let states = vec![GateState::Pending; config.gates.len()];
        Arc::new(Self {
            config,
            states: parking_lot::Mutex::new(states),
            publish_hook: parking_lot::Mutex::new(HookSlot::Empty),
            release_hook: parking_lot::Mutex::new(HookSlot::Empty),
            released: watch::channel(false).0,
            started: AtomicBool::new(false),
            local_endpoint_claimed: AtomicBool::new(false),
        })
    }

    /// See [`ValidationConfig::holds_card`].
    pub fn holds_card(&self) -> bool {
        self.config.holds_card()
    }

    /// Run `hook` once every publish check passes, before any ready check is
    /// called. A worker has at most one hook. Returns `false`
    /// once publishing has begun: the caller should then do the work itself.
    pub fn set_publish_hook(&self, hook: PublishHook) -> bool {
        set_hook(&self.publish_hook, hook)
    }

    /// Run `hook` once every check passes, before the worker is released. A
    /// worker has at most one hook. Returns `false` once release has begun:
    /// the caller should then do the work itself.
    pub fn set_release_hook(&self, hook: ReleaseHook) -> bool {
        set_hook(&self.release_hook, hook)
    }

    /// Claim the loopback endpoint the gates check. Only the first caller gets
    /// `true`: a worker serves one model to its gates.
    pub fn claim_local_endpoint(&self) -> bool {
        !self.local_endpoint_claimed.swap(true, Ordering::SeqCst)
    }

    /// True once every check has passed. A released worker is never held again.
    pub fn is_released(&self) -> bool {
        *self.released.borrow()
    }

    /// Resolves when the worker is released. Never resolves if a check fails.
    pub async fn wait_released(&self) {
        let _ = self
            .released
            .subscribe()
            .wait_for(|released| *released)
            .await;
    }

    /// Check states for the `/health` body, by gate and then URL.
    pub fn status_json(&self) -> serde_json::Value {
        let mut status = json!({"released": self.is_released(), "publish": {}, "ready": {}});
        for (spec, state) in self.config.gates.iter().zip(self.states.lock().iter()) {
            let entry = match state {
                GateState::Pending => json!({"state": "pending"}),
                GateState::Running => json!({"state": "running"}),
                GateState::Passed => json!({"state": "passed"}),
                GateState::Failed(reason) => json!({"state": "failed", "reason": reason}),
            };
            status[spec.gate.as_str()][spec.url.as_str()] = entry;
        }
        status
    }

    /// Start the checks, once; later calls do nothing. `cancel` stops them
    /// on shutdown, failing them.
    pub fn start(self: &Arc<Self>, env: GateEnv, cancel: CancellationToken) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move { this.run(env, cancel).await });
    }

    async fn run(&self, env: GateEnv, cancel: CancellationToken) {
        if !self.run_gate(Gate::Publish, &env, &cancel).await
            || !run_hook(&self.publish_hook, "Publishing held model cards").await
            || !self.run_gate(Gate::Ready, &env, &cancel).await
            || !run_hook(&self.release_hook, "Worker release").await
        {
            return;
        }
        tracing::info!("Worker validation passed; worker released");
        self.released.send_replace(true);
    }

    /// Call one gate's checks in parallel. True if every one passed.
    async fn run_gate(&self, gate: Gate, env: &GateEnv, cancel: &CancellationToken) -> bool {
        let body = env.body(gate);
        let runs = self
            .config
            .gates
            .iter()
            .enumerate()
            .filter(|(_, spec)| spec.gate == gate)
            .map(|(index, spec)| {
                let body = &body;
                async move {
                    self.set_state(index, GateState::Running);
                    let verdict = tokio::select! {
                        verdict = runner::run_gate(&spec.url, body) => verdict,
                        _ = cancel.cancelled() => Err("worker is shutting down".to_string()),
                    };
                    let passed = verdict.is_ok();
                    self.set_state(
                        index,
                        verdict.map_or_else(GateState::Failed, |()| GateState::Passed),
                    );
                    passed
                }
            });
        let passed = futures::future::join_all(runs)
            .await
            .into_iter()
            .all(|passed| passed);
        if !passed {
            tracing::error!(status = %self.status_json(), "Worker validation failed; worker stays held");
        }
        passed
    }

    fn set_state(&self, index: usize, state: GateState) {
        self.states.lock()[index] = state;
    }
}

fn set_hook(slot: &parking_lot::Mutex<HookSlot>, hook: ReleaseHook) -> bool {
    let mut slot = slot.lock();
    if matches!(*slot, HookSlot::Taken) {
        return false;
    }
    *slot = HookSlot::Set(hook);
    true
}

/// Take and run the hook in `slot`, if any. False if it failed.
async fn run_hook(slot: &parking_lot::Mutex<HookSlot>, what: &str) -> bool {
    let hook = std::mem::replace(&mut *slot.lock(), HookSlot::Taken);
    if let HookSlot::Set(hook) = hook
        && let Err(error) = hook().await
    {
        tracing::error!(
            error = format!("{error:#}"),
            "{what} failed; worker stays held"
        );
        return false;
    }
    true
}

/// A loopback check runner for tests.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::{Json, Router, extract::Path, http::StatusCode, routing::post};

    /// Answers `POST /pass` with 200, `/fail` with 500 and the reason
    /// `check failed`, and `/slow` with 200 after a second. Records every
    /// request's path and body.
    pub struct CheckServer {
        addr: SocketAddr,
        requests: Arc<parking_lot::Mutex<Vec<(String, serde_json::Value)>>>,
    }

    impl CheckServer {
        pub async fn start() -> Self {
            let requests = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let seen = Arc::clone(&requests);
            let app = Router::new().route(
                "/{check}",
                post(
                    move |Path(check): Path<String>, Json(body): Json<serde_json::Value>| {
                        let seen = Arc::clone(&seen);
                        async move {
                            seen.lock().push((check.clone(), body));
                            match check.as_str() {
                                "pass" => (StatusCode::OK, ""),
                                "slow" => {
                                    tokio::time::sleep(Duration::from_secs(1)).await;
                                    (StatusCode::OK, "")
                                }
                                _ => (StatusCode::INTERNAL_SERVER_ERROR, "check failed"),
                            }
                        }
                    },
                ),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await });
            Self { addr, requests }
        }

        /// The URL of one check, such as `pass`.
        pub fn url(&self, check: &str) -> String {
            format!("http://{}/{check}", self.addr)
        }

        /// Every request so far, as `(check, body)`.
        pub fn requests(&self) -> Vec<(String, serde_json::Value)> {
            self.requests.lock().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::CheckServer;
    use super::*;
    use std::time::Duration;

    fn validation(publish: &[&str], ready: &[&str]) -> Arc<WorkerValidation> {
        let config = ValidationConfig::parse(Some(publish.join(",")), Some(ready.join(",")));
        WorkerValidation::new(config.unwrap().unwrap())
    }

    async fn settled(validation: &WorkerValidation) -> serde_json::Value {
        for _ in 0..200 {
            let states = validation.states.lock().clone();
            if states
                .iter()
                .all(|state| matches!(state, GateState::Passed | GateState::Failed(_)))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        validation.status_json()
    }

    #[tokio::test]
    async fn one_failing_check_holds_the_worker() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("pass"), &checks.url("fail")], &[]);
        validation.start(GateEnv::default(), CancellationToken::new());
        let status = settled(&validation).await;
        assert!(!validation.is_released());
        assert_eq!(status["publish"][checks.url("pass")]["state"], "passed");
        let failed = &status["publish"][checks.url("fail")];
        assert_eq!(failed["state"], "failed");
        assert!(
            failed["reason"].as_str().unwrap().contains("check failed"),
            "{status}"
        );
    }

    #[tokio::test]
    async fn every_check_is_told_its_gate_endpoint_and_model() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("pass")], &[&checks.url("pass")]);
        let env = GateEnv {
            openai_base_url: Some("http://127.0.0.1:1234/v1".into()),
            model: Some("m".into()),
        };
        validation.start(env, CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        let bodies: Vec<_> = checks
            .requests()
            .into_iter()
            .map(|(_, body)| body)
            .collect();
        let expected =
            |gate| json!({"gate": gate, "base_url": "http://127.0.0.1:1234/v1", "models": ["m"]});
        assert_eq!(bodies, [expected("publish"), expected("ready")]);
    }

    #[tokio::test]
    async fn a_failed_release_hook_keeps_the_worker_held() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("pass")], &[]);
        assert!(validation.set_release_hook(Box::new(|| {
            Box::pin(async { Err(anyhow::anyhow!("discovery write failed")) })
        })));
        validation.start(GateEnv::default(), CancellationToken::new());
        settled(&validation).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!validation.is_released());
        assert!(!validation.set_release_hook(Box::new(|| Box::pin(async { Ok(()) }))));
    }

    /// Records the check states each hook sees, in order.
    fn recording_hooks(
        validation: &Arc<WorkerValidation>,
    ) -> Arc<parking_lot::Mutex<Vec<(&'static str, serde_json::Value)>>> {
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        for (name, release) in [("publish", false), ("release", true)] {
            let (seen, this) = (Arc::clone(&seen), Arc::clone(validation));
            let hook: ReleaseHook = Box::new(move || {
                Box::pin(async move {
                    seen.lock().push((name, this.status_json()));
                    Ok(())
                })
            });
            assert!(if release {
                validation.set_release_hook(hook)
            } else {
                validation.set_publish_hook(hook)
            });
        }
        seen
    }

    #[tokio::test]
    async fn ready_checks_are_called_only_once_the_card_is_published() {
        let checks = CheckServer::start().await;
        let (soak, warm) = (checks.url("slow"), checks.url("pass"));
        let validation = validation(&[soak.as_str()], &[warm.as_str()]);
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        let seen = seen.lock().clone();
        let names: Vec<_> = seen.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["publish", "release"]);
        assert_eq!(seen[0].1["publish"][&soak]["state"], "passed");
        assert_eq!(seen[0].1["ready"][&warm]["state"], "pending");
        assert_eq!(seen[1].1["ready"][&warm]["state"], "passed");
    }

    #[tokio::test]
    async fn a_failed_publish_check_never_publishes() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("fail")], &[&checks.url("pass")]);
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        for _ in 0..200 {
            if validation.status_json()["publish"][checks.url("fail")]["state"] == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!validation.is_released());
        assert!(seen.lock().is_empty());
        assert_eq!(
            validation.status_json()["ready"][checks.url("pass")]["state"],
            "pending"
        );
    }

    #[tokio::test]
    async fn with_only_ready_checks_the_card_is_published_first() {
        let checks = CheckServer::start().await;
        let warm = checks.url("pass");
        let validation = validation(&[], &[warm.as_str()]);
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        let seen = seen.lock().clone();
        assert_eq!(seen[0].1["ready"][&warm]["state"], "pending");
        assert_eq!(seen[1].1["ready"][&warm]["state"], "passed");
    }
}
