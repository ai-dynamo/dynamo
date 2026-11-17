// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker validation gates.
//!
//! A worker configured with `DYN_PUBLISH_GATE` calls each check URL, in
//! parallel, once its engine is serving. The model card's discovery write is
//! held while they run (see `dynamo_llm::validation_hold`), and `/health`
//! reports not ready until every check passes and the card is published.
//! `/live` is unaffected, so a held worker is not restarted.
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

pub use config::ValidationConfig;

/// Work that must finish after every check passes and before the worker
/// reports ready, such as writing held model cards. An error keeps the worker
/// held.
pub type PublishHook = Box<dyn FnOnce() -> BoxFuture<'static, anyhow::Result<()>> + Send>;

/// What the runtime tells every check, as the JSON body of its request. Only
/// what a check runner can't be configured with ahead of time: anything in
/// the deployment spec belongs in the runner's own configuration.
#[derive(Clone, Debug, Default)]
pub struct GateEnv {
    /// Loopback OpenAI-compatible endpoint serving this worker's model, on a
    /// port chosen at startup. Its `/models` lists the model to request.
    pub openai_base_url: Option<String>,
}

impl GateEnv {
    fn body(&self) -> serde_json::Value {
        json!({"base_url": self.openai_base_url})
    }
}

enum HookSlot {
    Empty,
    Set(PublishHook),
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
    /// One per configured check URL, in the same order.
    states: parking_lot::Mutex<Vec<GateState>>,
    publish_hook: parking_lot::Mutex<HookSlot>,
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
        let states = vec![GateState::Pending; config.urls.len()];
        Arc::new(Self {
            config,
            states: parking_lot::Mutex::new(states),
            publish_hook: parking_lot::Mutex::new(HookSlot::Empty),
            released: watch::channel(false).0,
            started: AtomicBool::new(false),
            local_endpoint_claimed: AtomicBool::new(false),
        })
    }

    /// Run `hook` once every check passes, before the worker is released. A
    /// worker has at most one hook. Returns `false` once publishing has begun:
    /// the caller should then do the work itself.
    pub fn set_publish_hook(&self, hook: PublishHook) -> bool {
        let mut slot = self.publish_hook.lock();
        if matches!(*slot, HookSlot::Taken) {
            return false;
        }
        *slot = HookSlot::Set(hook);
        true
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

    /// Check states for the `/health` body, by URL.
    pub fn status_json(&self) -> serde_json::Value {
        let mut status = json!({"released": self.is_released(), "publish": {}});
        for (url, state) in self.config.urls.iter().zip(self.states.lock().iter()) {
            let entry = match state {
                GateState::Pending => json!({"state": "pending"}),
                GateState::Running => json!({"state": "running"}),
                GateState::Passed => json!({"state": "passed"}),
                GateState::Failed(reason) => json!({"state": "failed", "reason": reason}),
            };
            status["publish"][url.as_str()] = entry;
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
        if !self.run_checks(&env, &cancel).await || !self.run_publish_hook().await {
            return;
        }
        tracing::info!("Worker validation passed; worker released");
        self.released.send_replace(true);
    }

    /// Call every check in parallel. True if every one passed.
    async fn run_checks(&self, env: &GateEnv, cancel: &CancellationToken) -> bool {
        let body = env.body();
        let runs = self.config.urls.iter().enumerate().map(|(index, url)| {
            let body = &body;
            async move {
                self.set_state(index, GateState::Running);
                let verdict = tokio::select! {
                    verdict = runner::run_gate(url, body) => verdict,
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

impl WorkerValidation {
    /// Take and run the publish hook, if any. False if it failed.
    async fn run_publish_hook(&self) -> bool {
        let hook = std::mem::replace(&mut *self.publish_hook.lock(), HookSlot::Taken);
        if let HookSlot::Set(hook) = hook
            && let Err(error) = hook().await
        {
            tracing::error!(
                error = format!("{error:#}"),
                "Publishing held model cards failed; worker stays held"
            );
            return false;
        }
        true
    }
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

    fn validation(urls: &[&str]) -> Arc<WorkerValidation> {
        let config = ValidationConfig::parse(Some(urls.join(",")));
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
        let validation = validation(&[&checks.url("pass"), &checks.url("fail")]);
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
    async fn every_check_is_told_the_loopback_endpoint() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("pass"), &checks.url("slow")]);
        let env = GateEnv {
            openai_base_url: Some("http://127.0.0.1:1234/v1".into()),
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
        let expected = json!({"base_url": "http://127.0.0.1:1234/v1"});
        assert_eq!(bodies, [expected.clone(), expected]);
    }

    #[tokio::test]
    async fn the_card_is_published_only_once_every_check_passes() {
        let checks = CheckServer::start().await;
        let (soak, smoke) = (checks.url("slow"), checks.url("pass"));
        let validation = validation(&[soak.as_str(), smoke.as_str()]);
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (recorded, this) = (Arc::clone(&seen), Arc::clone(&validation));
        assert!(validation.set_publish_hook(Box::new(move || {
            Box::pin(async move {
                recorded.lock().push(this.status_json());
                Ok(())
            })
        })));
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        let seen = seen.lock().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0]["publish"][&soak]["state"], "passed");
        assert_eq!(seen[0]["publish"][&smoke]["state"], "passed");
        assert_eq!(seen[0]["released"], false);
    }

    #[tokio::test]
    async fn a_failed_check_never_publishes() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("fail")]);
        let published = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&published);
        assert!(validation.set_publish_hook(Box::new(move || {
            Box::pin(async move {
                flag.store(true, Ordering::SeqCst);
                Ok(())
            })
        })));
        validation.start(GateEnv::default(), CancellationToken::new());
        settled(&validation).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!validation.is_released());
        assert!(!published.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_failed_publish_hook_keeps_the_worker_held() {
        let checks = CheckServer::start().await;
        let validation = validation(&[&checks.url("pass")]);
        assert!(validation.set_publish_hook(Box::new(|| {
            Box::pin(async { Err(anyhow::anyhow!("discovery write failed")) })
        })));
        validation.start(GateEnv::default(), CancellationToken::new());
        settled(&validation).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!validation.is_released());
        assert!(!validation.set_publish_hook(Box::new(|| Box::pin(async { Ok(()) }))));
    }
}
