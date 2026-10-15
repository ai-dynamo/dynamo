// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker validation gates.
//!
//! A worker configured with `DYN_WORKER_VALIDATION_GATES` runs each gate once
//! as soon as its engine is serving, and reports not ready on `/health` until
//! every gate passes. `/live` is unaffected, so a held worker is not restarted.
//!
//! Gates run in two phases, each in parallel. [`GatePhase::BeforePublish`]
//! gates run while the model card's discovery write is held (see
//! `dynamo_llm::validation_hold`); once they all pass the card is published,
//! and only then do the [`GatePhase::AfterPublish`] gates start.
//!
//! Verdicts live only in this process's memory: a restarted worker container
//! re-runs every gate. A failed gate keeps the worker held for the life of the
//! process; deleting the pod is the retry.

pub mod config;
mod runner;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::future::BoxFuture;
use serde_json::json;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub use config::{GateCheck, GatePhase, GateSpec, ValidationConfig};

/// Work that must finish after every gate passes and before the worker
/// reports ready, such as writing held model cards. An error keeps the worker
/// held.
pub type ReleaseHook = Box<dyn FnOnce() -> BoxFuture<'static, anyhow::Result<()>> + Send>;

/// Work that must finish after every `before_publish` gate passes and before
/// any `after_publish` gate starts, such as writing held model cards. An error
/// keeps the worker held.
pub type PublishHook = ReleaseHook;

/// Environment passed to every gate.
#[derive(Clone, Debug, Default)]
pub struct GateEnv {
    /// Loopback OpenAI-compatible endpoint serving this worker's model.
    pub openai_base_url: Option<String>,
    /// Model name to request from that endpoint.
    pub model: Option<String>,
}

impl GateEnv {
    fn vars(&self, gate: &str) -> Vec<(&'static str, String)> {
        let mut vars = vec![("DYN_VALIDATION_GATE", gate.to_string())];
        if let Some(url) = &self.openai_base_url {
            vars.push(("OPENAI_BASE_URL", url.clone()));
        }
        if let Some(model) = &self.model {
            vars.push(("DYN_VALIDATION_MODEL", model.clone()));
        }
        vars
    }
}

enum HookSlot {
    Empty,
    Set(ReleaseHook),
    Taken,
}

/// One gate's verdict so far.
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
    gates: parking_lot::Mutex<BTreeMap<String, GateState>>,
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
            .field("gates", &self.gates.lock())
            .finish_non_exhaustive()
    }
}

impl WorkerValidation {
    pub fn new(config: ValidationConfig) -> Arc<Self> {
        let gates = config
            .gates
            .iter()
            .map(|gate| (gate.name.clone(), GateState::Pending))
            .collect();
        Arc::new(Self {
            config,
            gates: parking_lot::Mutex::new(gates),
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

    /// Run `hook` once every `before_publish` gate passes, before any
    /// `after_publish` gate starts. A worker has at most one hook. Returns `false`
    /// once publishing has begun: the caller should then do the work itself.
    pub fn set_publish_hook(&self, hook: PublishHook) -> bool {
        set_hook(&self.publish_hook, hook)
    }

    /// Run `hook` once every gate passes, before the worker is released. A
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

    /// True once every gate has passed. A released worker is never held again.
    pub fn is_released(&self) -> bool {
        *self.released.borrow()
    }

    /// Resolves when the worker is released. Never resolves if a gate fails.
    pub async fn wait_released(&self) {
        let _ = self
            .released
            .subscribe()
            .wait_for(|released| *released)
            .await;
    }

    /// Gate state for the `/health` body.
    pub fn status_json(&self) -> serde_json::Value {
        let gates: serde_json::Map<_, _> = self
            .gates
            .lock()
            .iter()
            .map(|(name, state)| {
                let entry = match state {
                    GateState::Pending => json!({"state": "pending"}),
                    GateState::Running => json!({"state": "running"}),
                    GateState::Passed => json!({"state": "passed"}),
                    GateState::Failed(reason) => json!({"state": "failed", "reason": reason}),
                };
                (name.clone(), entry)
            })
            .collect();
        json!({"released": self.is_released(), "gates": gates})
    }

    /// Start every gate, once; later calls do nothing. `cancel` stops the
    /// gates on shutdown, failing them.
    pub fn start(self: &Arc<Self>, env: GateEnv, cancel: CancellationToken) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move { this.run(env, cancel).await });
    }

    async fn run(&self, env: GateEnv, cancel: CancellationToken) {
        if !self
            .run_gates(GatePhase::BeforePublish, &env, &cancel)
            .await
            || !run_hook(&self.publish_hook, "Publishing held model cards").await
            || !self.run_gates(GatePhase::AfterPublish, &env, &cancel).await
            || !run_hook(&self.release_hook, "Worker release").await
        {
            return;
        }
        tracing::info!("Worker validation passed; worker released");
        self.released.send_replace(true);
    }

    /// Run the gates of one phase in parallel. True if every one passed.
    async fn run_gates(&self, phase: GatePhase, env: &GateEnv, cancel: &CancellationToken) -> bool {
        let runs = self
            .config
            .gates
            .iter()
            .filter(|gate| gate.phase == phase)
            .map(|gate| async {
                self.set_state(&gate.name, GateState::Running);
                let vars = env.vars(&gate.name);
                let verdict = tokio::select! {
                    verdict = runner::run_gate(gate, &vars) => verdict,
                    _ = cancel.cancelled() => Err("worker is shutting down".to_string()),
                };
                let passed = verdict.is_ok();
                self.set_state(
                    &gate.name,
                    verdict.map_or_else(GateState::Failed, |()| GateState::Passed),
                );
                passed
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

    fn set_state(&self, gate: &str, state: GateState) {
        self.gates.lock().insert(gate.to_string(), state);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn settled(validation: &WorkerValidation) -> serde_json::Value {
        for _ in 0..200 {
            let gates = validation.gates.lock().clone();
            if gates
                .values()
                .all(|state| matches!(state, GateState::Passed | GateState::Failed(_)))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        validation.status_json()
    }

    #[tokio::test]
    async fn one_failing_gate_holds_the_worker() {
        let config = ValidationConfig::parse(Some(
            r#"{"good": {"exec": ["true"], "timeout_secs": 5}, "bad": {"exec": ["sh", "-c", "echo broken >&2; exit 1"], "timeout_secs": 5}}"#.into(),
        ));
        let validation = WorkerValidation::new(config.unwrap().unwrap());
        validation.start(GateEnv::default(), CancellationToken::new());
        let status = settled(&validation).await;
        assert!(!validation.is_released());
        assert_eq!(status["gates"]["good"]["state"], "passed");
        assert_eq!(status["gates"]["bad"]["state"], "failed");
        assert!(
            status["gates"]["bad"]["reason"]
                .as_str()
                .unwrap()
                .contains("broken")
        );
    }

    #[tokio::test]
    async fn a_failed_release_hook_keeps_the_worker_held() {
        let config = ValidationConfig::parse(Some(
            r#"{"g": {"exec": ["true"], "timeout_secs": 5}}"#.into(),
        ));
        let validation = WorkerValidation::new(config.unwrap().unwrap());
        assert!(validation.set_release_hook(Box::new(|| {
            Box::pin(async { Err(anyhow::anyhow!("discovery write failed")) })
        })));
        validation.start(GateEnv::default(), CancellationToken::new());
        settled(&validation).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!validation.is_released());
        assert!(!validation.set_release_hook(Box::new(|| Box::pin(async { Ok(()) }))));
    }

    /// Records the gate states each hook sees, in order.
    fn recording_hooks(
        validation: &Arc<WorkerValidation>,
    ) -> Arc<parking_lot::Mutex<Vec<(&'static str, serde_json::Value)>>> {
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        for (name, release) in [("publish", false), ("release", true)] {
            let (seen, this) = (Arc::clone(&seen), Arc::clone(validation));
            let hook: ReleaseHook = Box::new(move || {
                Box::pin(async move {
                    seen.lock()
                        .push((name, this.status_json()["gates"].clone()));
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

    fn two_phase(soak: &str) -> Arc<WorkerValidation> {
        let gates = format!(
            r#"{{"soak": {{"exec": ["sh", "-c", "{soak}"], "timeout_secs": 5}},
                "warm": {{"exec": ["true"], "timeout_secs": 5, "phase": "after_publish"}}}}"#
        );
        let config = ValidationConfig::parse(Some(gates));
        WorkerValidation::new(config.unwrap().unwrap())
    }

    #[tokio::test]
    async fn after_publish_gates_start_only_once_the_card_is_published() {
        let validation = two_phase("sleep 0.2");
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        let seen = seen.lock().clone();
        let names: Vec<_> = seen.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["publish", "release"]);
        assert_eq!(seen[0].1["soak"]["state"], "passed");
        assert_eq!(seen[0].1["warm"]["state"], "pending");
        assert_eq!(seen[1].1["warm"]["state"], "passed");
    }

    #[tokio::test]
    async fn a_failed_gate_before_publish_never_publishes() {
        let validation = two_phase("exit 1");
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        for _ in 0..200 {
            if validation.status_json()["gates"]["soak"]["state"] == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!validation.is_released());
        assert!(seen.lock().is_empty());
        assert_eq!(
            validation.status_json()["gates"]["warm"]["state"],
            "pending"
        );
    }

    #[tokio::test]
    async fn without_after_publish_every_gate_runs_before_either_hook() {
        let config = ValidationConfig::parse(Some(
            r#"{"a": {"exec": ["true"], "timeout_secs": 5}, "b": {"exec": ["sleep", "0.2"], "timeout_secs": 5}}"#.into(),
        ));
        let validation = WorkerValidation::new(config.unwrap().unwrap());
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        for (_, gates) in seen.lock().iter() {
            assert_eq!(gates["a"]["state"], "passed");
            assert_eq!(gates["b"]["state"], "passed");
        }
        assert_eq!(seen.lock().len(), 2);
    }

    #[tokio::test]
    async fn with_only_after_publish_gates_the_card_is_published_first() {
        let config = ValidationConfig::parse(Some(
            r#"{"warm": {"exec": ["true"], "timeout_secs": 5, "phase": "after_publish"}}"#.into(),
        ));
        let validation = WorkerValidation::new(config.unwrap().unwrap());
        let seen = recording_hooks(&validation);
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
        let seen = seen.lock().clone();
        assert_eq!(seen[0].1["warm"]["state"], "pending");
        assert_eq!(seen[1].1["warm"]["state"], "passed");
    }
}
