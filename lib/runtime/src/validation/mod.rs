// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker validation gates.
//!
//! A worker configured with `DYN_WORKER_VALIDATION_GATES` runs each gate once,
//! in parallel, as soon as its engine is serving, and reports not ready on
//! `/health` until every gate passes. `/live` is unaffected, so a held worker
//! is not restarted.
//!
//! Verdicts live only in this process's memory: a restarted worker container
//! re-runs every gate. A failed gate keeps the worker held for the life of the
//! process; deleting the pod is the retry.

pub mod config;
mod runner;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::json;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub use config::{GateCheck, GateSpec, ValidationConfig};

/// One gate's verdict so far.
#[derive(Clone, Debug, PartialEq, Eq)]
enum GateState {
    Pending,
    Running,
    Passed,
    Failed(String),
}

/// Validation state for one worker process, owned by the `DistributedRuntime`.
#[derive(Debug)]
pub struct WorkerValidation {
    config: ValidationConfig,
    gates: parking_lot::Mutex<BTreeMap<String, GateState>>,
    released: watch::Sender<bool>,
    started: AtomicBool,
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
            released: watch::channel(false).0,
            started: AtomicBool::new(false),
        })
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
    pub fn start(self: &Arc<Self>, cancel: CancellationToken) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move { this.run(cancel).await });
    }

    async fn run(&self, cancel: CancellationToken) {
        let runs = self.config.gates.iter().map(|gate| async {
            self.set_state(&gate.name, GateState::Running);
            let verdict = tokio::select! {
                verdict = runner::run_gate(gate) => verdict,
                _ = cancel.cancelled() => Err("worker is shutting down".to_string()),
            };
            let passed = verdict.is_ok();
            self.set_state(
                &gate.name,
                verdict.map_or_else(GateState::Failed, |()| GateState::Passed),
            );
            passed
        });
        if futures::future::join_all(runs)
            .await
            .into_iter()
            .all(|passed| passed)
        {
            tracing::info!("Worker validation passed; worker released");
            self.released.send_replace(true);
        } else {
            tracing::error!(status = %self.status_json(), "Worker validation failed; worker stays held");
        }
    }

    fn set_state(&self, gate: &str, state: GateState) {
        self.gates.lock().insert(gate.to_string(), state);
    }
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
        validation.start(CancellationToken::new());
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
}
