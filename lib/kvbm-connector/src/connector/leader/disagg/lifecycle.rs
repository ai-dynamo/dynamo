// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared lifecycle-watcher helper for the conditional-disagg coordinators.
//!
//! Both [`super::decode::RemotePrefillCoordinator`] and
//! [`super::prefill_coordinator::PrefillCoordinatorImpl`] spawn a near-
//! identical tokio task per request:
//!
//! 1. Subscribe to the session's [`LifecycleEvent`] stream.
//! 2. Drive it to a terminal `Detached` / `Failed` (cooperative or
//!    velo-heartbeat-driven), or fall back to a defensive watchdog.
//! 3. Run an `on_evict` closure that drops the per-request state map
//!    entry — releasing the last `Arc<dyn Session>` strong ref so the
//!    per-session sender task drains and exits.
//!
//! Today both coordinators inline that task. This helper consolidates
//! the pattern so future coordinator code (and the upcoming unified
//! leader's flow refactor) can share the timing + audit semantics
//! without duplicating the watch loop.
//!
//! ### Why a helper, not a refactor of the existing call sites
//!
//! The two existing call sites are stable — switching them to this
//! helper changes audit-event ordering and (subtly) error semantics.
//! The migration is intentionally split: introduce the helper now,
//! keep both inline copies running, and swap them after the unified
//! leader has soaked. See `/home/ryan/.claude/plans/cd-session-refactor.md`.

use std::time::Duration;

use futures::StreamExt;
use kvbm_engine::disagg::session::{LifecycleEvent, Session};
use std::sync::Arc;
use tokio::runtime::Handle;

/// Defensive watchdog used by both coordinators today.  Kept here as
/// the canonical default so callers can opt in without re-deriving it.
pub const LIFECYCLE_WATCHDOG: Duration = Duration::from_secs(60);

/// Outcome reported by [`spawn_lifecycle_watcher`]'s wait loop.  Returned
/// to the caller's `on_evict` closure so the audit/log trail can record
/// *why* eviction fired.
#[derive(Debug, Clone)]
pub enum LifecycleOutcome {
    /// Peer cooperatively detached (Frame::Finished rendezvous OR
    /// velo `Finalized` sentinel).  `reason` is the optional reason
    /// string the peer attached to the detach.
    Detached { reason: Option<String> },
    /// Session entered a terminal failed state (peer-reported error
    /// or velo heartbeat surfaced a Failed event).
    Failed { reason: String },
    /// Watchdog fired before any terminal event arrived.  Defensive
    /// signal — under normal operation Detached/Failed lands within
    /// milliseconds of the cooperative finalize on each side.
    WatchdogTimeout { watchdog: Duration },
    /// Lifecycle stream returned `None` without yielding a terminal
    /// event.  Possible if the session was dropped before the watcher
    /// observed Detached/Failed.  Treated as a clean detach for
    /// eviction purposes.
    StreamEnded,
}

/// Spawn a per-request lifecycle watcher.
///
/// Reads `session.lifecycle()` until the first `Detached` or `Failed`
/// event, or until `watchdog` elapses, whichever comes first.  Calls
/// `on_evict(outcome)` exactly once on the spawned task — the caller
/// uses that closure to drop per-request state map entries (which in
/// turn drops the last `Arc<dyn Session>` strong ref).
///
/// `audit_role` is the literal `"decode"` / `"prefill"` tag emitted
/// alongside the watchdog audit event so traces remain searchable
/// per role.  `request_id` and `session_id` (formatted by the caller)
/// are interpolated into the audit event for correlation.
///
/// The closure runs on the tokio runtime task — keep it small and
/// non-blocking.  Idempotent eviction is the caller's responsibility
/// (e.g. `DashMap::remove` returning `None` on a missing entry is the
/// expected pattern).
pub fn spawn_lifecycle_watcher<F>(
    runtime: &Handle,
    session: Arc<dyn Session>,
    audit_role: &'static str,
    request_id: String,
    session_id: String,
    watchdog: Duration,
    on_evict: F,
) where
    F: FnOnce(LifecycleOutcome) + Send + 'static,
{
    runtime.spawn(async move {
        let outcome = watch_until_terminal(&session, watchdog).await;
        match &outcome {
            LifecycleOutcome::Detached { reason } => {
                crate::audit!(
                    "lifecycle_detached",
                    role = audit_role,
                    request_id = %request_id,
                    session_id = %session_id,
                    reason = ?reason
                );
            }
            LifecycleOutcome::Failed { reason } => {
                crate::audit!(
                    "lifecycle_failed",
                    role = audit_role,
                    request_id = %request_id,
                    session_id = %session_id,
                    reason = %reason
                );
            }
            LifecycleOutcome::WatchdogTimeout { watchdog } => {
                crate::audit!(
                    "lifecycle_watchdog_fired",
                    role = audit_role,
                    request_id = %request_id,
                    session_id = %session_id,
                    watchdog_secs = watchdog.as_secs()
                );
                tracing::warn!(
                    role = audit_role,
                    request_id = %request_id,
                    session_id = %session_id,
                    watchdog_secs = watchdog.as_secs(),
                    "lifecycle watchdog fired without Detached; \
                     evicting per-request state (potential session leak signal)"
                );
            }
            LifecycleOutcome::StreamEnded => {
                crate::audit!(
                    "lifecycle_stream_ended",
                    role = audit_role,
                    request_id = %request_id,
                    session_id = %session_id
                );
            }
        }
        on_evict(outcome);
    });
}

async fn watch_until_terminal(
    session: &Arc<dyn Session>,
    watchdog: Duration,
) -> LifecycleOutcome {
    let result = tokio::time::timeout(watchdog, async {
        let mut lifecycle = session.lifecycle();
        while let Some(event) = lifecycle.next().await {
            match event {
                LifecycleEvent::Detached { reason } => {
                    return LifecycleOutcome::Detached { reason };
                }
                LifecycleEvent::Failed { reason } => {
                    return LifecycleOutcome::Failed { reason };
                }
                LifecycleEvent::Attached { .. } => continue,
            }
        }
        LifecycleOutcome::StreamEnded
    })
    .await;

    match result {
        Ok(outcome) => outcome,
        Err(_) => LifecycleOutcome::WatchdogTimeout { watchdog },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_default_is_60_secs() {
        assert_eq!(LIFECYCLE_WATCHDOG, Duration::from_secs(60));
    }

    #[test]
    fn outcome_variants_are_constructible() {
        let _detached = LifecycleOutcome::Detached {
            reason: Some("released".into()),
        };
        let _failed = LifecycleOutcome::Failed {
            reason: "boom".into(),
        };
        let _wd = LifecycleOutcome::WatchdogTimeout {
            watchdog: Duration::from_secs(1),
        };
        let _ended = LifecycleOutcome::StreamEnded;
    }
}
