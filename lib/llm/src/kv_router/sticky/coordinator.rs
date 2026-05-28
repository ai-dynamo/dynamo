// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coordination layer for sticky routing and backend session lifecycle.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_runtime::component::Component;

use crate::{
    preprocessor::PreprocessedRequest,
    protocols::{
        common::{preprocessor::RoutingHints, timing::RequestPhase},
        openai::nvext::SessionAction,
    },
};

use super::{
    lifecycle::{SessionCloseAction, SessionLifecycleController},
    router::{AffinityBinding, AffinityKind, InMemoryAffinityStore, StickySessionRouter},
};

pub struct SessionRoutingResult {
    pub deferred_close: Option<SessionCloseAction>,
}

pub struct StickySessionCoordinator {
    router: Arc<StickySessionRouter>,
    lifecycle: Arc<SessionLifecycleController>,
}

impl StickySessionCoordinator {
    pub fn new(component: Component) -> Self {
        let lifecycle = Arc::new(SessionLifecycleController::new(component));
        let on_expire = {
            let lifecycle = lifecycle.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                lifecycle
                    .clone()
                    .close_expired_session(session_id, worker_id);
            }) as Arc<dyn Fn(String, u64) + Send + Sync>
        };
        let router = Arc::new(StickySessionRouter::new(
            InMemoryAffinityStore::new_with_on_expire(Some(on_expire)),
        ));

        StickySessionCoordinator { router, lifecycle }
    }

    pub fn worker_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<WorkerWithDpRank> {
        let session_id = sticky_session_id_for_phase(request, phase)?;
        self.router.peek_session(session_id)
    }

    pub fn refresh_worker_for_phase(&self, request: &PreprocessedRequest, phase: RequestPhase) {
        let Some(session_id) = sticky_session_id_for_phase(request, phase) else {
            return;
        };
        let Some(sc) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.as_ref())
        else {
            return;
        };
        if sc.action.is_some() {
            return;
        }

        self.router.resolve_session(session_id);
    }

    pub fn unbind_for_phase<'a>(
        &self,
        request: &'a PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<(&'a str, Option<AffinityBinding>)> {
        let session_id = sticky_session_id_for_phase(request, phase)?;
        Some((session_id, self.router.unbind(session_id)))
    }

    pub async fn on_routed(
        &self,
        request: &PreprocessedRequest,
        worker: WorkerWithDpRank,
        context_id: &str,
    ) -> Result<SessionRoutingResult> {
        let sc = request
            .routing
            .as_ref()
            .and_then(|r| r.session_control.as_ref());

        let Some(sc) = sc else {
            return Ok(SessionRoutingResult {
                deferred_close: None,
            });
        };
        let Some(action) = sc.action.as_ref() else {
            return Ok(SessionRoutingResult {
                deferred_close: None,
            });
        };

        match action {
            SessionAction::Open => {
                let opened = self
                    .lifecycle
                    .open_session(&sc.session_id, sc.timeout, worker.worker_id, context_id)
                    .await?;
                if opened {
                    self.bind_with_kind(sc, worker, AffinityKind::EngineBacked);
                }
                Ok(SessionRoutingResult {
                    deferred_close: None,
                })
            }
            SessionAction::Bind => {
                self.bind_with_kind(sc, worker, AffinityKind::RouterOnly);
                Ok(SessionRoutingResult {
                    deferred_close: None,
                })
            }
            SessionAction::Close => {
                let should_close_worker_session = self
                    .router
                    .unbind(&sc.session_id)
                    .map(|binding| binding.kind == AffinityKind::EngineBacked)
                    .unwrap_or(true);
                let deferred_close = if should_close_worker_session {
                    self.lifecycle
                        .deferred_close(sc.session_id.clone(), worker.worker_id)
                        .await
                } else {
                    None
                };
                Ok(SessionRoutingResult { deferred_close })
            }
        }
    }

    fn bind_with_kind(
        &self,
        sc: &crate::protocols::openai::nvext::SessionControl,
        worker: WorkerWithDpRank,
        kind: AffinityKind,
    ) {
        let ttl = Duration::from_secs(sc.timeout);
        match kind {
            AffinityKind::RouterOnly => self.router.bind_router_only(&sc.session_id, worker, ttl),
            AffinityKind::EngineBacked => {
                self.router.bind_engine_session(&sc.session_id, worker, ttl)
            }
        }
    }
}

pub(crate) fn sticky_allowed_for_phase(
    phase: RequestPhase,
    routing: Option<&RoutingHints>,
) -> bool {
    let Some(routing) = routing else {
        return false;
    };
    if routing.session_control.is_none() {
        return false;
    }

    match phase {
        RequestPhase::Prefill => {
            routing.prefill_worker_id.is_none()
                && routing.prefill_dp_rank.is_none()
                && routing.backend_instance_id.is_none()
        }
        RequestPhase::Decode => {
            routing.decode_worker_id.is_none()
                && routing.dp_rank.is_none()
                && routing.backend_instance_id.is_none()
        }
        RequestPhase::Aggregated => {
            routing.backend_instance_id.is_none() && routing.dp_rank.is_none()
        }
    }
}

fn sticky_session_id_for_phase(request: &PreprocessedRequest, phase: RequestPhase) -> Option<&str> {
    let routing = request.routing.as_ref()?;
    if !sticky_allowed_for_phase(phase, Some(routing)) {
        return None;
    }

    routing
        .session_control
        .as_ref()
        .map(|sc| sc.session_id.as_str())
}

#[cfg(test)]
mod tests {
    use super::sticky_allowed_for_phase;
    use crate::protocols::common::{preprocessor::RoutingHints, timing::RequestPhase};
    use crate::protocols::openai::nvext::SessionControl;

    fn session_control() -> SessionControl {
        SessionControl {
            session_id: "sess-1".to_string(),
            action: None,
            timeout: 300,
        }
    }

    #[test]
    fn sticky_is_noop_without_session_control() {
        let routing = RoutingHints::default();
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Aggregated,
            Some(&routing)
        ));
    }

    #[test]
    fn sticky_allowed_when_only_session_control_is_present() {
        let routing = RoutingHints {
            session_control: Some(session_control()),
            ..Default::default()
        };
        assert!(sticky_allowed_for_phase(
            RequestPhase::Aggregated,
            Some(&routing)
        ));
        assert!(sticky_allowed_for_phase(
            RequestPhase::Prefill,
            Some(&routing)
        ));
        assert!(sticky_allowed_for_phase(
            RequestPhase::Decode,
            Some(&routing)
        ));
    }

    #[test]
    fn sticky_skips_phase_specific_explicit_pins() {
        let prefill = RoutingHints {
            session_control: Some(session_control()),
            prefill_worker_id: Some(1),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Prefill,
            Some(&prefill)
        ));

        let prefill_rank = RoutingHints {
            session_control: Some(session_control()),
            prefill_dp_rank: Some(2),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Prefill,
            Some(&prefill_rank)
        ));

        let decode = RoutingHints {
            session_control: Some(session_control()),
            decode_worker_id: Some(3),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Decode,
            Some(&decode)
        ));

        let decode_rank = RoutingHints {
            session_control: Some(session_control()),
            dp_rank: Some(4),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Decode,
            Some(&decode_rank)
        ));

        let aggregated = RoutingHints {
            session_control: Some(session_control()),
            backend_instance_id: Some(5),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Aggregated,
            Some(&aggregated)
        ));
    }
}
