// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};

use dynamo_kv_router::{
    QueueAdmissionDecision, QueueAdmissionEvent, QueueAdmissionId, QueueAdmissionPolicy,
    QueueAdmissionRequest, QueueAdmissionWorkerSnapshot,
};

#[derive(Default)]
struct SessionState {
    active_request_id: Option<String>,
    waiting: VecDeque<QueueAdmissionId>,
}

/// Admit at most one request per session while allowing different sessions to run concurrently.
#[derive(Default)]
pub(crate) struct SessionAdmissionPolicy {
    sessions: HashMap<String, SessionState>,
    request_sessions: HashMap<String, String>,
    admission_requests: HashMap<QueueAdmissionId, String>,
    request_admissions: HashMap<String, QueueAdmissionId>,
}

impl SessionAdmissionPolicy {
    fn finish(&mut self, request_id: &str, ready: &mut Vec<QueueAdmissionId>) {
        let Some(session_id) = self.request_sessions.remove(request_id) else {
            return;
        };
        if let Some(id) = self.request_admissions.remove(request_id) {
            self.admission_requests.remove(&id);
        }

        let mut remove_session = false;
        if let Some(state) = self.sessions.get_mut(&session_id) {
            if state.active_request_id.as_deref() == Some(request_id) {
                state.active_request_id = None;
                while let Some(next_id) = state.waiting.pop_front() {
                    let Some(next_request_id) = self.admission_requests.get(&next_id) else {
                        continue;
                    };
                    state.active_request_id = Some(next_request_id.clone());
                    ready.push(next_id);
                    break;
                }
            } else {
                state
                    .waiting
                    .retain(|id| self.admission_requests.contains_key(id));
            }
            remove_session = state.active_request_id.is_none() && state.waiting.is_empty();
        }
        if remove_session {
            self.sessions.remove(&session_id);
        }
    }
}

impl QueueAdmissionPolicy for SessionAdmissionPolicy {
    fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
        let Some(session_id) = request
            .session_context()
            .map(|context| context.session_id().to_owned())
        else {
            return QueueAdmissionDecision::Bypass;
        };
        let request_id = request.request_id().to_owned();
        let admission_id = request.id();
        self.request_sessions
            .insert(request_id.clone(), session_id.clone());
        self.admission_requests
            .insert(admission_id, request_id.clone());
        self.request_admissions
            .insert(request_id.clone(), admission_id);

        let state = self.sessions.entry(session_id).or_default();
        if state.active_request_id.is_none() {
            state.active_request_id = Some(request_id);
            QueueAdmissionDecision::Ready
        } else {
            state.waiting.push_back(admission_id);
            QueueAdmissionDecision::Defer
        }
    }

    fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
        match event {
            QueueAdmissionEvent::Completed { request_id, .. }
            | QueueAdmissionEvent::Aborted { request_id } => self.finish(request_id, ready),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::SessionContext;

    fn request<'a>(
        id: u64,
        request_id: &'a str,
        session: &'a SessionContext,
        snapshot: &'a QueueAdmissionWorkerSnapshot,
    ) -> QueueAdmissionRequest<'a> {
        QueueAdmissionRequest::new(
            QueueAdmissionId::new(id),
            request_id,
            16,
            Some(session),
            snapshot,
        )
    }

    #[test]
    fn terminal_event_promotes_the_next_request_in_the_same_session() {
        let session = SessionContext::new("session-a".to_string(), None, None, None, None);
        let snapshot = QueueAdmissionWorkerSnapshot::new(1, Vec::new());
        let mut policy = SessionAdmissionPolicy::default();
        assert_eq!(
            policy.admit(request(1, "request-a", &session, &snapshot)),
            QueueAdmissionDecision::Ready
        );
        assert_eq!(
            policy.admit(request(2, "request-b", &session, &snapshot)),
            QueueAdmissionDecision::Defer
        );

        let mut ready = Vec::new();
        policy.on_event(
            QueueAdmissionEvent::Completed {
                request_id: "request-a",
                context_tokens: None,
            },
            &mut ready,
        );
        assert_eq!(ready, [QueueAdmissionId::new(2)]);

        ready.clear();
        policy.on_event(
            QueueAdmissionEvent::Aborted {
                request_id: "request-b",
            },
            &mut ready,
        );
        assert!(ready.is_empty());
        assert!(policy.sessions.is_empty());
    }
}
