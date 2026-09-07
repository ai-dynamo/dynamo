// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example post-selection session policy that emits block-addressed KV actions.

use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_llm::entrypoint::HttpFrontend;
use dynamo_llm::kv_router::{
    KvHintAction, KvHintPolicy, KvHintPolicyContext, KvHintPolicyError, KvHintsEnvelope,
};

const PROTOCOL_VERSION: &str = "1.0";

#[derive(Clone, Copy, Debug)]
pub struct FixedRetention {
    pub priority: u64,
    pub ttl_seconds: f64,
}

/// Emits deferred eviction for explicit session intent and optional retention otherwise.
pub struct SessionKvHintPolicy {
    fixed_retention: Option<FixedRetention>,
    next_message_id: AtomicU64,
}

impl SessionKvHintPolicy {
    pub fn new(fixed_retention: Option<FixedRetention>) -> Result<Self, KvHintPolicyError> {
        if fixed_retention
            .is_some_and(|retain| !retain.ttl_seconds.is_finite() || retain.ttl_seconds <= 0.0)
        {
            return Err(KvHintPolicyError::new(
                "retention ttl_seconds must be finite and positive",
            ));
        }
        Ok(Self {
            fixed_retention,
            next_message_id: AtomicU64::new(0),
        })
    }

    fn envelope(
        &self,
        session_id: &str,
        selected_worker: u64,
        action_type: &str,
        mut payload: serde_json::Map<String, serde_json::Value>,
        block_hashes: impl IntoIterator<Item = u64>,
    ) -> KvHintsEnvelope {
        payload.insert(
            "execute_at".to_string(),
            serde_json::Value::String("request_completion".to_string()),
        );
        payload.insert(
            "include_current_request".to_string(),
            serde_json::Value::Bool(true),
        );
        payload.insert(
            "block_hashes".to_string(),
            serde_json::Value::Array(
                block_hashes
                    .into_iter()
                    .map(|hash| serde_json::Value::String(hash.to_string()))
                    .collect(),
            ),
        );
        let sequence = self.next_message_id.fetch_add(1, Ordering::Relaxed);
        let message_id = format!("{session_id}-{selected_worker}-{sequence}");
        KvHintsEnvelope {
            protocol_version: PROTOCOL_VERSION.to_string(),
            message_id: message_id.clone(),
            actions: vec![KvHintAction {
                action_id: message_id,
                action_type: action_type.to_string(),
                action_version: PROTOCOL_VERSION.to_string(),
                payload,
            }],
        }
    }
}

pub fn register(
    frontend: HttpFrontend,
    fixed_retention: Option<FixedRetention>,
) -> Result<HttpFrontend, KvHintPolicyError> {
    Ok(frontend.kv_hint_policy(SessionKvHintPolicy::new(fixed_retention)?))
}

impl KvHintPolicy for SessionKvHintPolicy {
    fn evaluate(
        &self,
        context: &KvHintPolicyContext<'_>,
    ) -> Result<Option<KvHintsEnvelope>, KvHintPolicyError> {
        let Some(agent) = context.agent_context else {
            return Ok(None);
        };
        let Some(lineage) = context
            .session_lineage
            .filter(|lineage| !lineage.is_empty())
        else {
            return Ok(None);
        };
        let block_hashes = lineage
            .unique_block_hashes()
            .into_iter()
            .map(|hash| hash.0)
            .collect::<Vec<_>>();

        let evict = agent
            .kv_hints
            .as_ref()
            .is_some_and(|hints| hints.evict_session);
        if evict {
            return Ok(Some(self.envelope(
                &agent.session_id,
                context.selected_worker.worker_id,
                "kv.evict",
                serde_json::Map::new(),
                block_hashes,
            )));
        }

        let Some(retain) = self.fixed_retention else {
            return Ok(None);
        };
        let mut payload = serde_json::Map::new();
        payload.insert("priority".to_string(), retain.priority.into());
        payload.insert(
            "ttl_seconds".to_string(),
            serde_json::Value::from(retain.ttl_seconds),
        );
        Ok(Some(self.envelope(
            &agent.session_id,
            context.selected_worker.worker_id,
            "kv.retain",
            payload,
            block_hashes,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::kv_router::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};
    use dynamo_llm::{
        kv_router::SessionLineageView,
        protocols::common::extensions::{AgentContext, KvHints},
    };

    fn context<'a>(
        agent_context: &'a AgentContext,
        lineage: &'a SessionLineageView,
    ) -> KvHintPolicyContext<'a> {
        KvHintPolicyContext {
            agent_context: Some(agent_context),
            selected_worker: WorkerWithDpRank::new(9, 0),
            session_lineage: Some(lineage),
        }
    }

    #[test]
    fn eviction_intent_emits_deferred_decimal_string_block_hashes() {
        let policy = SessionKvHintPolicy::new(None).unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .kv_hints(KvHints {
                evict_session: true,
            })
            .build()
            .unwrap();
        let lineage = SessionLineageView::new(vec![vec![
            ExternalSequenceBlockHash(11),
            ExternalSequenceBlockHash(u64::MAX),
        ]]);

        let envelope = policy
            .evaluate(&context(&agent, &lineage))
            .unwrap()
            .unwrap();

        assert_eq!(envelope.actions[0].action_type, "kv.evict");
        assert_eq!(
            envelope.actions[0].payload["execute_at"],
            "request_completion"
        );
        assert_eq!(envelope.actions[0].payload["include_current_request"], true);
        assert_eq!(
            envelope.actions[0].payload["block_hashes"],
            serde_json::json!(["11", u64::MAX.to_string()])
        );
    }

    #[test]
    fn fixed_retention_emits_priority_and_ttl() {
        let policy = SessionKvHintPolicy::new(Some(FixedRetention {
            priority: 3,
            ttl_seconds: 2.5,
        }))
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .build()
            .unwrap();
        let lineage = SessionLineageView::new(vec![vec![ExternalSequenceBlockHash(11)]]);

        let envelope = policy
            .evaluate(&context(&agent, &lineage))
            .unwrap()
            .unwrap();

        assert_eq!(envelope.actions[0].action_type, "kv.retain");
        assert_eq!(
            envelope.actions[0].payload["execute_at"],
            "request_completion"
        );
        assert_eq!(envelope.actions[0].payload["priority"], 3);
        assert_eq!(envelope.actions[0].payload["ttl_seconds"], 2.5);
    }
}
