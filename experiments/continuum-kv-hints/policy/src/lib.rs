// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example post-selection session policy that emits block-addressed KV actions.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
};

use dynamo_kv_router::kv_hints::{KvHint, KvHintAction};
use dynamo_llm::{
    entrypoint::HttpFrontend,
    kv_router::{KvHintPolicy, KvHintPolicyContext, KvHintPolicyError},
};

const ACTION_VERSION: &str = "1.0";

#[derive(Clone, Copy, Debug)]
pub struct FixedRetention {
    pub priority: u64,
    pub ttl_seconds: f64,
}

/// Applies sparse lifecycle actions to session-addressed KV.
pub struct SessionKvHintPolicy {
    spawn_retention: Option<FixedRetention>,
    evict_final_roots: bool,
    next_message_id: AtomicU64,
}

impl SessionKvHintPolicy {
    pub fn new(
        spawn_retention: Option<FixedRetention>,
        evict_final_roots: bool,
    ) -> Result<Self, KvHintPolicyError> {
        if spawn_retention
            .is_some_and(|retain| !retain.ttl_seconds.is_finite() || retain.ttl_seconds <= 0.0)
        {
            return Err(KvHintPolicyError::new(
                "retention ttl_seconds must be finite and positive",
            ));
        }
        Ok(Self {
            spawn_retention,
            evict_final_roots,
            next_message_id: AtomicU64::new(0),
        })
    }

    fn hint(
        &self,
        session_id: &str,
        selected_worker: u64,
        action_type: &str,
        mut payload: BTreeMap<String, serde_json::Value>,
        block_hashes: impl IntoIterator<Item = u64>,
    ) -> KvHint {
        payload.insert("execute_at".into(), "request_completion".into());
        payload.insert("include_current_request".into(), true.into());
        payload.insert(
            "block_hashes".into(),
            block_hashes
                .into_iter()
                .map(|hash| serde_json::Value::String(hash.to_string()))
                .collect::<Vec<_>>()
                .into(),
        );
        let sequence = self.next_message_id.fetch_add(1, Ordering::Relaxed);
        let message_id = format!("{session_id}-{selected_worker}-{sequence}");
        KvHint::new(
            message_id.clone(),
            vec![KvHintAction::new(
                message_id,
                action_type,
                ACTION_VERSION,
                payload,
            )],
        )
    }
}

pub fn register(
    frontend: HttpFrontend,
    spawn_retention: Option<FixedRetention>,
    evict_final_roots: bool,
) -> Result<HttpFrontend, KvHintPolicyError> {
    Ok(frontend.kv_hint_policy(SessionKvHintPolicy::new(
        spawn_retention,
        evict_final_roots,
    )?))
}

impl KvHintPolicy for SessionKvHintPolicy {
    fn evaluate(
        &self,
        context: &KvHintPolicyContext<'_>,
    ) -> Result<Option<KvHint>, KvHintPolicyError> {
        let Some(agent) = context.agent_context else {
            return Ok(None);
        };
        let block_hashes = context
            .session_lineage
            .map(|lineage| lineage.unique_block_hashes())
            .unwrap_or_default()
            .into_iter()
            .map(|hash| hash.0)
            .collect::<Vec<_>>();

        if self.evict_final_roots
            && agent.session_final == Some(true)
            && agent.parent_session_id.is_none()
        {
            tracing::info!(
                target: "continuum_kv_hints",
                session_id = %agent.session_id,
                worker_id = context.selected_worker.worker_id,
                lineage_block_count = block_hashes.len(),
                action_type = "kv.evict",
                "Emitting request-completion KV hint"
            );
            return Ok(Some(self.hint(
                &agent.session_id,
                context.selected_worker.worker_id,
                "kv.evict",
                BTreeMap::new(),
                block_hashes,
            )));
        }

        let Some(retain) = self
            .spawn_retention
            .filter(|_| agent.subagent_spawn == Some(true))
        else {
            return Ok(None);
        };
        tracing::info!(
            target: "continuum_kv_hints",
            session_id = %agent.session_id,
            worker_id = context.selected_worker.worker_id,
            lineage_block_count = block_hashes.len(),
            action_type = "kv.retain",
            priority = retain.priority,
            ttl_seconds = retain.ttl_seconds,
            "Emitting request-completion KV hint"
        );
        Ok(Some(self.hint(
            &agent.session_id,
            context.selected_worker.worker_id,
            "kv.retain",
            BTreeMap::from([
                ("priority".into(), retain.priority.into()),
                ("ttl_seconds".into(), retain.ttl_seconds.into()),
            ]),
            block_hashes,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};
    use dynamo_llm::{kv_router::SessionLineageView, protocols::common::extensions::AgentContext};

    fn context<'a>(
        agent_context: &'a AgentContext,
        lineage: Option<&'a SessionLineageView>,
    ) -> KvHintPolicyContext<'a> {
        KvHintPolicyContext {
            agent_context: Some(agent_context),
            selected_worker: WorkerWithDpRank::new(9, 0),
            session_lineage: lineage,
        }
    }

    #[test]
    fn final_session_emits_deferred_eviction() {
        let policy = SessionKvHintPolicy::new(None, true).unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .session_final(true)
            .build()
            .unwrap();
        let lineage = SessionLineageView::new(vec![vec![
            ExternalSequenceBlockHash(11),
            ExternalSequenceBlockHash(u64::MAX),
        ]]);

        let hint = policy
            .evaluate(&context(&agent, Some(&lineage)))
            .unwrap()
            .unwrap();

        assert_eq!(hint.protocol_version, "0.1");
        assert_eq!(hint.actions[0].action_type, "kv.evict");
        assert_eq!(hint.actions[0].payload["execute_at"], "request_completion");
        assert_eq!(hint.actions[0].payload["include_current_request"], true);
        assert_eq!(
            hint.actions[0].payload["block_hashes"],
            serde_json::json!(["11", u64::MAX.to_string()])
        );
    }

    #[test]
    fn first_request_can_target_its_current_blocks() {
        let policy = SessionKvHintPolicy::new(None, true).unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .session_final(true)
            .build()
            .unwrap();

        let hint = policy.evaluate(&context(&agent, None)).unwrap().unwrap();

        assert_eq!(
            hint.actions[0].payload["block_hashes"],
            serde_json::json!([])
        );
        assert_eq!(hint.actions[0].payload["include_current_request"], true);
    }

    #[test]
    fn spawn_retention_emits_priority_and_ttl() {
        let policy = SessionKvHintPolicy::new(
            Some(FixedRetention {
                priority: 3,
                ttl_seconds: 2.5,
            }),
            false,
        )
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .subagent_spawn(true)
            .build()
            .unwrap();

        let hint = policy.evaluate(&context(&agent, None)).unwrap().unwrap();

        assert_eq!(hint.actions[0].action_type, "kv.retain");
        assert_eq!(hint.actions[0].payload["priority"], 3);
        assert_eq!(hint.actions[0].payload["ttl_seconds"], 2.5);
    }

    #[test]
    fn ordinary_request_emits_no_retention() {
        let policy = SessionKvHintPolicy::new(
            Some(FixedRetention {
                priority: 3,
                ttl_seconds: 2.5,
            }),
            false,
        )
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .build()
            .unwrap();

        assert!(policy.evaluate(&context(&agent, None)).unwrap().is_none());
    }

    #[test]
    fn final_child_does_not_evict_shared_prefix() {
        let policy = SessionKvHintPolicy::new(None, true).unwrap();
        let agent = AgentContext::builder()
            .session_id("child-1".to_string())
            .parent_session_id("root-1".to_string())
            .session_final(true)
            .build()
            .unwrap();

        assert!(policy.evaluate(&context(&agent, None)).unwrap().is_none());
    }
}
