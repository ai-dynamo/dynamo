// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example post-selection session policy that emits block-addressed KV actions.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Mutex,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use dynamo_kv_router::{
    indexer::successful_block_removals,
    kv_hints::{KvHint, KvHintAction},
};
use dynamo_llm::{
    entrypoint::HttpFrontend,
    kv_router::{KvHintPolicy, KvHintPolicyContext, KvHintPolicyError},
};

const ACTION_VERSION: &str = "1.0";

#[derive(Clone, Copy, Debug)]
pub struct RetentionConfig {
    pub priority: u64,
    pub trigger: RetentionTrigger,
}

#[derive(Clone, Copy, Debug)]
pub struct OnlineRetentionConfig {
    pub priority: u64,
    pub fixed_ttl_seconds: f64,
    pub min_missing_blocks: usize,
    pub min_removal_pressure_per_second: f64,
    pub removal_pressure_window: Duration,
}

#[derive(Clone, Copy, Debug)]
pub enum RetentionTrigger {
    SubagentSpawn,
    InferredToolCall,
    SubagentSpawnOrInferredToolCall,
}

impl RetentionTrigger {
    fn matches(self, subagent_spawn: bool, inferred_tool_call: bool) -> Option<&'static str> {
        match self {
            Self::SubagentSpawn if subagent_spawn => Some("subagent_spawn"),
            Self::InferredToolCall if inferred_tool_call => Some("inferred_tool_call"),
            Self::SubagentSpawnOrInferredToolCall if subagent_spawn => Some("subagent_spawn"),
            Self::SubagentSpawnOrInferredToolCall if inferred_tool_call => {
                Some("inferred_tool_call")
            }
            _ => None,
        }
    }
}

/// Applies sparse lifecycle actions to session-addressed KV.
pub struct SessionKvHintPolicy {
    retention: Option<RetentionConfig>,
    online_retention: Option<OnlineRetentionConfig>,
    evict_final_roots: bool,
    next_message_id: AtomicU64,
    removal_pressure: Mutex<RemovalPressureWindow>,
}

#[derive(Debug, Default)]
struct RemovalPressureWindow {
    samples: VecDeque<(Instant, u64)>,
}

impl RemovalPressureWindow {
    fn observe(&mut self, now: Instant, count: u64, window: Duration) -> f64 {
        if self
            .samples
            .back()
            .is_some_and(|(_, previous)| count < *previous)
        {
            self.samples.clear();
        }
        self.samples.push_back((now, count));

        while self.samples.len() > 1 && self.samples[1].0 + window <= now {
            self.samples.pop_front();
        }

        let Some((start, start_count)) = self.samples.front().copied() else {
            return 0.0;
        };
        let elapsed = now.saturating_duration_since(start).as_secs_f64();
        if elapsed < 1.0 {
            return 0.0;
        }
        count.saturating_sub(start_count) as f64 / elapsed
    }
}

impl SessionKvHintPolicy {
    pub fn new(
        retention: Option<RetentionConfig>,
        evict_final_roots: bool,
    ) -> Result<Self, KvHintPolicyError> {
        Ok(Self {
            retention,
            online_retention: None,
            evict_final_roots,
            next_message_id: AtomicU64::new(0),
            removal_pressure: Mutex::new(RemovalPressureWindow::default()),
        })
    }

    pub fn new_online(config: OnlineRetentionConfig) -> Result<Self, KvHintPolicyError> {
        if !config.fixed_ttl_seconds.is_finite() || config.fixed_ttl_seconds <= 0.0 {
            return Err(KvHintPolicyError::new(
                "online retention fixed TTL must be finite and positive",
            ));
        }
        if !config.min_removal_pressure_per_second.is_finite()
            || config.min_removal_pressure_per_second < 0.0
        {
            return Err(KvHintPolicyError::new(
                "online retention pressure threshold must be finite and non-negative",
            ));
        }
        if config.removal_pressure_window.is_zero() {
            return Err(KvHintPolicyError::new(
                "online retention pressure window must be positive",
            ));
        }
        Ok(Self {
            retention: None,
            online_retention: Some(config),
            evict_final_roots: false,
            next_message_id: AtomicU64::new(0),
            removal_pressure: Mutex::new(RemovalPressureWindow::default()),
        })
    }

    fn hint(
        &self,
        session_id: &str,
        selected_worker: u64,
        action_type: &str,
        mut payload: BTreeMap<String, serde_json::Value>,
        block_hashes: impl IntoIterator<Item = u64>,
        include_current_request: bool,
    ) -> KvHint {
        payload.insert("execute_at".into(), "request_completion".into());
        payload.insert(
            "include_current_request".into(),
            include_current_request.into(),
        );
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
    retention: Option<RetentionConfig>,
    evict_final_roots: bool,
) -> Result<HttpFrontend, KvHintPolicyError> {
    Ok(frontend.kv_hint_policy(SessionKvHintPolicy::new(retention, evict_final_roots)?))
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
                true,
            )));
        }

        if let Some(online) = self.online_retention {
            let missing_blocks = context
                .prefix_blocks
                .saturating_sub(context.cached_prefix_blocks);
            let removal_pressure_per_second = self
                .removal_pressure
                .lock()
                .map_err(|_| KvHintPolicyError::new("removal-pressure sampler poisoned"))?
                .observe(
                    Instant::now(),
                    successful_block_removals(),
                    online.removal_pressure_window,
                );
            let continuation_expected = agent.session_final == Some(false);
            if !continuation_expected
                || missing_blocks < online.min_missing_blocks
                || removal_pressure_per_second < online.min_removal_pressure_per_second
            {
                return Ok(None);
            }

            tracing::info!(
                target: "continuum_kv_hints",
                session_id = %agent.session_id,
                worker_id = context.selected_worker.worker_id,
                prefix_blocks = context.prefix_blocks,
                cached_prefix_blocks = context.cached_prefix_blocks,
                missing_blocks,
                removal_pressure_per_second,
                removal_pressure_window_seconds = online.removal_pressure_window.as_secs_f64(),
                min_missing_blocks = online.min_missing_blocks,
                min_removal_pressure_per_second = online.min_removal_pressure_per_second,
                action_type = "kv.retain",
                priority = online.priority,
                ttl_seconds = online.fixed_ttl_seconds,
                ttl_source = "fixed",
                retention_reason = "online_missing_depth_and_eviction_pressure",
                "Emitting request-completion KV hint"
            );
            return Ok(Some(self.hint(
                &agent.session_id,
                context.selected_worker.worker_id,
                "kv.retain",
                BTreeMap::from([
                    ("priority".into(), online.priority.into()),
                    ("ttl_seconds".into(), online.fixed_ttl_seconds.into()),
                ]),
                block_hashes,
                true,
            )));
        }

        let Some((retain, retention_reason, ttl_ms)) = self.retention.and_then(|retain| {
            retain
                .trigger
                .matches(
                    agent.subagent_spawn == Some(true),
                    agent.inferred_tool_call == Some(true),
                )
                .zip(agent.retention_ttl_ms.filter(|ttl_ms| *ttl_ms > 0))
                .map(|(reason, ttl_ms)| (retain, reason, ttl_ms))
        }) else {
            return Ok(None);
        };
        let ttl_seconds = ttl_ms as f64 / 1000.0;
        let request_block_range = agent
            .retention_block_start
            .zip(agent.retention_block_count)
            .filter(|(_, count)| *count > 0);
        let (block_hashes, include_current_request) = if request_block_range.is_some() {
            (Vec::new(), false)
        } else {
            (block_hashes, true)
        };
        let lineage_count = context
            .session_lineage
            .map_or(0, |lineage| lineage.lineages().len());
        tracing::info!(
            target: "continuum_kv_hints",
            session_id = %agent.session_id,
            worker_id = context.selected_worker.worker_id,
            lineage_block_count = block_hashes.len(),
            lineage_count,
            action_type = "kv.retain",
            priority = retain.priority,
            ttl_seconds,
            ttl_source = "request",
            retention_reason,
            request_block_start = request_block_range.map(|range| range.0),
            request_block_count = request_block_range.map(|range| range.1),
            "Emitting request-completion KV hint"
        );
        let mut payload = BTreeMap::from([
            ("priority".into(), retain.priority.into()),
            ("ttl_seconds".into(), ttl_seconds.into()),
        ]);
        if let Some((start, count)) = request_block_range {
            payload.insert("current_request_block_start".into(), start.into());
            payload.insert("current_request_block_count".into(), count.into());
        }
        Ok(Some(self.hint(
            &agent.session_id,
            context.selected_worker.worker_id,
            "kv.retain",
            payload,
            block_hashes,
            include_current_request,
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
            prefix_blocks: 4,
            cached_prefix_blocks: 2,
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
            Some(RetentionConfig {
                priority: 3,
                trigger: RetentionTrigger::SubagentSpawn,
            }),
            false,
        )
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .subagent_spawn(true)
            .retention_ttl_ms(2500)
            .build()
            .unwrap();
        let lineage = SessionLineageView::new(vec![vec![
            ExternalSequenceBlockHash(11),
            ExternalSequenceBlockHash(12),
        ]]);

        let hint = policy
            .evaluate(&context(&agent, Some(&lineage)))
            .unwrap()
            .unwrap();

        assert_eq!(hint.actions[0].action_type, "kv.retain");
        assert_eq!(hint.actions[0].payload["priority"], 3);
        assert_eq!(hint.actions[0].payload["ttl_seconds"], 2.5);
        assert_eq!(
            hint.actions[0].payload["block_hashes"],
            serde_json::json!(["11", "12"])
        );
        assert_eq!(hint.actions[0].payload["include_current_request"], true);
    }

    #[test]
    fn spawn_without_oracle_ttl_emits_no_retention() {
        let policy = SessionKvHintPolicy::new(
            Some(RetentionConfig {
                priority: 3,
                trigger: RetentionTrigger::SubagentSpawn,
            }),
            false,
        )
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .subagent_spawn(true)
            .build()
            .unwrap();

        assert!(policy.evaluate(&context(&agent, None)).unwrap().is_none());
    }

    #[test]
    fn online_retention_uses_causal_signals_and_fixed_ttl() {
        let policy = SessionKvHintPolicy::new_online(OnlineRetentionConfig {
            priority: 7,
            fixed_ttl_seconds: 10.0,
            min_missing_blocks: 2,
            min_removal_pressure_per_second: 0.0,
            removal_pressure_window: Duration::from_secs(5),
        })
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .session_final(false)
            .retention_ttl_ms(123_456)
            .build()
            .unwrap();
        let lineage = SessionLineageView::new(vec![vec![
            ExternalSequenceBlockHash(11),
            ExternalSequenceBlockHash(12),
        ]]);

        let hint = policy
            .evaluate(&context(&agent, Some(&lineage)))
            .unwrap()
            .unwrap();

        assert_eq!(hint.actions[0].action_type, "kv.retain");
        assert_eq!(hint.actions[0].payload["priority"], 7);
        assert_eq!(hint.actions[0].payload["ttl_seconds"], 10.0);
        assert_eq!(hint.actions[0].payload["include_current_request"], true);
        assert_eq!(
            hint.actions[0].payload["block_hashes"],
            serde_json::json!(["11", "12"])
        );
    }

    #[test]
    fn online_retention_requires_expected_continuation() {
        let policy = SessionKvHintPolicy::new_online(OnlineRetentionConfig {
            priority: 7,
            fixed_ttl_seconds: 10.0,
            min_missing_blocks: 2,
            min_removal_pressure_per_second: 0.0,
            removal_pressure_window: Duration::from_secs(5),
        })
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .session_final(true)
            .build()
            .unwrap();

        assert!(policy.evaluate(&context(&agent, None)).unwrap().is_none());
    }

    #[test]
    fn trailing_removal_pressure_uses_only_the_configured_window() {
        let start = Instant::now();
        let mut window = RemovalPressureWindow::default();
        assert_eq!(window.observe(start, 100, Duration::from_secs(5)), 0.0);
        assert_eq!(
            window.observe(start + Duration::from_secs(2), 110, Duration::from_secs(5)),
            5.0
        );
        assert_eq!(
            window.observe(start + Duration::from_secs(7), 130, Duration::from_secs(5)),
            4.0
        );
    }

    #[test]
    fn inferred_tool_call_emits_retention() {
        let policy = SessionKvHintPolicy::new(
            Some(RetentionConfig {
                priority: 3,
                trigger: RetentionTrigger::InferredToolCall,
            }),
            false,
        )
        .unwrap();
        let agent = AgentContext::builder()
            .session_id("session-1".to_string())
            .inferred_tool_call(true)
            .retention_ttl_ms(2250)
            .build()
            .unwrap();

        let hint = policy.evaluate(&context(&agent, None)).unwrap().unwrap();

        assert_eq!(hint.actions[0].action_type, "kv.retain");
        assert_eq!(hint.actions[0].payload["ttl_seconds"], 2.25);
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
