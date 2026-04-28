// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent tool-event publisher backed by the Dynamo runtime event plane.

use anyhow::Result;
use dynamo_runtime::component::Namespace;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::agents::context::AgentContext;

use super::{
    AgentToolEvent, AgentTraceRecord, DEFAULT_TOOL_EVENTS_TOPIC, TraceEventSource, TraceEventType,
    TraceSchema,
};

/// Thin event-plane publisher for harness-originated agent tool events.
///
/// This mirrors the FPM/KV pattern: the agent domain owns the record validation
/// and topic choice, while transport, discovery registration, and serialization
/// stay in [`EventPublisher`].
pub struct AgentToolEventPublisher {
    inner: EventPublisher,
}

impl AgentToolEventPublisher {
    /// Create a publisher for the default namespace-scoped tool-events topic.
    pub async fn for_namespace(namespace: &Namespace) -> Result<Self> {
        Self::for_namespace_with_topic(namespace, DEFAULT_TOOL_EVENTS_TOPIC).await
    }

    /// Create a publisher for a namespace-scoped tool-events topic.
    pub async fn for_namespace_with_topic(
        namespace: &Namespace,
        topic: impl Into<String>,
    ) -> Result<Self> {
        let inner = EventPublisher::for_namespace(namespace, topic).await?;
        Ok(Self { inner })
    }

    /// Publish a tool lifecycle event with the standard agent trace envelope.
    pub async fn publish_tool_event(
        &self,
        event_type: TraceEventType,
        agent_context: AgentContext,
        tool: AgentToolEvent,
    ) -> Result<()> {
        let record = AgentTraceRecord {
            schema: TraceSchema::V1,
            event_type,
            event_time_unix_ms: super::current_time_unix_ms(),
            event_source: TraceEventSource::Harness,
            agent_context,
            request: None,
            tool: Some(tool),
        };
        self.publish_record(&record).await
    }

    /// Publish a prebuilt tool record.
    pub async fn publish_record(&self, record: &AgentTraceRecord) -> Result<()> {
        super::validate_tool_record(record)?;
        self.inner.publish(record).await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_runtime::config::environment_names::zmq_broker as broker_env;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::transports::event_plane::EventSubscriber;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::time::timeout;

    use super::*;
    use crate::agents::trace::{AgentToolStatus, AgentTraceRecord};

    fn agent_context() -> AgentContext {
        AgentContext {
            workflow_type_id: "ms_agent".to_string(),
            workflow_id: "run-1".to_string(),
            program_id: "run-1:agent".to_string(),
            parent_program_id: None,
        }
    }

    fn tool_event() -> AgentToolEvent {
        AgentToolEvent {
            tool_call_id: "tool-123".to_string(),
            tool_class: "web_search".to_string(),
            status: Some(AgentToolStatus::Succeeded),
            duration_ms: Some(12.5),
            output_tokens: Some(9),
            output_bytes: Some(64),
            tool_name_hash: None,
            error_type: None,
        }
    }

    fn valid_record() -> AgentTraceRecord {
        AgentTraceRecord {
            schema: TraceSchema::V1,
            event_type: TraceEventType::ToolEnd,
            event_time_unix_ms: 1,
            event_source: TraceEventSource::Harness,
            agent_context: agent_context(),
            request: None,
            tool: Some(tool_event()),
        }
    }

    #[test]
    fn validates_tool_event_records() {
        super::super::validate_tool_record(&valid_record()).expect("valid tool record should pass");

        let mut non_tool = valid_record();
        non_tool.event_type = TraceEventType::RequestEnd;
        assert!(super::super::validate_tool_record(&non_tool).is_err());

        let mut wrong_source = valid_record();
        wrong_source.event_source = TraceEventSource::Dynamo;
        assert!(super::super::validate_tool_record(&wrong_source).is_err());

        let mut missing_tool = valid_record();
        missing_tool.tool = None;
        assert!(super::super::validate_tool_record(&missing_tool).is_err());
    }

    #[tokio::test]
    async fn publishes_typed_tool_event_to_event_plane() -> Result<()> {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = Runtime::from_current()?;
                let drt =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
                let namespace =
                    drt.namespace(format!("agent-tool-events-{}", uuid::Uuid::new_v4()))?;

                let publisher = AgentToolEventPublisher::for_namespace(&namespace).await?;
                let mut subscriber =
                    EventSubscriber::for_namespace(&namespace, DEFAULT_TOOL_EVENTS_TOPIC)
                        .await?
                        .typed::<AgentTraceRecord>();

                tokio::time::sleep(Duration::from_millis(100)).await;

                publisher
                    .publish_tool_event(TraceEventType::ToolEnd, agent_context(), tool_event())
                    .await?;

                let (_envelope, record) = timeout(Duration::from_secs(2), subscriber.next())
                    .await?
                    .expect("event stream should stay open")?;

                assert_eq!(record.event_type, TraceEventType::ToolEnd);
                assert_eq!(record.event_source, TraceEventSource::Harness);
                assert_eq!(record.agent_context.workflow_id, "run-1");
                assert_eq!(record.tool.unwrap().tool_call_id, "tool-123");

                drt.shutdown();
                Ok(())
            },
        )
        .await
    }
}
