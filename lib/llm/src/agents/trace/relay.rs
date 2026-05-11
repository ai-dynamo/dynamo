// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent tool-event relay.
//!
//! Dynamo binds a local PULL socket so any number of external harness processes
//! can connect with PUSH sockets, validate domain records, then publish them to
//! the Dynamo event plane. The ZMQ wire format is multipart:
//! `[topic, seq_be_u64, msgpack(AgentTraceRecord)]`.

use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::utils::zmq::{PullSocket, bind_pull_socket, multipart_message};

use super::{AgentTraceRecord, DEFAULT_TOOL_EVENTS_TOPIC};

/// Relay from local agent tool-event ZMQ PUSH producers to the Dynamo event plane.
pub struct AgentToolEventRelay {
    cancel: CancellationToken,
}

impl AgentToolEventRelay {
    pub async fn start(
        component: Component,
        zmq_endpoint: String,
        zmq_topic: Option<String>,
        event_namespace: Option<String>,
        event_topic: Option<String>,
    ) -> Result<Self> {
        let rt = component.drt().runtime().secondary();
        let namespace = match event_namespace {
            Some(namespace) => component.drt().namespace(namespace)?,
            None => component.namespace().clone(),
        };
        let topic = event_topic.unwrap_or_else(|| DEFAULT_TOOL_EVENTS_TOPIC.to_string());
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let socket = bind_pull_socket(&zmq_endpoint).await?;
        tracing::info!(endpoint = %zmq_endpoint, "agent tool relay: bound");

        let publisher = EventPublisher::for_namespace(&namespace, topic).await?;

        rt.spawn(async move {
            Self::relay_loop(socket, zmq_topic, publisher, cancel_clone).await;
        });

        Ok(Self { cancel })
    }

    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    async fn relay_loop(
        mut socket: PullSocket,
        zmq_topic: Option<String>,
        publisher: EventPublisher,
        cancel: CancellationToken,
    ) {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::info!("agent tool relay: shutting down");
                    break;
                }
                result = socket.next() => {
                    match result {
                        Some(Ok(frames)) => {
                            let mut frames = multipart_message(frames);
                            if frames.len() != 3 {
                                tracing::warn!(
                                    "agent tool relay: unexpected ZMQ frame count: expected 3, got {}",
                                    frames.len()
                                );
                                continue;
                            }

                            if let Some(expected_topic) = zmq_topic.as_deref()
                                && frames[0].as_slice() != expected_topic.as_bytes()
                            {
                                tracing::debug!(
                                    expected_topic = expected_topic,
                                    topic = %String::from_utf8_lossy(&frames[0]),
                                    "agent tool relay: dropping record for unexpected topic"
                                );
                                continue;
                            }

                            let payload = frames.swap_remove(2);
                            let record = match rmp_serde::from_slice::<AgentTraceRecord>(&payload) {
                                Ok(record) => record,
                                Err(error) => {
                                    tracing::warn!(%error, bytes = payload.len(), "agent tool relay: failed to decode record");
                                    continue;
                                }
                            };

                            if let Err(error) = super::validate_tool_record(&record) {
                                tracing::warn!(%error, "agent tool relay: dropping invalid record");
                                continue;
                            }

                            if let Err(error) = publisher.publish(&record).await {
                                tracing::warn!(%error, "agent tool relay: event plane publish failed");
                            }
                        }
                        Some(Err(error)) => {
                            tracing::error!(%error, "agent tool relay: ZMQ recv failed");
                            break;
                        }
                        None => {
                            tracing::error!("agent tool relay: ZMQ stream ended");
                            break;
                        }
                    }
                }
            }
        }
    }
}

impl Drop for AgentToolEventRelay {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::time::Duration;

    use dynamo_runtime::config::environment_names::zmq_broker as broker_env;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::transports::event_plane::EventSubscriber;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::time::timeout;

    use super::*;
    use crate::agents::context::AgentContext;
    use crate::agents::trace::{
        AgentToolEvent, AgentToolStatus, TraceEventSource, TraceEventType, TraceSchema,
    };
    use crate::utils::zmq::{connect_push_socket, send_multipart_direct};

    fn reserve_open_port() -> TcpListener {
        TcpListener::bind("127.0.0.1:0").expect("failed to reserve TCP port")
    }

    fn endpoint_from_listener(listener: &TcpListener) -> String {
        format!(
            "tcp://127.0.0.1:{}",
            listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        )
    }

    fn valid_record() -> AgentTraceRecord {
        AgentTraceRecord {
            schema: TraceSchema::V1,
            event_type: TraceEventType::ToolEnd,
            event_time_unix_ms: 1,
            event_source: TraceEventSource::Harness,
            agent_context: AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id: "run-1".to_string(),
                program_id: "run-1:agent".to_string(),
                parent_program_id: None,
            },
            request: None,
            tool: Some(AgentToolEvent {
                tool_call_id: "tool-123".to_string(),
                tool_class: "web_search".to_string(),
                started_at_unix_ms: None,
                ended_at_unix_ms: None,
                status: Some(AgentToolStatus::Succeeded),
                duration_ms: Some(12.5),
                output_tokens: Some(9),
                output_bytes: Some(64),
                tool_name_hash: None,
                error_type: None,
            }),
        }
    }

    fn valid_record_with_tool_call_id(tool_call_id: &str) -> AgentTraceRecord {
        let mut record = valid_record();
        record.tool.as_mut().expect("tool event").tool_call_id = tool_call_id.to_string();
        record
    }

    #[tokio::test]
    async fn relays_zmq_tool_record_to_event_plane() -> Result<()> {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let reserved = reserve_open_port();
                let endpoint = endpoint_from_listener(&reserved);
                drop(reserved);

                let runtime = Runtime::from_current()?;
                let drt =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
                let namespace =
                    drt.namespace(format!("agent-tool-relay-{}", uuid::Uuid::new_v4()))?;
                let component = namespace.component("worker")?;
                let relay =
                    AgentToolEventRelay::start(component, endpoint.clone(), None, None, None)
                        .await?;
                let mut push_socket = connect_push_socket(&endpoint).await?;
                let mut subscriber =
                    EventSubscriber::for_namespace(&namespace, DEFAULT_TOOL_EVENTS_TOPIC)
                        .await?
                        .typed::<AgentTraceRecord>();

                tokio::time::sleep(Duration::from_millis(150)).await;

                let payload = rmp_serde::to_vec_named(&valid_record())?;
                for _ in 0..5 {
                    send_multipart_direct(
                        &mut push_socket,
                        vec![Vec::new(), 1u64.to_be_bytes().to_vec(), payload.clone()],
                    )
                    .await?;
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }

                let (_envelope, record) = timeout(Duration::from_secs(5), subscriber.next())
                    .await?
                    .expect("event stream should stay open")?;

                assert_eq!(record.event_type, TraceEventType::ToolEnd);
                assert_eq!(record.event_source, TraceEventSource::Harness);
                assert_eq!(record.agent_context.workflow_id, "run-1");
                assert_eq!(record.tool.unwrap().tool_call_id, "tool-123");

                relay.shutdown();
                drt.shutdown();
                Ok(())
            },
        )
        .await
    }

    #[tokio::test]
    async fn relays_records_from_multiple_zmq_push_producers() -> Result<()> {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let reserved = reserve_open_port();
                let endpoint = endpoint_from_listener(&reserved);
                drop(reserved);

                let runtime = Runtime::from_current()?;
                let drt =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
                let namespace =
                    drt.namespace(format!("agent-tool-relay-{}", uuid::Uuid::new_v4()))?;
                let component = namespace.component("worker")?;
                let relay =
                    AgentToolEventRelay::start(component, endpoint.clone(), None, None, None)
                        .await?;
                let mut first_push = connect_push_socket(&endpoint).await?;
                let mut second_push = connect_push_socket(&endpoint).await?;
                let mut subscriber =
                    EventSubscriber::for_namespace(&namespace, DEFAULT_TOOL_EVENTS_TOPIC)
                        .await?
                        .typed::<AgentTraceRecord>();

                tokio::time::sleep(Duration::from_millis(150)).await;

                let first_payload =
                    rmp_serde::to_vec_named(&valid_record_with_tool_call_id("tool-first"))?;
                let second_payload =
                    rmp_serde::to_vec_named(&valid_record_with_tool_call_id("tool-second"))?;
                send_multipart_direct(
                    &mut first_push,
                    vec![Vec::new(), 1u64.to_be_bytes().to_vec(), first_payload],
                )
                .await?;
                send_multipart_direct(
                    &mut second_push,
                    vec![Vec::new(), 1u64.to_be_bytes().to_vec(), second_payload],
                )
                .await?;

                let mut tool_call_ids = Vec::new();
                for _ in 0..2 {
                    let (_envelope, record) = timeout(Duration::from_secs(5), subscriber.next())
                        .await?
                        .expect("event stream should stay open")?;
                    tool_call_ids.push(record.tool.expect("tool event").tool_call_id);
                }
                tool_call_ids.sort();

                assert_eq!(tool_call_ids, vec!["tool-first", "tool-second"]);

                relay.shutdown();
                drt.shutdown();
                Ok(())
            },
        )
        .await
    }

    #[tokio::test]
    async fn start_returns_error_when_zmq_pull_bind_fails() -> Result<()> {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let reserved = reserve_open_port();
                let endpoint = endpoint_from_listener(&reserved);

                let runtime = Runtime::from_current()?;
                let drt =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
                let namespace =
                    drt.namespace(format!("agent-tool-relay-{}", uuid::Uuid::new_v4()))?;
                let component = namespace.component("worker")?;

                let result =
                    AgentToolEventRelay::start(component, endpoint, None, None, None).await;

                assert!(result.is_err());

                drt.shutdown();
                Ok(())
            },
        )
        .await
    }

    #[tokio::test]
    async fn relay_filters_records_by_zmq_topic() -> Result<()> {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let reserved = reserve_open_port();
                let endpoint = endpoint_from_listener(&reserved);
                drop(reserved);

                let runtime = Runtime::from_current()?;
                let drt =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
                let namespace =
                    drt.namespace(format!("agent-tool-relay-{}", uuid::Uuid::new_v4()))?;
                let component = namespace.component("worker")?;
                let relay = AgentToolEventRelay::start(
                    component,
                    endpoint.clone(),
                    Some("matching-tools".to_string()),
                    None,
                    None,
                )
                .await?;
                let mut push_socket = connect_push_socket(&endpoint).await?;
                let mut subscriber =
                    EventSubscriber::for_namespace(&namespace, DEFAULT_TOOL_EVENTS_TOPIC)
                        .await?
                        .typed::<AgentTraceRecord>();

                tokio::time::sleep(Duration::from_millis(150)).await;

                let dropped_payload =
                    rmp_serde::to_vec_named(&valid_record_with_tool_call_id("tool-dropped"))?;
                let accepted_payload =
                    rmp_serde::to_vec_named(&valid_record_with_tool_call_id("tool-accepted"))?;

                send_multipart_direct(
                    &mut push_socket,
                    vec![
                        b"other-tools".to_vec(),
                        1u64.to_be_bytes().to_vec(),
                        dropped_payload,
                    ],
                )
                .await?;
                send_multipart_direct(
                    &mut push_socket,
                    vec![
                        b"matching-tools".to_vec(),
                        2u64.to_be_bytes().to_vec(),
                        accepted_payload,
                    ],
                )
                .await?;

                let (_envelope, record) = timeout(Duration::from_secs(5), subscriber.next())
                    .await?
                    .expect("event stream should stay open")?;

                assert_eq!(
                    record.tool.expect("tool event").tool_call_id,
                    "tool-accepted"
                );

                assert!(
                    timeout(Duration::from_millis(200), subscriber.next())
                        .await
                        .is_err()
                );

                relay.shutdown();
                drt.shutdown();
                Ok(())
            },
        )
        .await
    }
}
