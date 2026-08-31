// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SATF workload artifacts and Dynamo request-trace conversion.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::mooncake::RollingHashIdMapper;
use crate::request_trace::load::{
    LoadedAgentTrace, RequestEntry, RequestTraceFinishReasonMetadata, ToolEntry,
    load_request_trace_records,
};

pub const SATF_FORMAT: &str = "satf";
pub const SATF_VERSION: &str = "2.0.0";
pub const SATF_TIME_BASE: &str = "us";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfTrace {
    pub format: String,
    pub format_version: String,
    pub time_base: String,
    pub block_size_tokens: usize,
    pub sessions: Vec<SatfSession>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer: Option<SatfProducer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfProducer {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfSession {
    pub session_id: String,
    pub nodes: Vec<SatfNode>,
    pub edges: Vec<SatfEdge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SatfNode {
    LlmInfer {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_us: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        min_start_delay_us: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        performance_metrics: Option<SatfPerformanceMetrics>,
        payload: SatfLlmInferPayload,
        #[serde(skip_serializing_if = "Option::is_none")]
        extensions: Option<Value>,
    },
    ToolCall {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        min_start_delay_us: Option<u64>,
        payload: SatfToolCallPayload,
        #[serde(skip_serializing_if = "Option::is_none")]
        extensions: Option<Value>,
    },
    ToolResult {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_us: Option<u64>,
        payload: SatfToolResultPayload,
        #[serde(skip_serializing_if = "Option::is_none")]
        extensions: Option<Value>,
    },
}

impl SatfNode {
    pub fn id(&self) -> &str {
        match self {
            Self::LlmInfer { id, .. } | Self::ToolCall { id, .. } | Self::ToolResult { id, .. } => {
                id
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SatfEdge {
    pub from: String,
    pub to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delay_after_predecessor_us: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfLlmInferPayload {
    pub input_tokens_total: u64,
    pub output_tokens: u64,
    pub kv_block_ids: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_parameters: Option<SatfRequestParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfRequestParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfToolCallPayload {
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfToolResultPayload {
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SatfPerformanceMetrics {
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inter_token_latency_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_delay_us: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_hit_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_latency_us: Option<f64>,
}

pub fn convert_request_trace_to_satf(paths: &[PathBuf]) -> Result<SatfTrace> {
    convert_loaded_trace(load_request_trace_records(paths)?)
}

pub fn write_satf<W: Write>(writer: W, trace: &SatfTrace, pretty: bool) -> Result<()> {
    if pretty {
        serde_json::to_writer_pretty(writer, trace).context("serializing pretty SATF JSON")?;
    } else {
        serde_json::to_writer(writer, trace).context("serializing SATF JSON")?;
    }
    Ok(())
}

fn convert_loaded_trace(mut loaded: LoadedAgentTrace) -> Result<SatfTrace> {
    loaded.requests.sort_by(|left, right| {
        (left.start_ms, left.end_ms, &left.request.request_id).cmp(&(
            right.start_ms,
            right.end_ms,
            &right.request.request_id,
        ))
    });

    let block_size_tokens = loaded.requests[0].replay.trace_block_size;
    if loaded
        .requests
        .iter()
        .any(|request| request.replay.trace_block_size != block_size_tokens)
    {
        bail!("SATF conversion requires one trace block size per artifact");
    }

    let artifact_start_unix_us = loaded
        .requests
        .iter()
        .map(request_start_us)
        .min()
        .expect("request-trace loading requires at least one request");

    let mut mapper = RollingHashIdMapper::new(block_size_tokens);
    for request in &mut loaded.requests {
        request.replay.input_sequence_hashes =
            mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
    }

    let mut requests_by_session: BTreeMap<String, Vec<RequestEntry>> = BTreeMap::new();
    for request in loaded.requests {
        requests_by_session
            .entry(session_id_for(&request))
            .or_default()
            .push(request);
    }

    let mut tools_by_session: HashMap<String, Vec<ToolEntry>> = HashMap::new();
    for tool in loaded.tools {
        tools_by_session
            .entry(tool.session_id.clone())
            .or_default()
            .push(tool);
    }

    let mut sessions = requests_by_session
        .into_iter()
        .map(|(session_id, requests)| {
            let tools = tools_by_session.remove(&session_id).unwrap_or_default();
            convert_session(session_id, requests, tools, artifact_start_unix_us)
        })
        .collect::<Result<Vec<_>>>()?;
    sessions.sort_by(|left, right| {
        session_first_start_us(left)
            .cmp(&session_first_start_us(right))
            .then_with(|| left.session_id.cmp(&right.session_id))
    });

    Ok(SatfTrace {
        format: SATF_FORMAT.to_string(),
        format_version: SATF_VERSION.to_string(),
        time_base: SATF_TIME_BASE.to_string(),
        block_size_tokens,
        sessions,
        producer: Some(SatfProducer {
            name: "dynamo-request-trace-converter".to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            commit: None,
        }),
        extensions: Some(vendor_extension(Value::Object(Map::from_iter([(
            "artifact_start_unix_us".to_string(),
            artifact_start_unix_us.into(),
        )])))),
    })
}

fn convert_session(
    session_id: String,
    mut requests: Vec<RequestEntry>,
    mut tools: Vec<ToolEntry>,
    artifact_start_unix_us: u64,
) -> Result<SatfSession> {
    requests.sort_by(|left, right| {
        (
            request_start_us(left),
            request_end_us(left),
            &left.request.request_id,
        )
            .cmp(&(
                request_start_us(right),
                request_end_us(right),
                &right.request.request_id,
            ))
    });
    tools.sort_by_key(|tool| (tool_start_us(tool), tool_end_us(tool)));

    let extensions = session_extensions(&requests[0]);
    let mut nodes = requests
        .iter()
        .enumerate()
        .map(|(index, request)| llm_node(request, (index == 0).then_some(artifact_start_unix_us)))
        .collect::<Vec<_>>();
    let mut edges = Vec::new();
    let request_index = requests
        .iter()
        .enumerate()
        .map(|(index, request)| (request.request.request_id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut tool_paths = HashSet::new();

    for (ordinal, tool) in tools.iter().enumerate() {
        let source_index = tool
            .claude
            .as_ref()
            .and_then(|metadata| {
                request_index
                    .get(metadata.source_request_id.as_str())
                    .copied()
            })
            .or_else(|| find_tool_source(&requests, tool));
        let consumer_index = tool
            .claude
            .as_ref()
            .and_then(|metadata| metadata.consumer_request_id.as_deref())
            .and_then(|request_id| request_index.get(request_id).copied())
            .or_else(|| find_tool_consumer(&requests, tool, source_index));

        let call_node_id = format!("tool_call:{ordinal}");
        let result_node_id = format!("tool_result:{ordinal}");
        let source_request = source_index.map(|index| &requests[index]);
        let tool_name = source_request
            .and_then(|request| tool_name_from_finish_metadata(request, &tool.tool_call_id))
            .unwrap_or_else(|| tool.tool_class.clone());

        nodes.push(SatfNode::ToolCall {
            id: call_node_id.clone(),
            min_start_delay_us: source_index
                .is_none()
                .then(|| tool_start_us(tool).saturating_sub(artifact_start_unix_us)),
            payload: SatfToolCallPayload {
                tool_name,
                tool_call_id: Some(tool.tool_call_id.clone()),
            },
            extensions: None,
        });
        nodes.push(SatfNode::ToolResult {
            id: result_node_id.clone(),
            duration_us: Some(tool_duration_us(tool)),
            payload: SatfToolResultPayload {
                tool_call_id: tool.tool_call_id.clone(),
                is_error: Some(tool_is_error(tool)),
            },
            extensions: Some(tool_extensions(tool)),
        });

        if let Some(source_index) = source_index {
            edges.push(edge(
                llm_node_id(&requests[source_index]),
                call_node_id.clone(),
                tool_start_us(tool).saturating_sub(request_end_us(&requests[source_index])),
            ));
        }
        edges.push(edge(call_node_id, result_node_id.clone(), 0));
        if let Some(consumer_index) = consumer_index {
            edges.push(edge(
                result_node_id,
                llm_node_id(&requests[consumer_index]),
                request_start_us(&requests[consumer_index]).saturating_sub(tool_end_us(tool)),
            ));
            if let Some(source_index) = source_index {
                tool_paths.insert((source_index, consumer_index));
            }
        }
    }

    for current_index in 1..requests.len() {
        let previous_index = current_index - 1;
        if tool_paths.contains(&(previous_index, current_index)) {
            continue;
        }
        edges.push(edge(
            llm_node_id(&requests[previous_index]),
            llm_node_id(&requests[current_index]),
            request_start_us(&requests[current_index])
                .saturating_sub(request_end_us(&requests[previous_index])),
        ));
    }

    Ok(SatfSession {
        session_id,
        nodes,
        edges,
        extensions,
    })
}

fn llm_node(request: &RequestEntry, artifact_start_unix_us: Option<u64>) -> SatfNode {
    let metrics = SatfPerformanceMetrics {
        source: "dynamo".to_string(),
        time_to_first_token_us: request.request.ttft_ms.map(|value| value * 1_000.0),
        inter_token_latency_us: request.request.avg_itl_ms.map(|value| value * 1_000.0),
        queue_delay_us: request
            .request
            .prefill_wait_time_ms
            .map(|value| value * 1_000.0),
        kv_cache_hit_rate: request.request.kv_hit_rate,
        kv_transfer_latency_us: request
            .request
            .kv_transfer_estimated_latency_ms
            .map(|value| value * 1_000.0),
    };
    let metrics_present = metrics.time_to_first_token_us.is_some()
        || metrics.inter_token_latency_us.is_some()
        || metrics.queue_delay_us.is_some()
        || metrics.kv_cache_hit_rate.is_some()
        || metrics.kv_transfer_latency_us.is_some();

    SatfNode::LlmInfer {
        id: llm_node_id(request),
        duration_us: Some(request_duration_us(request)),
        min_start_delay_us: artifact_start_unix_us
            .map(|start| request_start_us(request).saturating_sub(start)),
        performance_metrics: metrics_present.then_some(metrics),
        payload: SatfLlmInferPayload {
            input_tokens_total: request.replay.input_length as u64,
            output_tokens: request.request.output_tokens.unwrap_or_default(),
            kv_block_ids: request.replay.input_sequence_hashes.clone(),
            request_parameters: request
                .request
                .model
                .as_ref()
                .map(|model| SatfRequestParameters {
                    model: Some(model.clone()),
                }),
        },
        extensions: Some(request_extensions(request)),
    }
}

fn request_extensions(request: &RequestEntry) -> Value {
    let mut fields = Map::from_iter([(
        "request_id".to_string(),
        request.request.request_id.clone().into(),
    )]);
    insert_optional(
        &mut fields,
        "x_request_id",
        request.request.x_request_id.clone(),
    );
    insert_optional(&mut fields, "input_tokens", request.request.input_tokens);
    insert_optional(&mut fields, "cached_tokens", request.request.cached_tokens);
    insert_optional(
        &mut fields,
        "prefill_time_ms",
        request.request.prefill_time_ms,
    );
    insert_optional(&mut fields, "queue_depth", request.request.queue_depth);
    insert_optional(&mut fields, "worker", request.request.worker.clone());
    insert_optional(
        &mut fields,
        "finish_reason_metadata",
        request.request.finish_reason_metadata.clone(),
    );
    vendor_extension(Value::Object(fields))
}

fn tool_extensions(tool: &ToolEntry) -> Value {
    let mut fields = Map::from_iter([
        ("status".to_string(), tool.status.clone().into()),
        ("tool_class".to_string(), tool.tool_class.clone().into()),
    ]);
    insert_optional(&mut fields, "output_bytes", tool.output_bytes);
    insert_optional(&mut fields, "output_tokens", tool.output_tokens);
    insert_optional(&mut fields, "error_type", tool.error_type.clone());
    vendor_extension(Value::Object(fields))
}

fn session_extensions(request: &RequestEntry) -> Option<Value> {
    let context = request.agent_context.as_ref()?;
    let mut fields = Map::new();
    insert_optional(
        &mut fields,
        "parent_session_id",
        context.parent_session_id.clone(),
    );
    (!fields.is_empty()).then(|| vendor_extension(Value::Object(fields)))
}

fn vendor_extension(value: Value) -> Value {
    Value::Object(Map::from_iter([("nvidia.dynamo".to_string(), value)]))
}

fn insert_optional<T: Serialize>(fields: &mut Map<String, Value>, name: &str, value: Option<T>) {
    if let Some(value) = value {
        fields.insert(
            name.to_string(),
            serde_json::to_value(value).expect("serializing trace metadata cannot fail"),
        );
    }
}

fn session_id_for(request: &RequestEntry) -> String {
    request
        .agent_context
        .as_ref()
        .map(|context| context.session_id.clone())
        .unwrap_or_else(|| request.request.request_id.clone())
}

fn llm_node_id(request: &RequestEntry) -> String {
    format!("llm:{}", request.request.request_id)
}

fn edge(from: String, to: String, delay_us: u64) -> SatfEdge {
    SatfEdge {
        from,
        to,
        delay_after_predecessor_us: (delay_us > 0).then_some(delay_us),
    }
}

fn request_start_us(request: &RequestEntry) -> u64 {
    request
        .request
        .request_received_ms
        .map(|value| value.saturating_mul(1_000))
        .unwrap_or_else(|| nonnegative_us(request.start_ms))
}

fn request_duration_us(request: &RequestEntry) -> u64 {
    request
        .request
        .total_time_ms
        .map(ms_to_us)
        .unwrap_or_else(|| nonnegative_us(request.end_ms.saturating_sub(request.start_ms)))
}

fn request_end_us(request: &RequestEntry) -> u64 {
    request_start_us(request).saturating_add(request_duration_us(request))
}

fn tool_duration_us(tool: &ToolEntry) -> u64 {
    ms_to_us(tool.duration_ms)
}

fn tool_end_us(tool: &ToolEntry) -> u64 {
    nonnegative_us(tool.end_ms)
}

fn tool_start_us(tool: &ToolEntry) -> u64 {
    nonnegative_us(tool.start_ms)
}

fn nonnegative_us(value_ms: i64) -> u64 {
    u64::try_from(value_ms)
        .unwrap_or_default()
        .saturating_mul(1_000)
}

fn ms_to_us(value: f64) -> u64 {
    let value = (value.max(0.0) * 1_000.0).round();
    if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn find_tool_source(requests: &[RequestEntry], tool: &ToolEntry) -> Option<usize> {
    let tool_start_us = tool_start_us(tool);
    requests
        .iter()
        .enumerate()
        .rev()
        .find(|(_, request)| request_end_us(request) <= tool_start_us)
        .or_else(|| {
            requests
                .iter()
                .enumerate()
                .rev()
                .find(|(_, request)| request_start_us(request) <= tool_start_us)
        })
        .map(|(index, _)| index)
}

fn find_tool_consumer(
    requests: &[RequestEntry],
    tool: &ToolEntry,
    source_index: Option<usize>,
) -> Option<usize> {
    if tool
        .claude
        .as_ref()
        .is_some_and(|metadata| metadata.execution_mode == "background")
    {
        return None;
    }
    requests
        .iter()
        .enumerate()
        .skip(source_index.map_or(0, |index| index + 1))
        .find(|(_, request)| request_start_us(request) >= tool_end_us(tool))
        .map(|(index, _)| index)
}

fn tool_name_from_finish_metadata(request: &RequestEntry, tool_call_id: &str) -> Option<String> {
    request.request.finish_reason_metadata.as_ref().and_then(
        |metadata: &RequestTraceFinishReasonMetadata| {
            metadata
                .tool_calls
                .iter()
                .find(|call| call.id.as_deref() == Some(tool_call_id))
                .and_then(|call| call.name.clone())
        },
    )
}

fn tool_is_error(tool: &ToolEntry) -> bool {
    tool.error_type.is_some()
        || matches!(
            tool.status.as_str(),
            "error" | "failed" | "cancelled" | "canceled"
        )
}

fn session_first_start_us(session: &SatfSession) -> u64 {
    session
        .nodes
        .iter()
        .find_map(|node| match node {
            SatfNode::LlmInfer {
                min_start_delay_us, ..
            } => *min_start_delay_us,
            _ => None,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn request_trace_sessions_share_kv_ids() {
        let mut file = NamedTempFile::new().unwrap();
        for (request_id, session_id, timestamp) in
            [("req-a", "task-a", 1_000), ("req-b", "task-b", 1_100)]
        {
            writeln!(
                file,
                "{}",
                serde_json::json!({
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": timestamp + 100,
                    "agent_context": {"session_id": session_id},
                    "request": {
                        "request_id": request_id,
                        "request_received_ms": timestamp,
                        "output_tokens": 4,
                        "replay": {
                            "trace_block_size": 2,
                            "input_length": 3,
                            "input_sequence_hashes": [11, 22],
                        }
                    }
                })
            )
            .unwrap();
        }

        let trace = convert_request_trace_to_satf(&[file.path().to_path_buf()]).unwrap();

        assert_eq!(trace.sessions.len(), 2);
        assert_eq!(trace.sessions[0].session_id, "task-a");
        assert_eq!(trace.sessions[1].session_id, "task-b");
        let first_ids = llm_payload(&trace.sessions[0].nodes[0])
            .kv_block_ids
            .clone();
        let second_ids = llm_payload(&trace.sessions[1].nodes[0])
            .kv_block_ids
            .clone();
        assert_eq!(first_ids, second_ids);
        assert_eq!(first_ids, vec![0, 1]);
    }

    #[test]
    fn request_trace_tool_span_becomes_call_and_result_path() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1100,"agent_context":{{"session_id":"root"}},"request":{{"request_id":"req-1","model":"test","request_received_ms":1000,"total_time_ms":100.0,"output_tokens":4,"replay":{{"trace_block_size":2,"input_length":3,"input_sequence_hashes":[11,22]}},"finish_reason_metadata":{{"finish_reason":"tool_calls","tool_calls":[{{"choice_index":0,"tool_call_index":0,"id":"call-1","name":"search"}}]}}}}}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"tool_end","event_time_unix_ms":1200,"agent_context":{{"session_id":"root"}},"tool":{{"tool_call_id":"call-1","tool_class":"fallback","started_at_unix_ms":1110,"ended_at_unix_ms":1200,"status":"succeeded","duration_ms":50.0}}}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":1400,"agent_context":{{"session_id":"root"}},"request":{{"request_id":"req-2","request_received_ms":1300,"total_time_ms":100.0,"output_tokens":2,"replay":{{"trace_block_size":2,"input_length":4,"input_sequence_hashes":[11,33]}}}}}}"#
        )
        .unwrap();

        let trace = convert_request_trace_to_satf(&[file.path().to_path_buf()]).unwrap();
        let session = &trace.sessions[0];

        assert_eq!(session.nodes.len(), 4);
        assert_eq!(session.edges.len(), 3);
        assert!(session.edges.iter().any(|edge| {
            edge.from == "llm:req-1"
                && edge.to == "tool_call:0"
                && edge.delay_after_predecessor_us == Some(10_000)
        }));
        assert!(session.edges.iter().any(|edge| {
            edge.from == "tool_result:0"
                && edge.to == "llm:req-2"
                && edge.delay_after_predecessor_us == Some(100_000)
        }));
        let call = session
            .nodes
            .iter()
            .find(|node| node.id() == "tool_call:0")
            .unwrap();
        assert!(matches!(
            call,
            SatfNode::ToolCall {
                payload: SatfToolCallPayload { tool_name, .. },
                ..
            } if tool_name == "search"
        ));

        let encoded = serde_json::to_string(&trace).unwrap();
        let decoded: SatfTrace = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, trace);
    }

    fn llm_payload(node: &SatfNode) -> &SatfLlmInferPayload {
        match node {
            SatfNode::LlmInfer { payload, .. } => payload,
            other => panic!("expected llm_infer node, got {other:?}"),
        }
    }
}
