// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP proof that Dynamo hosts a two-step agent loop with outbound MCP.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
};
use serde_json::{Value, json};
use serial_test::serial;
use tokio::net::TcpListener;

#[path = "common/http_harness.rs"]
#[allow(dead_code)]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[path = "common/scripted_chat_engine.rs"]
#[allow(dead_code)]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture};

#[derive(Clone, Default)]
struct McpFixtureState {
    call_count: Arc<AtomicUsize>,
    arguments: Arc<Mutex<Vec<Value>>>,
    authorization: Arc<Mutex<Vec<Option<String>>>>,
}

struct McpFixture {
    endpoint: String,
    state: McpFixtureState,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for McpFixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl McpFixture {
    async fn start() -> Self {
        let state = McpFixtureState::default();
        let app = Router::new()
            .route("/mcp", post(handle_mcp))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            endpoint: format!("http://{address}/mcp"),
            state,
            task,
        }
    }
}

async fn handle_mcp(
    State(state): State<McpFixtureState>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    state.authorization.lock().unwrap().push(
        headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
    );
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let result = match request.get("method").and_then(Value::as_str) {
        Some("server/discover") => json!({
            "resultType": "complete",
            "supportedVersions": ["2026-07-28"],
            "capabilities": {"tools": {}},
            "ttlMs": 0,
            "cacheScope": "private"
        }),
        Some("tools/list") => json!({
            "resultType": "complete",
            "tools": [{
                "name": "list_directory",
                "description": "test tool",
                "inputSchema": tool_schema()
            }],
            "ttlMs": 0,
            "cacheScope": "private"
        }),
        Some("tools/call") => {
            state.call_count.fetch_add(1, Ordering::SeqCst);
            state
                .arguments
                .lock()
                .unwrap()
                .push(request.pointer("/params/arguments").cloned().unwrap());
            json!({
                "resultType": "complete",
                "content": [{"type": "text", "text": "a.txt"}],
                "structuredContent": {"entries": ["a.txt"]},
                "isError": false
            })
        }
        _ => {
            return (
                StatusCode::OK,
                Json(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32601, "message": "method not found"}
                })),
            )
                .into_response();
        }
    };
    (
        StatusCode::OK,
        Json(json!({"jsonrpc": "2.0", "id": id, "result": result})),
    )
        .into_response()
}

fn tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": false
    })
}

fn responses_tool() -> Value {
    json!({
        "type": "function",
        "name": "list_directory",
        "description": "test tool",
        "parameters": tool_schema(),
        "strict": true
    })
}

fn anthropic_tool() -> Value {
    json!({
        "name": "list_directory",
        "description": "test tool",
        "input_schema": tool_schema()
    })
}

fn configured_tools() -> String {
    json!([{
        "name": "list_directory",
        "remote_name": "list_directory",
        "description": "test tool",
        "input_schema": tool_schema(),
        "timeout_millis": 5000,
        "max_output_bytes": 4096
    }])
    .to_string()
}

fn assert_tool_result(
    request: &dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest,
) {
    let tool_result = request
        .inner
        .messages
        .iter()
        .find_map(|message| match message {
            ChatCompletionRequestMessage::Tool(tool) => Some(tool),
            _ => None,
        })
        .expect("second model step must contain the MCP result");
    assert!(matches!(
        &tool_result.content,
        ChatCompletionRequestToolMessageContent::Text(text)
            if text.contains("entries") && text.contains("a.txt")
    ));
}

#[tokio::test]
#[serial]
async fn responses_and_anthropic_complete_a_server_side_mcp_round() {
    let mcp = McpFixture::start().await;
    let directory = tempfile::tempdir().unwrap();
    let sqlite_path = directory.path().join("agent-runtime.sqlite");
    let environment = vec![
        ("DYN_ENABLE_AGENT_RT_POC", Some("true".to_owned())),
        ("DYN_AGENT_RT_AUTH_MODE", Some("local".to_owned())),
        ("DYN_AGENT_RT_PERMITTED_CONNECTORS", Some("mcp".to_owned())),
        ("DYN_AGENT_RT_MCP_ENDPOINT", Some(mcp.endpoint.clone())),
        (
            "DYN_AGENT_RT_MCP_BEARER_TOKEN",
            Some("fixture-secret".to_owned()),
        ),
        (
            "DYN_AGENT_RT_MCP_ALLOW_HTTP_LOOPBACK",
            Some("true".to_owned()),
        ),
        ("DYN_AGENT_RT_MCP_TOOLS_JSON", Some(configured_tools())),
        (
            "DYN_AGENT_RT_SQLITE_PATH",
            Some(sqlite_path.to_string_lossy().into_owned()),
        ),
        ("DYN_ENABLE_ANTHROPIC_API", Some("1".to_owned())),
        (
            "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS",
            Some("0".to_owned()),
        ),
    ];

    temp_env::async_with_vars(environment, async {
        let tool_script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        let text_script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([
            tool_script.clone(),
            text_script.clone(),
            tool_script,
            text_script,
        ])
        .await;

        let response = svc
            .client
            .post(format!("{}/v1/responses", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "input": "List /tmp",
                "stream": false,
                "tools": [responses_tool()]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(
            body["output"].as_array().unwrap().last().unwrap()["content"][0]["text"],
            "Pong."
        );

        let response = svc
            .client
            .post(format!("{}/v1/messages", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": false,
                "messages": [{"role": "user", "content": "List /tmp"}],
                "tools": [anthropic_tool()]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(
            body["content"].as_array().unwrap().last().unwrap()["text"],
            "Pong."
        );

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 4);
        assert_tool_result(&requests[1]);
        assert_tool_result(&requests[3]);
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        assert_eq!(mcp.state.call_count.load(Ordering::SeqCst), 2);
        assert_eq!(
            mcp.state.arguments.lock().unwrap().as_slice(),
            [json!({"path": "/tmp"}), json!({"path": "/tmp"})]
        );
        assert!(
            mcp.state
                .authorization
                .lock()
                .unwrap()
                .iter()
                .all(|header| header.as_deref() == Some("Bearer fixture-secret"))
        );

        svc.shutdown().await;
    })
    .await;
}
