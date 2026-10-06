// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    collections::HashSet,
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use protocol::ModelId;
use switchyard_runner::Runner;

use crate::request;

pub const MODEL_HEADER: &str = "x-gateway-model-name";

pub struct Router {
    runner: Runner,
    sessions: Mutex<HashSet<(String, String, Option<String>)>>,
}

impl Router {
    pub fn load(routes: impl AsRef<Path>) -> Result<Self> {
        // SDK parse/build access would allow validation of decision-only options before construction.
        Ok(Self::new(Runner::load(routes)?))
    }

    pub fn new(runner: Runner) -> Self {
        Self {
            runner,
            sessions: Mutex::new(HashSet::new()),
        }
    }

    pub async fn decide(
        &self,
        body: &[u8],
        headers: &http::HeaderMap,
    ) -> Result<(Vec<u8>, String)> {
        let started = Instant::now();
        let request = {
            let raw = crate::json::parse(body)?;
            request::decode(&raw, headers)?
        };
        let input_model = request
            .llm_request
            .model
            .as_ref()
            .context("missing request model")?
            .as_str();
        let route_id = ModelId::from(input_model);
        let route = self
            .runner
            .route(route_id.as_str())
            .context("model is outside the configured catalog")?;
        let agent = request.metadata.as_ref().and_then(|m| m.agent_id.clone());
        ensure!(
            agent.as_ref().is_none_or(|id| id.len() <= 256),
            "agent id exceeds 256 bytes"
        );
        if let Some(session) = request
            .metadata
            .as_ref()
            .and_then(|m| m.session_id.as_ref())
        {
            ensure!(session.len() <= 256, "session id exceeds 256 bytes");
            let identity = (route_id.as_str().to_owned(), session.clone(), agent);
            let mut sessions = self.sessions.lock().unwrap_or_else(|p| p.into_inner());
            if !sessions.contains(&identity) && sessions.len() >= 4096 {
                return Err(dynamo_ext_proc::PreprocessError::new(
                    503,
                    "session identity capacity exceeded",
                )
                .into());
            }
            sessions.insert(identity);
        }
        let mut original_ir = request.llm_request.clone();
        original_ir.model = None;
        let outcome = tokio::time::timeout(Duration::from_secs(1), route.decide(request))
            .await
            .map_err(|_| {
                dynamo_ext_proc::PreprocessError::new(504, "SDK routing deadline exceeded")
            })?
            .map_err(|_| dynamo_ext_proc::PreprocessError::new(500, "SDK routing failed"))?;
        ensure!(outcome.response.is_none(), "routing-only contract violated");
        let model = outcome.selected_model_id()?.clone();
        let mut selected_ir = outcome.request.llm_request;
        selected_ir.model = None;
        ensure!(
            selected_ir == original_ir,
            "router unexpectedly rewrote request semantics"
        );
        let output = crate::json::replace_model(body, model.as_str())?;
        let elapsed_us = started.elapsed().as_micros() as u64;
        tracing::info!(
            %model,
            elapsed_us,
            "routing decision"
        );
        Ok((output, model.as_str().to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::collections::HashMap;

    fn router() -> Router {
        Router::new(Runner::from_toml(include_str!("../config/routes.toml")).unwrap())
    }
    fn neutral() -> Value {
        json!({"model":"auto", "messages":[{"role":"user", "content":"Say hello"}], "stream":true, "temperature":0.2, "custom_vendor_field":{"preserve":true}, "nvext":{"custom":"keep"}})
    }
    fn recovery() -> Value {
        json!({"model":"auto", "messages":[
        {"role":"user", "content":"Fix the test failure"},
        {"role":"assistant", "content":null, "tool_calls":[{"id":"call-1", "type":"function", "function":{"name":"Bash", "arguments":"{\"command\":\"pytest tests/\"}"}}]},
        {"role":"tool", "tool_call_id":"call-1", "content":"MemoryError: out of memory"}
    ], "tools":[{"type":"function", "function":{"name":"Bash", "parameters":{"type":"object"}}}]})
    }
    fn session(id: &str) -> http::HeaderMap {
        let mut h = http::HeaderMap::new();
        h.insert("x-switchyard-session-id", id.parse().unwrap());
        h
    }

    #[tokio::test]
    async fn toml_controls_stage_policy() {
        let source =
            include_str!("../config/routes.toml").replace("efficient_first", "capable_first");
        let r = Router::new(Runner::from_toml(&source).unwrap());
        assert_eq!(
            r.decide(
                &serde_json::to_vec(&neutral()).unwrap(),
                &http::HeaderMap::new()
            )
            .await
            .unwrap()
            .1,
            "Qwen/Qwen3-1.7B"
        );
    }

    #[tokio::test]
    async fn named_route_selects_added_target_without_sdk_http_calls() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let source = include_str!("../config/routes.toml").replace(
            "http://switchyard-gateway/v1",
            &format!("http://{}/v1", listener.local_addr().unwrap()),
        );
        let source = format!(
            "{source}\n[targets.third]\nid = 'Another/Model'\nllm_client = 'dynamo'\n[routes.direct]\nid = 'direct-third'\ntype = 'passthrough'\ntarget = 'third'\n"
        );
        let r = Router::new(Runner::from_toml(&source).unwrap());
        let mut input = neutral();
        input["model"] = json!("direct-third");
        let (body, model) = r
            .decide(
                &serde_json::to_vec(&input).unwrap(),
                &http::HeaderMap::new(),
            )
            .await
            .unwrap();
        assert_eq!(model, "Another/Model");
        input["model"] = json!("Another/Model");
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), input);
        assert!(matches!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        ));
    }

    #[tokio::test]
    async fn named_routes_isolate_state_and_reject_oversized_agent_without_session() {
        let source = format!(
            "{}\n[routes.other]\nid = 'other'\ntype = 'stage_router'\nefficient_target = 'qwen-small'\ncapable_target = 'qwen-large'\npicker = 'efficient_first'\nconfidence_threshold = 0.5\n",
            include_str!("../config/routes.toml")
        );
        let r = Router::new(Runner::from_toml(&source).unwrap());
        let headers = session("shared-session");
        assert_eq!(
            r.decide(&serde_json::to_vec(&recovery()).unwrap(), &headers)
                .await
                .unwrap()
                .1,
            "Qwen/Qwen3-1.7B"
        );
        let mut quiet = neutral();
        quiet["model"] = json!("other");
        assert_eq!(
            r.decide(&serde_json::to_vec(&quiet).unwrap(), &headers)
                .await
                .unwrap()
                .1,
            "Qwen/Qwen3-0.6B"
        );
        quiet["model"] = json!("auto");
        assert_eq!(
            r.decide(&serde_json::to_vec(&quiet).unwrap(), &headers)
                .await
                .unwrap()
                .1,
            "Qwen/Qwen3-1.7B"
        );
        let mut headers = http::HeaderMap::new();
        headers.insert("x-switchyard-agent-id", "x".repeat(257).parse().unwrap());
        assert!(
            r.decide(&serde_json::to_vec(&quiet).unwrap(), &headers)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn actual_stage_router_routes_neutral_to_efficient_and_preserves_fields() {
        let input = neutral();
        let (body, model) = router()
            .decide(
                &serde_json::to_vec(&input).unwrap(),
                &http::HeaderMap::new(),
            )
            .await
            .unwrap();
        assert_eq!(model, "Qwen/Qwen3-0.6B");
        let mut expected = input;
        expected["model"] = json!("Qwen/Qwen3-0.6B");
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), expected);
    }

    #[tokio::test]
    async fn preserves_original_values_including_precise_numbers_and_nested_fields() {
        let input = br#"{
            "mo\u0064el":"auto",
            "messages":[ {"role":"user", "content":"Hello"} ],
            "large_integer":18446744073709551617,
            "precise_decimal":0.12345678901234567890123456789,
            "custom\u005ffield":{ "nested": [18446744073709551617, {"text":"\u0061"}] }
        }"#;
        let (output, model) = router()
            .decide(input, &http::HeaderMap::new())
            .await
            .unwrap();
        assert_eq!(model, "Qwen/Qwen3-0.6B");
        let original: HashMap<String, &serde_json::value::RawValue> =
            serde_json::from_slice(input).unwrap();
        let rewritten: HashMap<String, &serde_json::value::RawValue> =
            serde_json::from_slice(&output).unwrap();
        assert_eq!(original.len(), rewritten.len());
        for (key, value) in original {
            if key == "model" {
                assert_eq!(rewritten[&key].get(), r#""Qwen/Qwen3-0.6B""#);
            } else {
                assert_eq!(rewritten[&key].get(), value.get(), "changed {key}");
            }
        }
    }

    #[tokio::test]
    async fn actual_stage_router_escalates_critical_tool_failure_and_preserves_history() {
        let mut input = recovery();
        let (body, model) = router()
            .decide(
                &serde_json::to_vec(&input).unwrap(),
                &http::HeaderMap::new(),
            )
            .await
            .unwrap();
        assert_eq!(model, "Qwen/Qwen3-1.7B");
        input["model"] = json!("Qwen/Qwen3-1.7B");
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), input);
    }

    #[tokio::test]
    async fn same_session_holds_capable_without_leaking_to_other_sessions() {
        let r = router();
        let failure = serde_json::to_vec(&recovery()).unwrap();
        let quiet = serde_json::to_vec(&neutral()).unwrap();
        assert_eq!(
            r.decide(&failure, &session("session-a")).await.unwrap().1,
            "Qwen/Qwen3-1.7B"
        );
        assert_eq!(
            r.decide(&quiet, &session("session-b")).await.unwrap().1,
            "Qwen/Qwen3-0.6B"
        );
        assert_eq!(
            r.decide(&quiet, &session("session-a")).await.unwrap().1,
            "Qwen/Qwen3-1.7B"
        );
        assert_eq!(
            r.decide(&quiet, &session("session-a")).await.unwrap().1,
            "Qwen/Qwen3-1.7B"
        );
        assert_eq!(
            r.decide(&quiet, &session("session-a")).await.unwrap().1,
            "Qwen/Qwen3-0.6B"
        );
    }

    #[tokio::test]
    async fn rejects_unknown_catalog_and_media() {
        let r = router();
        let mut body = neutral();
        body["model"] = json!("attacker-model");
        assert_eq!(
            r.decide(&serde_json::to_vec(&body).unwrap(), &http::HeaderMap::new())
                .await
                .unwrap_err()
                .to_string(),
            "model is outside the configured catalog"
        );
        body["model"] = json!("auto");
        body["messages"][0]["content"] =
            json!([{"type":"image_url", "image_url":{"url":"https://example.com/image.png"}}]);
        assert!(
            r.decide(&serde_json::to_vec(&body).unwrap(), &http::HeaderMap::new())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn projects_malformed_historical_arguments_and_compacted_tool_output() {
        let r = router();
        let mut body = recovery();
        body["messages"][1]["tool_calls"][0]["function"]["arguments"] = json!("{bad json");
        assert_eq!(
            r.decide(&serde_json::to_vec(&body).unwrap(), &http::HeaderMap::new())
                .await
                .unwrap()
                .1,
            "Qwen/Qwen3-1.7B"
        );
        body["messages"] = json!([{"role":"tool", "tool_call_id":"historical", "content":"MemoryError: out of memory"}, {"role":"user", "content":"Recover"}]);
        assert_eq!(
            r.decide(&serde_json::to_vec(&body).unwrap(), &http::HeaderMap::new())
                .await
                .unwrap()
                .1,
            "Qwen/Qwen3-1.7B"
        );
    }
}
