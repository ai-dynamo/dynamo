// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for the standalone selection service (`python -m dynamo.select_service`).
//!
//! Router-only mode delegates all KV-aware worker selection and active-load
//! accounting to one or more selection-service replicas over HTTP. This module
//! defines local serialize/deserialize mirror types for the subset of the
//! service's wire contract that router-only V1 needs, so the EPP does not depend
//! on the selection service's internal types or its crate features. The mirror
//! types intentionally match the documented JSON field names
//! (`docs/components/router/standalone-selection.md`).
//!
//! V1 (aggregated, query-only) uses:
//! - `POST/PATCH/DELETE/GET /workers` to reconcile the worker catalog, and
//! - `POST /select` to obtain a worker + dp_rank for each request, and
//! - `GET /ready` for readiness gating.
//!
//! Reservation and lifecycle endpoints (`/select_and_reserve`, `/reservations*`)
//! are deliberately omitted until load-accurate routing is added.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Worker registration payload (`POST /workers`). Only the fields router-only
/// mode populates are included; the service defaults the rest.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerRegistration {
    pub worker_id: u64,
    pub model_name: String,
    pub endpoint: String,
    pub block_size: u32,
    pub data_parallel_size: u32,
    pub kv_events_endpoints: HashMap<u32, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_kv_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_batched_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stable_routing_id: Option<String>,
}

/// Partial worker update (`PATCH /workers/{id}`). Any `Some` field is applied;
/// the service re-reconciles the worker afterward.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WorkerPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_events_endpoints: Option<HashMap<u32, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_kv_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stable_routing_id: Option<String>,
}

/// Query-only selection request (`POST /select`). The prompt fields are sent
/// flat (the service flattens them into its internal prompt type). Sending raw
/// `token_ids` lets the service compute block/sequence hashes itself.
#[derive(Debug, Clone, Serialize)]
pub struct SelectRequest {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_id: Option<String>,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_worker_ids: Option<HashSet<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority_jump: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
}

/// Raw observability overlap summary (matched token counts).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OverlapSummary {
    #[serde(default)]
    pub longest_matched: u32,
    #[serde(default)]
    pub gpu: u32,
    #[serde(default)]
    pub cpu: u32,
    #[serde(default)]
    pub disk: u32,
}

/// Selection result returned by `/select`.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectResponse {
    #[serde(default)]
    pub selection_id: Option<String>,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub endpoint: String,
    pub block_size: u32,
    #[serde(default)]
    pub overlap: OverlapSummary,
    #[serde(default)]
    pub effective_prefill_tokens: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ReadyResponse {
    #[serde(default)]
    ready: bool,
}

/// Async client over a set of selection-service replica base URLs.
#[derive(Clone)]
pub struct SelectorClient {
    http: reqwest::Client,
    urls: Vec<String>,
}

impl SelectorClient {
    /// Build a client over the given replica base URLs (e.g. `http://selector:8092`).
    pub fn new(urls: Vec<String>) -> Result<Self> {
        if urls.is_empty() {
            bail!("SelectorClient requires at least one selection-service URL");
        }
        let http = reqwest::Client::builder()
            .build()
            .context("building selection-service HTTP client")?;
        let urls = urls
            .into_iter()
            .map(|u| u.trim_end_matches('/').to_string())
            .collect();
        Ok(Self { http, urls })
    }

    /// The configured replica base URLs.
    pub fn urls(&self) -> &[String] {
        &self.urls
    }

    /// Register/upsert a worker on every replica (`POST /workers`). Each replica
    /// owns its own catalog, so registration must fan out to all of them.
    pub async fn upsert_worker(&self, req: &WorkerRegistration) -> Result<()> {
        self.fan_out("POST /workers", |url| {
            let url = format!("{url}/workers");
            self.http.post(url).json(req)
        })
        .await
    }

    /// Patch a worker on every replica (`PATCH /workers/{id}`).
    pub async fn patch_worker(&self, worker_id: u64, patch: &WorkerPatch) -> Result<()> {
        self.fan_out("PATCH /workers/{id}", |url| {
            let url = format!("{url}/workers/{worker_id}");
            self.http.patch(url).json(patch)
        })
        .await
    }

    /// Delete a worker on every replica (`DELETE /workers/{id}`). A `404` on a
    /// replica that never saw the worker is treated as success.
    pub async fn delete_worker(&self, worker_id: u64) -> Result<()> {
        for url in &self.urls {
            let resp = self
                .http
                .delete(format!("{url}/workers/{worker_id}"))
                .send()
                .await
                .with_context(|| format!("DELETE /workers/{worker_id} to {url}"))?;
            if resp.status() == reqwest::StatusCode::NOT_FOUND {
                continue;
            }
            ensure_success(resp, "DELETE /workers/{id}", url).await?;
        }
        Ok(())
    }

    /// Query-only selection against a single replica (`POST /select`).
    pub async fn select(&self, req: &SelectRequest) -> Result<SelectResponse> {
        let url = self.pick_url();
        let resp = self
            .http
            .post(format!("{url}/select"))
            .json(req)
            .send()
            .await
            .with_context(|| format!("POST /select to {url}"))?;
        let resp = ensure_success(resp, "POST /select", url).await?;
        resp.json::<SelectResponse>()
            .await
            .context("decoding /select response")
    }

    /// Returns `true` if the given replica reports `ready: true` (`GET /ready`).
    pub async fn ready(&self, url: &str) -> Result<bool> {
        let resp = self
            .http
            .get(format!("{url}/ready"))
            .send()
            .await
            .with_context(|| format!("GET /ready to {url}"))?;
        if !resp.status().is_success() {
            return Ok(false);
        }
        Ok(resp
            .json::<ReadyResponse>()
            .await
            .map(|r| r.ready)
            .unwrap_or(false))
    }

    /// Returns `true` if any configured replica reports ready.
    pub async fn any_ready(&self) -> bool {
        for url in &self.urls {
            if self.ready(url).await.unwrap_or(false) {
                return true;
            }
        }
        false
    }

    fn pick_url(&self) -> &str {
        // V1: a single read target is sufficient; all replicas share an
        // eventually-consistent catalog/index. First URL keeps it deterministic.
        &self.urls[0]
    }

    async fn fan_out(
        &self,
        op: &str,
        build: impl Fn(&str) -> reqwest::RequestBuilder,
    ) -> Result<()> {
        for url in &self.urls {
            let resp = build(url)
                .send()
                .await
                .with_context(|| format!("{op} to {url}"))?;
            ensure_success(resp, op, url).await?;
        }
        Ok(())
    }
}

async fn ensure_success(resp: reqwest::Response, op: &str, url: &str) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    bail!("{op} to {url} failed with status {status}: {body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_registration_serializes_expected_fields() {
        let mut kv = HashMap::new();
        kv.insert(0u32, "tcp://10.0.0.1:5557".to_string());
        let req = WorkerRegistration {
            worker_id: 42,
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            endpoint: "http://10.0.0.1:8000".to_string(),
            block_size: 16,
            data_parallel_size: 1,
            kv_events_endpoints: kv,
            replay_endpoint: None,
            total_kv_blocks: None,
            max_num_batched_tokens: None,
            stable_routing_id: Some("vllm-0".to_string()),
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["worker_id"], 42);
        assert_eq!(v["block_size"], 16);
        assert_eq!(v["kv_events_endpoints"]["0"], "tcp://10.0.0.1:5557");
        assert_eq!(v["stable_routing_id"], "vllm-0");
        // None fields are omitted entirely.
        assert!(v.get("replay_endpoint").is_none());
        assert!(v.get("total_kv_blocks").is_none());
    }

    #[test]
    fn select_request_sends_flat_prompt() {
        let req = SelectRequest {
            model_name: "m".to_string(),
            selection_id: Some("s1".to_string()),
            token_ids: vec![1, 2, 3],
            allowed_worker_ids: Some(HashSet::from([7u64])),
            priority_jump: None,
            strict_priority: None,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["token_ids"], serde_json::json!([1, 2, 3]));
        assert_eq!(v["selection_id"], "s1");
        assert_eq!(v["allowed_worker_ids"], serde_json::json!([7]));
        assert!(v.get("priority_jump").is_none());
    }

    #[test]
    fn select_response_deserializes() {
        let body = serde_json::json!({
            "model_name": "m",
            "tenant_id": "default",
            "worker_id": 9,
            "dp_rank": 0,
            "endpoint": "http://10.0.0.1:8000",
            "block_size": 16,
            "overlap": {"longest_matched": 32, "gpu": 16, "cpu": 0, "disk": 0},
            "effective_prefill_tokens": 64
        });
        let resp: SelectResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.worker_id, 9);
        assert_eq!(resp.dp_rank, 0);
        assert_eq!(resp.endpoint, "http://10.0.0.1:8000");
        assert_eq!(resp.effective_prefill_tokens, 64);
        assert_eq!(resp.overlap.longest_matched, 32);
    }

    #[test]
    fn new_rejects_empty_urls() {
        assert!(SelectorClient::new(vec![]).is_err());
    }

    #[test]
    fn new_trims_trailing_slash() {
        let c = SelectorClient::new(vec!["http://a:8092/".to_string()]).unwrap();
        assert_eq!(c.urls(), &["http://a:8092".to_string()]);
    }
}
