// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for the standalone selection service (`python -m dynamo.select_service`).
//!
//! Production/replicated path: the EPP delegates worker selection to one or more
//! selection-service replicas over HTTP. The wire types live in
//! [`crate::selection_backend`]; this module only adds the HTTP transport and a
//! [`SelectionBackend`] implementation. Built only with the `selector-http`
//! feature.
//!
//! V1 (aggregated, query-only) uses:
//! - `POST/PATCH/DELETE/GET /workers` to reconcile the worker catalog,
//! - `POST /select` to obtain a worker + dp_rank for each request, and
//! - `GET /ready` for readiness gating.

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::selection_backend::{
    SelectRequest, SelectResponse, SelectionBackend, WorkerPatch, WorkerRegistration,
};

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

/// HTTP-backed [`SelectionBackend`] (replicated/production mode).
pub struct HttpSelectionBackend {
    client: SelectorClient,
}

impl HttpSelectionBackend {
    /// Build from selection-service replica base URLs.
    pub fn new(urls: Vec<String>) -> Result<Self> {
        Ok(Self {
            client: SelectorClient::new(urls)?,
        })
    }
}

#[tonic::async_trait]
impl SelectionBackend for HttpSelectionBackend {
    async fn upsert_worker(&self, reg: &WorkerRegistration) -> Result<()> {
        self.client.upsert_worker(reg).await
    }

    async fn delete_worker(&self, worker_id: u64) -> Result<()> {
        self.client.delete_worker(worker_id).await
    }

    async fn select(&self, req: &SelectRequest) -> Result<SelectResponse> {
        self.client.select(req).await
    }

    async fn any_ready(&self) -> bool {
        self.client.any_ready().await
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
    fn new_rejects_empty_urls() {
        assert!(SelectorClient::new(vec![]).is_err());
    }

    #[test]
    fn new_trims_trailing_slash() {
        let c = SelectorClient::new(vec!["http://a:8092/".to_string()]).unwrap();
        assert_eq!(c.urls(), &["http://a:8092".to_string()]);
    }
}
