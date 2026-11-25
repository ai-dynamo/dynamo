// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One rank's fault-tolerance API, behind [`EngineFt`] so the controller can
//! be tested without HTTP and hosted elsewhere. [`HttpEngine`] talks to a
//! vLLM front end (the Python front end in vLLM 0.34.0; the Rust front end
//! serves the same routes).

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use reqwest::{Client, Url};
use tokio::sync::OnceCell;

use crate::protocol::{ApplyRequest, EngineReport, ModelList, StatusResponse, wake_request};

/// Why an order did not get a 2xx answer.
#[derive(Debug, thiserror::Error)]
pub enum ApplyError {
    /// The front end answered 4xx: it understood and declined the order.
    #[error("refused with HTTP {status}: {body}")]
    Refused { status: u16, body: String },
    /// No usable answer (transport error, timeout or 5xx). The order may or
    /// may not have arrived; it is never resent.
    #[error("no answer: {0}")]
    NoAnswer(String),
}

/// One rank's fault-tolerance API.
#[async_trait]
pub trait EngineFt: Send + Sync {
    /// One status poll, reduced to the front end's single engine.
    async fn status(&self) -> Result<EngineReport>;
    /// Sends one recovery order. Callers never call this twice for a round,
    /// and implementations must not retry it.
    async fn apply(&self, order: &ApplyRequest) -> Result<(), ApplyError>;
    /// Sends one tiny request so an idle engine runs a step and notices a dead
    /// peer. Its answer does not matter.
    async fn wake(&self) -> Result<()>;
}

/// The HTTP client the controller uses: direct connections only. Proxy
/// settings in the environment (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`)
/// are ignored, so a proxy meant for other traffic never sits between the
/// controller and the ranks.
pub fn http_client() -> reqwest::Result<Client> {
    Client::builder().no_proxy().build()
}

/// [`EngineFt`] over HTTP.
#[derive(Debug)]
pub struct HttpEngine {
    client: Client,
    status_url: Url,
    apply_url: Url,
    models_url: Url,
    completions_url: Url,
    call_timeout: Duration,
    wake_timeout: Duration,
    /// The served model, fetched on the first wake.
    model: OnceCell<String>,
}

/// Longest error body kept from a refused order.
const MAX_BODY: usize = 512;

impl HttpEngine {
    /// `base` is the front end's base URL, such as `http://10.0.0.5:8000`.
    pub fn new(
        client: Client,
        base: &Url,
        call_timeout: Duration,
        wake_timeout: Duration,
    ) -> Result<Self> {
        let mut base = base.clone();
        if !base.path().ends_with('/') {
            let path = format!("{}/", base.path());
            base.set_path(&path);
        }
        let join = |path: &str| {
            base.join(path)
                .with_context(|| format!("cannot join {path} to {base}"))
        };
        Ok(Self {
            status_url: join("v1/fault_tolerance/status")?,
            apply_url: join("v1/fault_tolerance/apply")?,
            models_url: join("v1/models")?,
            completions_url: join("v1/completions")?,
            client,
            call_timeout,
            wake_timeout,
            model: OnceCell::new(),
        })
    }

    async fn served_model(&self) -> Result<String> {
        let response = self
            .client
            .get(self.models_url.clone())
            .timeout(self.call_timeout)
            .send()
            .await
            .with_context(|| format!("GET {}", self.models_url))?
            .error_for_status()
            .with_context(|| format!("GET {}", self.models_url))?;
        let models: ModelList = response
            .json()
            .await
            .with_context(|| format!("GET {}: bad body", self.models_url))?;
        models
            .data
            .into_iter()
            .next()
            .map(|model| model.id)
            .ok_or_else(|| anyhow!("{} lists no model", self.models_url))
    }
}

#[async_trait]
impl EngineFt for HttpEngine {
    async fn status(&self) -> Result<EngineReport> {
        let response = self
            .client
            .get(self.status_url.clone())
            .timeout(self.call_timeout)
            .send()
            .await?;
        let code = response.status();
        if !code.is_success() {
            bail!("HTTP {code}");
        }
        let body: StatusResponse = response.json().await.context("bad status body")?;
        body.into_report().map_err(|error| anyhow!(error))
    }

    async fn apply(&self, order: &ApplyRequest) -> Result<(), ApplyError> {
        let response = self
            .client
            .post(self.apply_url.clone())
            .timeout(self.call_timeout)
            .json(order)
            .send()
            .await
            .map_err(|error| ApplyError::NoAnswer(error.to_string()))?;
        let code = response.status();
        if code.is_success() {
            return Ok(());
        }
        let mut body = response
            .text()
            .await
            .unwrap_or_else(|error| format!("<unreadable body: {error}>"));
        if body.len() > MAX_BODY {
            let mut end = MAX_BODY;
            while !body.is_char_boundary(end) {
                end -= 1;
            }
            body.truncate(end);
        }
        if code.is_client_error() {
            Err(ApplyError::Refused {
                status: code.as_u16(),
                body,
            })
        } else {
            Err(ApplyError::NoAnswer(format!("HTTP {code}: {body}")))
        }
    }

    async fn wake(&self) -> Result<()> {
        let model = self.model.get_or_try_init(|| self.served_model()).await?;
        let sent = self
            .client
            .post(self.completions_url.clone())
            .timeout(self.wake_timeout)
            .json(&wake_request(model))
            .send()
            .await;
        match sent {
            Ok(response) => {
                tracing::debug!(url = %self.completions_url, status = %response.status(), "wake answered");
                Ok(())
            }
            Err(error) if error.is_timeout() => {
                // Expected: an engine waiting on a dead peer does not answer.
                tracing::debug!(url = %self.completions_url, "wake timed out");
                Ok(())
            }
            Err(error) => Err(error).with_context(|| format!("POST {}", self.completions_url)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_keep_the_base_path() {
        let engine = HttpEngine::new(
            Client::new(),
            &"http://10.0.0.5:8000".parse().unwrap(),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            engine.status_url.as_str(),
            "http://10.0.0.5:8000/v1/fault_tolerance/status"
        );
        let prefixed = HttpEngine::new(
            Client::new(),
            &"http://gw.local/rank-1".parse().unwrap(),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            prefixed.apply_url.as_str(),
            "http://gw.local/rank-1/v1/fault_tolerance/apply"
        );
        assert_eq!(
            prefixed.completions_url.as_str(),
            "http://gw.local/rank-1/v1/completions"
        );
    }
}
