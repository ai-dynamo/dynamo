// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::transports::nats;
use std::sync::Arc;
use tokio::sync::broadcast;

use super::{bus, handle::AuditRecord};

#[async_trait]
pub trait AuditSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, rec: &AuditRecord);
}

pub struct StderrSink;
#[async_trait]
impl AuditSink for StderrSink {
    fn name(&self) -> &'static str {
        "stderr"
    }
    async fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_string(rec) {
            Ok(js) => {
                tracing::info!(target="dynamo_llm::audit", log_type="audit", record=%js, "audit")
            }
            Err(e) => tracing::warn!("audit: serialize failed: {e}"),
        }
    }
}

pub struct NatsSink {
    js: jetstream::Context,
    subject: String,
}

impl NatsSink {
    pub fn new(nats_client: dynamo_runtime::transports::nats::Client) -> Self {
        let subject = std::env::var("DYN_AUDIT_NATS_SUBJECT")
            .unwrap_or_else(|_| "dynamo.audit.v1".to_string());
        Self {
            js: nats_client.jetstream().clone(),
            subject,
        }
    }
}

#[async_trait]
impl AuditSink for NatsSink {
    fn name(&self) -> &'static str {
        "nats"
    }

    async fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_vec(rec) {
            Ok(bytes) => {
                if let Err(e) = self.js.publish(self.subject.clone(), bytes.into()).await {
                    tracing::warn!("nats: publish failed: {e}");
                }
            }
            Err(e) => tracing::warn!("nats: serialize failed: {e}"),
        }
    }
}

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn AuditSink>>> {
    let cfg = std::env::var("DYN_AUDIT_SINKS").unwrap_or_else(|_| "stderr".into());
    let mut out: Vec<Arc<dyn AuditSink>> = Vec::new();
    for name in cfg.split(',').map(|s| s.trim().to_lowercase()) {
        match name.as_str() {
            "stderr" | "" => out.push(Arc::new(StderrSink)),
            "nats" => {
                let nats_client = nats::ClientOptions::default()
                    .connect()
                    .await
                    .context("Attempting to connect NATS sink from env var DYN_AUDIT_SINKS")?;
                out.push(Arc::new(NatsSink::new(nats_client)));
            }
            // "pg"   => out.push(Arc::new(PostgresSink::from_env())),
            other => tracing::warn!(%other, "audit: unknown sink ignored"),
        }
    }
    Ok(out)
}

/// spawn one worker per sink; each subscribes to the bus (off hot path)
pub async fn spawn_workers_from_env() -> anyhow::Result<()> {
    let sinks = parse_sinks_from_env().await?;
    for sink in sinks {
        let name = sink.name();
        let mut rx: broadcast::Receiver<AuditRecord> = bus::subscribe();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(rec) => sink.emit(&rec).await,
                    Err(broadcast::error::RecvError::Lagged(n)) => tracing::warn!(
                        sink = name,
                        dropped = n,
                        "audit bus lagged; dropped records"
                    ),
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }
    tracing::info!("Audit sinks ready.");
    Ok(())
}
