// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_nats::jetstream;
use async_trait::async_trait;
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
    pub fn new(nats_client: &dynamo_runtime::transports::nats::Client) -> Self {
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

fn parse_sinks_from_env(
    nats_client: Option<&dynamo_runtime::transports::nats::Client>,
) -> Vec<Arc<dyn AuditSink>> {
    let cfg = std::env::var("DYN_AUDIT_SINKS").unwrap_or_else(|_| "stderr".into());
    let mut out: Vec<Arc<dyn AuditSink>> = Vec::new();
    for name in cfg.split(',').map(|s| s.trim().to_lowercase()) {
        match name.as_str() {
            "stderr" | "" => out.push(Arc::new(StderrSink)),
            "nats" => {
                if let Some(client) = nats_client {
                    out.push(Arc::new(NatsSink::new(client)));
                } else {
                    tracing::warn!(
                        "NATS sink requested but no DistributedRuntime NATS client available; skipping"
                    );
                }
            }
            // "pg"   => out.push(Arc::new(PostgresSink::from_env())),
            other => tracing::warn!(%other, "audit: unknown sink ignored"),
        }
    }
    out
}

/// spawn one worker per sink; each subscribes to the bus (off hot path)
pub fn spawn_workers_from_env(drt: Option<&dynamo_runtime::DistributedRuntime>) {
    let nats_client = drt.and_then(|d| d.nats_client());
    let sinks = parse_sinks_from_env(nats_client);
    for sink in sinks {
        let name = sink.name();
        let mut rx: broadcast::Receiver<Arc<AuditRecord>> = bus::subscribe();
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
}
