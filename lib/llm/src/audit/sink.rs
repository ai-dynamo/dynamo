// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use tokio::sync::broadcast;

use super::{bus, handle::AuditRecord};

pub trait AuditSink: Send + Sync {
    fn name(&self) -> &'static str;
    fn emit(&self, rec: &AuditRecord);
}

pub struct StderrSink;
impl AuditSink for StderrSink {
    fn name(&self) -> &'static str {
        "stderr"
    }
    fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_string(rec) {
            Ok(js) => {
                tracing::info!(target="dynamo_llm::audit", log_type="audit", record=%js, "audit")
            }
            Err(e) => tracing::warn!("audit: serialize failed: {e}"),
        }
    }
}

fn parse_sinks_from_env() -> Vec<Arc<dyn AuditSink>> {
    let cfg = std::env::var("DYN_AUDIT_SINKS").unwrap_or_else(|_| "stderr".into());
    let mut out: Vec<Arc<dyn AuditSink>> = Vec::new();
    for name in cfg.split(',').map(|s| s.trim().to_lowercase()) {
        match name.as_str() {
            "stderr" | "" => out.push(Arc::new(StderrSink)),
            // "nats" => out.push(Arc::new(NatsSink::from_env())),
            // "pg"   => out.push(Arc::new(PostgresSink::from_env())),
            other => tracing::warn!(%other, "audit: unknown sink ignored"),
        }
    }
    out
}

/// spawn one worker per sink; each subscribes to the bus (off hot path)
pub fn spawn_workers_from_env() {
    let sinks = parse_sinks_from_env();
    for sink in sinks {
        let name = sink.name();
        let mut rx: broadcast::Receiver<Arc<AuditRecord>> = bus::subscribe();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(rec) => sink.emit(&rec),
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
